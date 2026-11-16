"""EmberLearner: the cold-aware downstream learner (EMBER §3.2-3.3).

HTLNet backbone -- shared field embeddings, cross-field fusion, sequential task
towers ``h^(t) = phi_t(r^(t-1), l^(t-1), E)`` -- with the two EMBER components
grafted on:

  * **RouteFuse** replaces the plain concatenation of shared embeddings with the
    direction-aware fusion ``E_DA`` (:mod:`ember2.routefuse`).
  * **CaliChain** blends the simulator confidence prior into each binary logit
    *before* it forms the next label state (:mod:`ember2.calichain`).

The chain is built from a :class:`~ember2.config.TaskSpec`, so ``T`` binary
conversion targets followed by one continuous core target works for any ``T``
(the paper uses ``T = 2`` throughout).

Gradient flow through the chain, and why it differs from the released code:

  * The *representation* transfer ``r^(t)`` is detached -- HTLNet's stop-gradient
    (Eq. 15), so a later task cannot reshape an earlier tower.
  * The *tower logit* is detached before the Eq. 5 blend, for the same reason.
  * The *label state* ``l^(t)`` is NOT detached. It is defined as the
    differentiable label state, and it is the only path by which ``alpha_t`` and
    the label-embedding tables receive gradient. Detaching it (as the released
    code does) freezes ``alpha_t`` at its initialization permanently.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .blocks import (
    InformationFusionUnit,
    LabelEmbeddingUnit,
    SharedFieldEmbedding,
    TaskTower,
)
from .calichain import ConfidencePrior
from .coldsynth import ColdIdSynthesis
from .config import FieldSpec, LearnerConfig, TaskSpec
from .routefuse import RouteFuse


class EmberLearner(nn.Module):
    def __init__(
        self,
        fields: FieldSpec,
        tasks: TaskSpec,
        cfg: Optional[LearnerConfig] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or LearnerConfig()
        self.fields, self.tasks, self.cfg = fields, tasks, cfg
        T = tasks.n_binary

        self.shared_emb = SharedFieldEmbedding(fields.vocab_sizes, cfg.emb_dim)
        shared_dim = self.shared_emb.out_dim

        self.routefuse = (
            RouteFuse(
                emb_dim=cfg.emb_dim,
                n_user=fields.n_user,
                n_item=fields.n_item,
                n_ctx=fields.n_ctx,
                user_id_fields=fields.user_id_fields,
                item_id_fields=fields.item_id_fields,
                n_heads=cfg.n_heads,
                reliability=cfg.reliability,
            )
            if cfg.use_routefuse
            else None
        )
        self.confidence_prior = (
            ConfidencePrior(T, signed=cfg.signed_confidence_prior)
            if cfg.use_confidence_prior
            else None
        )
        self.synth = (
            ColdIdSynthesis(
                emb_dim=cfg.emb_dim,
                n_user=fields.n_user,
                n_item=fields.n_item,
                user_id_fields=fields.user_id_fields,
                item_id_fields=fields.item_id_fields,
                hidden=cfg.synth_hidden,
                pretend_cold=cfg.synth_pretend_cold,
            )
            if cfg.cold_id_synthesis
            else None
        )
        # Warm support per ID code, installed by `set_support`. Both the
        # reliability signal and the synthesis blend read it, and both must read
        # the *same* numbers -- see `forward`.
        self.register_buffer("support_user", torch.zeros(0), persistent=False)
        self.register_buffer("support_item", torch.zeros(0), persistent=False)

        rep_dim = cfg.tower_hidden[0]
        ifu_dim = cfg.ifu_hidden if cfg.ifu_hidden is not None else cfg.tower_hidden[0]
        self.ifu_dim = ifu_dim

        # Binary task chain. Task 0 sees only E_DA; task t>0 additionally sees
        # the fused preceding representations and label states.
        self.towers = nn.ModuleList()
        self.leus = nn.ModuleList()
        self.ifu_rep = nn.ModuleList()
        self.ifu_le = nn.ModuleList()
        for t in range(T):
            if t == 0:
                in_dim = shared_dim
                self.ifu_rep.append(nn.Identity())
                self.ifu_le.append(nn.Identity())
            else:
                in_dim = shared_dim + 2 * ifu_dim
                self.ifu_rep.append(InformationFusionUnit(rep_dim, ifu_dim))
                self.ifu_le.append(InformationFusionUnit(cfg.label_emb_dim, ifu_dim))
            self.towers.append(TaskTower(in_dim, cfg.tower_hidden, cfg.dropout))
            self.leus.append(
                LabelEmbeddingUnit(cfg.label_emb_dim, cfg.tau_init, cfg.tau_min)
            )

        # Continuous core target: consumes every preceding task, no label state
        # of its own (nothing downstream reads it).
        self.ifu_rep_core = InformationFusionUnit(rep_dim, ifu_dim)
        self.ifu_le_core = InformationFusionUnit(cfg.label_emb_dim, ifu_dim)
        self.tower_core = TaskTower(shared_dim + 2 * ifu_dim, cfg.tower_hidden, cfg.dropout)

    # ── plumbing ────────────────────────────────────────────────────
    def set_support(self, support_user: torch.Tensor, support_item: torch.Tensor) -> None:
        """Install warm-training row counts per ID code (index 0 = padding = 0)."""
        self.support_user = support_user.float()
        self.support_item = support_item.float()
        if self.routefuse is not None:
            self.routefuse.set_support(self.support_user, self.support_item)

    @property
    def needs_support(self) -> bool:
        return self.cfg.cold_id_synthesis or self.cfg.reliability == "support"

    def _row_support(self, feature_ids: torch.Tensor):
        if self.support_user.numel() == 0:
            raise RuntimeError(
                "this configuration needs set_support(...) -- call train() or "
                "install warm support counts yourself"
            )
        u = self.support_user[feature_ids[:, list(self.fields.user_id_fields)]].mean(-1)
        i = self.support_item[feature_ids[:, list(self.fields.item_id_fields)]].mean(-1)
        return u, i

    def drop_ids(self, feature_ids: torch.Tensor) -> torch.Tensor:
        """Mask one side's sparse ID fields to the padding code on some rows.

        A cold eval row is exactly this: one side carries an ID the model never
        trained, the other side is warm, and both sides keep their metadata.
        Training only on rows where both IDs are informative never asks the
        towers to rank without one, so they never learn to. Code 0 is pinned to
        the zero vector, so a masked ID contributes nothing -- the same thing an
        untrained ID embedding approximately does, but exactly.

        The side is chosen per row, never both: dropping both would describe a
        row that does not occur at evaluation.
        """
        p = float(self.cfg.id_dropout)
        if p <= 0:
            return feature_ids
        b = feature_ids.shape[0]
        dev = feature_ids.device
        hit = torch.rand(b, device=dev) < p
        drop_user = hit & (torch.rand(b, device=dev) < 0.5)
        drop_item = hit & ~drop_user

        out = feature_ids.clone()
        for f in self.fields.user_id_fields:
            out[:, f] = torch.where(drop_user, torch.zeros_like(out[:, f]), out[:, f])
        for f in self.fields.item_id_fields:
            out[:, f] = torch.where(drop_item, torch.zeros_like(out[:, f]), out[:, f])
        return out

    def set_tau(self, value: float) -> None:
        for leu in self.leus:
            leu.set_tau(value)

    @property
    def tau(self) -> float:
        return float(self.leus[0].tau)

    def shared_parameters(self) -> List[nn.Parameter]:
        """The field embedding tables -- the parameters every task reads."""
        return list(self.shared_emb.parameters())

    def task_parameters(self) -> List[nn.Parameter]:
        shared = {id(p) for p in self.shared_parameters()}
        return [p for p in self.parameters() if id(p) not in shared]

    # ── forward ─────────────────────────────────────────────────────
    def forward(
        self,
        feature_ids: torch.Tensor,
        q: Optional[torch.Tensor] = None,
        aug_mask: Optional[torch.Tensor] = None,
        y_binary: Optional[torch.Tensor] = None,
        id_override: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the hybrid-target chain.

        Args:
          feature_ids: ``(B, n_fields)`` categorical codes.
          q:           ``(B, T+1)`` per-task simulator confidence, or ``None``.
                       Only the first ``T`` columns are read here (the core
                       target has no label state to prime); the core column is
                       used by the loss (Eq. 4).
          aug_mask:    ``(B,)`` bool marking rows from ``A_LLM``. The Eq. 5 prior
                       applies only to those; warm rows keep their raw logit.
          y_binary:    ``(B, T)`` the simulator's pseudo-labels, used to sign the
                       prior (see :class:`~ember2.calichain.ConfidencePrior`).
          id_override: ``{field index: (B, d) tensor}`` replacing that field's
                       looked-up embedding, for methods that *generate* ID
                       embeddings rather than learning them.

        At evaluation time call with ``q=None`` -- serving sees no simulator.
        """
        if self.training and self.cfg.id_dropout > 0:
            feature_ids = self.drop_ids(feature_ids)

        field_emb = self.shared_emb(feature_ids)
        if id_override:
            # Replace the embedding of specific ID fields with externally
            # supplied vectors. Embedding-generation methods need the injected
            # vector to stay in the autograd graph of *their* generator, which a
            # table lookup cannot provide.
            field_emb = field_emb.clone()
            for field, vec in id_override.items():
                field_emb[:, field, :] = vec

        # Synthesis may declare a row cold that the raw counts call warm
        # (`pretend_cold`). RouteFuse has to see the same declaration, or the
        # gate is told a row is warm whose ID embedding has just been replaced.
        support = None
        if self.needs_support:
            su, si = self._row_support(feature_ids)
            if self.synth is not None:
                field_emb, su, si = self.synth(field_emb, su, si)
                self._last_support = (su.detach(), si.detach())
            support = torch.stack([su, si], dim=-1)

        e = (
            self.routefuse(field_emb, feature_ids=feature_ids, support=support)
            if self.routefuse is not None
            else field_emb.reshape(field_emb.shape[0], -1)
        )

        logits: List[torch.Tensor] = []
        reps: List[torch.Tensor] = []
        label_states: List[torch.Tensor] = []

        for t, (tower, leu) in enumerate(zip(self.towers, self.leus)):
            if t == 0:
                x = e
            else:
                x = torch.cat(
                    [e, self.ifu_rep[t](reps), self.ifu_le[t](label_states)], dim=-1
                )
            logit, rep = tower(x)
            logits.append(logit)
            reps.append(rep.detach())          # HTLNet stop-gradient on transfer

            # Eq. 5 on a detached logit: the chain does not reshape this tower,
            # but alpha_t and the label table stay differentiable.
            o = logit.detach()
            if self.confidence_prior is not None:
                q_t = q[:, t] if q is not None else None
                y_t = y_binary[:, t] if y_binary is not None else None
                o = self.confidence_prior.blend(o, q_t, t=t, mask=aug_mask, y=y_t)
            label_states.append(leu(torch.sigmoid(o)))

        x_core = torch.cat(
            [e, self.ifu_rep_core(reps), self.ifu_le_core(label_states)], dim=-1
        )
        core_pred, core_rep = self.tower_core(x_core)

        return {
            "binary_logits": torch.stack(logits, dim=1),        # (B, T)
            "binary_probs": torch.sigmoid(torch.stack(logits, dim=1)),
            "core_pred": core_pred,                             # (B,)
            "reps": reps,
            "core_rep": core_rep,
            "label_states": label_states,
        }

    # ── diagnostics ─────────────────────────────────────────────────
    def diagnostics(self) -> dict:
        out: dict = {"tau_leu": self.tau}
        if self.routefuse is not None:
            out.update(self.routefuse.diagnostics())
        if self.synth is not None and getattr(self, "_last_support", None) is not None:
            su, si = self._last_support
            out.update({k: round(v, 4) for k, v in self.synth.diagnostics(su, si).items()})
        if self.confidence_prior is not None:
            out["alpha_t"] = [round(a, 4) for a in self.confidence_prior.alphas()]
        return out
