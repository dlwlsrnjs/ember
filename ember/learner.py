"""EmberHTLNet: the cold-aware downstream learner (EMBER paper §3.2-3.3).

HTLNet backbone with two EMBER components grafted on:

  * **RouteFuse** replaces the plain shared-embedding concatenation with a
    direction-aware fusion ``E_DA`` (warm-side context routed into the cold
    side). See :mod:`ember.routefuse`.
  * **CaliChain** blends the simulator confidence prior into each binary
    logit *before* it forms the next label state in the task-dependence
    chain (Eq. 5). The per-sample confidence loss weighting (Eq. 4) is applied
    by the training loop using the helpers in :mod:`ember.calichain`.

Task order (KuaiRand layout): ``click`` (binary) -> ``long_view`` (binary) ->
``watch_time`` (continuous core target). For Kaggle / Tenrec the two binary
heads carry the dataset-specific conversion labels and the core head carries
the continuous outcome; the structure is identical.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .htlnet import (
    InformationFusionUnit,
    LabelEmbeddingUnit,
    SharedFieldEmbedding,
    TaskTower,
)
from .calichain import ConfidencePrior
from .routefuse import RouteFuse


class EmberHTLNet(nn.Module):
    def __init__(
        self,
        field_vocab_sizes: List[int],
        n_user_fields: int,
        n_item_fields: int,
        n_ctx_fields: int,
        shared_emb_dim: int = 10,
        label_emb_dim: int = 10,
        tower_hidden_dims: Tuple[int, ...] = (128, 64, 32),
        ifu_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        n_heads: int = 2,
        use_routefuse: bool = True,
        use_confidence_prior: bool = True,
    ) -> None:
        super().__init__()
        assert (
            n_user_fields + n_item_fields + n_ctx_fields == len(field_vocab_sizes)
        ), "field block sizes must sum to the number of fields"

        self.use_routefuse = use_routefuse
        self.use_confidence_prior = use_confidence_prior

        self.shared_emb = SharedFieldEmbedding(field_vocab_sizes, shared_emb_dim)
        shared_dim = self.shared_emb.out_dim

        if use_routefuse:
            self.routefuse: Optional[RouteFuse] = RouteFuse(
                emb_dim=shared_emb_dim,
                n_user=n_user_fields,
                n_item=n_item_fields,
                n_ctx=n_ctx_fields,
                n_heads=n_heads,
            )
        else:
            self.routefuse = None

        # CaliChain confidence prior over the two binary tasks.
        self.confidence_prior = ConfidencePrior(n_tasks=2) if use_confidence_prior else None

        if ifu_hidden_dim is None:
            ifu_hidden_dim = tower_hidden_dims[0]
        self.ifu_hidden_dim = ifu_hidden_dim

        # Tower 1: click
        self.tower_click = TaskTower(shared_dim, tower_hidden_dims, dropout)
        self.leu_click = LabelEmbeddingUnit(label_emb_dim)

        # Tower 2: long_view
        self.ifu_rep_for_lv = InformationFusionUnit(tower_hidden_dims[0], ifu_hidden_dim)
        self.ifu_le_for_lv = InformationFusionUnit(label_emb_dim, ifu_hidden_dim)
        lv_in_dim = shared_dim + ifu_hidden_dim * 2
        self.tower_lv = TaskTower(lv_in_dim, tower_hidden_dims, dropout)
        self.leu_lv = LabelEmbeddingUnit(label_emb_dim)

        # Tower 3: watch_time (continuous core target, no LEU)
        self.ifu_rep_for_wt = InformationFusionUnit(tower_hidden_dims[0], ifu_hidden_dim)
        self.ifu_le_for_wt = InformationFusionUnit(label_emb_dim, ifu_hidden_dim)
        wt_in_dim = shared_dim + ifu_hidden_dim * 2
        self.tower_wt = TaskTower(wt_in_dim, tower_hidden_dims, dropout)

    def set_temperature(self, value: float) -> None:
        self.leu_click.set_temperature(value)
        self.leu_lv.set_temperature(value)

    def forward(
        self,
        feature_ids: torch.Tensor,
        q_click: Optional[torch.Tensor] = None,
        q_long_view: Optional[torch.Tensor] = None,
        aug_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.

        ``q_click`` / ``q_long_view`` are per-sample simulator confidences for
        the binary tasks (``(B,)`` in ``[0, 1]``) and ``aug_mask`` (``(B,)``
        bool) marks augmented rows. The CaliChain prior (Eq. 5) blends the
        confidence into the label-state logits *only on augmented rows*; warm
        rows keep their raw logit. At evaluation time pass nothing (no-op).
        """
        field_emb = self.shared_emb(feature_ids)  # (B, n_fields, d)
        if self.routefuse is not None:
            e = self.routefuse(field_emb)          # (B, n_fields * d)
        else:
            e = field_emb.reshape(field_emb.shape[0], -1)

        # Tower 1: click
        click_logit, click_rep = self.tower_click(e)
        click_logit = click_logit.squeeze(-1)
        # CaliChain prior blend BEFORE forming the label state (Eq. 5).
        if self.confidence_prior is not None:
            click_logit_chain = self.confidence_prior.blend(click_logit, q_click, t=0, mask=aug_mask)
        else:
            click_logit_chain = click_logit
        click_prob = torch.sigmoid(click_logit)
        click_prob_chain = torch.sigmoid(click_logit_chain)
        click_le = self.leu_click(click_prob_chain.detach())

        # Tower 2: long_view (stop-gradient on transferred info, HTLNet Eq. 15)
        click_rep_sg = click_rep.detach()
        click_le_sg = click_le.detach()
        lv_rep_in = self.ifu_rep_for_lv([click_rep_sg])
        lv_le_in = self.ifu_le_for_lv([click_le_sg])
        lv_input = torch.cat([e, lv_rep_in, lv_le_in], dim=-1)
        lv_logit, lv_rep = self.tower_lv(lv_input)
        lv_logit = lv_logit.squeeze(-1)
        if self.confidence_prior is not None:
            lv_logit_chain = self.confidence_prior.blend(lv_logit, q_long_view, t=1, mask=aug_mask)
        else:
            lv_logit_chain = lv_logit
        lv_prob = torch.sigmoid(lv_logit)
        lv_prob_chain = torch.sigmoid(lv_logit_chain)
        lv_le = self.leu_lv(lv_prob_chain.detach())

        # Tower 3: watch_time (core)
        lv_rep_sg = lv_rep.detach()
        lv_le_sg = lv_le.detach()
        wt_rep_in = self.ifu_rep_for_wt([click_rep_sg, lv_rep_sg])
        wt_le_in = self.ifu_le_for_wt([click_le_sg, lv_le_sg])
        wt_input = torch.cat([e, wt_rep_in, wt_le_in], dim=-1)
        wt_pred, wt_rep = self.tower_wt(wt_input)

        return {
            "click_logit": click_logit,
            "click_prob": click_prob,
            "long_view_logit": lv_logit,
            "long_view_prob": lv_prob,
            "watch_time_pred": wt_pred.squeeze(-1),
            "click_rep": click_rep,
            "long_view_rep": lv_rep,
            "watch_time_rep": wt_rep,
            "click_le": click_le,
            "long_view_le": lv_le,
        }

    def diagnostics(self) -> dict:
        out = {}
        if self.routefuse is not None:
            out.update(self.routefuse.gate_diagnostics())
        if self.confidence_prior is not None:
            out["alpha_t"] = self.confidence_prior.alphas()
        return out
