"""HTLNet backbone blocks, reimplemented.

"Touch the Core: Exploring Task Dependence Among Hybrid Targets for
Recommendation" (Tang et al., RecSys 2024). EMBER instantiates its cold-aware
learner on this backbone, so these are the pieces RouteFuse and CaliChain graft
onto:

  * :class:`SharedFieldEmbedding` -- one table per categorical field, returned
    un-concatenated so RouteFuse can slice user / item / context blocks.
  * :class:`TaskTower` -- MLP that also exposes its first hidden state, which is
    what the Information Fusion Unit transfers to later tasks.
  * :class:`LabelEmbeddingUnit` -- turns a predicted probability into the
    *differentiable label state* the next task consumes.
  * :class:`InformationFusionUnit` -- attention fusion over preceding-task
    representations / label states.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedFieldEmbedding(nn.Module):
    """``(B, n_fields)`` int codes -> ``(B, n_fields, emb_dim)``.

    Code ``0`` is padding and is pinned to the zero vector, so an unknown /
    absent categorical value contributes nothing rather than a random vector.
    """

    def __init__(self, vocab_sizes: Sequence[int], emb_dim: int = 10) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_fields = len(vocab_sizes)
        self.embs = nn.ModuleList(
            [nn.Embedding(int(v) + 1, emb_dim, padding_idx=0) for v in vocab_sizes]
        )
        for e in self.embs:
            nn.init.xavier_uniform_(e.weight)
            with torch.no_grad():
                e.weight[0].zero_()
        self.out_dim = self.n_fields * emb_dim

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([e(feature_ids[:, i]) for i, e in enumerate(self.embs)], dim=1)


class TaskTower(nn.Module):
    """MLP returning ``(head_output, first_hidden)``.

    The first hidden state is returned separately because HTLNet transfers it
    (not the logit) to downstream tasks through the IFU.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.first_hidden_dim = hidden_dims[0]
        blocks: List[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layer: List[nn.Module] = [nn.Linear(cur, h), nn.ReLU()]
            if dropout > 0:
                layer.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*layer))
            cur = h
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(cur, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks[0](x)
        first = h
        for blk in self.blocks[1:]:
            h = blk(h)
        return self.head(h).squeeze(-1), first


class LabelEmbeddingUnit(nn.Module):
    """Differentiable label state ``l^(t)`` from a predicted probability.

    A 2-row label table read out with a temperature-scaled softmax over
    ``[log p, log(1-p)] / tau``. Annealing ``tau`` down sharpens the read-out
    from a soft mixture toward a hard label look-up as training proceeds.
    """

    def __init__(
        self,
        label_emb_dim: int = 10,
        tau_init: float = 10.0,
        tau_min: float = 0.5,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(2, label_emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.register_buffer("tau", torch.tensor(float(tau_init)))
        self.tau_min = float(tau_min)
        self.label_emb_dim = label_emb_dim

    def set_tau(self, value: float) -> None:
        self.tau.fill_(max(float(value), self.tau_min))

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        p = prob.clamp(1e-7, 1 - 1e-7)
        tau = self.tau.clamp(min=self.tau_min)
        logits = torch.stack([torch.log(p) / tau, torch.log1p(-p) / tau], dim=-1)
        return F.softmax(logits, dim=-1) @ self.emb.weight


class InformationFusionUnit(nn.Module):
    """Attention fusion over a list of preceding-task vectors (HTLNet Eq. 13/14).

    With a single input this degenerates to a linear projection, which is the
    intended behaviour for the first dependent task.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(in_dim, hidden_dim)
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, items: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(items) == 1:
            return self.value(items[0])
        u = torch.stack(list(items), dim=1)                     # (B, n_prev, in_dim)
        attn = (self.query(u) * self.key(u)).sum(-1) / self.scale  # (B, n_prev)
        w = F.softmax(attn, dim=-1).unsqueeze(-1)
        return (w * self.value(u)).sum(1)
