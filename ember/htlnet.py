"""Base HTLNet building blocks (Tang et al., RecSys 2024).

"Touch the Core: Exploring Task Dependence Among Hybrid Targets for
Recommendation" (https://doi.org/10.1145/3640457.3688101).

EMBER instantiates its cold-aware learner on top of HTLNet, so we vendor the
original blocks here unchanged:

  * ``SharedFieldEmbedding`` — one ``nn.Embedding`` per categorical field.
  * ``TaskTower`` — 3-layer MLP that also exposes its first-hidden state for
    the Information Fusion Unit.
  * ``LabelEmbeddingUnit`` (LEU) — a 2-row label table read out with a
    temperature-scaled softmax over the predicted probability.
  * ``InformationFusionUnit`` (IFU) — attention fusion of preceding-task
    representations / label states.

The novel EMBER components (RouteFuse, CaliChain) live in their own modules
and consume / wrap these blocks; see :mod:`ember.routefuse`,
:mod:`ember.calichain`, and :mod:`ember.learner`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedFieldEmbedding(nn.Module):
    """One ``nn.Embedding`` per categorical field.

    Unlike the original ``SharedEmbedding`` (which immediately concatenates),
    this returns the per-field embeddings as a ``(B, n_fields, emb_dim)``
    tensor so RouteFuse can slice them into user / item / context blocks.
    """

    def __init__(self, field_vocab_sizes: List[int], emb_dim: int = 10) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_fields = len(field_vocab_sizes)
        self.embs = nn.ModuleList(
            [nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0) for vs in field_vocab_sizes]
        )
        for e in self.embs:
            nn.init.xavier_uniform_(e.weight)
            with torch.no_grad():
                e.weight[0].zero_()
        self.out_dim = self.n_fields * emb_dim

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        """``(B, n_fields)`` int ids -> ``(B, n_fields, emb_dim)``."""
        parts = [e(feature_ids[:, i]) for i, e in enumerate(self.embs)]
        return torch.stack(parts, dim=1)


class TaskTower(nn.Module):
    """3-layer MLP ``[128, 64, 32] -> 1`` returning ``(logit, first_hidden)``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        cur = in_dim
        self.first_hidden_dim = hidden_dims[0]
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = h
        self.mlp_body = nn.Sequential(*layers)
        self.first_layer = nn.Sequential(layers[0], layers[1])
        self.out_head = nn.Linear(cur, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.first_layer(x)
        h_full = h1
        for layer in self.mlp_body[2:]:
            h_full = layer(h_full)
        return self.out_head(h_full), h1


class LabelEmbeddingUnit(nn.Module):
    """Per-task 2-row label embedding read out with a temperature softmax."""

    def __init__(
        self,
        label_emb_dim: int = 10,
        init_temperature: float = 10.0,
        min_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(2, label_emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.register_buffer("temperature", torch.tensor(init_temperature, dtype=torch.float32))
        self.min_temperature = min_temperature
        self.label_emb_dim = label_emb_dim

    def set_temperature(self, value: float) -> None:
        self.temperature.fill_(max(value, self.min_temperature))

    def forward(self, predicted_prob: torch.Tensor) -> torch.Tensor:
        p = predicted_prob.clamp(min=1e-7, max=1 - 1e-7)
        log_p = torch.log(p)
        log_1mp = torch.log(1 - p)
        tau = self.temperature.clamp(min=self.min_temperature)
        logits = torch.stack([log_p / tau, log_1mp / tau], dim=-1)
        weights = F.softmax(logits, dim=-1)
        return weights @ self.emb.weight


class InformationFusionUnit(nn.Module):
    """Attention fusion of a list of preceding-task vectors (HTLNet Eq. 13/14)."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.h1 = nn.Linear(in_dim, hidden_dim)
        self.h2 = nn.Linear(in_dim, hidden_dim)
        self.h3 = nn.Linear(in_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, items: List[torch.Tensor]) -> torch.Tensor:
        if len(items) == 1:
            return self.h1(items[0])
        u = torch.stack(items, dim=1)
        h2 = self.h2(u)
        h3 = self.h3(u)
        logits = (h2 * h3).sum(dim=-1) / self.scale
        weights = F.softmax(logits, dim=-1)
        h1_u = self.h1(u)
        return (weights.unsqueeze(-1) * h1_u).sum(dim=1)
