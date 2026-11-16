"""Support-gated ID embedding synthesis -- what to put where the cold ID isn't.

RouteFuse's answer to a cold entity is to route *around* it: detect that the ID
embedding is unreliable and pull context from the warm side. Measured on the
calibrated harness, replacing its detection signal (embedding norm -> warm
support count) is worth more than everything else tried here. But detection is
only half the problem. A cold ID's embedding is still whatever initialization
left there -- noise the towers must learn to ignore rather than a representation
of the entity.

This module fills the slot instead of routing around it. A small generator maps
the entity's own *content* fields into ID-embedding space, and the two are
blended by how much logged support the ID actually has::

    w      = sigmoid(a * log1p(support) + b)          # a, b learnable, per side
    e_id   = w * E[id] + (1 - w) * gen(content fields)

An entity with thousands of warm rows keeps its learned ID embedding (w -> 1);
one with none is described entirely by its content (w -> 0), through a generator
trained on entities where both are available. Nothing here is a test-time
quantity: support counts come from D_warm, exactly like the reliability signal.

**Why this needs pretend-cold rows.** The blend weight is a function of support,
so on warm rows w is near 1 and the generator receives almost no gradient -- it
would be trained only on the rows where it is never used. ``pretend_cold``
forces support to zero for a fraction of *training* rows (one side per row), so
the generator learns on entities whose labels we actually have. This is the same
mechanism as ``id_dropout``, aimed at a generator instead of at the towers: not
"learn to cope without the ID" but "learn what the ID would have been".
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


class ColdIdSynthesis(nn.Module):
    """Blend each side's ID embedding with one generated from its content."""

    def __init__(
        self,
        emb_dim: int,
        n_user: int,
        n_item: int,
        user_id_fields: Sequence[int],
        item_id_fields: Sequence[int],
        hidden: int = 64,
        pretend_cold: float = 0.0,
    ) -> None:
        super().__init__()
        self.user_id_fields = tuple(user_id_fields)
        self.item_id_fields = tuple(item_id_fields)
        self.user_attr_fields = tuple(j for j in range(n_user) if j not in self.user_id_fields)
        self.item_attr_fields = tuple(
            j for j in range(n_user, n_user + n_item) if j not in self.item_id_fields
        )
        if not self.user_attr_fields or not self.item_attr_fields:
            raise ValueError("synthesis needs at least one content field per side")
        self.pretend_cold = float(pretend_cold)

        def gen(n_attr: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(n_attr * emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, emb_dim)
            )

        self.gen_user = gen(len(self.user_attr_fields))
        self.gen_item = gen(len(self.item_attr_fields))
        # w = sigmoid(a * log1p(support) + b); a > 0 so more support means more
        # trust in the learned ID. Initialized so a warm entity (log1p ~ 5) sits
        # near w = 0.9 and a cold one (log1p = 0) near w = 0.1.
        self.a = nn.Parameter(torch.full((2,), 0.9))
        self.b = nn.Parameter(torch.full((2,), -2.2))

    def _blend_weight(self, support: torch.Tensor, side: int) -> torch.Tensor:
        a = self.a[side].clamp_min(0.0)
        return torch.sigmoid(a * torch.log1p(support.clamp_min(0)) + self.b[side])

    def forward(
        self,
        field_emb: torch.Tensor,
        support_u: torch.Tensor,
        support_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, F, d)`` -> field embeddings with synthesized IDs, plus the
        (possibly zeroed) support each side was blended with.

        The returned support is what downstream consumers must use: a
        pretend-cold row has to look cold to RouteFuse as well, or the gate is
        told the row is warm while its ID embedding has been replaced.
        """
        b, _, d = field_emb.shape
        out = field_emb.clone()

        if self.training and self.pretend_cold > 0:
            hit = torch.rand(b, device=field_emb.device) < self.pretend_cold
            fake_u = hit & (torch.rand(b, device=field_emb.device) < 0.5)
            fake_i = hit & ~fake_u
            support_u = torch.where(fake_u, torch.zeros_like(support_u), support_u)
            support_i = torch.where(fake_i, torch.zeros_like(support_i), support_i)

        for side, (attr_fields, id_fields, generator) in enumerate(
            (
                (self.user_attr_fields, self.user_id_fields, self.gen_user),
                (self.item_attr_fields, self.item_id_fields, self.gen_item),
            )
        ):
            support = support_u if side == 0 else support_i
            content = field_emb[:, list(attr_fields), :].reshape(b, -1)
            synth = generator(content)                                   # (B, d)
            w = self._blend_weight(support, side).view(b, 1)
            for j in id_fields:
                out[:, j, :] = w * field_emb[:, j, :] + (1.0 - w) * synth

        return out, support_u, support_i

    def diagnostics(self, support_u: torch.Tensor, support_i: torch.Tensor) -> dict:
        with torch.no_grad():
            return {
                "w_user": float(self._blend_weight(support_u, 0).mean()),
                "w_item": float(self._blend_weight(support_i, 1).mean()),
            }
