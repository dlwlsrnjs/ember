"""RouteFuse: direction-aware information fusion (EMBER paper §3.2, Eq. 2-3).

HTLNet's task-dependence chain assumes reliable field representations, which
fails on cold rows: a sparse cold-ID field embedding stays near its
initialization, yet the chain still transfers it into the long-view and
watch-time heads. RouteFuse estimates a per-sample reliability for each side
from the (z-normalized) mean embedding norm and routes cross-field attention
*from the dense warm side into the sparse cold side*.

Given the shared field embeddings grouped into user / item / context blocks
``U, V, C`` (each ``(B, n_*, d)``):

    rho_u, rho_i = z-norm( mean ||U|| ),  z-norm( mean ||V|| )      # reliability
    tau          = softplus(eta) + 1e-3                             # learnable temp
    g_{u->i}     = sigmoid( (rho_u - rho_i) / tau )
    g_{i->u}     = sigmoid( (rho_i - rho_u) / tau )

    H_{i<-u}     = g_{u->i} * MHA(V, U, U)      # warm user context -> cold item
    H_{u<-i}     = g_{i->u} * MHA(U, V, V)      # warm item context -> cold user
    E_DA         = concat( U + H_{u<-i}, V + H_{i<-u}, C )

The asymmetric gate recovers the augmentation direction without an explicit
flag (the paper reports ``g_{u->i}=0.70`` vs. ``g_{i->u}=0.31`` on cold-item
rows), and the norm proxy correlates with log interaction frequency.

``E_DA`` is flattened back to ``(B, n_fields * d)`` and consumed by the task
towers exactly like HTLNet's shared embedding output, so the rest of the
learner is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _znorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Z-normalize a ``(B,)`` reliability score across the batch."""
    mu = x.mean()
    sd = x.std(unbiased=False) + eps
    return (x - mu) / sd


class RouteFuse(nn.Module):
    """Direction-aware cross-field fusion over user / item / context blocks.

    Args:
      emb_dim:    per-field embedding dim ``d``.
      n_user:     number of user-side fields (block ``U``).
      n_item:     number of item-side fields (block ``V``).
      n_ctx:      number of context fields (block ``C``); these are passed
                  through untouched.
      n_heads:    attention heads for the cross-field MHA.
    """

    def __init__(
        self,
        emb_dim: int,
        n_user: int,
        n_item: int,
        n_ctx: int,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_user = n_user
        self.n_item = n_item
        self.n_ctx = n_ctx

        # Cross-field multi-head attention, one per routing direction.
        self.mha_u_to_i = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.mha_i_to_u = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)

        # Learnable gate temperature tau = softplus(eta) + 1e-3.
        self.eta = nn.Parameter(torch.zeros(()))

        # Indices of the sparse ID fields used as the cold proxy. By HTLNet
        # convention field 0 is user_id and field ``n_user`` is item/video_id;
        # their embedding norm is the per-sample reliability signal.
        self.user_id_field = 0
        self.item_id_field = n_user

        # Exposed for diagnostics (paper reports mean gate values per split).
        self.register_buffer("_last_g_u_to_i", torch.zeros(()), persistent=False)
        self.register_buffer("_last_g_i_to_u", torch.zeros(()), persistent=False)

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.eta) + 1e-3

    def forward(self, field_emb: torch.Tensor) -> torch.Tensor:
        """``(B, n_fields, d)`` -> fused ``(B, n_fields * d)``.

        Field order is assumed to be ``[user fields | item fields | context
        fields]`` (the HTLNet KuaiRand layout), matching ``n_user``,
        ``n_item``, ``n_ctx``.
        """
        b = field_emb.shape[0]
        u = field_emb[:, : self.n_user, :]
        v = field_emb[:, self.n_user : self.n_user + self.n_item, :]
        c = field_emb[:, self.n_user + self.n_item :, :]

        # Per-sample reliability from the sparse ID-field embedding norm.
        rho_u = _znorm(field_emb[:, self.user_id_field, :].norm(dim=-1))
        rho_i = _znorm(field_emb[:, self.item_id_field, :].norm(dim=-1))

        tau = self.tau
        g_u_to_i = torch.sigmoid((rho_u - rho_i) / tau).view(b, 1, 1)  # warm-u -> cold-i
        g_i_to_u = torch.sigmoid((rho_i - rho_u) / tau).view(b, 1, 1)  # warm-i -> cold-u

        # Route warm-side context into the cold side (Eq. 3).
        h_i_from_u, _ = self.mha_u_to_i(v, u, u)  # query item, attend over user
        h_u_from_i, _ = self.mha_i_to_u(u, v, v)  # query user, attend over item
        v_fused = v + g_u_to_i * h_i_from_u
        u_fused = u + g_i_to_u * h_u_from_i

        if c.shape[1] > 0:
            fused = torch.cat([u_fused, v_fused, c], dim=1)
        else:
            fused = torch.cat([u_fused, v_fused], dim=1)

        with torch.no_grad():
            self._last_g_u_to_i = g_u_to_i.mean().detach()
            self._last_g_i_to_u = g_i_to_u.mean().detach()

        return fused.reshape(b, -1)

    def gate_diagnostics(self) -> dict:
        """Mean routing-gate values from the most recent forward pass."""
        return {
            "g_u_to_i": float(self._last_g_u_to_i),
            "g_i_to_u": float(self._last_g_i_to_u),
            "tau": float(self.tau.detach()),
        }
