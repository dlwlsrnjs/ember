"""RouteFuse: direction-aware information fusion (EMBER §3.2, Eq. 2-3).

HTLNet's task chain assumes reliable field representations. On a cold row that
assumption fails: the sparse ID embedding of a new item has received (almost)
no gradient, so it sits near its initialization -- yet the chain still transfers
it into every downstream head. RouteFuse measures that unreliability and routes
cross-field attention *from the dense warm side into the sparse cold side*.

    rho_u, rho_i = znorm( mean ||E_j|| over user/item ID fields )   # reliability
    tau          = softplus(eta) + 1e-3                             # learnable
    g_{u->i}     = sigmoid( (rho_u - rho_i) / tau )                 # Eq. 2
    g_{i->u}     = sigmoid( (rho_i - rho_u) / tau )

    H_{i<-u}     = g_{u->i} * MHA(V, U, U)     # warm user context -> cold item
    H_{u<-i}     = g_{i->u} * MHA(U, V, V)     # warm item context -> cold user
    E_DA         = concat( U + H_{u<-i}, V + H_{i<-u}, C )          # Eq. 3

Two implementation notes that the equations leave open:

**Which fields feed rho.** Only the *sparse ID* fields. Content/metadata fields
are populated for cold entities too (that is the whole point of a cold entity
keeping its metadata), so their embedding norms carry no coldness signal and
would dilute rho toward noise. See :class:`ember2.config.FieldSpec`.

**How rho is normalized.** Eq. 2 z-normalizes, but does not say over what. Doing
it per batch makes the gate depend on batch composition -- an all-warm batch
re-centers to rho=0 on both sides and every gate collapses to 0.5, and eval
results would shift with shuffling. We therefore track running mean/variance
(BatchNorm-style): batch statistics while training, frozen running statistics at
eval. Pass ``batch_znorm=True`` for the literal per-batch reading.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteFuse(nn.Module):
    """Direction-aware cross-field fusion over user / item / context blocks."""

    def __init__(
        self,
        emb_dim: int,
        n_user: int,
        n_item: int,
        n_ctx: int,
        user_id_fields: Sequence[int] = (0,),
        item_id_fields: Optional[Sequence[int]] = None,
        n_heads: int = 2,
        momentum: float = 0.1,
        batch_znorm: bool = False,
        eps: float = 1e-5,
        reliability: str = "norm",
    ) -> None:
        super().__init__()
        self.n_user, self.n_item, self.n_ctx = n_user, n_item, n_ctx
        self.user_id_fields = tuple(user_id_fields)
        self.item_id_fields = tuple(item_id_fields if item_id_fields is not None else (n_user,))
        self.batch_znorm = bool(batch_znorm)
        self.momentum = float(momentum)
        self.eps = float(eps)
        if reliability not in ("norm", "support"):
            raise ValueError(f"unknown reliability signal {reliability!r}")
        self.reliability = reliability
        # Warm training support per ID code, installed by `set_support`. Empty
        # until then; `support` mode without it is a configuration error, not a
        # silent fallback to zeros.
        self.register_buffer("support_user", torch.zeros(0), persistent=False)
        self.register_buffer("support_item", torch.zeros(0), persistent=False)

        self.mha_u_to_i = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.mha_i_to_u = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)

        # tau = softplus(eta) + 1e-3, learnable (Eq. 2).
        self.eta = nn.Parameter(torch.zeros(()))

        # Running reliability statistics, one pair per side.
        self.register_buffer("running_mean", torch.zeros(2))
        self.register_buffer("running_var", torch.ones(2))

        # Per-sample gates from the most recent forward, kept so an eval loop can
        # aggregate them by warm/cold split instead of reporting one batch mean.
        self._last: Optional[Dict[str, torch.Tensor]] = None

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.eta) + 1e-3

    def set_support(self, support_user: torch.Tensor, support_item: torch.Tensor) -> None:
        """Install warm-training row counts per ID code (``reliability='support'``).

        Index 0 is the padding code and must count as zero support, so a masked
        or unknown ID reads as maximally unreliable.
        """
        self.support_user = support_user.float()
        self.support_item = support_item.float()

    def _raw_reliability(
        self,
        field_emb: torch.Tensor,
        feature_ids: Optional[torch.Tensor],
        support: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, 2)`` un-normalized reliability, [user, item].

        ``support`` overrides the table lookup: the caller may have declared a
        row cold that the raw counts call warm (synthesis' pretend-cold rows).
        """
        if self.reliability == "support":
            if support is None:
                if feature_ids is None or self.support_user.numel() == 0:
                    raise RuntimeError(
                        "reliability='support' needs feature_ids and set_support(...)"
                    )
                u = self.support_user[feature_ids[:, list(self.user_id_fields)]].mean(-1)
                i = self.support_item[feature_ids[:, list(self.item_id_fields)]].mean(-1)
                support = torch.stack([u, i], dim=-1)
            # log1p: support is heavy-tailed, and what the gate needs is the
            # order-of-magnitude difference between "a few rows" and "none".
            return torch.log1p(support.clamp_min(0))

        norms = field_emb.norm(dim=-1)                                   # (B, n_fields)
        return torch.stack(
            [
                norms[:, list(self.user_id_fields)].mean(-1),
                norms[:, list(self.item_id_fields)].mean(-1),
            ],
            dim=-1,
        )

    def _reliability(
        self,
        field_emb: torch.Tensor,
        feature_ids: Optional[torch.Tensor] = None,
        support: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, n_fields, d)`` -> normalized ``(B, 2)`` reliability [user, item]."""
        rho = self._raw_reliability(field_emb, feature_ids, support)      # (B, 2)

        if self.batch_znorm:
            mean, var = rho.mean(0), rho.var(0, unbiased=False)
        elif self.training:
            mean, var = rho.mean(0), rho.var(0, unbiased=False)
            with torch.no_grad():
                m = self.momentum
                self.running_mean.mul_(1 - m).add_(m * mean.detach())
                self.running_var.mul_(1 - m).add_(m * var.detach())
        else:
            mean, var = self.running_mean, self.running_var
        return (rho - mean) / (var + self.eps).sqrt()

    def forward(
        self,
        field_emb: torch.Tensor,
        feature_ids: Optional[torch.Tensor] = None,
        support: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, n_fields, d)`` -> fused ``(B, n_fields * d)``."""
        b = field_emb.shape[0]
        u = field_emb[:, : self.n_user, :]
        v = field_emb[:, self.n_user : self.n_user + self.n_item, :]
        c = field_emb[:, self.n_user + self.n_item :, :]

        rho = self._reliability(field_emb, feature_ids, support)
        rho_u, rho_i = rho[:, 0], rho[:, 1]
        tau = self.tau

        g_u_to_i = torch.sigmoid((rho_u - rho_i) / tau)   # warm user -> cold item
        g_i_to_u = torch.sigmoid((rho_i - rho_u) / tau)   # warm item -> cold user

        h_i_from_u, _ = self.mha_u_to_i(v, u, u)          # item queries user block
        h_u_from_i, _ = self.mha_i_to_u(u, v, v)          # user queries item block
        v_fused = v + g_u_to_i.view(b, 1, 1) * h_i_from_u
        u_fused = u + g_i_to_u.view(b, 1, 1) * h_u_from_i

        parts = [u_fused, v_fused] + ([c] if c.shape[1] > 0 else [])
        fused = torch.cat(parts, dim=1)

        self._last = {
            "g_u_to_i": g_u_to_i.detach(),
            "g_i_to_u": g_i_to_u.detach(),
            "rho_u": rho_u.detach(),
            "rho_i": rho_i.detach(),
        }
        return fused.reshape(b, -1)

    def last_gates(self) -> Optional[Dict[str, torch.Tensor]]:
        """Per-sample gates from the last forward, for split-wise diagnostics."""
        return self._last

    def diagnostics(self) -> Dict[str, float]:
        out = {"tau": float(self.tau.detach())}
        if self._last is not None:
            out["g_u_to_i"] = float(self._last["g_u_to_i"].mean())
            out["g_i_to_u"] = float(self._last["g_i_to_u"].mean())
        return out
