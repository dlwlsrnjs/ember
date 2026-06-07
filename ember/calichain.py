"""CaliChain: distillation-calibrated objective (EMBER paper §3.3, Eq. 4-5).

The simulator is a *soft teacher*: every synthetic row carries a per-task
confidence ``q`` (teacher-forced valid-label token log-probability by default).
CaliChain consumes that confidence twice, so noisy pseudo-labels neither
dominate the loss nor cascade through HTLNet's task-dependence chain.

1. **Per-sample confidence weighting (Eq. 4).** Each augmented-row loss is
   scaled by its task confidence, so low-confidence synthetic labels
   contribute less::

       L_aug_k(a) = q_a^(k) * loss_k( yhat_a^(k), y_a^(k),LLM )

   Warm logged rows use ``q = 1`` (real labels are fully trusted); the whole
   augmented block is additionally scaled by ``lambda_a``.

2. **Confidence prior in the task chain (Eq. 5).** Before the predicted logit
   feeds the next label state, it is blended with the confidence prior::

       o_tilde_a^(t) = alpha_t * o_a^(t)
                       + (1 - alpha_t) * logit( clip(q_a^(t), eps, 1-eps) )

   with a learnable, task-specific ``alpha_t = clip(sigmoid(a_t), 0.05, 0.95)``
   initialized at 0.7. The learner relies more on simulator priors exactly
   where cold rows are hardest (learned ``alpha_t`` decreases from the primary
   binary target toward the continuous core target).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _logit(q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    q = q.clamp(eps, 1.0 - eps)
    return torch.log(q) - torch.log1p(-q)


class ConfidencePrior(nn.Module):
    """Learnable per-task confidence-prior blend (Eq. 5).

    One scalar ``a_t`` per binary task. ``alpha_t = clip(sigmoid(a_t),
    0.05, 0.95)`` is initialized to 0.7 by setting ``a_t = logit(0.7)``.
    """

    def __init__(self, n_tasks: int, init_alpha: float = 0.7) -> None:
        super().__init__()
        a0 = float(torch.logit(torch.tensor(init_alpha)))
        self.a = nn.Parameter(torch.full((n_tasks,), a0))

    def alpha(self, t: int) -> torch.Tensor:
        return torch.sigmoid(self.a[t]).clamp(0.05, 0.95)

    def blend(
        self,
        logit: torch.Tensor,
        q: Optional[torch.Tensor],
        t: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend predicted ``logit`` (B,) with the confidence prior for task ``t``.

        ``q`` is the per-sample task confidence (B,) in ``[0, 1]``. The prior is
        applied only to *augmented* rows: ``mask`` (B,) bool selects them, and
        warm rows keep their raw logit. When ``q`` is ``None`` the call is a
        no-op (used at evaluation time, where there is no simulator confidence).
        """
        if q is None:
            return logit
        a_t = self.alpha(t)
        blended = a_t * logit + (1.0 - a_t) * _logit(q)
        if mask is None:
            return blended
        return torch.where(mask, blended, logit)

    def alphas(self) -> list:
        return [float(self.alpha(t)) for t in range(self.a.numel())]


def confidence_weighted_bce(
    logit: torch.Tensor,
    target: torch.Tensor,
    q: Optional[torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    """Confidence- and source-weighted BCE (Eq. 4).

    Args:
      logit:  (B,) predicted logits.
      target: (B,) binary labels (real or simulated).
      q:      (B,) per-sample confidence, or ``None`` for all-ones.
      weight: (B,) per-sample source weight (e.g. ``lambda_a`` on augmented
              rows, ``1`` on warm rows).
    """
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if q is not None:
        per = per * q
    return (per * weight).sum() / weight.sum().clamp_min(1.0)


def confidence_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    q: Optional[torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    """Confidence- and source-weighted MSE for the continuous core target."""
    per = (pred - target) ** 2
    if q is not None:
        per = per * q
    return (per * weight).sum() / weight.sum().clamp_min(1.0)
