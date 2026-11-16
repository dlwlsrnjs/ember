"""CaliChain: distillation-calibrated objective (EMBER §3.3, Eq. 4-5).

The simulator is a *soft teacher*: every synthetic row carries a per-task
confidence ``q``. CaliChain spends that confidence twice.

**(1) Per-sample loss weighting (Eq. 4).**

    L_aug_k(a) = q_a^(k) * loss_k( yhat_a^(k), y_a^(k),LLM ),  k in {1..T, r}

Warm logged rows use ``q = 1`` (real labels are trusted fully); the whole
augmented block is additionally scaled by ``lambda_a``.

**(2) Confidence prior in the task chain (Eq. 5).** Before a logit becomes the
next label state, it is blended with the prior:

    o~_a^(t) = alpha_t * o_a^(t) + (1 - alpha_t) * logit( clip(q_a^(t), eps, 1-eps) )

with learnable per-task ``alpha_t = clip(sigmoid(a_t), 0.05, 0.95)``, init 0.7.

**Gradient path for alpha_t.** HTLNet stops the gradient on the transfer between
task towers, and the natural way to write that is to detach the probability fed
into the label-embedding unit. Doing it *after* the blend severs ``alpha_t`` from
the graph entirely and freezes it at its initialization forever -- the released
EMBER code has exactly that bug, so its reported "learned alpha_t decreases from
0.70-0.76 to 0.51-0.58" cannot arise from it.

The fix is to detach the *incoming tower logit* rather than the blend output:
``o^(t)`` is detached before Eq. 5, so no gradient flows back into the producing
tower (HTLNet's intent is preserved), while ``alpha_t`` and the label-embedding
table stay differentiable through the downstream tasks' losses. That is what
:meth:`ConfidencePrior.blend` expects -- pass it an already-detached logit.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def prob_to_logit(q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    q = q.clamp(eps, 1.0 - eps)
    return torch.log(q) - torch.log1p(-q)


class ConfidencePrior(nn.Module):
    """Learnable per-task confidence-prior blend (Eq. 5).

    **On the sign of the prior.** Eq. 5 adds ``logit(q)`` to the logit, but ``q``
    is the simulator's confidence *in the label it emitted*, not its probability
    that the label is positive. Taken literally, a pseudo-label of 0 held with
    confidence 0.9 contributes ``logit(0.9) = +2.2`` -- pushing the label state
    toward positive, the opposite of what the teacher said. In the caches built
    here 15.7% of augmented rows carry a negative secondary label, so the prior
    is mis-signed on all of them, and that label state is exactly what feeds the
    continuous core tower.

    ``signed=True`` (default) converts confidence into the teacher's belief
    before blending::

        p_sim = y_LLM * q + (1 - y_LLM) * (1 - q)
        o~    = alpha_t * o + (1 - alpha_t) * logit(p_sim)

    which is what "distillation-calibrated" should mean: blend toward the
    teacher's posterior, not toward the magnitude of its certainty. Pass
    ``signed=False`` to reproduce Eq. 5 as written.
    """

    def __init__(
        self,
        n_tasks: int,
        init_alpha: float = 0.7,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
        signed: bool = True,
    ) -> None:
        super().__init__()
        a0 = float(torch.logit(torch.tensor(float(init_alpha))))
        self.a = nn.Parameter(torch.full((n_tasks,), a0))
        self.alpha_min, self.alpha_max = float(alpha_min), float(alpha_max)
        self.signed = bool(signed)

    def alpha(self, t: int) -> torch.Tensor:
        return torch.sigmoid(self.a[t]).clamp(self.alpha_min, self.alpha_max)

    def alphas(self) -> list:
        return [float(self.alpha(t).detach()) for t in range(self.a.numel())]

    def blend(
        self,
        logit: torch.Tensor,
        q: Optional[torch.Tensor],
        t: int,
        mask: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blend a (detached) tower ``logit`` with the confidence prior.

        Args:
          logit: ``(B,)`` tower logit for task ``t``. Detach before calling if
                 the chain should not backprop into the producing tower.
          q:     ``(B,)`` simulator confidence in ``[0, 1]``; ``None`` is a no-op
                 (evaluation, where no simulator confidence exists).
          mask:  ``(B,)`` bool selecting augmented rows. Warm rows keep their raw
                 logit -- a logged label needs no prior.
          y:     ``(B,)`` the simulator's pseudo-label for task ``t``. Required
                 for ``signed`` mode; without it the prior falls back to the
                 literal Eq. 5 form.
        """
        if q is None:
            return logit
        p = q if (not self.signed or y is None) else y * q + (1.0 - y) * (1.0 - q)
        a_t = self.alpha(t)
        blended = a_t * logit + (1.0 - a_t) * prob_to_logit(p)
        if mask is None:
            return blended
        return torch.where(mask, blended, logit)


def _reduce(per_sample: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean with a weight-sum denominator.

    The denominator is the weight mass, not the row count, so that dropping the
    augmented block (``lambda_a = 0``) reproduces warm-only training exactly
    instead of silently shrinking the effective learning rate.
    """
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)


def confidence_weighted_bce(
    logit: torch.Tensor,
    target: torch.Tensor,
    q: Optional[torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    """Confidence- and source-weighted BCE (Eq. 4) for one binary task."""
    per = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if q is not None:
        per = per * q
    return _reduce(per, weight)


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
    return _reduce(per, weight)


def hybrid_losses(
    binary_logits: torch.Tensor,
    core_pred: torch.Tensor,
    binary_targets: torch.Tensor,
    core_target: torch.Tensor,
    q: Optional[torch.Tensor],
    is_aug: torch.Tensor,
    lambda_a: float = 1.0,
    use_confidence_weighting: bool = True,
    aug_soft_targets: bool = False,
) -> dict:
    """Assemble the per-task hybrid objective over ``D_warm u A_LLM``.

    Args:
      binary_logits:  ``(B, T)``
      core_pred:      ``(B,)`` prediction of the *normalized* core target
      binary_targets: ``(B, T)``
      core_target:    ``(B,)`` normalized core target
      q:              ``(B, T+1)`` per-task confidence, or ``None``
      is_aug:         ``(B,)`` 1.0 for rows drawn from ``A_LLM``
      lambda_a:       source weight of the augmented block

    Returns a dict of named scalar losses (one per task); the caller sums them.
    Warm rows always get ``q = 1`` and source weight ``1``, whatever the cache
    happens to store for them.
    """
    aug = is_aug > 0
    weight = torch.where(aug, torch.full_like(is_aug, float(lambda_a)), torch.ones_like(is_aug))

    n_binary = binary_logits.shape[1]
    out = {}
    for t in range(n_binary):
        q_t = None
        if use_confidence_weighting and q is not None:
            q_t = torch.where(aug, q[:, t], torch.ones_like(q[:, t]))
        target_t = binary_targets[:, t]
        if aug_soft_targets and q is not None:
            # Distillation target: on an augmented row, fit the simulator's
            # posterior instead of the label it happened to argmax to. A pair it
            # called positive at q=0.6 then asks for 0.6, not 1.0 -- the same
            # information Eq. 4 spends on a loss weight, spent on the target
            # instead, where it cannot be confused with "this row matters less".
            y, qq = binary_targets[:, t], q[:, t]
            soft = y * qq + (1.0 - y) * (1.0 - qq)
            target_t = torch.where(aug, soft, y)
        out[f"binary_{t}"] = confidence_weighted_bce(
            binary_logits[:, t], target_t, q_t, weight
        )

    q_core = None
    if use_confidence_weighting and q is not None:
        q_core = torch.where(aug, q[:, n_binary], torch.ones_like(q[:, n_binary]))
    out["core"] = confidence_weighted_mse(core_pred, core_target, q_core, weight)
    return out
