"""Shared-parameter gradient processing (HTLNet Algorithm 1).

The field embedding tables are read by every task, so their gradient is a sum of
competing signals. HTLNet resolves that in two steps, both applied only to the
shared parameters (task-specific parameters keep plain backprop of their own
loss):

  * **Direction (Eq. 16)** -- PCGrad-style projection: if a preceding task's
    gradient conflicts with the core task's (negative inner product), project
    out the conflicting component, scaled by ``alpha``.
  * **Magnitude (Eq. 17)** -- rescale each preceding task's gradient toward the
    core gradient's norm, blended by ``gamma`` and clipped to ``[1/c, c]`` so a
    near-zero task gradient cannot explode.

This is orthogonal to EMBER's contribution; it is reimplemented so the learner
optimizes under the same rules as the HTLNet baseline it is compared against.

Implementation note: per-task gradients are taken with ``torch.autograd.grad``,
which returns them without touching ``.grad``. The released EMBER code instead
calls ``backward()`` once per task and snapshots/restores every task parameter's
``.grad`` around it -- same result, but it accumulates and rewinds the entire
parameter set on every step.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn


class SharedGradientProcessor:
    def __init__(
        self,
        shared_params: Iterable[nn.Parameter],
        core_task: str,
        preceding_tasks: Sequence[str],
        alpha: float = 0.5,
        gamma: float = 0.1,
        clip_c: float = 10.0,
    ) -> None:
        self.params: List[nn.Parameter] = [p for p in shared_params if p.requires_grad]
        self.core_task = core_task
        self.preceding_tasks = list(preceding_tasks)
        self.alpha, self.gamma, self.clip_c = float(alpha), float(gamma), float(clip_c)

    def _flat_grad(self, loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss, self.params, retain_graph=True, allow_unused=True
        )
        return torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).reshape(-1)
                for p, g in zip(self.params, grads)
            ]
        )

    def _write_back(self, flat: torch.Tensor) -> None:
        off = 0
        for p in self.params:
            n = p.numel()
            chunk = flat[off : off + n].view_as(p)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            off += n

    @torch.no_grad()
    def _combine(self, per_task: Dict[str, torch.Tensor]) -> torch.Tensor:
        g_core = per_task[self.core_task]
        core_sq = g_core.dot(g_core) + 1e-12
        core_norm = core_sq.sqrt()

        total = g_core.clone()
        for name in self.preceding_tasks:
            g = per_task[name].clone()
            dot = g.dot(g_core)
            if dot < 0:                                   # Eq. 16: conflict
                g = g - self.alpha * (dot / core_sq) * g_core
            scale = (core_norm / g.norm().clamp_min(1e-12)).clamp(
                1.0 / self.clip_c, self.clip_c
            )                                             # Eq. 17: magnitude
            g = self.gamma * scale * g + (1.0 - self.gamma) * g
            total = total + g
        return total

    def process(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Overwrite the shared parameters' ``.grad`` with the combined gradient.

        Call *after* ``total_loss.backward(retain_graph=True)`` so the
        task-specific parameters already hold their own gradients; only the
        shared block is replaced.
        """
        per_task = {name: self._flat_grad(loss) for name, loss in losses.items()}
        combined = self._combine(per_task)
        self._write_back(combined)
        return {
            "core_grad_norm": float(per_task[self.core_task].norm()),
            "combined_grad_norm": float(combined.norm()),
        }
