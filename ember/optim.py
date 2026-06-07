"""HTLNet shared-parameter gradient processor (HTLNet Algorithm 1).

For SHARED parameters (the field embedding tables every task reads), the
per-task gradients are combined with PCGrad-style direction-conflict
resolution (Eq. 16) and magnitude balancing toward the core gradient
(Eq. 17). Task-specific parameters keep standard backprop of their own loss.

This is unchanged from HTLNet and orthogonal to EMBER's RouteFuse / CaliChain;
it is vendored so the learner trains under the same optimization as the
baseline it is compared against.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn


class HTLGradientProcessor:
    def __init__(
        self,
        shared_params: List[nn.Parameter],
        core_task: str = "watch_time",
        preceding_tasks: Iterable[str] = ("click", "long_view"),
        alpha: float = 0.5,
        gamma: float = 0.1,
        clip_c: float = 10.0,
    ) -> None:
        self.shared_params = [p for p in shared_params if p.requires_grad]
        self.core_task = core_task
        self.preceding_tasks = list(preceding_tasks)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.clip_c = float(clip_c)

    def _flatten(self, params: List[nn.Parameter]) -> torch.Tensor:
        flats = []
        for p in params:
            if p.grad is None:
                flats.append(torch.zeros_like(p.data).reshape(-1))
            else:
                flats.append(p.grad.detach().reshape(-1))
        return torch.cat(flats, dim=0)

    def _unflatten_into_grads(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.shared_params:
            n = p.numel()
            chunk = flat[offset : offset + n].view_as(p.data)
            if p.grad is None:
                p.grad = chunk.clone()
            else:
                p.grad.copy_(chunk)
            offset += n

    def _compute_per_task_grads(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for tname, loss in losses.items():
            for p in self.shared_params:
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward(retain_graph=True)
            out[tname] = self._flatten(self.shared_params).clone()
        for p in self.shared_params:
            if p.grad is not None:
                p.grad.zero_()
        return out

    def process_step(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        per_task_grad = self._compute_per_task_grads(losses)
        g_core = per_task_grad[self.core_task]
        g_core_norm_sq = float(g_core.dot(g_core).item()) + 1e-12
        g_core_norm = g_core_norm_sq ** 0.5

        diag = {"core_norm": g_core_norm}
        adjusted: List[torch.Tensor] = []
        for tname in self.preceding_tasks:
            g_t = per_task_grad[tname].clone()
            t_norm = float(g_t.norm().item()) + 1e-12
            dot = float(g_t.dot(g_core).item())
            if dot < 0:
                g_t = g_t - self.alpha * (dot / g_core_norm_sq) * g_core
            weight = g_core_norm / max(t_norm, 1e-12)
            weight = min(max(weight, 1.0 / self.clip_c), self.clip_c)
            g_t = self.gamma * weight * g_t + (1.0 - self.gamma) * g_t
            adjusted.append(g_t)

        g_s = g_core.clone()
        for g in adjusted:
            g_s = g_s + g
        self._unflatten_into_grads(g_s)
        diag["combined_norm"] = float(g_s.norm().item())
        return diag


def split_params(model: nn.Module, shared_names: Iterable[str]) -> tuple:
    shared_set = set(shared_names)
    shared: List[nn.Parameter] = []
    task: List[nn.Parameter] = []
    for n, p in model.named_parameters():
        if any(n.startswith(s) for s in shared_set):
            shared.append(p)
        else:
            task.append(p)
    return shared, task
