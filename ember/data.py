"""Dataset and cache loading for the EMBER learner.

The learner trains on ``D_warm union A_LLM``:

  * **warm rows** — real logged interactions after every cold-entity
    interaction has been removed (strict cold-start split). Their per-task
    confidence is ``1`` and their source weight is ``1``.
  * **augmented rows** (``A_LLM``) — simulator-generated hybrid label tuples
    with per-task confidence ``q`` from the simulator and source weight
    ``lambda_a`` (Eq. 4).

A combined NPZ produced by ``filter/build_augmented_cache.py`` is expected to
hold, for each split, the feature ids, the three targets, the per-task
confidences, and an ``is_aug`` flag. Warm-only caches (no augmentation) are
also supported — missing confidence arrays default to ones.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


def load_cache(path: str) -> Dict[str, np.ndarray]:
    """Load a (possibly augmented) HTL cache NPZ into a plain dict."""
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files if not k.startswith("_")}
    if "_wt_stats" in z.files:
        out["wt_mean"] = float(z["_wt_stats"][0])
        out["wt_std"] = float(z["_wt_stats"][1])
    return out


class EmberHTLDataset(Dataset):
    """Returns ``(feature_ids, click, long_view, watch_time, q_click,
    q_long_view, q_wt, weight)`` per row.

    ``lambda_a`` scales the source weight of augmented rows; warm rows always
    get weight ``1`` and confidence ``1``.
    """

    def __init__(
        self,
        feature_ids: np.ndarray,
        click: np.ndarray,
        long_view: np.ndarray,
        watch_time: np.ndarray,
        q_click: np.ndarray | None = None,
        q_long_view: np.ndarray | None = None,
        q_watch_time: np.ndarray | None = None,
        is_aug: np.ndarray | None = None,
        lambda_a: float = 1.0,
    ) -> None:
        n = len(feature_ids)
        self.feature_ids = torch.as_tensor(feature_ids, dtype=torch.long)
        self.click = torch.as_tensor(click, dtype=torch.float32)
        self.long_view = torch.as_tensor(long_view, dtype=torch.float32)
        self.watch_time = torch.as_tensor(watch_time, dtype=torch.float32)

        def _q(arr):
            return torch.ones(n, dtype=torch.float32) if arr is None else torch.as_tensor(arr, dtype=torch.float32)

        self.q_click = _q(q_click)
        self.q_long_view = _q(q_long_view)
        self.q_watch_time = _q(q_watch_time)

        if is_aug is None:
            self.weight = torch.ones(n, dtype=torch.float32)
        else:
            is_aug_t = torch.as_tensor(is_aug, dtype=torch.float32)
            self.weight = torch.where(is_aug_t > 0, torch.full((n,), float(lambda_a)), torch.ones(n))
        self.is_aug = torch.zeros(n, dtype=torch.float32) if is_aug is None else torch.as_tensor(is_aug, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.feature_ids)

    def __getitem__(self, idx: int):
        return (
            self.feature_ids[idx],
            self.click[idx],
            self.long_view[idx],
            self.watch_time[idx],
            self.q_click[idx],
            self.q_long_view[idx],
            self.q_watch_time[idx],
            self.weight[idx],
            self.is_aug[idx],
        )
