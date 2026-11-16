"""Significance testing that the seed count can actually support.

The submitted paper claims ``p < 0.05`` from Wilcoxon signed-rank over five
seeds. That is unattainable: with n = 5 paired differences there are 2^5 = 32
sign assignments, so the smallest two-sided exact p is 2/32 = 0.0625. No
arrangement of five seeds produces a two-sided p below 0.05 -- the test cannot
reject at that level regardless of how large the effect is.

Two ways out, both implemented here:

* **More seeds.** :func:`min_two_sided_p` reports the floor for a given n, so
  the seed budget can be chosen before running rather than defended afterwards:
  n = 6 reaches 0.03125, n = 8 reaches 0.0078.
* **Stop treating seeds as the sample.** The evaluation ranks thousands of
  candidate lists, and a paired bootstrap *over those lists* estimates the
  uncertainty of the metric itself rather than the uncertainty of the training
  run. :func:`paired_bootstrap` does that, pairing by group so both variants are
  resampled on the same lists.

Report both: the bootstrap interval says whether the metric difference is real
on this data, the seed-level test says whether it survives retraining.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def min_two_sided_p(n: int) -> float:
    """Smallest two-sided exact Wilcoxon/sign-test p attainable with ``n`` pairs."""
    if n <= 0:
        return 1.0
    return min(1.0, 2.0 / (2 ** n))


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap the paired mean difference ``a - b`` over evaluation groups.

    ``a`` and ``b`` must be per-group metric values *for the same groups in the
    same order* -- pairing is what removes the between-group variance that would
    otherwise swamp the difference.

    Returns the observed difference, a percentile interval, and the two-sided
    bootstrap p-value (the fraction of resamples on the other side of zero,
    doubled). The p-value floor here is 1/n_boot, not 2/2^n_seeds.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must align: {a.shape} vs {b.shape}")
    d = a - b
    n = len(d)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": 1.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided: how often does the resampled difference land on the far side of
    # zero from the observed one.
    tail = float((means <= 0).mean() if d.mean() > 0 else (means >= 0).mean())
    return {
        "diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(1.0, 2 * tail)),
        "n_groups": int(n),
    }


def sign_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Exact two-sided sign test over paired seed-level values.

    Reported alongside the attainable floor, so a non-significant result at
    n = 5 reads as "this many seeds cannot show it" rather than "no effect".
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "p": 1.0, "floor": 1.0, "wins": 0}
    wins = int((d > 0).sum())
    k = min(wins, n - wins)
    # Two-sided exact binomial under p = 0.5.
    from math import comb

    p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)
    return {"n": n, "p": float(p), "floor": min_two_sided_p(n), "wins": wins}


def per_group_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    k: int = 20,
    keep: Optional[Dict[int, str]] = None,
    case: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-candidate-list Recall@k, and the group ids they belong to.

    The bootstrap needs the individual list scores, not their mean; ``keep`` and
    ``case`` restrict to one cold-start case.
    """
    from .metrics import recall_ndcg_at_k

    order = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[order]
    gids, vals = [], []
    for idx in np.split(order, np.flatnonzero(np.diff(g_sorted)) + 1):
        g = int(group_ids[idx[0]])
        if keep is not None and case is not None and keep.get(g) != case:
            continue
        r, _ = recall_ndcg_at_k(scores[idx], labels[idx], k)
        if r is not None:
            gids.append(g)
            vals.append(r)
    return np.asarray(vals, float), np.asarray(gids, np.int64)
