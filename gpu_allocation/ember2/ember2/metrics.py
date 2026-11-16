"""Evaluation metrics (EMBER §4.1 protocol).

The paper's headline numbers are **Recall@20 / NDCG@20 on overall / warm / cold
splits, ranking by the primary intent logit** (ColdLLM-style layout), plus RMSE
on training-split-normalized continuous targets (NRMSE).

Ranking is per candidate list: each evaluation group is one user's candidate
set, scored and ranked by the first binary task's logit. A group belongs to the
cold split if its target entity is cold -- see :func:`split_groups`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ── single ranking list ─────────────────────────────────────────────
def recall_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 20):
    """Recall@k and binary-gain NDCG@k for one candidate list.

    Returns ``(None, None)`` when the list has no relevant item -- such a list
    carries no ranking information and is excluded from the mean rather than
    counted as a zero.
    """
    n_rel = int(labels.sum())
    if n_rel == 0:
        return None, None
    k = min(k, len(scores))
    topk = np.argsort(-scores, kind="stable")[:k]
    hits = labels[topk]

    recall = float(hits.sum()) / n_rel
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[: min(n_rel, k)].sum())
    return recall, (dcg / idcg if idcg > 0 else 0.0)


# ── grouped ranking evaluation ──────────────────────────────────────
def evaluate_ranking(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    group_split: Optional[Dict[int, str]] = None,
    k: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Mean Recall@k / NDCG@k over candidate lists, broken out by split.

    Args:
      scores:      ``(N,)`` ranking score (the primary intent logit).
      labels:      ``(N,)`` binary relevance.
      group_ids:   ``(N,)`` candidate-list id per row.
      group_split: group id -> ``"warm"`` / ``"cold"``. Every group is always
                   counted in ``"overall"`` as well.
    """
    order = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[order]
    bounds = np.flatnonzero(np.diff(g_sorted)) + 1
    chunks = np.split(order, bounds)

    acc: Dict[str, list] = {"overall": []}
    for idx in chunks:
        r, n = recall_ndcg_at_k(scores[idx], labels[idx], k)
        if r is None:
            continue
        acc["overall"].append((r, n))
        if group_split is not None:
            s = group_split.get(int(group_ids[idx[0]]))
            if s is not None:
                acc.setdefault(s, []).append((r, n))

    out: Dict[str, Dict[str, float]] = {}
    for split, vals in acc.items():
        if not vals:
            out[split] = {"n_groups": 0}
            continue
        arr = np.asarray(vals)
        out[split] = {
            "n_groups": len(vals),
            f"recall@{k}": float(arr[:, 0].mean()),
            f"ndcg@{k}": float(arr[:, 1].mean()),
        }
    return out


def split_groups(
    group_ids: np.ndarray,
    row_is_cold: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[int, str]:
    """Label each candidate list ``warm`` or ``cold``.

    A group is cold if any of its *relevant* rows involves a cold entity -- that
    is the row whose rank the metric actually measures. With ``labels=None``,
    falls back to any row in the group being cold.
    """
    split: Dict[int, str] = {}
    order = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[order]
    for idx in np.split(order, np.flatnonzero(np.diff(g_sorted)) + 1):
        sel = idx if labels is None else idx[labels[idx] > 0]
        if len(sel) == 0:
            sel = idx
        split[int(group_ids[idx[0]])] = "cold" if bool(row_is_cold[sel].any()) else "warm"
    return split


def case_groups(group_ids: np.ndarray, row_case: np.ndarray) -> Dict[int, str]:
    """Label each candidate list with its cold-start *case*.

    ``warm`` / ``cold_item`` / ``cold_user`` / ``cold_both``. A candidate list is
    built entirely within one case (see :func:`ember2.rawdata.build_eval_split`),
    so the first row decides it.

    Reporting one ``cold`` number over all three hides that they are different
    problems: the double-cold case has no warm side to route evidence from, so a
    direction-aware gate cannot help there even in principle.
    """
    out: Dict[int, str] = {}
    order = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[order]
    for idx in np.split(order, np.flatnonzero(np.diff(g_sorted)) + 1):
        out[int(group_ids[idx[0]])] = str(row_case[idx[0]])
    return out


# ── pointwise metrics ───────────────────────────────────────────────
def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via rank statistic; NaN when only one class is present."""
    y = np.asarray(y).astype(np.float64)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(p), dtype=np.float64)
    order = np.argsort(p, kind="stable")
    sp = np.asarray(p)[order]
    # Average ranks within ties, otherwise tied scores would inflate AUC.
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE on the *already normalized* core target (paper reports NRMSE)."""
    if len(pred) == 0:
        return float("nan")
    return float(np.sqrt(((pred - target) ** 2).mean()))


def evaluate_pointwise(
    binary_probs: np.ndarray,
    core_pred: np.ndarray,
    binary_targets: np.ndarray,
    core_target: np.ndarray,
    binary_names: Sequence[str],
    core_name: str,
    masks: Optional[Dict[str, np.ndarray]] = None,
    is_observed: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-task AUC / LogLoss / NRMSE, optionally per split mask.

    Binary metrics use every row -- sampled negatives are legitimate label-0
    examples under the ranking protocol. The continuous target does not work
    that way: a sampled negative has no logged core value, so ``is_observed``
    restricts NRMSE to real logged rows.
    """
    masks = {"overall": np.ones(len(core_pred), bool), **(masks or {})}
    out: Dict[str, Dict[str, float]] = {}
    for split, m in masks.items():
        d: Dict[str, float] = {"n": int(m.sum())}
        if d["n"]:
            for t, name in enumerate(binary_names):
                d[f"{name}_auc"] = auc(binary_targets[m, t], binary_probs[m, t])
                d[f"{name}_logloss"] = log_loss(binary_targets[m, t], binary_probs[m, t])
            core_m = m if is_observed is None else (m & is_observed)
            d[f"n_{core_name}"] = int(core_m.sum())
            d[f"{core_name}_nrmse"] = nrmse(core_pred[core_m], core_target[core_m])
        out[split] = d
    return out
