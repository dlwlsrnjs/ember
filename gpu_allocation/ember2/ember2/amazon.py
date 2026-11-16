"""Amazon Reviews 2023 (a single category) -> :class:`RawDataset`.

The third dataset, chosen to vary two things at once relative to MovieLens-1M:
the domain (retail products rather than films) and the provenance of the
content (seller-written titles and descriptions rather than curated metadata),
while keeping the two properties our setting needs. Ratings are explicit, so
the hybrid targets follow the same construction as MovieLens-1M --

    like = rating >= 4,  love = rating == 5,  core = rating in [1,5]

-- and items carry natural language a simulator can read.

Reviews and item metadata arrive as gzipped JSON Lines, one object per line and
hundreds of megabytes per category, so both files are streamed rather than
loaded. The whole category is read by default: the file is ordered neither by time nor
by item, so a prefix is not a defensible sample. ``k_core`` repeatedly drops
users and items with too few interactions,
the standard preprocessing for this corpus; without it the long tail of
single-review accounts dominates the cold split and every method looks alike.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import TaskSpec
from .rawdata import RawDataset, TEST, TRAIN, VAL


def _stream(path: str, keys: Tuple[str, ...], limit: Optional[int] = None):
    with gzip.open(path, "rt") as f:
        for n, line in enumerate(f):
            if limit is not None and n >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield {k: obj.get(k) for k in keys}


def _k_core(u: np.ndarray, i: np.ndarray, k: int) -> np.ndarray:
    """Indices surviving iterative k-core filtering on both sides."""
    keep = np.ones(len(u), bool)
    while True:
        cu = np.bincount(u[keep], minlength=u.max() + 1)
        ci = np.bincount(i[keep], minlength=i.max() + 1)
        drop = keep & ((cu[u] < k) | (ci[i] < k))
        if not drop.any():
            return np.flatnonzero(keep)
        keep &= ~drop


def load_amazon(
    review_path: str,
    meta_path: str,
    k_core: int = 10,
    cold_frac: float = 0.2,
    seed: int = 0,
    max_reviews: Optional[int] = None,
    anonymize: bool = False,
) -> RawDataset:
    rng = np.random.default_rng(seed)

    users, items, stars, ts = [], [], [], []
    for r in _stream(review_path, ("user_id", "parent_asin", "rating", "timestamp"),
                     max_reviews):
        if r["user_id"] is None or r["parent_asin"] is None or r["rating"] is None:
            continue
        users.append(r["user_id"])
        items.append(r["parent_asin"])
        stars.append(float(r["rating"]))
        ts.append(int(r["timestamp"] or 0))

    u_codes, u_idx = np.unique(np.asarray(users), return_inverse=True)
    i_codes, i_idx = np.unique(np.asarray(items), return_inverse=True)
    stars = np.asarray(stars, np.float32)
    ts = np.asarray(ts, np.int64)

    keep = _k_core(u_idx, i_idx, k_core)
    u_idx, i_idx, stars, ts = u_idx[keep], i_idx[keep], stars[keep], ts[keep]
    # re-index after filtering so the embedding tables stay dense
    u_keep, u_idx = np.unique(u_idx, return_inverse=True)
    i_keep, i_idx = np.unique(i_idx, return_inverse=True)
    u_codes, i_codes = u_codes[u_keep], i_codes[i_keep]
    n_users, n_items = len(u_codes), len(i_codes)

    like = (stars >= 4).astype(np.float32)
    love = (stars >= 5).astype(np.float32)

    day = ts // 86_400_000 if ts.max() > 10 ** 12 else ts // 86_400
    dow = (day % 7 + 1).astype(np.int64)
    span = np.clip((day - day.min()) // 90 + 1, 1, 24).astype(np.int64)
    ctx = np.stack([dow, span], 1)
    ctx_vocab = (7, 24)

    t1, t2 = np.quantile(ts, [0.70, 0.85])
    period = np.where(ts <= t1, TRAIN, np.where(ts <= t2, VAL, TEST)).astype(np.int64)

    # ── item metadata: title, price band, category, rating volume ──
    wanted = {code: k for k, code in enumerate(i_codes)}
    title = ["unknown"] * n_items
    cat = np.zeros(n_items, np.int64)
    price_b = np.zeros(n_items, np.int64)
    vol_b = np.zeros(n_items, np.int64)
    cat_codes: Dict[str, int] = {}
    prices = np.full(n_items, np.nan)
    volumes = np.zeros(n_items)
    for m in _stream(meta_path, ("parent_asin", "title", "main_category", "price",
                                 "rating_number", "categories")):
        k = wanted.get(m["parent_asin"])
        if k is None:
            continue
        if m["title"]:
            title[k] = str(m["title"])[:120]
        c = m["main_category"] or (m["categories"][0] if m.get("categories") else "unknown")
        cat[k] = cat_codes.setdefault(str(c), len(cat_codes) + 1)
        try:
            prices[k] = float(m["price"])
        except (TypeError, ValueError):
            pass
        volumes[k] = float(m["rating_number"] or 0)

    def bucket(x, n=8):
        ok = np.isfinite(x) & (x > 0)
        out = np.zeros(len(x), np.int64)
        if ok.sum() > n:
            edges = np.quantile(x[ok], np.linspace(0, 1, n + 1)[1:-1])
            out[ok] = np.digitize(x[ok], edges) + 1
        return out

    price_b, vol_b = bucket(prices), bucket(volumes)
    item_attrs = np.stack([cat, price_b, vol_b], 1)

    # ── user attributes: activity and generosity, from D_warm only ──
    # No demographics exist in this corpus, so the profile is behavioural
    # summary statistics computed on the *training period* alone; using the
    # evaluation period here would leak.
    tr = period == TRAIN
    cnt = np.bincount(u_idx[tr], minlength=n_users).astype(np.float64)
    mean_star = np.zeros(n_users)
    np.add.at(mean_star, u_idx[tr], stars[tr])
    mean_star = np.where(cnt > 0, mean_star / np.maximum(cnt, 1), 0.0)
    act_b, gen_b = bucket(cnt, 6), bucket(mean_star, 6)
    user_attrs = np.stack([act_b, gen_b], 1)

    inv_cat = {v: k for k, v in cat_codes.items()}
    if anonymize:
        item_text = [f"item #{k}, codes [{cat[k]},{price_b[k]},{vol_b[k]}]"
                     for k in range(n_items)]
        user_text = [f"user #{k}, codes [{act_b[k]},{gen_b[k]}]" for k in range(n_users)]
    else:
        item_text = [
            f"{title[k]}, category: {inv_cat.get(int(cat[k]), 'unknown')}"
            + (f", price ${prices[k]:.0f}" if np.isfinite(prices[k]) else "")
            for k in range(n_items)
        ]
        user_text = [
            f"reviewer with {int(cnt[k])} reviews in the training period, "
            f"average rating given {mean_star[k]:.1f} of 5"
            for k in range(n_users)
        ]

    cold_users = np.sort(rng.choice(n_users, int(n_users * cold_frac), replace=False))
    cold_items = np.sort(rng.choice(n_items, int(n_items * cold_frac), replace=False))
    vocab = int(max(item_attrs.max(), user_attrs.max(), 1))

    return RawDataset(
        user=u_idx.astype(np.int64),
        item=i_idx.astype(np.int64),
        binary=np.stack([like, love], 1),
        core=stars,
        ctx=ctx,
        period=period,
        user_attrs=user_attrs,
        item_attrs=item_attrs,
        user_text=user_text,
        item_text=item_text,
        cold_users=cold_users,
        cold_items=cold_items,
        tasks=TaskSpec(binary=["like", "love"], core="rating"),
        attr_vocab=vocab,
        ctx_vocab=ctx_vocab,
        n_users=n_users,
        n_items=n_items,
    )
