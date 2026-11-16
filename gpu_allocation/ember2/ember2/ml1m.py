"""MovieLens-1M -> :class:`~ember2.rawdata.RawDataset`.

The second dataset for the dLLM-simulator experiments, chosen for the property
KuaiRand-Pure lacked out of the box: **content a language model can actually
read**. Items are movie titles plus genres; users carry demographic profiles
(age band, gender, occupation). If the content-readability precondition is
real, the simulator should discriminate here without any supplementary file.

Hybrid targets from the 1-5 star ratings, mirroring the click -> long_view ->
watch_time chain:

    like  = rating >= 4          (primary binary intent)
    love  = rating == 5          (nested conversion, love => like)
    core  = rating in [1, 5]     (continuous target)

Chronological 70/15/15 split by timestamp quantiles, strict 20%/20% cold split,
same as the other loaders. ``anonymize=True`` replaces titles/genres/demographics
with opaque codes -- the ablation that reproduces the KuaiRand content finding
on a second dataset.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .config import TaskSpec
from .rawdata import RawDataset, TEST, TRAIN, VAL

AGE_TEXT = {1: "under 18", 18: "18-24", 25: "25-34", 35: "35-44",
            45: "45-49", 50: "50-55", 56: "56+"}
OCCUPATIONS = [
    "other", "academic/educator", "artist", "clerical/admin", "college/grad student",
    "customer service", "doctor/health care", "executive/managerial", "farmer",
    "homemaker", "K-12 student", "lawyer", "programmer", "retired",
    "sales/marketing", "scientist", "self-employed", "technician/engineer",
    "tradesman/craftsman", "unemployed", "writer",
]


def load_ml1m(
    root: str,
    cold_frac: float = 0.2,
    seed: int = 0,
    anonymize: bool = False,
) -> RawDataset:
    rng = np.random.default_rng(seed)

    def rows(name):
        with open(os.path.join(root, name), encoding="latin-1") as f:
            return [line.rstrip("\n").split("::") for line in f]

    ratings = rows("ratings.dat")                 # user, movie, rating, ts
    movies = rows("movies.dat")                   # movie, title, genres
    users = rows("users.dat")                     # user, gender, age, occ, zip

    u_raw = np.array([int(r[0]) for r in ratings], np.int64)
    i_raw = np.array([int(r[1]) for r in ratings], np.int64)
    stars = np.array([float(r[2]) for r in ratings], np.float32)
    ts = np.array([int(r[3]) for r in ratings], np.int64)

    uids, u_idx = np.unique(u_raw, return_inverse=True)
    iids, i_idx = np.unique(i_raw, return_inverse=True)
    n_users, n_items = len(uids), len(iids)

    like = (stars >= 4).astype(np.float32)
    love = (stars >= 5).astype(np.float32)

    # Context: rating year-quarter and day-of-week from the timestamp.
    day = ts // 86400
    dow = (day % 7 + 1).astype(np.int64)
    quarter = ((ts - ts.min()) // (86400 * 90) + 1).astype(np.int64)
    ctx = np.stack([dow, np.clip(quarter, 1, 16)], 1)
    ctx_vocab = (7, 16)

    t1, t2 = np.quantile(ts, [0.70, 0.85])
    period = np.where(ts <= t1, TRAIN, np.where(ts <= t2, VAL, TEST)).astype(np.int64)

    # ── attributes ──────────────────────────────────────────────────
    GENRES = sorted({g for m in movies for g in m[2].split("|")})
    g_code = {g: k + 1 for k, g in enumerate(GENRES)}

    m_row = {int(m[0]): m for m in movies}
    year = np.zeros(n_items, np.int64)
    item_attrs = np.zeros((n_items, 4), np.int64)     # 3 genre slots + year bucket
    item_text: List[str] = []
    for k, iid in enumerate(iids):
        m = m_row.get(int(iid))
        title, genres = (m[1], m[2].split("|")) if m else ("unknown", [])
        for j, g in enumerate(genres[:3]):
            item_attrs[k, j] = g_code[g]
        y = title[-5:-1] if title.endswith(")") and title[-5:-1].isdigit() else "0"
        year[k] = int(y)
        item_attrs[k, 3] = max(1, (int(y) - 1918) // 10 + 1) if y != "0" else 1
        if anonymize:
            item_text.append(f"movie #{k}, genre codes {[g_code.get(g, 0) for g in genres]}")
        else:
            item_text.append(f"{title}, genres: {'|'.join(genres) or 'unknown'}")

    u_row = {int(u[0]): u for u in users}
    user_attrs = np.zeros((n_users, 3), np.int64)     # gender, age band, occupation
    user_text = []
    ages = sorted(AGE_TEXT)
    for k, uid in enumerate(uids):
        u = u_row.get(int(uid))
        if u is None:
            user_text.append("no profile")
            continue
        gender, age, occ = u[1], int(u[2]), int(u[3])
        user_attrs[k, 0] = 1 if gender == "M" else 2
        user_attrs[k, 1] = ages.index(age) + 1
        user_attrs[k, 2] = occ + 1
        if anonymize:
            user_text.append(f"user #{k}, codes [{user_attrs[k,0]},{user_attrs[k,1]},{user_attrs[k,2]}]")
        else:
            user_text.append(f"{'male' if gender == 'M' else 'female'}, age {AGE_TEXT[age]}, "
                             f"occupation: {OCCUPATIONS[occ] if occ < len(OCCUPATIONS) else occ}")

    vocab = int(max(item_attrs.max(), user_attrs.max()))

    cold_users = np.sort(rng.choice(n_users, int(n_users * cold_frac), replace=False))
    cold_items = np.sort(rng.choice(n_items, int(n_items * cold_frac), replace=False))

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
