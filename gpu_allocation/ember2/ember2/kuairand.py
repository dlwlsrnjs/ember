"""KuaiRand-Pure -> :class:`~ember2.rawdata.RawDataset`.

The paper's primary dataset (§4.1): short-video logs with the exact hybrid
target EMBER models -- ``is_click`` / ``long_view`` binary conversions and
``play_time_ms`` as the continuous core. This loader reproduces the harness
protocol on the real data:

* **Interactions**: the standard-policy logs (both periods). The random-exposure
  log is left out -- it exists for unbiased evaluation research, and mixing
  policies would change the label distribution mid-dataset.
* **Chronological periods**: TRAIN / VAL / TEST by date (70 / 15 / 15). The
  synthetic generator assigns periods iid, but real logs drift; a time split is
  the honest version.
* **Strict cold split**: 20% of users and items (seeded) have all their
  interactions removed from ``D_warm``, exactly as `rawdata.make_synthetic`.
* **Attributes**: categorical user/item features label-encoded per column into a
  shared 1-based code space (0 = padding). Continuous features are bucketed.

What this loader is for: the learner-side comparisons (norm-rho vs support-rho,
RouteFuse on/off, no-augmentation baselines). It does not provide an LLM
simulator, so augmentation arms are out of scope here.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TaskSpec
from .rawdata import RawDataset, TEST, TRAIN, VAL

LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
USER_F = "user_features_pure.csv"
ITEM_F = "video_features_basic_pure.csv"

# Categorical user columns to keep (all present in user_features_pure.csv).
USER_CAT = [
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
] + [f"onehot_feat{i}" for i in range(18)]
ITEM_CAT = ["video_type", "upload_type", "visible_status", "music_type", "tag"]


def _read_csv(path: str, usecols: Sequence[str]) -> Dict[str, np.ndarray]:
    """Minimal CSV reader -- no pandas dependency for the mock environment."""
    import csv

    cols: Dict[str, List[str]] = {c: [] for c in usecols}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in usecols:
                cols[c].append(row.get(c, ""))
    return {c: np.asarray(v, dtype=object) for c, v in cols.items()}


def _encode(col: np.ndarray, max_vocab: int) -> Tuple[np.ndarray, int]:
    """Label-encode one categorical column, 1-based; rare tail folds into one code.

    ``tag`` is multi-valued ("11,26"); the first tag is the primary category and
    is what we keep. Missing values get their own code rather than padding 0 --
    "unknown" is information on real data.
    """
    first = np.asarray([str(v).split(",")[0].strip() if str(v).strip() else "<na>" for v in col])
    values, counts = np.unique(first, return_counts=True)
    order = values[np.argsort(-counts)]
    table = {v: min(i + 1, max_vocab) for i, v in enumerate(order)}
    return np.asarray([table[v] for v in first], np.int64), min(len(order), max_vocab)


def _bucket(x: np.ndarray, n: int) -> np.ndarray:
    """Quantile-bucket a continuous column into 1..n."""
    x = np.asarray(x, np.float64)
    edges = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, n + 1)[1:-1])
    return (np.digitize(x, edges) + 1).astype(np.int64)


def load_kuairand(
    root: str,
    cold_frac: float = 0.2,
    attr_vocab: int = 64,
    core_cap_s: float = 300.0,
    seed: int = 0,
    max_rows: Optional[int] = None,
) -> RawDataset:
    """Build a :class:`RawDataset` from an extracted KuaiRand-Pure directory.

    ``core_cap_s`` clips watch time: play_time has a heavy tail (background
    autoplay), and NRMSE on an unclipped tail measures outliers, not predictions.
    ``max_rows`` subsamples interactions uniformly (seeded) for quick runs.
    """
    data = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
    rng = np.random.default_rng(seed)

    # ── interactions ────────────────────────────────────────────────
    want = ["user_id", "video_id", "date", "hourmin", "is_click", "long_view",
            "play_time_ms", "tab"]
    parts = [_read_csv(os.path.join(data, f), want) for f in LOGS]
    log = {c: np.concatenate([p[c] for p in parts]) for c in want}
    n = len(log["user_id"])
    if max_rows is not None and n > max_rows:
        keep = rng.choice(n, max_rows, replace=False)
        log = {c: v[keep] for c, v in log.items()}
        n = max_rows

    u_raw = log["user_id"].astype(np.int64)
    i_raw = log["video_id"].astype(np.int64)
    users, u_idx = np.unique(u_raw, return_inverse=True)
    items, i_idx = np.unique(i_raw, return_inverse=True)

    click = log["is_click"].astype(np.float32)
    long_view = log["long_view"].astype(np.float32)
    core = np.clip(log["play_time_ms"].astype(np.float64) / 1000.0, 0, core_cap_s
                   ).astype(np.float32)

    # Context: day-of-week and hour -- cheap, present on every row, and not an
    # entity attribute. `tab` (where the impression happened) as a third field.
    date = log["date"].astype(np.int64)
    dow = (date % 7 + 1).astype(np.int64)
    hour = (log["hourmin"].astype(np.int64) // 100 + 1).astype(np.int64)
    tab, tab_vocab = _encode(log["tab"], 16)
    ctx = np.stack([dow, hour, tab], 1)
    ctx_vocab = (7, 25, tab_vocab)

    # Chronological periods, 70 / 15 / 15 *by rows*: quantiles over row dates,
    # so uneven daily volume cannot starve the evaluation periods.
    t1, t2 = np.quantile(date, [0.70, 0.85])
    period = np.where(date <= t1, TRAIN, np.where(date <= t2, VAL, TEST)).astype(np.int64)

    # ── entity attributes ───────────────────────────────────────────
    uf = _read_csv(os.path.join(data, USER_F), ["user_id"] + USER_CAT)
    vf = _read_csv(os.path.join(data, ITEM_F),
                   ["video_id", "video_duration"] + ITEM_CAT)

    def attr_matrix(feat, id_col, ids, cat_cols, extra=()):
        row_of = {int(v): k for k, v in enumerate(feat[id_col].astype(np.int64))}
        take = np.asarray([row_of.get(int(e), -1) for e in ids])
        cols, vocab = [], 1
        for c in cat_cols:
            enc, v = _encode(feat[c], attr_vocab)
            enc = np.concatenate([enc, [0]])           # -1 -> padding 0
            cols.append(enc[take])
            vocab = max(vocab, v)
        for e in extra:
            e = np.concatenate([e, [0]])
            cols.append(e[take])
            vocab = max(vocab, int(e.max()))
        return np.stack(cols, 1), vocab

    user_attrs, v1 = attr_matrix(uf, "user_id", users, USER_CAT)
    dur_bucket = _bucket(np.where(vf["video_duration"] == "", "0",
                                  vf["video_duration"]).astype(np.float64), 10)
    item_attrs, v2 = attr_matrix(vf, "video_id", items, ITEM_CAT, extra=(dur_bucket,))
    vocab = max(v1, v2, attr_vocab)

    # ── strict cold split ───────────────────────────────────────────
    n_users, n_items = len(users), len(items)
    cold_users = rng.choice(n_users, int(n_users * cold_frac), replace=False)
    cold_items = rng.choice(n_items, int(n_items * cold_frac), replace=False)

    # Text profiles for the LLM stages. Only attributes -- never behaviour, so a
    # cold entity's profile is exactly as informative as a warm one's, which is
    # the premise the simulator is supposed to work under.
    u_row = {int(v): k for k, v in enumerate(uf["user_id"].astype(np.int64))}
    i_row = {int(v): k for k, v in enumerate(vf["video_id"].astype(np.int64))}

    def utext(e):
        k = u_row.get(int(e))
        if k is None:
            return "no profile"
        return (f"activity={uf['user_active_degree'][k]}, "
                f"follows={uf['follow_user_num_range'][k]}, "
                f"fans={uf['fans_user_num_range'][k]}, "
                f"registered={uf['register_days_range'][k]}, "
                f"video_author={uf['is_video_author'][k]}")

    def itext(e):
        k = i_row.get(int(e))
        if k is None:
            return "no profile"
        dur = vf["video_duration"][k]
        dur_s = f"{float(dur) / 1000:.0f}s" if str(dur).strip() else "?"
        return (f"type={vf['video_type'][k]}, upload={vf['upload_type'][k]}, "
                f"duration={dur_s}, music={vf['music_type'][k]}, "
                f"tags={vf['tag'][k] or 'none'}")

    user_text = [utext(e) for e in users]
    item_text = [itext(e) for e in items]

    return RawDataset(
        user=u_idx.astype(np.int64),
        item=i_idx.astype(np.int64),
        binary=np.stack([click, long_view], 1),
        core=core,
        ctx=ctx,
        period=period,
        user_attrs=user_attrs.astype(np.int64),
        item_attrs=item_attrs.astype(np.int64),
        user_text=user_text,
        item_text=item_text,
        cold_users=np.sort(cold_users),
        cold_items=np.sort(cold_items),
        tasks=TaskSpec(binary=["click", "long_view"], core="watch_time"),
        attr_vocab=int(vocab),
        ctx_vocab=ctx_vocab,
        n_users=n_users,
        n_items=n_items,
    )


def join_captions(raw: RawDataset, root: str, captions_path: str, max_chars: int = 200) -> int:
    """Append real video captions to ``raw.item_text`` in place; returns hit count.

    The caption file covers tens of millions of videos; stream it keeping only
    the ids present in this dataset. Join key: ``final_video_id`` == the Pure
    log's ``video_id``.
    """
    import csv

    data = os.path.join(root, "data") if os.path.isdir(os.path.join(root, "data")) else root
    ids = np.concatenate([
        _read_csv(os.path.join(data, f), ["video_id"])["video_id"].astype(np.int64)
        for f in LOGS
    ])
    wanted = {int(v): k for k, v in enumerate(np.unique(ids))}
    hit = 0
    with open(captions_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                vid = int(row.get("final_video_id") or row.get("video_id"))
            except (TypeError, ValueError):
                continue
            k = wanted.get(vid)
            if k is None:
                continue
            blurb = (row.get("caption") or "").strip() or (row.get("show_cover_text") or "").strip()
            if blurb:
                raw.item_text[k] = raw.item_text[k] + f", caption: {blurb[:max_chars]}"
                hit += 1
    return hit
