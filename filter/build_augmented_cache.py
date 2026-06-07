"""Build the EMBER training cache: merge simulated rows (A_LLM) for both cold
directions into the cold-removed warm HTL NPZ (D_warm), carrying each
synthetic row's per-task confidence and an ``is_aug`` flag.

The output cache is consumed directly by ``train_ember.py``: warm rows get
confidence ``1`` and ``is_aug=0``; augmented rows keep the simulator's
``q_click / q_long_view / q_watch_time`` and ``is_aug=1``, which CaliChain
uses for per-sample loss weighting (Eq. 4) and the confidence prior (Eq. 5).

Inputs:
  data/<dataset>_htl_cache.npz                 (cold-removed warm cache)
  data/ember_simulated_*_cold_item.jsonl       (cold-ITEM sims w/ confidence)
  data/ember_simulated_*_cold_user.jsonl       (cold-USER sims, optional)

Approach (mirrors the v1 builder but per-row template lookup is robust to
sparse train coverage):
  1. Re-derive raw→code dicts for user_id / video_id with the SAME top_k
     values used in v2 preprocess (default 30000 / 30000).
  2. Build feature-row templates by scanning ALL of train (each unique
     user_code stores its user-side block; each unique video_code stores
     its video-side block).
  3. For ids without train coverage (true cold), fall back to a "default"
     template synthesised from the user_features.csv / video_features.csv
     row keyed by the RAW id — using the same _to_int_safe/_factorize_str
     logic as the original preprocessor would have.
  4. For context fields (date_dow, hour_bucket, tab, is_rand), sample a
     random train row's context block.

Output:
  data/kuairand_pure_htl_cache_v2_aug.npz
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# Inline _cap_top_k (no torch dependency)
def _cap_top_k(values, top_k):
    counts = values.dropna().astype(int).value_counts(sort=True, ascending=False)
    keep = counts.index[:top_k].tolist()
    return {int(v): i + 1 for i, v in enumerate(keep)}, len(keep) + 1


def derive_id_maps(log_csvs, top_k_user=30000, top_k_video=30000):
    frames = [pd.read_csv(c, usecols=["user_id", "video_id"],
                            dtype={"user_id": np.int64, "video_id": np.int64})
              for c in log_csvs]
    log = pd.concat(frames, ignore_index=True)
    u2c, n_u = _cap_top_k(log["user_id"], top_k_user)
    v2c, n_v = _cap_top_k(log["video_id"], top_k_video)
    return u2c, v2c, n_u, n_v


def build_templates(feature_ids_train, user_col, video_col, context_start,
                     n_user_codes, n_video_codes):
    n_fields = feature_ids_train.shape[1]
    user_template = np.zeros((n_user_codes, context_start - 0), dtype=np.int64)  # cols 0..context_start-1, BUT we'll only use 0..video_col-1
    user_template = np.zeros((n_user_codes, video_col), dtype=np.int64)          # cols 0..video_col-1
    video_template = np.zeros((n_video_codes, context_start - video_col), dtype=np.int64)
    user_filled = np.zeros(n_user_codes, dtype=bool)
    video_filled = np.zeros(n_video_codes, dtype=bool)
    for i in range(len(feature_ids_train)):
        uc = int(feature_ids_train[i, user_col])
        vc = int(feature_ids_train[i, video_col])
        if not user_filled[uc]:
            user_template[uc] = feature_ids_train[i, 0:video_col]
            user_filled[uc] = True
        if not video_filled[vc]:
            video_template[vc] = feature_ids_train[i, video_col:context_start]
            video_filled[vc] = True
    return user_template, video_template, user_filled, video_filled


def load_jsonl(p: str) -> List[Dict]:
    rows = []
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--htl_npz", default="./data/kuairand_pure_htl_cache_v2.npz")
    p.add_argument("--log_csvs", nargs="+",
                    default=["./data/kuairand_pure/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv"])
    p.add_argument("--sim_item_jsonl", default="./llm_cold_item/data/coldllm_simulated.jsonl",
                    help="cold-item simulated interactions (set to '' to skip)")
    p.add_argument("--sim_user_jsonl", default="./llm_cold_item/data/coldllm_simulated_users.jsonl",
                    help="cold-user simulated interactions (set to '' to skip)")
    p.add_argument("--out_npz", default="./data/kuairand_pure_htl_cache_v2_aug.npz")
    p.add_argument("--keep_negatives", action="store_true",
                    help="Also include click=0 as hard negatives.")
    p.add_argument("--top_k_user", type=int, default=30000)
    p.add_argument("--top_k_video", type=int, default=30000)
    p.add_argument("--user_field", default="user_id")
    p.add_argument("--video_field", default="video_id")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.random.seed(args.seed)

    print(f"[A] Load HTL NPZ", flush=True)
    raw = np.load(args.htl_npz, allow_pickle=True)
    feat_train = raw["feature_ids_train"].astype(np.int64)
    c_train  = raw["click_train"].astype(np.float32)
    lv_train = raw["long_view_train"].astype(np.float32)
    wt_train = raw["watch_time_train"].astype(np.float32)
    field_names = list(map(str, raw["field_names"]))
    user_col  = field_names.index(args.user_field)
    video_col = field_names.index(args.video_field)
    n_fields = feat_train.shape[1]
    context_start = video_col + 8  # 8 video-side fields per ITEM_FEAT_COLS
    print(f"  train: {feat_train.shape}, user_col={user_col}, video_col={video_col}, "
          f"context_start={context_start}", flush=True)

    sim_item = load_jsonl(args.sim_item_jsonl)
    sim_user = load_jsonl(args.sim_user_jsonl)
    print(f"  loaded sim_item: {len(sim_item):,}, sim_user: {len(sim_user):,}", flush=True)

    print(f"\n[B] Re-derive raw→code maps (top_k_user={args.top_k_user}, top_k_video={args.top_k_video})",
          flush=True)
    u2c, v2c, n_u, n_v = derive_id_maps(args.log_csvs, args.top_k_user, args.top_k_video)
    print(f"  n_user_codes={n_u}, n_video_codes={n_v}", flush=True)

    print(f"\n[C] Build per-code feature templates from train", flush=True)
    user_template, video_template, user_filled, video_filled = build_templates(
        feat_train, user_col, video_col, context_start, n_u, n_v,
    )
    print(f"  user templates: {int(user_filled.sum()):,}/{n_u:,} filled", flush=True)
    print(f"  video templates: {int(video_filled.sum()):,}/{n_v:,} filled", flush=True)

    # Compose all simulated rows into one list
    all_sims = []
    for r in sim_item:
        all_sims.append(("item", r))
    for r in sim_user:
        all_sims.append(("user", r))
    keep_rows = [(src, r) for (src, r) in all_sims
                  if args.keep_negatives or int(r.get("click", 0)) == 1]
    n_aug = len(keep_rows)
    print(f"\n[D] Composing {n_aug:,} augmented rows "
          f"(item-side: {sum(1 for s,_ in keep_rows if s=='item'):,}, "
          f"user-side: {sum(1 for s,_ in keep_rows if s=='user'):,})", flush=True)

    ctx_indices = np.random.randint(0, len(feat_train), size=n_aug)
    new_feat = np.zeros((n_aug, n_fields), dtype=np.int64)
    new_c  = np.zeros(n_aug, dtype=np.float32)
    new_lv = np.zeros(n_aug, dtype=np.float32)
    new_wt = np.zeros(n_aug, dtype=np.float32)
    new_qc  = np.ones(n_aug, dtype=np.float32)   # per-task simulator confidence
    new_qlv = np.ones(n_aug, dtype=np.float32)
    new_qwt = np.ones(n_aug, dtype=np.float32)
    j = 0; skip_uc = skip_vc = skip_template = 0
    for src, r in keep_rows:
        u_raw = int(r["user_raw_id"]); v_raw = int(r["video_raw_id"])
        uc = u2c.get(u_raw); vc = v2c.get(v_raw)
        if uc is None: skip_uc += 1; continue
        if vc is None: skip_vc += 1; continue
        if not user_filled[uc] or not video_filled[vc]:
            # Template missing for truly cold-in-train ids; skip rather than emit zero rows
            skip_template += 1
            continue
        new_feat[j, 0:video_col] = user_template[uc]
        new_feat[j, video_col:context_start] = video_template[vc]
        new_feat[j, context_start:n_fields] = feat_train[ctx_indices[j], context_start:n_fields]
        new_c[j] = float(int(r.get("click", 0)))
        new_lv[j] = float(int(r.get("long_view", 0)))
        new_wt[j] = float(int(r.get("watch_time_seconds", 0)))
        # Per-task confidence (default 1.0 if the simulator did not emit it).
        new_qc[j]  = float(r.get("q_click", 1.0))
        new_qlv[j] = float(r.get("q_long_view", 1.0))
        new_qwt[j] = float(r.get("q_watch_time", 1.0))
        j += 1
    new_feat = new_feat[:j]; new_c = new_c[:j]; new_lv = new_lv[:j]; new_wt = new_wt[:j]
    new_qc = new_qc[:j]; new_qlv = new_qlv[:j]; new_qwt = new_qwt[:j]
    print(f"  composed: {j:,}  (skipped uc={skip_uc:,}, vc={skip_vc:,}, "
          f"template={skip_template:,})", flush=True)

    print(f"\n[E] Concatenate + save", flush=True)
    n_warm = feat_train.shape[0]
    aug_feat = np.concatenate([feat_train, new_feat], axis=0)
    aug_c    = np.concatenate([c_train, new_c])
    aug_lv   = np.concatenate([lv_train, new_lv])
    aug_wt   = np.concatenate([wt_train, new_wt])
    # CaliChain bookkeeping: warm rows -> confidence 1, is_aug 0.
    aug_qc  = np.concatenate([np.ones(n_warm, np.float32), new_qc])
    aug_qlv = np.concatenate([np.ones(n_warm, np.float32), new_qlv])
    aug_qwt = np.concatenate([np.ones(n_warm, np.float32), new_qwt])
    is_aug  = np.concatenate([np.zeros(n_warm, np.float32), np.ones(len(new_c), np.float32)])
    print(f"  augmented train: {n_warm:,} → {aug_feat.shape[0]:,} "
          f"(+{new_feat.shape[0]:,}, +{new_feat.shape[0]/max(n_warm,1)*100:.2f}%)",
          flush=True)

    out = {k: raw[k] for k in raw.files}
    out["feature_ids_train"] = aug_feat
    out["click_train"] = aug_c
    out["long_view_train"] = aug_lv
    out["watch_time_train"] = aug_wt
    out["q_click_train"] = aug_qc
    out["q_long_view_train"] = aug_qlv
    out["q_watch_time_train"] = aug_qwt
    out["is_aug_train"] = is_aug
    np.savez_compressed(args.out_npz, **out)
    print(f"\nSaved → {args.out_npz}  ({os.path.getsize(args.out_npz)/1e6:.2f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
