"""EMBER Stage 1 simulation — batched vLLM inference (paper §3.1).

For each cold entity, the coupled filter's top-K partners are simulated by the
direction-specific LoRA adapter, which emits a hybrid label tuple
``(click, long_view, watch_time)`` plus per-task confidence ``q`` from
teacher-forced valid-label token log-probabilities (CaliChain's default
confidence estimator). Continuous batching + paged attention keep all LLM calls
offline. Positive primary-behavior predictions become augmented rows.

Usage example:
  python3 simulation/simulate.py \
    --adapter_dir ./llm_cold_item/data/lora_adapter_v3_both_no_reasoning \
    --filter_mode coupled --filter_backbone text \
    --filter_ckpt ./llm_cold_item/data/coupled_filter_qwen_paper.pt \
    --user_text_emb_npz ./llm_cold_item/data/text_emb_user_qwen_norm.npz \
    --video_text_emb_npz ./llm_cold_item/data/text_emb_video_qwen_norm.npz \
    --focus_at_train both --direction cold_item \
    --cold_video_ids_npz ./llm_cold_item/data/cold_warm_split_paper.npz \
    --top_k_users 30 --batch_size 256 \
    --out_dir ./llm_cold_item/data --out_suffix paper_vllm
"""
from __future__ import annotations
import argparse, json, os, sys, time, re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "filter"))
from build_lora_data import (  # noqa: E402
    SYSTEM_PROMPT_NO_REASONING, SYSTEM_PROMPT_WITH_REASONING,
    build_user_prompt_full,
    build_history_lines, build_video_clicker_lines,
    build_user_behavior_summary, build_video_behavior_summary,
    filter_user_history_by_sim, filter_video_clickers_by_sim,
)
from train_coupled_filter import CoupledFilter  # noqa: E402


def _per_token_probs(out, tok) -> List[Tuple[str, float]]:
    """Decode a vLLM output into a list of ``(token_text, prob)`` pairs.

    ``prob = exp(logprob)`` of the actually-generated token at each position.
    Returns ``[]`` if logprobs are unavailable so callers fall back cleanly.
    """
    try:
        token_ids = list(out.outputs[0].token_ids)
        lps = out.outputs[0].logprobs
        if not lps:
            return []
        pairs: List[Tuple[str, float]] = []
        for tid, lp in zip(token_ids, lps):
            entry = lp.get(tid) if isinstance(lp, dict) else None
            logp = getattr(entry, "logprob", None) if entry is not None else None
            prob = float(np.exp(logp)) if logp is not None else float("nan")
            pairs.append((tok.decode([tid]), prob))
        return pairs
    except Exception:
        return []


def _task_confidence(tok_probs: List[Tuple[str, float]]) -> Dict[str, float]:
    """Per-task confidence = prob of the first value token after each JSON key.

    Teacher-forced valid-label-token confidence (paper §3.3, default estimator).
    Falls back to the mean generated-token probability (sequence confidence)
    when a key/value cannot be located.
    """
    seq = [p for _, p in tok_probs if p == p]  # drop NaNs
    seq_conf = float(np.mean(seq)) if seq else 1.0
    keys = {"click": "click", "long_view": "long_view", "watch_time": "watch_time"}
    conf = {"click": seq_conf, "long_view": seq_conf, "watch_time": seq_conf}
    joined = "".join(t for t, _ in tok_probs)
    for task, key in keys.items():
        kpos = joined.find(key)
        if kpos < 0:
            continue
        # Walk tokens until we pass the key, then take the first token whose
        # text contains a digit (the label/value) and use its probability.
        run = ""
        passed_key = False
        for text, prob in tok_probs:
            run += text
            if not passed_key:
                if len(run) >= kpos + len(key):
                    passed_key = True
                continue
            if any(ch.isdigit() for ch in text) and prob == prob:
                conf[task] = float(prob)
                break
    return conf


def cold_videos(log, thr):
    return log["video_id"].value_counts().pipe(lambda s: s[s < thr]).index.values.astype(np.int64)


def cold_users(log, thr):
    return log["user_id"].value_counts().pipe(lambda s: s[s < thr]).index.values.astype(np.int64)


def cosine_topk(query, pool, pool_ids, top_k):
    qn = query / (np.linalg.norm(query) + 1e-8)
    pn = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-8)
    s = pn @ qn
    return pool_ids[np.argpartition(-s, min(top_k, len(s) - 1))[:top_k]]


def cf_topk_users(model, vid_emb, user_pool, user_ids, top_k, device):
    with torch.no_grad():
        u = model.F_U(user_pool.to(device))
        v = model.F_V(vid_emb.to(device))
        s = (u @ v).cpu().numpy()
    return user_ids[np.argpartition(-s, min(top_k, len(s) - 1))[:top_k]]


def cf_topk_videos(model, user_emb, video_pool, video_ids, top_k, device):
    with torch.no_grad():
        u = model.F_U(user_emb.to(device))
        v = model.F_V(video_pool.to(device))
        s = (v @ u).cpu().numpy()
    return video_ids[np.argpartition(-s, min(top_k, len(s) - 1))[:top_k]]


def load_coupled_filter(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = CoupledFilter(emb_dim=cfg["emb_dim"], hidden=cfg["hidden"],
                       out=cfg["out"], dropout=0.0,
                       user_emb_dim=cfg.get("user_emb_dim"),
                       video_emb_dim=cfg.get("video_emb_dim"))
    m.load_state_dict(ckpt["model_state"])
    m.eval(); m.to(device)
    return m


def parse_json_safe(txt: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", txt or "")
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    out = {}
    m1 = re.search(r'"?click"?\s*[:=]\s*([01])', txt or "")
    m2 = re.search(r'"?long_view"?\s*[:=]\s*([01])', txt or "")
    m3 = re.search(r'"?watch_time_seconds"?\s*[:=]\s*([0-9]+)', txt or "")
    if m1: out["click"] = int(m1.group(1))
    if m2: out["long_view"] = int(m2.group(1))
    if m3: out["watch_time_seconds"] = int(m3.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--focus_at_train", choices=["cold_item","cold_user","both"], required=True)
    p.add_argument("--with_reasoning", action="store_true")
    p.add_argument("--log_csvs", nargs="+",
                    default=["./data/kuairand_pure/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv"])
    p.add_argument("--video_meta_csv", default="./data/kuairand_pure/enriched_video_metadata_pure.csv")
    p.add_argument("--user_profile_csv", default="./llm_cold_item/data/user_profile_text.csv")
    p.add_argument("--user_features_csv",
                    default="./data/kuairand_pure/KuaiRand-Pure/data/user_features_pure.csv")
    p.add_argument("--video_text_emb_npz", default="./llm_cold_item/data/text_emb_video_pure.npz")
    p.add_argument("--user_text_emb_npz",  default="./llm_cold_item/data/text_emb_user_pure.npz")
    p.add_argument("--filter_mode", choices=["cosine","coupled"], default="coupled")
    p.add_argument("--filter_backbone", choices=["text","mf","lightgcn","ngcf"], default="text")
    p.add_argument("--filter_ckpt", default=None)
    p.add_argument("--direction", choices=["cold_item","cold_user","both"], default="both")
    p.add_argument("--cold_video_thr", type=int, default=50)
    p.add_argument("--cold_user_thr",  type=int, default=20)
    p.add_argument("--cold_video_ids_npz", default=None)
    p.add_argument("--cold_user_ids_npz", default=None)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_k_users", type=int, default=None)
    p.add_argument("--top_k_videos", type=int, default=None)
    p.add_argument("--top_l", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256, help="vLLM handles larger batch")
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--out_dir", default="./llm_cold_item/data")
    p.add_argument("--out_suffix", default="paper_vllm")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--gpu_mem_util", type=float, default=0.55,
                    help="vLLM gpu_memory_utilization (lower if sharing GPU)")
    p.add_argument("--max_model_len", type=int, default=2048)
    args = p.parse_args()

    print(f"Args: {vars(args)}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n[A] Load log + texts", flush=True)
    frames = [pd.read_csv(c,
                            usecols=["user_id","video_id","time_ms","is_click","long_view","play_time_ms"],
                            dtype={"user_id":np.int64,"video_id":np.int64,"time_ms":np.int64,
                                   "is_click":np.int64,"long_view":np.int64,"play_time_ms":np.int64})
              for c in args.log_csvs]
    log = pd.concat(frames, ignore_index=True).sort_values("time_ms").reset_index(drop=True)
    train_log = log.iloc[:int(len(log) * args.train_ratio)]
    pos_log = train_log[train_log["is_click"] == 1].copy()

    vm = pd.read_csv(args.video_meta_csv)
    video_views = dict(zip(vm["video_id"].astype(int), vm["semantic_view_en"].astype(str)))
    video_cats  = dict(zip(vm["video_id"].astype(int), vm["category_en"].astype(str)))
    video_caps  = dict(zip(vm["video_id"].astype(int), vm["caption_en"].astype(str)))
    up = pd.read_csv(args.user_profile_csv)
    user_profiles = dict(zip(up["user_id"].astype(int), up["profile_en"].astype(str)))
    uf = pd.read_csv(args.user_features_csv)
    user_feat_lookup = {int(r["user_id"]): r.to_dict() for _, r in uf.iterrows()}

    print(f"\n[B] Build per-user history + per-video clickers", flush=True)
    wt_max = 60
    user_history: Dict[int, List[Tuple[int,int,int,int]]] = {}
    video_clickers: Dict[int, List[Tuple[int,int,int,int]]] = {}
    for uid, grp in pos_log.groupby("user_id"):
        items = []
        for _, r in grp.sort_values("time_ms").iterrows():
            wt = max(0, min(wt_max, int(r["play_time_ms"]) // 1000))
            items.append((int(r["time_ms"]), int(r["video_id"]), wt, int(r["long_view"])))
        user_history[int(uid)] = items
    for vid, grp in pos_log.groupby("video_id"):
        items = []
        for _, r in grp.sort_values("time_ms").iterrows():
            wt = max(0, min(wt_max, int(r["play_time_ms"]) // 1000))
            items.append((int(r["time_ms"]), int(r["user_id"]), wt, int(r["long_view"])))
        video_clickers[int(vid)] = items
    user_summaries = {int(uid): build_user_behavior_summary(grp, video_cats)
                       for uid, grp in pos_log.groupby("user_id")}
    video_summaries = {int(vid): build_video_behavior_summary(grp, user_feat_lookup)
                        for vid, grp in train_log.groupby("video_id")}

    print(f"\n[C] Load text embeddings + Coupled Filter", flush=True)
    vte = np.load(args.video_text_emb_npz)
    v_emb = vte["text_emb"]; v_ids = vte["ids"]
    v_text_arr = np.zeros((int(v_ids.max())+1, v_emb.shape[1]), dtype=np.float32)
    v_text_arr[v_ids] = v_emb
    ute = np.load(args.user_text_emb_npz)
    u_emb = ute["text_emb"]; u_ids = ute["ids"]
    u_text_arr = np.zeros((int(u_ids.max())+1, u_emb.shape[1]), dtype=np.float32)
    u_text_arr[u_ids] = u_emb

    if args.filter_mode == "coupled":
        cf = load_coupled_filter(args.filter_ckpt, device)
        print(f"  Coupled Filter loaded from {args.filter_ckpt}", flush=True)
        u_filter_emb_t = torch.from_numpy(u_text_arr).float()
        v_filter_emb_t = torch.from_numpy(v_text_arr).float()

    # Cold detection
    pair_sources = []
    if args.direction in ("cold_item", "both"):
        if args.cold_video_ids_npz:
            cvs = np.load(args.cold_video_ids_npz)["cold_vids"].astype(np.int64)
            print(f"  [paper-cold] explicit cold vids: {len(cvs):,}", flush=True)
        else:
            cvs = cold_videos(train_log, args.cold_video_thr)
        valid_users = log["user_id"].unique().astype(np.int64)
        max_user_dim = u_filter_emb_t.shape[0] if args.filter_mode == "coupled" else len(u_text_arr)
        valid_users = valid_users[(valid_users >= 0) & (valid_users < max_user_dim)]
        pairs = []
        K = args.top_k_users or args.top_k
        if args.filter_mode == "coupled":
            u_pool_t = u_filter_emb_t[valid_users]
            for vid in cvs:
                tops = cf_topk_users(cf, v_filter_emb_t[int(vid)], u_pool_t, valid_users, K, device)
                for u in tops: pairs.append((int(u), int(vid)))
        else:
            u_pool = u_text_arr[valid_users]
            for vid in cvs:
                tops = cosine_topk(v_text_arr[vid], u_pool, valid_users, K)
                for u in tops: pairs.append((int(u), int(vid)))
        print(f"  cold-item pairs: {len(pairs):,}", flush=True)
        pair_sources.append(("cold_item", pairs))

    if args.direction in ("cold_user", "both"):
        if args.cold_user_ids_npz:
            cus = np.load(args.cold_user_ids_npz)["cold_uids"].astype(np.int64)
            print(f"  [paper-cold] explicit cold uids: {len(cus):,}", flush=True)
        else:
            cus = cold_users(train_log, args.cold_user_thr)
        valid_videos = log["video_id"].unique().astype(np.int64)
        max_v_dim = v_filter_emb_t.shape[0] if args.filter_mode == "coupled" else len(v_text_arr)
        valid_videos = valid_videos[(valid_videos >= 0) & (valid_videos < max_v_dim)]
        pairs = []
        K = args.top_k_videos or args.top_k
        if args.filter_mode == "coupled":
            v_pool_t = v_filter_emb_t[valid_videos]
            for uid in cus:
                tops = cf_topk_videos(cf, u_filter_emb_t[int(uid)], v_pool_t, valid_videos, K, device)
                for vid in tops: pairs.append((int(uid), int(vid)))
        else:
            v_pool = v_text_arr[valid_videos]
            for uid in cus:
                tops = cosine_topk(u_text_arr[uid], v_pool, valid_videos, K)
                for vid in tops: pairs.append((int(uid), int(vid)))
        print(f"  cold-user pairs: {len(pairs):,}", flush=True)
        pair_sources.append(("cold_user", pairs))

    # Free CF GPU memory before vLLM loads
    if args.filter_mode == "coupled":
        del cf
        torch.cuda.empty_cache()

    print(f"\n[D] Load vLLM (continuous batching, paged attention)", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=32,        # our LoRA r=16, alpha=32
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype="float16",
        trust_remote_code=True,
    )
    lora_req = LoRARequest("kuair_adapter", 1, args.adapter_dir)
    # logprobs=1 lets us read each generated token's log-probability, the basis
    # of CaliChain's teacher-forced valid-label-token confidence (paper §3.3).
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, top_p=1.0,
                        top_k=-1, logprobs=1)

    sys_p = SYSTEM_PROMPT_WITH_REASONING if args.with_reasoning else SYSTEM_PROMPT_NO_REASONING

    def build_prompt_for_pair(uid, vid):
        user_prof = user_profiles.get(uid, "")
        user_beh  = user_summaries.get(uid, "No prior click history yet.")
        v_text    = video_views.get(vid, "")
        v_cat     = video_cats.get(vid, "")
        v_cap     = video_caps.get(vid, "")
        v_beh     = video_summaries.get(vid, "")
        if not user_prof or not v_text: return None
        hist = [h for h in user_history.get(uid, []) if h[1] != vid]
        if vid < len(v_text_arr):
            hist_filt = filter_user_history_by_sim(hist, v_text_arr[vid], v_text_arr, args.top_l)
        else:
            hist_filt = hist[-args.top_l:]
        hist_lines = build_history_lines(hist_filt, video_views)
        clk = [c for c in video_clickers.get(vid, []) if c[1] != uid]
        if uid < len(u_text_arr):
            clk_filt = filter_video_clickers_by_sim(clk, u_text_arr[uid], u_text_arr, args.top_l)
        else:
            clk_filt = clk[-args.top_l:]
        clk_lines = build_video_clicker_lines(clk_filt, user_feat_lookup)
        return build_user_prompt_full(user_prof, user_beh, hist_lines,
                                       v_text, v_cat, v_cap, v_beh, clk_lines,
                                       focus=args.focus_at_train,
                                       reasoning_request=args.with_reasoning)

    for tag, pairs in pair_sources:
        out_jsonl = os.path.join(args.out_dir,
                                  f"coldllm_simulated_{args.out_suffix}_{tag}.jsonl")
        seen = set()
        if os.path.exists(out_jsonl):
            with open(out_jsonl) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        seen.add((int(r["user_raw_id"]), int(r["video_raw_id"])))
                    except Exception: pass
            print(f"  [{tag}] resume: {len(seen):,}", flush=True)
        todo = [(u, v) for u, v in pairs if (u, v) not in seen]
        print(f"\n[E-{tag}] Inference on {len(todo):,} pairs (vLLM batch={args.batch_size})", flush=True)
        t0 = time.time(); done = 0; n_yes = 0
        with open(out_jsonl, "a", encoding="utf-8") as fout:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i:i + args.batch_size]
                prompts = []; kept = []
                for u, v in batch:
                    pr = build_prompt_for_pair(u, v)
                    if pr is None: continue
                    chat = tok.apply_chat_template(
                        [{"role":"system","content":sys_p},
                         {"role":"user","content":pr}],
                        tokenize=False, add_generation_prompt=True)
                    prompts.append(chat); kept.append((u, v))
                if not prompts: continue
                outputs = llm.generate(prompts, sp, lora_request=lora_req)
                for (u, v), out in zip(kept, outputs):
                    txt = out.outputs[0].text
                    parsed = parse_json_safe(txt)
                    click = int(parsed.get("click", 0)); click = 1 if click > 0 else 0
                    lv    = int(parsed.get("long_view", 0)); lv = 1 if lv > 0 else 0
                    wt    = int(parsed.get("watch_time_seconds", 0))
                    wt    = max(0, min(60, wt))
                    # CaliChain per-task confidence from token logprobs (§3.3).
                    conf = _task_confidence(_per_token_probs(out, tok))
                    rec = {"user_raw_id": int(u), "video_raw_id": int(v),
                           "click": click, "long_view": lv, "watch_time_seconds": wt,
                           "llm_yes": click,
                           "q_click": conf["click"], "q_long_view": conf["long_view"],
                           "q_watch_time": conf["watch_time"]}
                    fout.write(json.dumps(rec) + "\n")
                    done += 1
                    if click == 1: n_yes += 1
                if done % args.batch_size == 0:
                    el = time.time() - t0
                    print(f"  [{tag}] {done:,}/{len(todo):,} rate={done/max(el,1e-6):.1f}/s "
                          f"yes={n_yes/max(done,1)*100:.1f}% elapsed={el:.0f}s", flush=True)
                    fout.flush()
        print(f"  [{tag}] DONE total={done:,} yes={n_yes:,}", flush=True)


if __name__ == "__main__":
    main()
