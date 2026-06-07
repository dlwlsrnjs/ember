"""Build Knowledge-Distillation training data for the Coupled Filter.

We sample random (user, video) pairs from log_random's train period and
query gpt-4o-mini with the ColdLLM Yes/No prompt. The resulting Ẑ_{ui}
becomes the soft label for the filter's KD loss (L_coupled in paper §4.3.2).

Inputs (already on disk):
  - llm_cold_item/data/user_combined_text.csv  (user profile + behavior)
  - data/kuairand_pure/enriched_video_metadata_pure.csv  (video EN text)
  - data/kuairand_pure/.../log_random_4_22_to_5_08_pure.csv  (for valid IDs)

Output:
  llm_cold_item/data/filter_kd_labels.jsonl
    rows: {user_raw_id, video_raw_id, llm_yes (0/1), source (pos|unobs)}

We include:
  - 50% from real positives (gpt-4o-mini should mostly say Yes → calibration)
  - 50% from random unobserved pairs (mostly No → calibration)
This balanced sampling reduces label noise.
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from openai import AsyncOpenAI
except ImportError:
    print("openai required", file=sys.stderr); raise


SYSTEM_PROMPT = (
    "You are a short-video recommender simulator. Given a user's profile + "
    "behavior summary and a video's description, predict whether the user "
    "would click on this video. Respond with exactly 'Yes' or 'No'."
)


def build_prompt(user_text: str, video_text: str, video_cat: str) -> str:
    return (
        f"### USER\n{user_text}\n\n"
        f"### VIDEO\nCategory: {video_cat}\nDescription: {video_text}\n\n"
        f"### TASK\nWill the user click this video? Answer Yes or No."
    )


async def query_one(client, sem, model, user_text, video_text, video_cat,
                     user_id, video_id, source, max_retries=3):
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(user_text, video_text, video_cat)},
                    ],
                    temperature=0.0,
                    max_completion_tokens=4,
                    timeout=20.0,
                )
                txt = (resp.choices[0].message.content or "").strip().lower()
                llm_yes = 1 if txt.startswith("yes") else 0
                return {"user_raw_id": int(user_id), "video_raw_id": int(video_id),
                         "llm_yes": llm_yes, "source": source}
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"user_raw_id": int(user_id), "video_raw_id": int(video_id),
                             "llm_yes": -1, "source": source}
                await asyncio.sleep(1.5 * (attempt + 1))


async def run_all(records, model, concurrency, out_jsonl, resume_set):
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    pending = []
    for r in records:
        key = (r["user_id"], r["video_id"])
        if key in resume_set: continue
        pending.append(query_one(client, sem, model,
                                   r["user_text"], r["video_text"], r["video_cat"],
                                   r["user_id"], r["video_id"], r["source"]))
    print(f"\n  Querying {len(pending):,} pairs (model={model}, concurrency={concurrency})…",
          flush=True)
    if not pending:
        print("  (nothing to do)", flush=True); return
    t0 = time.time(); done = [0]; n_yes = [0]
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(pending):
            rec = await coro
            f.write(json.dumps(rec) + "\n")
            done[0] += 1
            if rec["llm_yes"] == 1: n_yes[0] += 1
            if done[0] % 500 == 0:
                el = time.time() - t0
                print(f"    {done[0]:,}/{len(pending):,} done in {el:.0f}s "
                      f"({done[0]/max(el,1e-6):.1f}/s, yes={n_yes[0]/max(done[0],1)*100:.1f}%)",
                      flush=True); f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_csv",
                    default="./data/kuairand_pure/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv")
    p.add_argument("--user_combined_csv", default="./llm_cold_item/data/user_combined_text.csv")
    p.add_argument("--video_meta_csv", default="./data/kuairand_pure/enriched_video_metadata_pure.csv")
    p.add_argument("--out_jsonl", default="./llm_cold_item/data/filter_kd_labels.jsonl")
    p.add_argument("--n_pos", type=int, default=10000)
    p.add_argument("--n_unobs", type=int, default=10000)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[A] Load log + texts", flush=True)
    log = pd.read_csv(args.log_csv,
                       usecols=["user_id", "video_id", "time_ms", "is_click"],
                       dtype={"user_id": np.int64, "video_id": np.int64,
                              "time_ms": np.int64, "is_click": np.int64}) \
        .sort_values("time_ms").reset_index(drop=True)
    train_log = log.iloc[:int(len(log) * args.train_ratio)]
    pos = train_log[train_log["is_click"] == 1][["user_id", "video_id"]].values
    print(f"  train_period {len(train_log):,}, positives {len(pos):,}", flush=True)

    uc = pd.read_csv(args.user_combined_csv)
    user_texts = dict(zip(uc["user_id"].astype(int), uc["combined_en"].astype(str)))
    vm = pd.read_csv(args.video_meta_csv)
    video_texts = dict(zip(vm["video_id"].astype(int), vm["semantic_view_en"].astype(str)))
    video_cats  = dict(zip(vm["video_id"].astype(int), vm["category_en"].astype(str)))
    valid_users  = np.array(list(user_texts.keys()), dtype=np.int64)
    valid_videos = np.array(list(video_texts.keys()), dtype=np.int64)

    user_pos_set: Dict[int, set] = {}
    for u, v in pos:
        user_pos_set.setdefault(int(u), set()).add(int(v))

    print(f"[B] Sample {args.n_pos:,} positives + {args.n_unobs:,} unobserved", flush=True)
    records = []
    # positives
    idx = rng.choice(len(pos), min(args.n_pos, len(pos)), replace=False)
    for i in idx:
        u, v = int(pos[i, 0]), int(pos[i, 1])
        if u in user_texts and v in video_texts:
            records.append({"user_id": u, "video_id": v,
                             "user_text": user_texts[u], "video_text": video_texts[v],
                             "video_cat": video_cats.get(v, ""), "source": "pos"})
    # unobserved
    cnt = 0; max_try = args.n_unobs * 5
    tried = 0
    while cnt < args.n_unobs and tried < max_try:
        tried += 1
        u = int(rng.choice(valid_users)); v = int(rng.choice(valid_videos))
        if v in user_pos_set.get(u, set()): continue
        records.append({"user_id": u, "video_id": v,
                         "user_text": user_texts[u], "video_text": video_texts[v],
                         "video_cat": video_cats.get(v, ""), "source": "unobs"})
        cnt += 1
    print(f"  composed {len(records):,} records", flush=True)

    resume_set: set = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    resume_set.add((int(r["user_raw_id"]), int(r["video_raw_id"])))
                except Exception: pass
        print(f"[C] Resume: {len(resume_set):,} already done", flush=True)

    asyncio.run(run_all(records, args.model, args.concurrency, args.out_jsonl, resume_set))

    # Summary
    n_yes_pos = 0; n_pos_total = 0; n_yes_unobs = 0; n_unobs_total = 0
    with open(args.out_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "pos":
                n_pos_total += 1; n_yes_pos += (r["llm_yes"] == 1)
            else:
                n_unobs_total += 1; n_yes_unobs += (r["llm_yes"] == 1)
    print(f"\nSummary:", flush=True)
    print(f"  positives: {n_pos_total:,} total, LLM yes={n_yes_pos:,} ({n_yes_pos/max(n_pos_total,1)*100:.1f}%)",
          flush=True)
    print(f"  unobserved: {n_unobs_total:,} total, LLM yes={n_yes_unobs:,} ({n_yes_unobs/max(n_unobs_total,1)*100:.1f}%)",
          flush=True)
    print(f"  → {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
