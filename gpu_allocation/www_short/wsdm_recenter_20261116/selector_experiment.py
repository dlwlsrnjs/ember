"""Fixed-positive negative-selection controls in the original WSDM learner.

prepare: reconstruct warm-only retrieval hardness and freeze all training rows.
worker: claim one of nine predeclared jobs, train on validation only, save the
selected checkpoint, then evaluate the already-consumed historical test.
No new teacher inference; no hyperparameter search; no test-based selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "ember2"))
from ember2.cache import build_cache
from ember2.config import FilterConfig, LearnerConfig
from ember2.filterstage import (HashingTextEncoder, build_filter_training_data,
                               retrieve_candidates, train_coupled_filter, whiten)
from ember2.ml1m import load_ml1m
from ember2.simulate import COLD_ITEM, COLD_USER, SimulatedRow, balance_negatives
from ember2.train import evaluate, train, warm_support
from ember2.metrics import recall_ndcg_at_k

OUT = HERE / "selector_run_v1"
ARMS = ("llm", "random", "hard")
JOBS = [(s, arm) for arm in ARMS for s in range(3)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    tmp.replace(path)


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pack(r):
    return dict(u=int(r.user), i=int(r.item), d=r.direction,
                b=list(map(float, r.binary)), c=float(r.core), q=list(map(float, r.q)))


def unpack(d):
    return SimulatedRow(d["u"], d["i"], d["d"], np.asarray(d["b"], np.float32),
                        d["c"], np.asarray(d["q"], np.float32))


def key(r):
    return int(r.user), int(r.item), r.direction


def bucket(r):
    assert r.direction in (COLD_ITEM, COLD_USER)
    return r.direction, int(r.item if r.direction == COLD_ITEM else r.user)


def raw_data(seed):
    return load_ml1m(str(ROOT / "data_kuairand/ml-1m"), seed=seed)


def config():
    return LearnerConfig(n_epochs=20, reliability="support",
                         use_confidence_weighting=False, use_confidence_prior=False)


class ObservedTeacher:
    def __init__(self, raw):
        idx = np.flatnonzero(raw.warm_training_mask() & (raw.binary[:, 0] > 0))
        self.positive = set(zip(raw.user[idx].tolist(), raw.item[idx].tolist()))

    def judge(self, pairs):
        return np.asarray([float(tuple(p) in self.positive) for p in pairs], np.float32)


def freeze_protocol():
    paths = [Path(__file__), ROOT / "www_short/two_sided_certificates/single_thread.py"]
    paths += [ROOT / f"ember2/ember2/{n}.py" for n in
              ("train", "cache", "config", "dataset", "ml1m", "rawdata", "simulate",
               "filterstage", "learner", "gradproc", "calichain", "metrics",
               "routefuse", "blocks", "coldsynth")]
    paths += [ROOT / f"data_kuairand/ml-1m/{name}.dat" for name in ("ratings", "users", "movies")]
    paths += [ROOT / f"ember2/runs/ar_fresh_ml1m_s{s}/dllm_rows.json" for s in range(3)]
    protocol = {
        "version": "original-WSDM-fixed-P-selector-v1",
        "created_unix": time.time(), "seeds": [0, 1, 2], "arms": list(ARMS),
        "jobs": [dict(seed=s, arm=a) for s, a in JOBS],
        "cfg": asdict(config()), "filter_cfg": asdict(FilterConfig()),
        "fixed": ["queried Q", "positive pairs and their positions", "negative-label slots",
                  "negative count by cold entity and direction", "context RNG", "schema",
                  "uniform q", "warm-mean core", "support gate", "training seed", "validation lists"],
        "negative_universe": "all queried records minus fixed retained positive pairs, within cold-entity/direction bucket; not restricted by teacher negative verdict",
        "random_rng_seed": "seed+73031, ascending bucket order",
        "hardness": "reconstructed warm-only coupled-filter dot score, descending, lexicographic u/i tie break",
        "selected_checkpoint": "earliest maximum validation overall Recall@20, original patience5/min_epochs10",
        "test_access": "only after selected checkpoint and its metadata are saved",
        "test_scope": "post-hoc additional contrasts on previously consumed historical test; NOT independent confirmation",
        "reproduction_policy": "retain each first completed fit; report historical baseline difference; never rerun to obtain a desired match or sign",
        "predeclared_outcomes": ["cold Recall@20 primary", "cold NDCG@20", "warm Recall@20"],
        "not_fixed": ["opposite warm-side degree", "historical retriever checkpoint equivalence"],
        "thread_policy": "OMP/BLAS1 via existing wrapper; torch intra/inter-op1; original DataLoader workers2",
        "file_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    path = OUT / "FROZEN_PROTOCOL.json"
    assert not path.exists(), "Preserve completed runs; use a new version directory for changed designs"
    json_write(path, protocol)


def verify_frozen():
    p = OUT / "FROZEN_PROTOCOL.json"
    protocol = json.loads(p.read_text())
    for name, digest in protocol["file_sha256"].items():
        assert sha(ROOT / name) == digest, f"Frozen source/data changed: {name}"
    return sha(p)


def prepare_seed(seed, device):
    dest = OUT / f"seed{seed}"
    assert not (dest / "PREPARED.json").exists()
    dest.mkdir(parents=True, exist_ok=True)
    raw = raw_data(seed)
    warm = raw.warm_training_mask()
    warm_rate, warm_core = float(raw.binary[warm, 0].mean()), float(raw.core[warm].mean())
    source = ROOT / f"ember2/runs/ar_fresh_ml1m_s{seed}/dllm_rows.json"
    rows = [replace(unpack(r), core=warm_core, q=np.ones(3, np.float32))
            for r in json.loads(source.read_text())]
    assert len(rows) == len({key(r) for r in rows}) == 12000
    for r in rows:
        assert bool(raw.is_cold_user[r.user]) != bool(raw.is_cold_item[r.item])
        assert (r.direction == COLD_ITEM) == bool(raw.is_cold_item[r.item])
    cap = int(sum(r.binary[0] > 0 for r in rows))
    retained = balance_negatives(rows, warm_rate, np.random.default_rng(seed))
    if len(retained) > cap:
        idx = np.random.default_rng(seed).choice(len(retained), cap, replace=False)
        retained = [retained[k] for k in sorted(idx.tolist())]
    assert len(retained) == cap
    positive = {key(r) for r in retained if r.binary[0] > 0}
    negative_positions = defaultdict(list)
    for index, r in enumerate(retained):
        if r.binary[0] <= 0:
            negative_positions[bucket(r)].append(index)
    eligible = defaultdict(list)
    for r in rows:
        if key(r) not in positive:
            eligible[bucket(r)].append(r)
    for b, slots in negative_positions.items():
        assert len(eligible[b]) >= len(slots)

    print(f"PREPARE seed{seed}: warm{int(warm.sum())}, B{cap}, P{len(positive)}, N{cap-len(positive)}", flush=True)
    encoder = HashingTextEncoder(dim=256, seed=seed)
    ue = whiten(encoder.encode(raw.user_text), 128)
    ie = whiten(encoder.encode(raw.item_text), 128)
    filt_data = build_filter_training_data(raw, ObservedTeacher(raw), 200_000, 20_000,
                                         np.random.default_rng(seed))
    filt = train_coupled_filter(ue, ie, filt_data, FilterConfig(), device=device, seed=seed)
    dev = torch.device(device)
    with torch.no_grad():
        zu = filt.z_user(torch.as_tensor(ue, device=dev))
        zi = filt.z_item(torch.as_tensor(ie, device=dev))
        u = torch.tensor([r.user for r in rows], device=dev)
        i = torch.tensor([r.item for r in rows], device=dev)
        scores = (zu[u] * zi[i]).sum(-1).cpu().numpy()
    score = {key(r): float(v) for r, v in zip(rows, scores)}
    teacher_accept = {key(r): bool(r.binary[0] > 0) for r in rows}
    rebuilt = retrieve_candidates(filt, ue, ie, raw, 30, device=device)
    rebuilt_keys = {(int(u), int(i), d) for d, pairs in rebuilt.items() for u, i in pairs}
    overlap = sum(key(r) in rebuilt_keys for r in rows)
    torch.save({"state_dict": {n: t.detach().cpu() for n, t in filt.state_dict().items()},
                "cfg": asdict(FilterConfig()), "seed": seed}, dest / "retriever.pt")
    np.save(dest / "query_hardness.npy", scores)
    del filt, zu, zi
    torch.cuda.empty_cache()

    rng = np.random.default_rng(seed + 73031)
    selected = {"llm": retained, "random": list(retained), "hard": list(retained)}
    for b in sorted(negative_positions):
        positions = negative_positions[b]
        pool = sorted(eligible[b], key=key)
        random_idx = rng.choice(len(pool), len(positions), replace=False)
        random_rows = [pool[j] for j in random_idx]
        hard_rows = sorted(pool, key=lambda r: (-score[key(r)], key(r)))[:len(positions)]
        for arm, chosen in (("random", random_rows), ("hard", hard_rows)):
            for pos, r in zip(positions, chosen):
                selected[arm][pos] = replace(r, binary=np.zeros(2, np.float32),
                                            core=warm_core, q=np.ones(3, np.float32))
    audits = {}
    for arm in ARMS:
        use = selected[arm]
        assert len(use) == cap == len({key(r) for r in use})
        assert {key(r) for r in use if r.binary[0] > 0} == positive
        assert np.array_equal(np.stack([r.binary for r in use]),
                              np.stack([r.binary for r in retained]))
        assert Counter(bucket(r) for r in use if r.binary[0] <= 0) == Counter(
            bucket(r) for r in retained if r.binary[0] <= 0)
        for index, base in enumerate(retained):
            if base.binary[0] > 0:
                assert pack(base) == pack(use[index])
        path = dest / f"{arm}_rows.json"
        json_write(path, [pack(r) for r in use])
        negative = [r for r in use if r.binary[0] <= 0]
        audits[arm] = {
            "records_sha256": sha(path), "n_aug": cap, "n_positive": len(positive),
            "n_negative": len(negative), "cold_entity_direction_buckets": len(negative_positions),
            "mean_negative_hardness": float(np.mean([score[key(r)] for r in negative])),
            "negative_teacher_primary_accepts": sum(teacher_accept[key(r)] for r in negative),
        }
    json_write(dest / "PREPARED.json", {
        "seed": seed, "source_sha256": sha(source), "warm_mask_sha256": array_sha(warm),
        "reconstructed_top30_query_overlap": overlap / len(rows),
        "reconstructed_top30_query_overlap_count": overlap,
        "historical_retriever_checkpoint_equivalence": "not established",
        "score_sha256": sha(dest / "query_hardness.npy"), "arms": audits,
    })
    print(f"PREPARED seed{seed}: reconstructed top30 overlap {overlap}/12000", flush=True)


@torch.no_grad()
def per_list_metrics(model, split, device, dest):
    model.eval()
    chunks = []
    for start in range(0, len(split.feature_ids), 4096):
        f = torch.as_tensor(split.feature_ids[start:start+4096], device=device)
        chunks.append(model(f)["binary_logits"][:, 0].cpu().numpy())
    scores = np.concatenate(chunks)
    starts = np.r_[0, np.flatnonzero(np.diff(split.group_id)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    g, user, cold, cases, recall, ndcg = [], [], [], [], [], []
    for lo, hi in zip(starts, ends):
        labels = split.binary[lo:hi, 0]
        r, n = recall_ndcg_at_k(scores[lo:hi], labels, 20)
        if r is None:
            continue
        relevant = labels > 0
        g.append(int(split.group_id[lo])); user.append(int(split.feature_ids[lo, 0] - 1))
        cold.append(bool(split.is_cold[lo:hi][relevant].any()))
        cases.append(int(split.case[lo])); recall.append(r); ndcg.append(n)
    np.savez_compressed(dest, group_id=g, user=user, cold=cold, case=cases,
                        recall20=recall, ndcg20=ndcg)
    return {name: {"recall@20": float(np.asarray(recall)[mask].mean()),
                   "ndcg@20": float(np.asarray(ndcg)[mask].mean())}
            for name, mask in (("overall", np.ones(len(g), bool)),
                               ("cold", np.asarray(cold)), ("warm", ~np.asarray(cold)))}


def fit(seed, arm, device, label):
    raw = raw_data(seed)
    prep = OUT / f"seed{seed}"
    row_path = prep / f"{arm}_rows.json"
    audit = json.loads((prep / "PREPARED.json").read_text())
    assert sha(row_path) == audit["arms"][arm]["records_sha256"]
    rows = [unpack(d) for d in json.loads(row_path.read_text())]
    cache = build_cache(raw, rows, np.random.default_rng(seed), 100, 1500, with_test=True)
    historical_test, cache.test = cache.test, None
    dest = OUT / "fits" / f"s{seed}_{arm}"
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "RESULT.json").exists()
    cfg = config()
    inputs = dict(seed=seed, arm=arm, rows_sha256=sha(row_path),
                  features_sha256=array_sha(cache.train_feature_ids),
                  labels_sha256=array_sha(cache.train_binary),
                  context_sha256=array_sha(cache.train_feature_ids[:, -2:]),
                  validation_features_sha256=array_sha(cache.val.feature_ids),
                  test_features_sha256=array_sha(historical_test.feature_ids),
                  test_labels_sha256=array_sha(historical_test.binary),
                  cfg=asdict(cfg), worker=label, started_unix=time.time(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  torch_num_threads=torch.get_num_threads(),
                  torch_num_interop_threads=torch.get_num_interop_threads(),
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    json_write(dest / "INPUTS.json", inputs)
    print(f"START FIT s{seed}/{arm} device{device}", flush=True)
    started = time.monotonic()
    result = train(cache, cfg, device=device, seed=seed, verbose=True, return_model=True)
    model = result.pop("model")
    selected = max(result["history"], key=lambda h: h["val"]["overall"]["recall@20"])
    checkpoint = dest / "selected.pt"
    support_user, support_item = warm_support(cache)
    torch.save({"state_dict": {n: t.detach().cpu() for n, t in model.state_dict().items()},
                "cfg": asdict(cfg), "fields": asdict(cache.fields), "tasks": asdict(cache.tasks),
                "core_mean": cache.core_mean, "core_std": cache.core_std,
                "warm_support_user": support_user, "warm_support_item": support_item,
                "seed": seed, "arm": arm}, checkpoint)
    selection = dict(seed=seed, arm=arm, criterion="validation overall recall@20",
                     chosen_epoch=selected["epoch"], val=result["val"],
                     checkpoint_sha256=sha(checkpoint), frozen_before_test_unix=time.time(),
                     test_used_in_fit_or_selection=False)
    json_write(dest / "SELECTION.json", selection)
    json_write(dest / "VALIDATION_ONLY.json", result)
    # No checkpoint/gate/hyperparameter changes are allowed below this line.
    evaluated = evaluate(model, historical_test, cache, torch.device(device))
    per = per_list_metrics(model, historical_test, device, dest / "test_per_list.npz")
    for name in ("overall", "cold", "warm"):
        for metric in ("recall@20", "ndcg@20"):
            assert abs(per[name][metric] - evaluated.ranking[name][metric]) < 1e-10
    result["historical_test"] = asdict(evaluated)
    result["selection"] = selection
    result["fit_and_eval_wall_seconds"] = time.monotonic() - started
    result["test_scope"] = "post hoc on already-consumed historical test, not independent confirmation"
    if arm == "llm":
        old = json.loads((ROOT / "ember2/runs/record_ablation_fresh_matched.json").read_text())
        previous = old[str(seed)]["binary+uniform"]["cold_recall@20"]
        result["historical_retained_reproduction"] = {
            "old_cold_recall20": previous,
            "new_cold_recall20": evaluated.ranking["cold"]["recall@20"],
            "difference": evaluated.ranking["cold"]["recall@20"] - previous}
    json_write(dest / "RESULT.json", result)
    print(f"DONE FIT s{seed}/{arm}: {evaluated.brief()} epoch{selected['epoch']}", flush=True)
    del model, result, cache, raw
    torch.cuda.empty_cache()


def worker(device, label):
    verify_frozen()
    for seed in range(3):
        assert (OUT / f"seed{seed}/PREPARED.json").exists(), "Prepare all records before any fit"
    claims = OUT / "claims"
    claims.mkdir(exist_ok=True)
    for seed, arm in JOBS:
        path = claims / f"s{seed}_{arm}.json"
        try:
            with path.open("x") as stream:
                json.dump(dict(seed=seed, arm=arm, worker=label, pid=os.getpid(),
                               claimed_unix=time.time()), stream)
        except FileExistsError:
            continue
        fit(seed, arm, device, label)
    print(f"WORKER COMPLETE {label}", flush=True)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "worker"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--label", default="worker")
    args = parser.parse_args()
    if args.stage == "prepare":
        freeze_protocol()
        for seed in range(3):
            prepare_seed(seed, args.device)
        verify_frozen()
    else:
        worker(args.device, args.label)


if __name__ == "__main__":
    main()
