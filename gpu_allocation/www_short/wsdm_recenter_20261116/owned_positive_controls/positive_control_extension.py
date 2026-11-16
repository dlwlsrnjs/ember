"""Immutable post-hoc positive/ordinary-negative references for the WSDM study.

This module does not alter or run the old selector_experiment.main(). Root
launches preparation, then at most two workers through single_thread.py.
No new teacher inference, optimization search, or outcome-based rerun occurs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RECENTER = HERE.parent
sys.path.insert(0, str(RECENTER))
import selector_experiment as sel
from ember2.cache import build_cache
from ember2.dataset import Cache
from ember2.simulate import COLD_ITEM, COLD_USER, SimulatedRow
from ember2.train import evaluate, train, warm_support

ROOT = sel.ROOT
RUN = HERE / "run_v1"
ARMS = ("none", "fixedP_only", "fixedP_repeatB", "accepted_all_plus_random")
SEEDS = (0, 1, 2)
JOBS = tuple((seed, arm) for arm in ARMS for seed in SEEDS)
REPEAT_OFFSET = 92051
ORDINARY_OFFSET = 93071
TRAIN_ARRAYS = ("train_feature_ids", "train_binary", "train_core", "train_q", "train_is_aug")
SPLIT_ARRAYS = ("feature_ids", "binary", "core", "group_id", "is_cold", "is_observed", "case")


def plain(value):
    return json.loads(json.dumps(value))


def relative(path):
    return str(Path(path).relative_to(ROOT))


def split_pins(split):
    return {name: sel.array_sha(getattr(split, name)) for name in SPLIT_ARRAYS}


def train_pins(cache):
    return {name: sel.array_sha(getattr(cache, name)) for name in TRAIN_ARRAYS}


def clone_training(cache, indices=None):
    data = {name: (getattr(cache, name).copy() if indices is None else
                   getattr(cache, name)[indices].copy()) for name in TRAIN_ARRAYS}
    return replace(cache, **data)


def assert_same_evaluation(cache, reference):
    assert split_pins(cache.val) == split_pins(reference.val)
    assert split_pins(cache.test) == split_pins(reference.test)
    assert cache.core_mean == reference.core_mean and cache.core_std == reference.core_std
    assert asdict(cache.fields) == asdict(reference.fields)
    assert asdict(cache.tasks) == asdict(reference.tasks)


def cache_pairs(cache, mask):
    ids = cache.train_feature_ids[mask]
    return [(int(u) - 1, int(i) - 1) for u, i in zip(ids[:, 0], ids[:, cache.fields.n_user])]


def coverage(raw, pairs):
    cu, ci = raw.is_cold_user, raw.is_cold_item
    counts = Counter()
    for user, item in pairs:
        assert bool(cu[user]) != bool(ci[item])
        counts[(COLD_ITEM, item) if ci[item] else (COLD_USER, user)] += 1
    return counts


def summarize_cache(raw, cache, original):
    aug = cache.train_is_aug > 0
    positive = aug & (cache.train_binary[:, 0] > 0)
    negative = aug & ~positive
    pairs = cache_pairs(cache, aug)
    old_coverage = coverage(raw, cache_pairs(original, original.train_is_aug > 0))
    current = coverage(raw, pairs)
    return {
        "n_warm": cache.n_warm, "n_aug": cache.n_aug,
        "n_aug_positive": int(positive.sum()), "n_aug_negative": int(negative.sum()),
        "n_unique_aug_pairs": len(set(pairs)),
        "duplicate_aug_pair_rows": len(pairs) - len(set(pairs)),
        "aug_cold_entity_direction_buckets": len(current),
        "old_retained_buckets_lost": len(set(old_coverage) - set(current)),
        "old_retained_buckets_added": len(set(current) - set(old_coverage)),
        "same_cold_entity_direction_row_counts_as_original": current == old_coverage,
        "train_array_sha256": train_pins(cache),
        "validation_sha256": split_pins(cache.val),
        "historical_test_sha256": split_pins(cache.test),
        "core_mean": cache.core_mean, "core_std": cache.core_std,
    }


def freeze_protocol():
    old_protocol_sha = sel.verify_frozen()
    assert not RUN.exists(), "Preserve prior attempts; do not overwrite or reroll run_v1"
    for seed, arm in sel.JOBS:
        assert (sel.OUT / f"fits/s{seed}_{arm}/RESULT.json").exists(), "Finish all original nine fits first"
    old = json.loads((sel.OUT / "FROZEN_PROTOCOL.json").read_text())
    pins = dict(old["file_sha256"])
    paths = [Path(__file__), HERE / "PRE_EXECUTION_PROTOCOL_KO.txt", sel.OUT / "FROZEN_PROTOCOL.json"]
    for seed in SEEDS:
        paths += [sel.OUT / f"seed{seed}/llm_rows.json", sel.OUT / f"seed{seed}/PREPARED.json",
                  sel.OUT / f"fits/s{seed}_llm/INPUTS.json"]
        for arm in sel.ARMS:
            paths += [sel.OUT / f"fits/s{seed}_{arm}/RESULT.json",
                      sel.OUT / f"fits/s{seed}_{arm}/SELECTION.json"]
    pins.update({relative(path): sel.sha(path) for path in paths})
    protocol = {
        "version": "WSDM-positive-controls-posthoc-extension-v1",
        "created_unix": time.time(), "old_selector_protocol_sha256": old_protocol_sha,
        "seeds": list(SEEDS), "arms": list(ARMS),
        "jobs": [dict(seed=seed, arm=arm) for seed, arm in JOBS],
        "cfg": asdict(sel.config()),
        "selection": "earliest maximum validation overall Recall@20, original min10/patience5/max20",
        "test_access": "after selected checkpoint, validation result, and selection metadata are saved",
        "test_scope": "post-hoc extension on consumed historical outcomes; no unused cohort claim",
        "reason_for_extension": "positive-only, augmentation-volume, and ordinary-negative alternatives raised after external review; not preregistered before earlier results",
        "none": "warm log only; subset of full historical LLM cache",
        "fixedP_only": "warm + exact retained positive feature rows including their original sampled contexts",
        "fixedP_repeatB": {
            "rule": "keep every original positive slot; replace every negative slot by a uniform-with-replacement fixed-P pair, retaining the target slot's original context",
            "rng": f"seed+{REPEAT_OFFSET}", "budget": "same B as old retained mixture",
            "limits": "duplicate positives; no negative labels; cold-entity coverage/weights differ; positive-weight reference, not a coverage-matched selector",
        },
        "accepted_all_plus_random": {
            "positive_set": "all primary-positive teacher records in original 12K queried Q, preserving two conversion labels",
            "negative_quota": "per cold-entity/direction bucket: round(P_bucket*(1-warm_primary_rate)/warm_primary_rate), capped at eligible count if necessary",
            "eligible": "all warm opposite-side entities except that bucket's all-accepted positive counterparts; no teacher-negative filtering or non-warm outcomes",
            "sampling": "without replacement; sorted buckets and sorted opposite-side IDs",
            "rng": f"seed+{ORDINARY_OFFSET}",
            "contexts": "fresh build_cache RNG(seed) independent of sampling RNG; same original evaluation lists",
            "limits": "changes candidate universe, positive set, and augmentation volume; unmatched practical reference, not fixed-Q causal control or native ColdLLM reproduction",
        },
        "common_augmented_fields": "two conversion labels; uniform q; warm-mean continuous target; confidence paths off; fixed support gate",
        "all_first_fits": "no desired-sign/baseline-match retries; preserve claims, failures, and first completed fits",
        "predeclared_reporting": "all four arms and all three seeds; coldR20, coldN20, warmR20 and full stored ranking metrics",
        "thread_policy": "existing single_thread.py wrapper plus torch intra/inter-op1; root at most two science workers; old DataLoader workers2",
        "file_sha256": pins,
    }
    RUN.mkdir()
    sel.json_write(RUN / "FROZEN_PROTOCOL.json", protocol)


def verify_protocol():
    path = RUN / "FROZEN_PROTOCOL.json"
    protocol = json.loads(path.read_text())
    assert protocol["arms"] == list(ARMS) and protocol["seeds"] == list(SEEDS)
    assert protocol["cfg"] == plain(asdict(sel.config()))
    for name, digest in protocol["file_sha256"].items():
        assert sel.sha(ROOT / name) == digest, f"Frozen extension input changed: {name}"
    assert sel.verify_frozen() == protocol["old_selector_protocol_sha256"]
    return sel.sha(path)


def practical_random_rows(raw, seed, warm_rate, warm_core):
    source = ROOT / f"ember2/runs/ar_fresh_ml1m_s{seed}/dllm_rows.json"
    accepted = [replace(sel.unpack(row), core=warm_core, q=np.ones(3, np.float32))
                for row in json.loads(source.read_text()) if row["b"][0] > 0]
    by_bucket = defaultdict(list)
    for row in accepted:
        by_bucket[sel.bucket(row)].append(row)
    warm_users = np.flatnonzero(~raw.is_cold_user)
    warm_items = np.flatnonzero(~raw.is_cold_item)
    rng = np.random.default_rng(seed + ORDINARY_OFFSET)
    negatives, audits = [], []
    positive_pairs = {(r.user, r.item) for r in accepted}
    assert 0 < warm_rate < 1 and len(positive_pairs) == len(accepted)
    for direction, entity in sorted(by_bucket):
        p_rows = by_bucket[direction, entity]
        opposite = warm_users if direction == COLD_ITEM else warm_items
        excluded = {r.user if direction == COLD_ITEM else r.item for r in p_rows}
        eligible = opposite[~np.isin(opposite, sorted(excluded))]
        requested = int(round(len(p_rows) * (1.0 - warm_rate) / warm_rate))
        actual = min(requested, len(eligible))
        selected = eligible[rng.choice(len(eligible), actual, replace=False)]
        for counterpart in selected:
            user, item = ((int(counterpart), entity) if direction == COLD_ITEM else
                          (entity, int(counterpart)))
            assert (user, item) not in positive_pairs
            negatives.append(SimulatedRow(user, item, direction, np.zeros(2, np.float32),
                                          warm_core, np.ones(3, np.float32)))
        audits.append(dict(direction=direction, cold_entity=int(entity),
                           n_positive=len(p_rows), eligible_counterparts=len(eligible),
                           requested_negative=requested, actual_negative=actual))
    assert len({sel.key(r) for r in negatives}) == len(negatives)
    return accepted + negatives, {
        "source_sha256": sel.sha(source), "warm_primary_rate": warm_rate,
        "n_accepted_all": len(accepted), "n_random_negative": len(negatives),
        "negative_quotas_clipped_buckets": sum(x["actual_negative"] < x["requested_negative"] for x in audits),
        "buckets": audits,
        "scope": "unmatched practical reference; different universe/positive set/volume; not native ColdLLM",
    }


def prepare_seed(seed):
    dest = RUN / f"seed{seed}"
    dest.mkdir()
    raw = sel.raw_data(seed)
    old_rows = sel.OUT / f"seed{seed}/llm_rows.json"
    old_audit = json.loads((sel.OUT / f"seed{seed}/PREPARED.json").read_text())
    assert sel.sha(old_rows) == old_audit["arms"]["llm"]["records_sha256"]
    rows = [sel.unpack(row) for row in json.loads(old_rows.read_text())]
    original = build_cache(raw, rows, np.random.default_rng(seed), 100, 1500, with_test=True)
    old_inputs = json.loads((sel.OUT / f"fits/s{seed}_llm/INPUTS.json").read_text())
    for name, array in (("features_sha256", original.train_feature_ids),
                        ("labels_sha256", original.train_binary),
                        ("context_sha256", original.train_feature_ids[:, -2:]),
                        ("validation_features_sha256", original.val.feature_ids),
                        ("test_features_sha256", original.test.feature_ids),
                        ("test_labels_sha256", original.test.binary)):
        assert sel.array_sha(array) == old_inputs[name], (seed, name)
    warm = original.train_is_aug <= 0
    positive = ~warm & (original.train_binary[:, 0] > 0)
    negative = ~warm & ~positive
    p_idx, n_idx = np.flatnonzero(positive), np.flatnonzero(negative)
    assert len(p_idx) and len(n_idx)
    cases = {
        "none": clone_training(original, np.flatnonzero(warm)),
        "fixedP_only": clone_training(original, np.flatnonzero(warm | positive)),
        "fixedP_repeatB": clone_training(original),
    }
    repeated = cases["fixedP_repeatB"]
    sampled_p = np.random.default_rng(seed + REPEAT_OFFSET).choice(p_idx, len(n_idx), replace=True)
    original_context = original.train_feature_ids[n_idx, -2:].copy()
    repeated.train_feature_ids[n_idx] = original.train_feature_ids[sampled_p]
    repeated.train_feature_ids[n_idx, -2:] = original_context
    repeated.train_binary[n_idx] = original.train_binary[sampled_p]
    repeated.train_core[~warm] = original.core_mean
    repeated.train_q[~warm] = 1.0
    assert np.array_equal(repeated.train_feature_ids[positive], original.train_feature_ids[positive])
    assert np.array_equal(repeated.train_feature_ids[:, -2:], original.train_feature_ids[:, -2:])
    assert np.all(repeated.train_binary[~warm, 0] == 1)
    fixed_only = cases["fixedP_only"]
    assert np.array_equal(fixed_only.train_feature_ids[fixed_only.train_is_aug > 0],
                          original.train_feature_ids[positive])
    assert np.array_equal(fixed_only.train_binary[fixed_only.train_is_aug > 0],
                          original.train_binary[positive])
    assert cases["none"].n_aug == 0 and fixed_only.n_aug == len(p_idx)
    assert repeated.n_aug == original.n_aug
    np.savez_compressed(dest / "repeat_mapping.npz", negative_full_training_positions=n_idx,
                        sampled_positive_full_training_positions=sampled_p)
    practical_rows, practical_audit = practical_random_rows(raw, seed, float(raw.binary[raw.warm_training_mask(), 0].mean()), original.core_mean)
    sel.json_write(dest / "accepted_all_plus_random_rows.json", [sel.pack(r) for r in practical_rows])
    cases["accepted_all_plus_random"] = build_cache(raw, practical_rows, np.random.default_rng(seed),
                                                   100, 1500, with_test=True)
    audits = {}
    for arm in ARMS:
        cache = cases[arm]
        assert_same_evaluation(cache, original)
        aug = cache.train_is_aug > 0
        assert np.all(cache.train_q == 1)
        assert np.all(cache.train_core[aug] == np.float32(original.core_mean))
        assert np.array_equal(cache.train_feature_ids[~aug], original.train_feature_ids[warm])
        assert np.array_equal(cache.train_binary[~aug], original.train_binary[warm])
        path = dest / f"{arm}_cache.npz"
        temp = dest / f"{arm}_cache.tmp.{os.getpid()}.npz"
        cache.save(str(temp))
        assert not path.exists()
        temp.replace(path)
        audits[arm] = {**summarize_cache(raw, cache, original), "cache_sha256": sel.sha(path)}
        print(f"PREPARED s{seed}/{arm}: warm{cache.n_warm} aug{cache.n_aug} P{audits[arm]['n_aug_positive']} N{audits[arm]['n_aug_negative']}", flush=True)
    sel.json_write(dest / "PREPARED.json", {
        "seed": seed, "source_retained_rows_sha256": sel.sha(old_rows),
        "source_llm_inputs_sha256": sel.sha(sel.OUT / f"fits/s{seed}_llm/INPUTS.json"),
        "original_train_sha256": train_pins(original),
        "original_validation_sha256": split_pins(original.val),
        "original_historical_test_sha256": split_pins(original.test),
        "original_budget_B": original.n_aug, "fixed_P_count": len(p_idx),
        "repeat_mapping_sha256": sel.sha(dest / "repeat_mapping.npz"),
        "practical_rows_sha256": sel.sha(dest / "accepted_all_plus_random_rows.json"),
        "practical_random": practical_audit, "arms": audits,
    })


def freeze_preparation():
    files = []
    for seed in SEEDS:
        dest = RUN / f"seed{seed}"
        files += [dest / "PREPARED.json", dest / "repeat_mapping.npz",
                  dest / "accepted_all_plus_random_rows.json"]
        files += [dest / f"{arm}_cache.npz" for arm in ARMS]
    path = RUN / "PREPARATION_MANIFEST.json"
    assert not path.exists()
    sel.json_write(path, {"frozen_protocol_sha256": verify_protocol(),
                         "completed_unix": time.time(),
                         "file_sha256": {relative(p): sel.sha(p) for p in files}})


def verify_preparation():
    protocol_sha = verify_protocol()
    path = RUN / "PREPARATION_MANIFEST.json"
    prepared = json.loads(path.read_text())
    assert prepared["frozen_protocol_sha256"] == protocol_sha
    for name, digest in prepared["file_sha256"].items():
        assert sel.sha(ROOT / name) == digest, f"Frozen prepared cache changed: {name}"
    return protocol_sha, sel.sha(path)


def fit(seed, arm, device, label):
    protocol_sha, prep_sha = verify_preparation()
    prep_dir = RUN / f"seed{seed}"
    path = prep_dir / f"{arm}_cache.npz"
    audit = json.loads((prep_dir / "PREPARED.json").read_text())
    assert sel.sha(path) == audit["arms"][arm]["cache_sha256"]
    cache = Cache.load(str(path))
    assert train_pins(cache) == audit["arms"][arm]["train_array_sha256"]
    assert split_pins(cache.val) == audit["original_validation_sha256"]
    assert split_pins(cache.test) == audit["original_historical_test_sha256"]
    historical, cache.test = cache.test, None
    dest = RUN / "fits" / f"s{seed}_{arm}"
    dest.mkdir(parents=True)
    cfg = sel.config()
    inputs = {
        "seed": seed, "arm": arm, "worker": label, "started_unix": time.time(),
        "frozen_protocol_sha256": protocol_sha, "preparation_manifest_sha256": prep_sha,
        "cache_sha256": sel.sha(path), "train_array_sha256": train_pins(cache),
        "validation_sha256": split_pins(cache.val), "historical_test_sha256": split_pins(historical),
        "cfg": asdict(cfg), "record_audit": audit["arms"][arm],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    sel.json_write(dest / "INPUTS.json", inputs)
    print(f"START EXTENSION s{seed}/{arm} {device} aug{cache.n_aug}", flush=True)
    started = time.monotonic()
    result = train(cache, cfg, device=device, seed=seed, verbose=True, return_model=True)
    model = result.pop("model")
    values = [h["val"]["overall"]["recall@20"] for h in result["history"]]
    assert all(math.isfinite(value) for value in values)
    selected = max(result["history"], key=lambda h: h["val"]["overall"]["recall@20"])
    assert selected["val"]["overall"]["recall@20"] == result["val"]["ranking"]["overall"]["recall@20"]
    support_user, support_item = warm_support(cache)
    checkpoint = dest / "selected.pt"
    torch.save({
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "cfg": asdict(cfg), "fields": asdict(cache.fields), "tasks": asdict(cache.tasks),
        "core_mean": cache.core_mean, "core_std": cache.core_std,
        "warm_support_user": support_user, "warm_support_item": support_item,
        "seed": seed, "arm": arm, "cache_sha256": inputs["cache_sha256"],
        "frozen_protocol_sha256": protocol_sha,
    }, checkpoint)
    selection = {
        "seed": seed, "arm": arm, "criterion": "validation overall Recall@20",
        "chosen_epoch": selected["epoch"], "val": result["val"],
        "checkpoint_sha256": sel.sha(checkpoint), "frozen_before_test_unix": time.time(),
        "test_used_in_fit_or_selection": False,
    }
    sel.json_write(dest / "SELECTION.json", selection)
    sel.json_write(dest / "VALIDATION_ONLY.json", result)
    # No fitting or checkpoint/hyperparameter/arm selection is allowed below.
    evaluated = evaluate(model, historical, cache, torch.device(device))
    per = sel.per_list_metrics(model, historical, device, dest / "test_per_list.npz")
    for split in ("overall", "cold", "warm"):
        for metric in ("recall@20", "ndcg@20"):
            assert abs(per[split][metric] - evaluated.ranking[split][metric]) < 1e-10
    result.update({
        "historical_test": asdict(evaluated), "selection": selection,
        "fit_and_eval_wall_seconds": time.monotonic() - started,
        "test_scope": "post-hoc extension on consumed historical test; no independent confirmation",
        "arm_scope": audit["arms"][arm],
    })
    sel.json_write(dest / "RESULT.json", result)
    print(f"DONE EXTENSION s{seed}/{arm}: {evaluated.brief()} epoch{selected['epoch']}", flush=True)
    del model, result, cache
    torch.cuda.empty_cache()


def worker(device, label):
    verify_preparation()
    claims = RUN / "claims"
    claims.mkdir(exist_ok=True)
    for seed, arm in JOBS:
        path = claims / f"s{seed}_{arm}.json"
        try:
            with path.open("x") as stream:
                json.dump(dict(seed=seed, arm=arm, worker=label, pid=os.getpid(),
                               claimed_unix=time.time()), stream)
        except FileExistsError:
            continue
        try:
            fit(seed, arm, device, label)
        except BaseException as error:
            sel.json_write(claims / f"s{seed}_{arm}_FAILURE.json", {
                "seed": seed, "arm": arm, "worker": label, "failed_unix": time.time(),
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "automatic_retry": False,
            })
            raise
    print(f"EXTENSION WORKER COMPLETE {label}; collection must require all12 RESULT files", flush=True)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "worker"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", default="worker")
    args = parser.parse_args()
    if args.stage == "prepare":
        freeze_protocol()
        for seed in SEEDS:
            prepare_seed(seed)
        freeze_preparation()
        verify_preparation()
        print("ALL EXTENSION INPUTS PREPARED; root may start at most two workers", flush=True)
    else:
        worker(args.device, args.label)


if __name__ == "__main__":
    main()
