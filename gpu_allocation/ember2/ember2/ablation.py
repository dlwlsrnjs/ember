"""The Table 2 ablation ladder, plus the two fixes on top of it.

Within a seed every variant shares one dataset and one cold split, so
differences come from the ablated component and nothing else. Stage 1-2 runs
once per cache key and is reused by the learner-side variants on top of it; each
seed regenerates split, filter, simulation and learner from scratch, as the
paper does.

Read the absolute numbers as nothing at all -- the simulator here is a fixture
(see :class:`ember2.simulate.MockSimulator`). What the ladder checks is that each
component is wired in and that removing it moves the result consistently across
seeds. Report it with the spread; single-seed gaps between neighbouring rungs
are smaller than the run-to-run noise.
"""
from __future__ import annotations

import json
import os

import numpy as np
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from .config import LearnerConfig
from .dataset import Cache
from .pipeline import PipelineConfig, build_stage12_cache
from .rawdata import make_synthetic
from .train import train

# The package defaults are the *fixed* configuration (balanced negatives, signed
# prior, progress tau anneal). Table 2 is a claim about the
# paper's configuration, so every rung up to and including "Full EMBER (paper)"
# pins those three back to what §3 specifies; the rungs below turn the fixes on
# one at a time. Leaving them at their defaults would label the improved model
# "Full EMBER" and silently make the paper's ladder unreproducible from here.
PAPER = {"signed_confidence_prior": False, "tau_schedule": "steps"}

# (label, cache key, PipelineConfig kwargs, learner-config overrides). The cache
# key has to name the augmentation mode *and* the negatives policy: two rungs can
# share a simulation mode and still need different caches.
LADDER = [
    ("no augmentation (cold-removed)", "none",
     dict(sim="none"),                                {"lambda_a": 0.0, **PAPER}),
    ("random cold pairs",              "random",
     dict(sim="directional", pairs="random", negatives="none"),          PAPER),
    ("coupled-filter only",            "filter_only",
     dict(sim="filter_only"),                                            PAPER),
    ("hard-label LLM (no direction)",  "hard_label",
     dict(sim="hard_label", negatives="none"),                           PAPER),
    ("direction-specific LLM",         "pos_only",
     dict(sim="directional", negatives="none"),
     {"use_routefuse": False, "use_confidence_prior": False,
      "use_confidence_weighting": False, **PAPER}),
    ("  w/o RouteFuse",                "pos_only",
     dict(sim="directional", negatives="none"),  {"use_routefuse": False, **PAPER}),
    ("  w/o confidence prior",         "pos_only",
     dict(sim="directional", negatives="none"),  {"use_confidence_prior": False, **PAPER}),
    ("  w/o confidence weighting",     "pos_only",
     dict(sim="directional", negatives="none"),  {"use_confidence_weighting": False, **PAPER}),
    ("Full EMBER (paper)",             "pos_only",
     dict(sim="directional", negatives="none"),                          PAPER),
    # The paper's stated evidence for discarding negatives is a measurement of
    # *all* of them (cold R@20 0.1006 -> 0.0934). Both ends have to be on the
    # ladder or "all or nothing is a false choice" is an assertion, not a result.
    ("  + all negatives",              "all_neg",
     dict(sim="directional", negatives="all"),                           PAPER),
    ("  + balanced negatives",         "balanced",
     dict(sim="directional", negatives="balanced"),                      PAPER),
    ("  + signed prior",               "balanced",
     dict(sim="directional", negatives="balanced"),
     {"signed_confidence_prior": True, "tau_schedule": "steps"}),
    ("Fixed EMBER (+ tau anneal)",     "balanced",
     dict(sim="directional", negatives="balanced"),                      {}),
    # Reviewer control (R2): same machinery, a teacher that is not an LLM. The
    # warm-only learner labels its own retrieved cold pairs, so the only thing
    # that differs from the rung above is who produced the hybrid tuple.
    ("non-LLM teacher (model)",        "model_teacher",
     dict(sim="model_teacher", negatives="balanced"),                    PAPER),
    # Beyond the paper. These are learner-side and cost
    # no LLM calls, so the last two rungs -- the same changes with A_LLM removed
    # entirely -- are the ones that say whether the augmentation still pays.
    ("  + support reliability",        "balanced",
     dict(sim="directional", negatives="balanced"),  {"reliability": "support"}),
    ("  + cold id synthesis",          "balanced",
     dict(sim="directional", negatives="balanced"),
     {"reliability": "support", "cold_id_synthesis": True}),
    ("no aug + support",               "none",
     dict(sim="none"),   {"lambda_a": 0.0, "reliability": "support"}),
    ("no aug + support + synthesis",   "none",
     dict(sim="none"),   {"lambda_a": 0.0, "reliability": "support",
                          "cold_id_synthesis": True}),
]


def run_ablation(
    out_dir: str = "./runs",
    seeds: Sequence[int] = (0, 1, 2),
    device: str = "cuda:0",
    epochs: int = 25,
    emb_dim: int = 10,
    verbose: bool = True,
) -> List[Dict]:
    """Run the ladder over several seeds and report mean +/- std.

    Multiple seeds are not optional here. Single-seed gaps between neighbouring
    rungs are smaller than the run-to-run spread, so a one-seed ladder orders
    the variants by luck. The paper uses five seeds and regenerates the split,
    filter, simulation and learner for each; this does the same, with fewer.
    """
    base = LearnerConfig(n_epochs=epochs, emb_dim=emb_dim)
    per_seed: Dict[int, Dict[str, Dict]] = {}

    for seed in seeds:
        raw, oracle = make_synthetic(seed=seed)
        caches: Dict[str, Cache] = {}
        per_seed[seed] = {}

        for label, key, pipe_kw, overrides in LADDER:
            if key not in caches:
                print(f"\n{'=' * 72}\n### seed {seed} | building cache: {key} ({pipe_kw})\n{'=' * 72}", flush=True)
                caches[key] = build_stage12_cache(
                    PipelineConfig(seed=seed, device=device, **pipe_kw),
                    raw, oracle, verbose=verbose,
                )

            print(f"\n--- seed {seed} | {label} ---", flush=True)
            res = train(caches[key], replace(base, **overrides), device=device,
                        seed=seed, verbose=verbose)
            block = res.get("test") or res["val"]
            rank, point = block["ranking"], block["pointwise"]
            per_seed[seed][label] = {
                "n_aug": caches[key].n_aug,
                "cold_recall@20": rank.get("cold", {}).get("recall@20"),
                "cold_ndcg@20": rank.get("cold", {}).get("ndcg@20"),
                "warm_recall@20": rank.get("warm", {}).get("recall@20"),
                "overall_recall@20": rank.get("overall", {}).get("recall@20"),
                "cold_core_nrmse": point.get("cold", {}).get(f"{caches[key].tasks.core}_nrmse"),
                # Kept per seed so the per-task hybrid table and the cold-case
                # breakdown can be built from the same artifact -- reporting only
                # primary-task ranking is what makes a hybrid-target claim
                # unverifiable.
                "pointwise": point,
                "by_case": block.get("by_case", {}),
                "diagnostics": res["diagnostics"],
                "gates": block["gates"],
            }

    # ── aggregate ───────────────────────────────────────────────────
    METRICS = ["cold_recall@20", "cold_ndcg@20", "warm_recall@20",
               "overall_recall@20", "cold_core_nrmse"]
    results: List[Dict] = []
    for label, key, pipe_kw, overrides in LADDER:
        row: Dict = {"variant": label, "cache": key, "pipeline": pipe_kw,
                     "overrides": overrides, "seeds": list(seeds),
                     "n_aug": per_seed[seeds[0]][label]["n_aug"]}
        for m in METRICS:
            vals = [per_seed[s][label][m] for s in seeds
                    if isinstance(per_seed[s][label][m], float)]
            row[m] = float(np.mean(vals)) if vals else None
            row[f"{m}_std"] = float(np.std(vals)) if vals else None
        row["per_seed"] = {s: per_seed[s][label] for s in seeds}
        results.append(row)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ablation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    def cell(mu, sd, w=17):
        return f"{'--':>{w}}" if mu is None else f"{mu:>{w - 8}.4f}+-{sd:<6.4f}"

    print(f"\n{'=' * 108}")
    print(f"ablation over seeds {list(seeds)}, {epochs} epochs, test split (mean +- std)")
    print(
        f"{'variant':<34}{'n_aug':>8}{'cold R@20':>17}{'cold N@20':>17}"
        f"{'warm R@20':>17}{'cold NRMSE':>15}"
    )
    print("-" * 108)
    base_cold = results[0]["cold_recall@20"]
    for r in results:
        delta = ""
        if base_cold and r["cold_recall@20"] is not None:
            delta = f"  ({(r['cold_recall@20'] / base_cold - 1) * 100:+.1f}%)"
        print(
            f"{r['variant']:<34}{r['n_aug']:>8,}"
            f"{cell(r['cold_recall@20'], r['cold_recall@20_std'])}"
            f"{cell(r['cold_ndcg@20'], r['cold_ndcg@20_std'])}"
            f"{cell(r['warm_recall@20'], r['warm_recall@20_std'])}"
            f"{cell(r['cold_core_nrmse'], r['cold_core_nrmse_std'], 15)}{delta}"
        )
    print("=" * 108)
    print(f"saved -> {path}")
    return results
