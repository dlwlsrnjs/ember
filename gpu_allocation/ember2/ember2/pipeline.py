"""End-to-end driver for the three EMBER stages.

With ``--backend mock`` every stage runs on the synthetic dataset using test
doubles in place of the 7B simulator and the KD teacher, so the whole method --
prompt construction with direction masking, coupled filtering, cache assembly,
RouteFuse + CaliChain training, and the ranking protocol -- executes without
model weights or a licensed dataset.

The ablation switches follow the paper's Table 2:

  ``--pairs random``        random cold pairs instead of filter-retrieved ones
  ``--sim filter_only``     augment with retrieved pairs, no LLM labels
  ``--sim hard_label``      single hard primary label, no direction, no confidence
  ``--sim directional``     full direction-specific hybrid tuples (default)
  ``--no-routefuse`` / ``--no-confidence-prior`` / ``--no-confidence-weighting``
  ``--lambda-a 0``          drop the augmented block entirely
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cache import build_cache
from .config import FilterConfig, LearnerConfig
from .dataset import Cache
from .filterstage import (
    HashingTextEncoder,
    MockKDTeacher,
    build_filter_training_data,
    retrieve_candidates,
    train_coupled_filter,
    whiten,
)
from .rawdata import RawDataset, make_synthetic
from .simulate import (
    COLD_ITEM,
    COLD_USER,
    BehaviorIndex,
    MockSimulator,
    PromptBuilder,
    SimulatedRow,
    build_sft_examples,
    balance_negatives,
    keep_positive_rows,
    simulate_pairs,
)
from .train import train


@dataclass
class PipelineConfig:
    seed: int = 0
    device: str = "cuda:0"
    # Stage 2
    top_k: int = 30
    n_bpr: int = 200_000
    n_kd: int = 20_000
    filter_epochs: int = 3
    emb_dim: int = 256
    whiten_dim: int = 128
    pairs: str = "filter"          # filter | random
    # Stage 1
    sim: str = "directional"       # directional | hard_label | filter_only |
                                   # model_teacher (non-LLM control) | none
    teacher_epochs: int = 10       # model_teacher only
    negatives: str = "balanced"    # balanced | none (paper) | all
    sim_noise: float = 0.9
    n_sft_positive: int = 2000
    n_sft_unobserved: int = 2000
    with_reasoning: bool = False
    # Stage 3
    n_candidates: int = 100
    max_eval_groups: Optional[int] = 1500


def _random_pairs(raw: RawDataset, top_k: int, rng) -> Dict[str, List[Tuple[int, int]]]:
    """Uniformly sampled cold pairs -- the paper's `random cold pairs` control."""
    warm_u = np.setdiff1d(np.arange(raw.n_users), raw.cold_users)
    warm_i = np.setdiff1d(np.arange(raw.n_items), raw.cold_items)
    return {
        COLD_ITEM: [
            (int(u), int(i))
            for i in raw.cold_items
            for u in rng.choice(warm_u, top_k, replace=False)
        ],
        COLD_USER: [
            (int(u), int(i))
            for u in raw.cold_users
            for i in rng.choice(warm_i, top_k, replace=False)
        ],
    }


def _warm_model_teacher(raw, rng, cfg, log):
    """Train the warm-only learner and hand it back as a non-LLM soft teacher.

    The control for "is it the LLM, or is it the augmentation machinery?" -- see
    :class:`ember2.simulate.WarmModelTeacher`. The teacher is trained on
    ``D_warm`` alone, with no augmented rows and no simulator, so it is exactly
    the no-augmentation baseline being asked to label its own cold pairs.
    """
    from .config import LearnerConfig
    from .simulate import WarmModelTeacher

    log(f"[3b] training the non-LLM teacher (warm-only, {cfg.teacher_epochs} epochs)")
    warm_cache = build_cache(raw, [], rng, cfg.n_candidates, cfg.max_eval_groups)
    tcfg = LearnerConfig(n_epochs=cfg.teacher_epochs, min_epochs=cfg.teacher_epochs,
                         lambda_a=0.0)
    res = train(warm_cache, tcfg, device=cfg.device, seed=cfg.seed, verbose=False,
                return_model=True)
    model = res["model"]
    blk = res.get("test") or res["val"]
    log(f"     teacher cold R@20 {blk['ranking'].get('cold', {}).get('recall@20', float('nan')):.4f}")
    return WarmModelTeacher(model, warm_cache, raw, device=cfg.device)


def _filter_only_rows(
    candidates: Dict[str, List[Tuple[int, int]]], raw: RawDataset
) -> List[SimulatedRow]:
    """Retrieved pairs asserted as positives, with no LLM and no confidence.

    Isolates how much of the gain is retrieval alone versus simulated behaviour.
    """
    T = raw.tasks.n_binary
    rows = []
    for direction, pairs in candidates.items():
        for u, i in pairs:
            b = np.zeros(T, np.float32)
            b[0] = 1.0
            rows.append(
                SimulatedRow(
                    user=u, item=i, direction=direction,
                    binary=b, core=0.0, q=np.ones(T + 1, np.float32),
                )
            )
    return rows


def _to_hard_label(rows: Sequence[SimulatedRow]) -> List[SimulatedRow]:
    """Collapse hybrid tuples to a single hard label with no confidence.

    This is the ColdLLM-style control: the simulator's extra heads and its
    calibration are thrown away, keeping only `did this interaction happen`.
    """
    out = []
    for r in rows:
        b = np.zeros_like(r.binary)
        b[0] = r.binary[0]
        out.append(
            replace(r, binary=b, core=0.0, q=np.ones_like(r.q), direction=COLD_ITEM)
        )
    return out


def build_stage12_cache(
    cfg: Optional[PipelineConfig] = None,
    raw: Optional[RawDataset] = None,
    oracle: Optional[Dict] = None,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Cache:
    """Stages 1-2: simulate cold behaviour and assemble ``D_warm u A_LLM``."""
    cfg = cfg or PipelineConfig()
    rng = np.random.default_rng(cfg.seed)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ── data ────────────────────────────────────────────────────────
    if raw is None:
        raw, oracle = make_synthetic(seed=cfg.seed)
    if oracle is None:
        raise ValueError("mock backends need the synthetic oracle; pass one or omit `raw`")
    log(
        f"[0] dataset: {raw.n_users:,} users / {raw.n_items:,} items / "
        f"{len(raw.user):,} interactions | "
        f"cold: {len(raw.cold_users):,} users, {len(raw.cold_items):,} items | "
        f"D_warm: {int(raw.warm_training_mask().sum()):,} rows"
    )

    # ── Stage 2a: text embeddings, whitened ─────────────────────────
    log("[1] encoding user/item text and whitening")
    enc = HashingTextEncoder(dim=cfg.emb_dim, seed=cfg.seed)
    user_emb = whiten(enc.encode(raw.user_text), cfg.whiten_dim)
    item_emb = whiten(enc.encode(raw.item_text), cfg.whiten_dim)
    i, j = rng.integers(0, len(user_emb), 2000), rng.integers(0, len(user_emb), 2000)
    rand_cos = float(np.abs((user_emb[i] * user_emb[j]).sum(1)).mean())
    log(f"    mean |cos| between random user pairs after whitening: {rand_cos:.3f}")

    index = BehaviorIndex(raw)
    builder = PromptBuilder(raw, index, user_emb, item_emb, top_l=5)

    # ── Stage 1a: direction-masked SFT data ─────────────────────────
    sft: Dict[str, list] = {}
    if cfg.sim == "directional":
        for direction in (COLD_ITEM, COLD_USER):
            sft[direction] = build_sft_examples(
                raw, builder, direction,
                cfg.n_sft_positive, cfg.n_sft_unobserved, rng, cfg.with_reasoning,
            )
        log(
            f"[2] SFT data: "
            + ", ".join(f"{d} {len(v):,} examples" for d, v in sft.items())
            + (" (reasoning-SFT)" if cfg.with_reasoning else "")
        )

    # ── Stage 2b: coupled filter and retrieval ──────────────────────
    if cfg.pairs == "random":
        log("[3] candidate pairs: uniform random (ablation control)")
        candidates = _random_pairs(raw, cfg.top_k, rng)
    else:
        log("[3] training coupled filter (BPR + KD + InfoNCE)")
        fcfg = FilterConfig(
            n_bpr_triplets=cfg.n_bpr, n_epochs=cfg.filter_epochs, top_k=cfg.top_k
        )
        data = build_filter_training_data(
            raw, MockKDTeacher(oracle, seed=cfg.seed), cfg.n_bpr, cfg.n_kd, rng
        )
        filt = train_coupled_filter(
            user_emb, item_emb, data, fcfg, device=cfg.device, seed=cfg.seed, verbose=verbose
        )
        candidates = retrieve_candidates(
            filt, user_emb, item_emb, raw, cfg.top_k, device=cfg.device
        )
    n_pairs = sum(len(v) for v in candidates.values())
    saved = raw.n_users * len(raw.cold_items) + len(raw.cold_users) * raw.n_items
    log(f"    {n_pairs:,} candidate pairs retrieved (vs {saved:,} exhaustive: {n_pairs / saved * 100:.2f}%)")

    # ── Stage 1b: simulate ──────────────────────────────────────────
    if cfg.sim == "none":
        sim_rows: List[SimulatedRow] = []
        log("[4] simulation skipped (no augmentation)")
    elif cfg.sim == "filter_only":
        sim_rows = _filter_only_rows(candidates, raw)
        log(f"[4] filter-only augmentation: {len(sim_rows):,} rows")
    else:
        if cfg.sim == "model_teacher":
            backend = _warm_model_teacher(raw, rng, cfg, log)
        else:
            backend = MockSimulator(raw, oracle, noise=cfg.sim_noise, seed=cfg.seed)
        rows: List[SimulatedRow] = []
        for direction, pairs in candidates.items():
            rows += simulate_pairs(pairs, builder, backend, direction)
        yes_rate = sum(1 for r in rows if r.binary[0] > 0) / max(len(rows), 1)
        if cfg.negatives == "none":
            sim_rows = keep_positive_rows(rows)
        elif cfg.negatives == "all":
            sim_rows = list(rows)
        else:
            warm_pos_rate = float(raw.binary[raw.warm_training_mask(), 0].mean())
            sim_rows = balance_negatives(rows, warm_pos_rate, rng)
        if cfg.sim == "hard_label":
            sim_rows = _to_hard_label(sim_rows)
        by_dir = {d: sum(1 for r in sim_rows if r.direction == d) for d in {r.direction for r in sim_rows}}
        mean_q = float(np.mean([r.q[0] for r in sim_rows])) if sim_rows else float("nan")
        pos_rate = float(np.mean([r.binary[0] for r in sim_rows])) if sim_rows else float("nan")
        log(
            f"[4] simulated {len(rows):,} pairs (yes-rate {yes_rate * 100:.1f}%) "
            f"-> kept {len(sim_rows):,} rows [negatives={cfg.negatives}] "
            f"pos-rate {pos_rate:.3f} (D_warm {float(raw.binary[raw.warm_training_mask(), 0].mean()):.3f}) "
            f"{by_dir} | mean q_primary {mean_q:.3f}"
        )

    # ── cache ───────────────────────────────────────────────────────
    cache = build_cache(raw, sim_rows, rng, cfg.n_candidates, cfg.max_eval_groups)
    log(f"[5] cache: {cache.summary()}")
    if cache_path:
        cache.save(cache_path)
        log(f"    saved -> {cache_path}")
    return cache


def run_pipeline(
    cfg: Optional[PipelineConfig] = None,
    learner_cfg: Optional[LearnerConfig] = None,
    raw: Optional[RawDataset] = None,
    oracle: Optional[Dict] = None,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """All three stages: simulate, filter, cache, then train the learner."""
    cfg = cfg or PipelineConfig()
    cache = build_stage12_cache(cfg, raw, oracle, cache_path, verbose)
    if verbose:
        print("[6] training the cold-aware learner", flush=True)
    result = train(
        cache, learner_cfg or LearnerConfig(), device=cfg.device, seed=cfg.seed, verbose=verbose
    )
    result["n_augmented"] = cache.n_aug
    return result
