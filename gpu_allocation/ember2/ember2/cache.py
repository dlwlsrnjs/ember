"""Build the augmented training cache ``D_warm u A_LLM`` (EMBER Eq. 1).

Warm rows are the real logged interactions that survive the strict cold-start
split. Augmented rows are the simulator's hybrid label tuples, carrying their
per-task confidence and an ``is_aug`` flag so CaliChain can tell them apart.

A simulated pair has no logged context of its own, so one is sampled from the
warm training rows. That is a deliberate marginalization over context, not a
detail: the simulator was asked about a (user, item) affinity, and pinning the
row to an arbitrary fixed context would inject a spurious constant field.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .dataset import Cache
from .rawdata import TEST, VAL, RawDataset, build_eval_split
from .simulate import SimulatedRow


def build_cache(
    raw: RawDataset,
    sim_rows: Sequence[SimulatedRow],
    rng: np.random.Generator,
    n_candidates: int = 100,
    max_eval_groups: Optional[int] = None,
    with_test: bool = True,
    eval_seed: int = 12345,
) -> Cache:
    """Assemble the cache.

    ``rng`` is used only for the augmented block. The evaluation splits get
    their own generator seeded by ``eval_seed``, so every ablation variant is
    scored on the *identical* candidate lists -- otherwise the split would
    silently depend on how many augmented rows were drawn before it, and
    variants would be compared on different test sets.
    """
    warm_mask = raw.warm_training_mask()
    warm_rows = np.flatnonzero(warm_mask)
    if len(warm_rows) == 0:
        raise ValueError("D_warm is empty -- check the cold split")

    feat_warm = raw.compose_rows(raw.user[warm_rows], raw.item[warm_rows], raw.ctx[warm_rows])
    bin_warm = raw.binary[warm_rows].astype(np.float32)
    core_warm = raw.core[warm_rows].astype(np.float32)

    T = raw.tasks.n_binary
    n_warm = len(warm_rows)
    q_warm = np.ones((n_warm, T + 1), np.float32)      # logged labels are trusted fully
    aug_flag_warm = np.zeros(n_warm, np.float32)

    if sim_rows:
        u_aug = np.array([r.user for r in sim_rows], np.int64)
        i_aug = np.array([r.item for r in sim_rows], np.int64)
        ctx_aug = raw.ctx[warm_rows[rng.integers(0, n_warm, len(sim_rows))]]
        feat_aug = raw.compose_rows(u_aug, i_aug, ctx_aug)
        bin_aug = np.stack([r.binary for r in sim_rows]).astype(np.float32)
        core_aug = np.array([r.core for r in sim_rows], np.float32)
        q_aug = np.stack([r.q for r in sim_rows]).astype(np.float32)
        aug_flag = np.ones(len(sim_rows), np.float32)
    else:
        feat_aug = np.zeros((0, feat_warm.shape[1]), np.int64)
        bin_aug = np.zeros((0, T), np.float32)
        core_aug = np.zeros(0, np.float32)
        q_aug = np.zeros((0, T + 1), np.float32)
        aug_flag = np.zeros(0, np.float32)

    # Normalization statistics come from the warm block only. Fitting them on
    # simulated targets would let the simulator's scale drift into the metric.
    core_mean = float(core_warm.mean())
    core_std = float(core_warm.std()) + 1e-8

    eval_rng = np.random.default_rng(eval_seed)

    return Cache(
        fields=raw.field_spec(),
        tasks=raw.tasks,
        train_feature_ids=np.concatenate([feat_warm, feat_aug]),
        train_binary=np.concatenate([bin_warm, bin_aug]),
        train_core=np.concatenate([core_warm, core_aug]),
        train_q=np.concatenate([q_warm, q_aug]),
        train_is_aug=np.concatenate([aug_flag_warm, aug_flag]),
        val=build_eval_split(raw, VAL, eval_rng, n_candidates, max_eval_groups),
        test=build_eval_split(raw, TEST, eval_rng, n_candidates, max_eval_groups) if with_test else None,
        core_mean=core_mean,
        core_std=core_std,
    )
