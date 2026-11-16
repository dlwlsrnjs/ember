"""The training cache ``D_warm u A_LLM`` and its on-disk format.

  * **warm rows** -- real logged interactions, after every interaction of a cold
    entity has been removed (strict cold-start split). Confidence ``1``, source
    weight ``1``, ``is_aug = 0``.
  * **augmented rows** (``A_LLM``) -- simulator-generated hybrid label tuples
    carrying per-task confidence ``q`` and ``is_aug = 1``, weighted by
    ``lambda_a`` (Eq. 4).

Evaluation splits additionally carry ranking groups (one candidate list per
user) and a per-row cold flag, which is what the paper's overall / warm / cold
Recall@20 breakdown needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FieldSpec, TaskSpec

SCHEMA_VERSION = 1


# Which cold-start case a candidate list is. A strict 20%/20% split produces
# three distinct cold problems, and "cold" averaged over them hides that the
# double-cold case has no warm side to route evidence from at all.
CASES = ("warm", "cold_item", "cold_user", "cold_both")
CASE_CODE = {name: i for i, name in enumerate(CASES)}


@dataclass
class EvalSplit:
    """A ranking evaluation split.

    ``is_observed`` marks rows that come from a real logged interaction. The
    rest are sampled negatives, present only to give the ranking metric
    something to rank against. Their ``core`` value is a fabricated zero, so
    including them in the continuous-target error would measure an assumption
    rather than a prediction -- NRMSE is reported over observed rows only.

    ``case`` carries the :data:`CASES` code of the row's candidate list.
    """

    feature_ids: np.ndarray      # (N, F) int64
    binary: np.ndarray           # (N, T) float32
    core: np.ndarray             # (N,)   float32, raw units
    group_id: np.ndarray         # (N,)   int64, candidate-list id
    is_cold: np.ndarray          # (N,)   bool
    is_observed: np.ndarray      # (N,)   bool
    case: Optional[np.ndarray] = None    # (N,) int8, index into CASES

    def __len__(self) -> int:
        return len(self.feature_ids)

    def case_names(self) -> np.ndarray:
        """Per-row case name, derived from ``is_cold`` when unset (old caches)."""
        if self.case is None:
            return np.where(self.is_cold, "cold", "warm")
        return np.asarray(CASES, dtype=object)[self.case]


@dataclass
class Cache:
    """Everything Stage 3 needs, in one object."""

    fields: FieldSpec
    tasks: TaskSpec

    train_feature_ids: np.ndarray   # (N, F) int64
    train_binary: np.ndarray        # (N, T) float32
    train_core: np.ndarray          # (N,)   float32, raw units
    train_q: np.ndarray             # (N, T+1) float32
    train_is_aug: np.ndarray        # (N,)   float32

    val: EvalSplit
    test: Optional[EvalSplit] = None

    # Core-target normalization, computed on the *warm training rows only*
    # (test-period statistics must not leak into training).
    core_mean: float = 0.0
    core_std: float = 1.0

    def normalize_core(self, x: np.ndarray) -> np.ndarray:
        return (x - self.core_mean) / self.core_std

    @property
    def n_aug(self) -> int:
        return int((self.train_is_aug > 0).sum())

    @property
    def n_warm(self) -> int:
        return int((self.train_is_aug <= 0).sum())

    def summary(self) -> str:
        return (
            f"D_warm {self.n_warm:,} + A_LLM {self.n_aug:,} "
            f"= {len(self.train_core):,} train rows | "
            f"val {len(self.val):,} ({self.val.is_cold.mean() * 100:.1f}% cold rows) | "
            f"{self.fields.n_fields} fields "
            f"(u{self.fields.n_user}/i{self.fields.n_item}/c{self.fields.n_ctx}) | "
            f"tasks {self.tasks.binary} -> {self.tasks.core}"
        )

    # ── io ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        blob: Dict[str, np.ndarray] = {
            "schema_version": np.array(SCHEMA_VERSION),
            "vocab_sizes": np.asarray(self.fields.vocab_sizes, dtype=np.int64),
            "block_sizes": np.array(
                [self.fields.n_user, self.fields.n_item, self.fields.n_ctx], dtype=np.int64
            ),
            "field_names": np.array(self.fields.names or [], dtype=object),
            "user_id_fields": np.asarray(self.fields.user_id_fields, dtype=np.int64),
            "item_id_fields": np.asarray(self.fields.item_id_fields, dtype=np.int64),
            "binary_names": np.array(self.tasks.binary, dtype=object),
            "core_name": np.array(self.tasks.core),
            "core_stats": np.array([self.core_mean, self.core_std], dtype=np.float64),
            "train_feature_ids": self.train_feature_ids,
            "train_binary": self.train_binary,
            "train_core": self.train_core,
            "train_q": self.train_q,
            "train_is_aug": self.train_is_aug,
        }
        for name, split in (("val", self.val), ("test", self.test)):
            if split is None:
                continue
            blob[f"{name}_feature_ids"] = split.feature_ids
            blob[f"{name}_binary"] = split.binary
            blob[f"{name}_core"] = split.core
            blob[f"{name}_group_id"] = split.group_id
            blob[f"{name}_is_cold"] = split.is_cold
            blob[f"{name}_is_observed"] = split.is_observed
            if split.case is not None:
                blob[f"{name}_case"] = split.case
        np.savez_compressed(path, **blob)

    @staticmethod
    def load(path: str) -> "Cache":
        z = np.load(path, allow_pickle=True)
        ver = int(z["schema_version"])
        if ver != SCHEMA_VERSION:
            raise ValueError(f"cache schema v{ver}, expected v{SCHEMA_VERSION}")
        nu, ni, nc = (int(x) for x in z["block_sizes"])
        names = [str(s) for s in z["field_names"]] or None
        fields = FieldSpec(
            vocab_sizes=[int(v) for v in z["vocab_sizes"]],
            n_user=nu,
            n_item=ni,
            n_ctx=nc,
            names=names,
            user_id_fields=tuple(int(v) for v in z["user_id_fields"]),
            item_id_fields=tuple(int(v) for v in z["item_id_fields"]),
        )
        tasks = TaskSpec(binary=[str(s) for s in z["binary_names"]], core=str(z["core_name"]))

        def _split(prefix: str) -> Optional[EvalSplit]:
            if f"{prefix}_feature_ids" not in z.files:
                return None
            return EvalSplit(
                feature_ids=z[f"{prefix}_feature_ids"],
                binary=z[f"{prefix}_binary"],
                core=z[f"{prefix}_core"],
                group_id=z[f"{prefix}_group_id"],
                is_cold=z[f"{prefix}_is_cold"],
                is_observed=z[f"{prefix}_is_observed"],
                case=z[f"{prefix}_case"] if f"{prefix}_case" in z.files else None,
            )

        mean, std = (float(x) for x in z["core_stats"])
        return Cache(
            fields=fields,
            tasks=tasks,
            train_feature_ids=z["train_feature_ids"],
            train_binary=z["train_binary"],
            train_core=z["train_core"],
            train_q=z["train_q"],
            train_is_aug=z["train_is_aug"],
            val=_split("val"),
            test=_split("test"),
            core_mean=mean,
            core_std=std,
        )


class TrainDataset(Dataset):
    """``D_warm u A_LLM`` rows, with the core target already normalized."""

    def __init__(self, cache: Cache) -> None:
        self.feature_ids = torch.as_tensor(cache.train_feature_ids, dtype=torch.long)
        self.binary = torch.as_tensor(cache.train_binary, dtype=torch.float32)
        self.core = torch.as_tensor(cache.normalize_core(cache.train_core), dtype=torch.float32)
        self.q = torch.as_tensor(cache.train_q, dtype=torch.float32)
        self.is_aug = torch.as_tensor(cache.train_is_aug, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.feature_ids)

    def __getitem__(self, i: int):
        return self.feature_ids[i], self.binary[i], self.core[i], self.q[i], self.is_aug[i]


class EvalDataset(Dataset):
    """Feature ids only -- serving sees no simulator confidence."""

    def __init__(self, split: EvalSplit) -> None:
        self.feature_ids = torch.as_tensor(split.feature_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.feature_ids)

    def __getitem__(self, i: int):
        return self.feature_ids[i]
