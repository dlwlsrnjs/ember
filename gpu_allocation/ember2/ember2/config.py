"""Specs shared across the EMBER stages.

The paper formulates each instance as ``x = (u, i, c)`` with hybrid targets
``y = (y^(1), ..., y^(T), y^(r))``: ``T`` binary conversion labels followed by
one continuous core outcome. ``T = 2`` in every dataset the paper uses, but
nothing here hardcodes that -- the task chain is built from :class:`TaskSpec`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class TaskSpec:
    """The hybrid label tuple.

    Args:
      binary: names of the ``T`` preceding conversion targets, in chain order
              (earliest first, e.g. ``["click", "long_view"]``).
      core:   name of the continuous core target ``y^(r)`` (e.g. ``watch_time``).
    """

    binary: List[str]
    core: str

    @property
    def n_binary(self) -> int:
        return len(self.binary)

    @property
    def all_names(self) -> List[str]:
        return list(self.binary) + [self.core]


@dataclass
class FieldSpec:
    """Categorical field layout of one instance row.

    Rows are laid out as ``[user block | item block | context block]``. The
    block boundaries are what RouteFuse slices on, so the order is load-bearing.

    ``vocab_sizes[j]`` is the largest code used by field ``j``; the embedding
    table gets ``vocab_sizes[j] + 1`` rows because code ``0`` is reserved for
    padding / unknown.

    ``user_id_fields`` / ``item_id_fields`` mark the *sparse ID* fields whose
    embedding norm RouteFuse reads as its reliability proxy. They default to the
    first field of each block (the HTLNet convention: field 0 is ``user_id``,
    field ``n_user`` is ``item_id``). Content/metadata fields must NOT be listed
    here -- they are populated for cold entities too, so their norms carry no
    coldness signal.
    """

    vocab_sizes: List[int]
    n_user: int
    n_item: int
    n_ctx: int
    names: Optional[List[str]] = None
    user_id_fields: Tuple[int, ...] = (0,)
    item_id_fields: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.n_user + self.n_item + self.n_ctx != len(self.vocab_sizes):
            raise ValueError(
                f"block sizes {self.n_user}+{self.n_item}+{self.n_ctx} "
                f"!= n_fields {len(self.vocab_sizes)}"
            )
        if self.item_id_fields is None:
            self.item_id_fields = (self.n_user,)
        for j in self.user_id_fields:
            if not 0 <= j < self.n_user:
                raise ValueError(f"user id field {j} outside the user block")
        for j in self.item_id_fields:
            if not self.n_user <= j < self.n_user + self.n_item:
                raise ValueError(f"item id field {j} outside the item block")
        if self.names is not None and len(self.names) != len(self.vocab_sizes):
            raise ValueError("names must align with vocab_sizes")

    @property
    def n_fields(self) -> int:
        return len(self.vocab_sizes)


@dataclass
class LearnerConfig:
    """Stage-3 hyperparameters (paper §5)."""

    emb_dim: int = 10
    label_emb_dim: int = 10
    tower_hidden: Tuple[int, ...] = (128, 64, 32)
    ifu_hidden: Optional[int] = None      # defaults to tower_hidden[0]
    n_heads: int = 2
    dropout: float = 0.0

    # Ablation switches (paper Table 2).
    use_routefuse: bool = True
    use_confidence_prior: bool = True     # Eq. 5
    signed_confidence_prior: bool = True   # blend toward the teacher's
                                           # belief, not |confidence|
    use_confidence_weighting: bool = True  # Eq. 4

    # ── beyond the paper: three candidate changes, all off by default ──
    #
    # `id_dropout` masks one side's sparse ID fields to the padding code on a
    # fraction of *training* rows, which is exactly the condition an eval cold row
    # presents (one side has an ID the model has never learned). Without it the
    # towers only ever see rows where both IDs are informative, and nothing forces
    # them to be able to rank from content alone. Drop probability, per row; the
    # masked side is chosen uniformly so the two cold directions are covered
    # equally. This is the DropoutNet idea, not the paper's.
    id_dropout: float = 0.0
    # `reliability="support"` feeds RouteFuse the warm training support count of
    # each ID instead of the embedding norm. The norm is a *proxy* for "this ID
    # has not been trained", and augmentation breaks it: augmented rows train the
    # cold IDs, the norm gap closes, and the measured gate collapses to 0.5.
    # The support count is the quantity the proxy was standing in for.
    # `reliability="auto"` picks between the two on validation cold Recall@20,
    # per dataset and per seed, instead of the author fixing it by hand. The
    # signal that works depends on whether augmentation is present -- exactly
    # the thing an ablation varies -- so a hand-fixed gate silently favours
    # whichever arm it was chosen on. `train()` resolves "auto" before the
    # learner is built; RouteFuse never sees it.
    reliability: str = "norm"             # norm (paper) | support | auto
    # `cold_id_synthesis` goes one step past `reliability="support"`: instead of
    # only detecting that an ID is untrained and routing around it, generate a
    # stand-in for it from the entity's content and blend the two by support.
    # `synth_pretend_cold` is the fraction of training rows that are told their
    # support is zero -- without it the generator is only ever trained on rows
    # where it is not used. See :mod:`ember2.coldsynth`.
    cold_id_synthesis: bool = False
    synth_pretend_cold: float = 0.5
    synth_hidden: int = 64
    # `aug_soft_targets` uses the simulator's posterior as the *target* on
    # augmented rows (a distillation loss) instead of a hard pseudo-label
    # downweighted by confidence (Eq. 4). Same information, no hard commitment.
    aug_soft_targets: bool = False

    # Optimization.
    lr: float = 1e-3
    l2_shared: float = 1e-5
    l2_task: float = 1e-5
    batch_size: int = 1024
    n_epochs: int = 20
    # Ranking metrics sit on a plateau for the first several epochs before the
    # id embeddings separate, then climb steeply. Early stopping must not fire
    # during that warm-up: with patience=3 and no floor, a run that plateaus one
    # epoch longer than another is killed at epoch 4 and scores near random,
    # which turns an ablation ladder into a measurement of which run got lucky.
    min_epochs: int = 10
    patience: int = 5
    lambda_a: float = 1.0                 # weight on the A_LLM block
    grad_clip: float = 5.0

    # HTLNet Algorithm 1 shared-gradient processing.
    use_grad_processor: bool = True
    pcgrad_alpha: float = 0.5
    balance_gamma: float = 0.1
    balance_clip: float = 10.0

    # LEU temperature annealing. The label state is read out with a softmax
    # over [log p, log(1-p)] / tau, so tau controls how much the chain can carry
    # at all: at tau=9 a fully confident p=0.99 reads out as weights
    # [0.625, 0.375], almost indistinguishable from p=0.01's [0.375, 0.625].
    #
    # A step-count schedule cannot know how long the run is. "Decay by 0.9 every
    # 5000 steps" fires exactly once in a 7,520-step run, ending at tau=9.0 with
    # the label state -- and therefore Eq. 5 -- effectively switched off.
    # "progress" instead anneals geometrically from tau_init to tau_min across
    # the whole run, so the chain is soft early and sharp by the end regardless
    # of dataset size.
    tau_schedule: str = "progress"        # progress | steps
    tau_init: float = 10.0
    tau_min: float = 0.5
    tau_decay_every: int = 5000           # `steps` schedule only
    tau_decay_rate: float = 0.9           # `steps` schedule only


@dataclass
class FilterConfig:
    """Stage-2 coupled-filter hyperparameters (paper §3.1, §5)."""

    hidden_dim: int = 512
    out_dim: int = 200
    dropout: float = 0.1
    n_bpr_triplets: int = 500_000
    lambda_kd: float = 1.0
    lambda_nce: float = 1.0
    nce_temperature: float = 0.1
    n_epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    top_k: int = 30
