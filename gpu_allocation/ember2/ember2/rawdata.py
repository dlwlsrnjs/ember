"""Raw dataset representation, the strict cold-start split, and a synthetic
generator for exercising the whole pipeline without a licensed dataset.

:class:`RawDataset` is the interface every stage consumes. To plug in
KuaiRand-Pure / Kaggle Acquire-Valued-Shoppers / Tenrec, produce one of these
and the rest of the pipeline is unchanged.

**Strict cold-start split (§2).** 20% of items and 20% of users are designated
cold and *all* their training-period interactions are removed. Cold entities
keep their categorical metadata and their text, and they remain in validation /
test -- that is the only way they can be evaluated at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FieldSpec, TaskSpec
from .dataset import CASE_CODE, EvalSplit

TRAIN, VAL, TEST = 0, 1, 2


@dataclass
class RawDataset:
    """Interactions plus the side information every stage needs.

    Attributes:
      user / item:   ``(N,)`` entity indices per interaction (0-based).
      binary:        ``(N, T)`` conversion labels.
      core:          ``(N,)`` continuous core outcome, raw units.
      ctx:           ``(N, n_ctx)`` context codes (already 1-based; 0 = unknown).
      period:        ``(N,)`` one of TRAIN / VAL / TEST.
      user_attrs:    ``(n_users, A_u)`` categorical metadata codes, 1-based.
      item_attrs:    ``(n_items, A_i)`` categorical metadata codes, 1-based.
      user_text / item_text: natural-language profiles for the LLM stages.
      cold_users / cold_items: index arrays of the held-cold entities.
    """

    user: np.ndarray
    item: np.ndarray
    binary: np.ndarray
    core: np.ndarray
    ctx: np.ndarray
    period: np.ndarray

    user_attrs: np.ndarray
    item_attrs: np.ndarray
    user_text: List[str]
    item_text: List[str]

    cold_users: np.ndarray
    cold_items: np.ndarray

    tasks: TaskSpec
    attr_vocab: int
    ctx_vocab: Sequence[int]
    n_users: int
    n_items: int

    # ── derived ─────────────────────────────────────────────────────
    @property
    def is_cold_user(self) -> np.ndarray:
        m = np.zeros(self.n_users, bool)
        m[self.cold_users] = True
        return m

    @property
    def is_cold_item(self) -> np.ndarray:
        m = np.zeros(self.n_items, bool)
        m[self.cold_items] = True
        return m

    def field_spec(self) -> FieldSpec:
        """Field layout ``[user_id | user attrs | item_id | item attrs | ctx]``.

        Field 0 and field ``n_user`` are the sparse IDs RouteFuse reads.
        """
        a_u, a_i = self.user_attrs.shape[1], self.item_attrs.shape[1]
        vocab = (
            [self.n_users]
            + [self.attr_vocab] * a_u
            + [self.n_items]
            + [self.attr_vocab] * a_i
            + list(self.ctx_vocab)
        )
        names = (
            ["user_id"]
            + [f"u_attr_{j}" for j in range(a_u)]
            + ["item_id"]
            + [f"i_attr_{j}" for j in range(a_i)]
            + [f"ctx_{j}" for j in range(len(self.ctx_vocab))]
        )
        return FieldSpec(
            vocab_sizes=vocab,
            n_user=1 + a_u,
            n_item=1 + a_i,
            n_ctx=len(self.ctx_vocab),
            names=names,
            user_id_fields=(0,),
            item_id_fields=(1 + a_u,),
        )

    def compose_rows(self, users: np.ndarray, items: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        """``(u, i, c)`` triples -> ``(N, n_fields)`` categorical codes.

        Entity ids are stored 1-based so code 0 stays reserved for padding. A
        cold entity keeps its id and its metadata; what it lacks is training
        rows, so its id embedding never moves from initialization -- exactly the
        signal RouteFuse's reliability proxy reads.
        """
        return np.concatenate(
            [
                users[:, None] + 1,
                self.user_attrs[users],
                items[:, None] + 1,
                self.item_attrs[items],
                ctx,
            ],
            axis=1,
        ).astype(np.int64)

    def warm_training_mask(self) -> np.ndarray:
        """Training-period rows that touch no cold entity -- this is ``D_warm``."""
        return (
            (self.period == TRAIN)
            & ~self.is_cold_user[self.user]
            & ~self.is_cold_item[self.item]
        )


# ── evaluation splits ───────────────────────────────────────────────
def build_eval_split(
    raw: RawDataset,
    period: int,
    rng: np.random.Generator,
    n_candidates: int = 100,
    max_groups: Optional[int] = None,
) -> EvalSplit:
    """Candidate lists for the warm and cold ranking problems.

    Each group is one user's positives plus sampled negatives, padded to
    ``n_candidates``, all sharing the positive row's context.

    **Negatives are drawn from the same pool as the group's positives** -- cold
    items are ranked against cold items, warm against warm. Mixing the pools
    would make the metric dominated by the *calibration offset* between cold and
    warm scores rather than by ranking quality: augmented rows are all positives,
    so a cold id can acquire a systematic score offset, and a mixed list would
    then let cold items crowd the top-20 of every warm group (measured: warm
    Recall@20 collapsing to 0.03, far below the 0.19 random baseline, while cold
    rose to 0.37 -- an artifact in both directions, not a result).

    A warm user contributes two groups, one per pool. A cold user contributes one
    cold group: the coldness is already on the user side, so its items come from
    the warm pool.
    """
    rows = np.flatnonzero(raw.period == period)
    if len(rows) == 0:
        raise ValueError(f"no rows in period {period}")

    by_user: Dict[int, List[int]] = {}
    for r in rows:
        by_user.setdefault(int(raw.user[r]), []).append(int(r))

    users = sorted(by_user)
    if max_groups is not None and len(users) > max_groups:
        users = [users[i] for i in rng.choice(len(users), max_groups, replace=False)]

    is_cold_u, is_cold_i = raw.is_cold_user, raw.is_cold_item
    warm_pool = np.setdiff1d(np.arange(raw.n_items), raw.cold_items)
    cold_pool = raw.cold_items
    g_users, g_items, g_ctx, g_bin, g_core, g_gid, g_obs = [], [], [], [], [], [], []
    g_case: List[np.ndarray] = []
    gid = 0

    for u in users:
        idx = np.asarray(by_user[u])
        pos_all = idx[raw.binary[idx, 0] > 0]
        if len(pos_all) == 0:                  # no relevant item -> no signal
            continue
        ctx_row = raw.ctx[pos_all[0]]
        seen = set(raw.item[idx].tolist())

        # A 20%/20% split produces three cold cases, and they are not the same
        # problem: warm-user/cold-item and cold-user/warm-item each have a warm
        # side to route evidence *from*, while cold-user/cold-item has none --
        # the direction-aware gate has nothing to work with there. They are kept
        # as separate cases so the metric can report them separately instead of
        # averaging a solvable case with an unsolvable one.
        if is_cold_u[u]:
            cases = [
                ("cold_user", warm_pool, pos_all[~is_cold_i[raw.item[pos_all]]]),
                ("cold_both", cold_pool, pos_all[is_cold_i[raw.item[pos_all]]]),
            ]
        else:
            cases = [
                ("warm", warm_pool, pos_all[~is_cold_i[raw.item[pos_all]]]),
                ("cold_item", cold_pool, pos_all[is_cold_i[raw.item[pos_all]]]),
            ]

        for case, pool, pos in cases:
            if len(pos) == 0 or len(pool) <= len(pos):
                continue
            if len(pos) > n_candidates // 2:
                pos = pos[rng.choice(len(pos), n_candidates // 2, replace=False)]

            avail = pool[~np.isin(pool, list(seen))]
            n_neg = min(max(0, n_candidates - len(pos)), len(avail))
            neg_items = rng.choice(avail, n_neg, replace=False)

            n_rows = len(pos) + n_neg
            g_users.append(np.full(n_rows, u))
            g_items.append(np.concatenate([raw.item[pos], neg_items.astype(np.int64)]))
            g_ctx.append(np.tile(ctx_row, (n_rows, 1)))
            g_bin.append(
                np.concatenate(
                    [raw.binary[pos], np.zeros((n_neg, raw.binary.shape[1]), np.float32)]
                )
            )
            g_core.append(np.concatenate([raw.core[pos], np.zeros(n_neg, np.float32)]))
            g_gid.append(np.full(n_rows, gid, dtype=np.int64))
            # Only the positives are real logged rows; the negatives' core value
            # is a placeholder zero and must stay out of the NRMSE.
            g_obs.append(np.concatenate([np.ones(len(pos), bool), np.zeros(n_neg, bool)]))
            g_case.append(np.full(n_rows, CASE_CODE[case], dtype=np.int8))
            gid += 1

    users_arr = np.concatenate(g_users).astype(np.int64)
    items_arr = np.concatenate(g_items).astype(np.int64)
    return EvalSplit(
        feature_ids=raw.compose_rows(users_arr, items_arr, np.concatenate(g_ctx)),
        binary=np.concatenate(g_bin).astype(np.float32),
        core=np.concatenate(g_core).astype(np.float32),
        group_id=np.concatenate(g_gid),
        is_cold=(is_cold_u[users_arr] | is_cold_i[items_arr]),
        is_observed=np.concatenate(g_obs),
        case=np.concatenate(g_case),
    )


# ── synthetic generator ─────────────────────────────────────────────
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic(
    n_users: int = 3000,
    n_items: int = 1500,
    n_interactions: int = 800_000,
    latent_dim: int = 8,
    n_user_attrs: int = 12,
    n_item_attrs: int = 8,
    attr_vocab: int = 16,
    ctx_vocab: Sequence[int] = (7, 24, 5, 2),
    cold_frac: float = 0.2,
    attr_noise: float = 2.0,
    signal: float = 3.0,
    base_logit: float = -2.5,
    seed: int = 0,
) -> Tuple[RawDataset, Dict[str, np.ndarray]]:
    """A latent-factor dataset with the paper's cold-start structure.

    This exists so every stage -- SFT data construction, filtering, simulation,
    cache building, learner training and the ranking protocol -- can be run and
    tested end to end. The absolute metric values mean nothing; what it validates
    is that the method is implemented correctly and that the pieces compose.

    Metadata is a noisy quantization of the latent factors, which is what makes a
    cold entity predictable *at all*: it is the content signal that survives when
    the behaviour is deleted.

    ``attr_noise`` controls how good a proxy the metadata is for behaviour, and
    it is the single most consequential knob for anything cold-start. At low
    noise the categorical attributes are a near-lossless quantization of the
    latent factors, a cold entity is almost fully described by its content, and
    there is no cold-start gap left for simulated behaviour to close -- the
    no-augmentation baseline already reaches ~79% of the oracle ceiling. That is
    the opposite of the regime the paper targets, where content encodes "what it
    looks like, not who will engage" and the cold-removed baseline scores under
    half its warm value. The default is calibrated to that ratio; see
    ``scripts/calibrate_attr_noise.py``.

    ``signal`` and ``base_logit`` set how separable the task is. They matter: at
    a ~50% base click rate every label is close to a coin flip, the gap between a
    random ranker and an oracle collapses, and an ablation ladder run on top of
    it measures noise. The defaults put the click rate near 15% with a wide score
    spread, leaving real headroom between the two.

    Returns the dataset and an ``oracle`` dict (latent factors and the true score
    function) used only by the test doubles in :mod:`ember2.simulate` and
    :mod:`ember2.filterstage`.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, latent_dim))
    V = rng.normal(size=(n_items, latent_dim))
    ub = rng.normal(scale=0.5, size=n_users)
    ib = rng.normal(scale=0.5, size=n_items)

    def score(u, i):
        return (
            signal * (U[u] * V[i]).sum(-1) / np.sqrt(latent_dim) + ub[u] + ib[i] + base_logit
        )

    def attrs(latent, n_attrs, noise=None):
        noise = attr_noise if noise is None else noise
        cols = []
        for j in range(n_attrs):
            x = latent[:, j % latent.shape[1]] + rng.normal(0, noise, len(latent))
            cols.append(np.clip(((x + 3) / 6 * attr_vocab).astype(np.int64), 0, attr_vocab - 1) + 1)
        return np.stack(cols, 1)

    user_attrs = attrs(U, n_user_attrs)
    item_attrs = attrs(V, n_item_attrs)

    user_text = [
        "User profile: " + ", ".join(f"trait_{j}={v}" for j, v in enumerate(row))
        for row in user_attrs
    ]
    item_text = [
        "Item metadata: " + ", ".join(f"tag_{j}={v}" for j, v in enumerate(row))
        for row in item_attrs
    ]

    u_idx = rng.integers(0, n_users, n_interactions)
    i_idx = rng.integers(0, n_items, n_interactions)
    s = score(u_idx, i_idx)
    click = (rng.random(n_interactions) < _sigmoid(s)).astype(np.float32)
    long_view = (rng.random(n_interactions) < _sigmoid(0.8 * s - 0.6)).astype(np.float32) * click
    core = (
        np.clip(np.log1p(np.exp(s)) * 12.0 + rng.normal(0, 2.0, n_interactions), 0, None) * click
    ).astype(np.float32)

    ctx = np.stack([rng.integers(1, v + 1, n_interactions) for v in ctx_vocab], 1)
    period = rng.choice([TRAIN, VAL, TEST], n_interactions, p=[0.7, 0.15, 0.15])

    n_cold_u, n_cold_i = int(n_users * cold_frac), int(n_items * cold_frac)
    cold_users = np.sort(rng.permutation(n_users)[:n_cold_u])
    cold_items = np.sort(rng.permutation(n_items)[:n_cold_i])

    raw = RawDataset(
        user=u_idx.astype(np.int64),
        item=i_idx.astype(np.int64),
        binary=np.stack([click, long_view], 1),
        core=core,
        ctx=ctx.astype(np.int64),
        period=period.astype(np.int64),
        user_attrs=user_attrs,
        item_attrs=item_attrs,
        user_text=user_text,
        item_text=item_text,
        cold_users=cold_users,
        cold_items=cold_items,
        tasks=TaskSpec(binary=["click", "long_view"], core="watch_time"),
        attr_vocab=attr_vocab,
        ctx_vocab=list(ctx_vocab),
        n_users=n_users,
        n_items=n_items,
    )
    oracle = {
        "U": U, "V": V, "u_bias": ub, "i_bias": ib,
        "latent_dim": latent_dim, "signal": signal, "base_logit": base_logit,
    }
    return raw, oracle


def oracle_score(oracle: Dict[str, np.ndarray], u: np.ndarray, i: np.ndarray) -> np.ndarray:
    """The synthetic dataset's true affinity, shared by every test double.

    Kept in one place so the fixtures in :mod:`ember2.simulate` and
    :mod:`ember2.filterstage` cannot silently drift from the generator that
    produced the labels.
    """
    o = oracle
    return (
        float(o.get("signal", 1.0)) * (o["U"][u] * o["V"][i]).sum(-1) / np.sqrt(o["latent_dim"])
        + o["u_bias"][u]
        + o["i_bias"][i]
        + float(o.get("base_logit", 0.0))
    )
