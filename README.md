# EMBER

**Direction- and Confidence-Aware LLM Behavior Simulation for Cold-Start Multi-Task Recommendation**

Reference implementation of EMBER (*Emulated Multi-target Behavior Evidence for
Recommendation*). EMBER fine-tunes **direction-specific LoRA simulators** that
emit full hybrid label tuples with per-task confidence, retrieves candidate
pairs with a **coupled filter**, and trains a cold-aware downstream learner with
two components — **RouteFuse** (direction-aware cross-field fusion) and
**CaliChain** (a distillation-calibrated objective). On KuaiRand-Pure, Kaggle
Acquire-Valued-Shoppers, and Tenrec under strict cold-start splits, EMBER yields
16–36% relative gains on cold Recall@20 / NDCG@20 over a strong HTLNet-based
baseline without degrading warm performance.

All LLM calls are **offline**: serving uses only the trained recommender.

---

## Problem

Each instance `x = (u, i, c)` has hybrid targets `y = (y^(1), …, y^(T), y^(r))`,
where the `y^(t) ∈ {0,1}` are preceding conversion labels (e.g. click,
long-view) and `y^(r) ∈ ℝ≥0` is the continuous core outcome (watch time / spend);
`T = 2` in all datasets. Under **strict cold-start splits**, 20% of items and
20% of users are designated cold and *all* their training-period interactions
are removed. Cold entities keep only text / categorical metadata. Training
solves

```
Θ* = argmin_Θ  L( f_Θ ; D_warm ∪ A_LLM )
```

where `A_LLM` holds LLM-simulated rows for both cold directions. At serving time
`ŷ = f_Θ*(u, i, c)` requires no LLM calls.

Existing LLM cold-start simulators emit a single hard yes/no label and ignore
both the structure of hybrid label tuples and the *direction* of coldness.
EMBER addresses both.

---

## Method

```
Stage 1: Simulation         Stage 2: Cache              Stage 3: Learner
direction-specific LoRA      coupled filter over          RouteFuse: g_{u→i}, g_{i→u}
(cold-item / cold-user)  →   D_warm ∪ A_LLM           →   route warm → cold
→ hybrid tuple + conf. q     → top-K pairs simulated      CaliChain: q·L + q-prior in chain
```

### Stage 1 — Direction-specific LLM simulation (`simulation/`)

A Qwen2.5-7B-Instruct simulator is LoRA-fine-tuned to map a structured user–item
context into a full hybrid label tuple `(click, long_view, watch_time)` **with
per-task confidence**. We train **separate LoRA adapters** for the cold-item and
cold-user directions: on the cold side, the behavior block is replaced by an
explicit `insufficient_history` marker, so the simulator trains under the same
missing-field pattern it sees at inference. A *reasoning-SFT* variant adds a
brief rationale before the JSON label.

- `build_lora_data.py` — builds direction-masked SFT data (cold-item / cold-user
  × with/without reasoning) with symmetric user/item blocks and top-`L`
  text-similar histories.
- `train_lora_simulator.py` — LoRA SFT of Qwen2.5-7B (4-bit NF4).
- `simulate.py` — batched vLLM inference that emits the hybrid tuple **and**
  per-task confidence `q` from teacher-forced valid-label token log-probabilities
  (CaliChain's default confidence estimator).

### Stage 2 — Coupled filter & augmented cache (`filter/`)

Querying the LLM for every cold pair is infeasible, so a coupled filter
retrieves candidates first. User/item texts are encoded by mean-pooled, whitened
LLM hidden states; mappers `F_U`, `F_I` score `s(u,i) = z_uᵀ z_i` and are trained
with `L_BPR + L_KD + λ_nce · L_InfoNCE`, where `L_KD` is BCE distillation against
a teacher LLM's yes/no plausibility judgment. Only the top-`K` partners per cold
entity are simulated. Predicted negatives serve only as a filter (adding them to
the cache skews class balance and hurts cold Recall@20).

- `extract_text_embeddings.py` — mean-pooled last-layer Qwen hidden states.
- `whiten.py` — whitening + L2-normalization (anisotropy fix).
- `build_kd_labels.py` — teacher LLM yes/no KD labels.
- `train_coupled_filter.py` — BPR + KD + InfoNCE.
- `build_augmented_cache.py` — merges simulated rows into `D_warm`, carrying
  per-task confidence and an `is_aug` flag for CaliChain.

### Stage 3 — Cold-aware learner (`ember/`, `train_ember.py`)

The learner is instantiated on HTLNet (shared field embeddings → cross-field
fusion → sequential task towers).

**RouteFuse** (`ember/routefuse.py`, paper §3.2 Eq. 2–3). HTLNet's task chain
assumes reliable field representations, which fails on cold rows. RouteFuse
estimates per-sample reliability from the z-normalized mean ID-embedding norm
(`ρ_u, ρ_i` — sparse cold fields stay near initialization) and forms asymmetric
gates with learnable temperature `τ = softplus(η) + 1e-3`:

```
g_{u→i} = σ((ρ_u − ρ_i)/τ),   g_{i→u} = σ((ρ_i − ρ_u)/τ)
H_{i←u} = g_{u→i} · MHA(V, U, U),   H_{u←i} = g_{i→u} · MHA(U, V, V)
E_DA    = concat(U + H_{u←i},  V + H_{i←u},  C)
```

so warm-side evidence is routed into the cold side. The gate recovers the
augmentation direction without an explicit flag.

**CaliChain** (`ember/calichain.py`, paper §3.3 Eq. 4–5). Simulator confidence is
consumed twice: (1) as a **per-sample loss weight** on augmented rows so
low-confidence pseudo-labels contribute less; and (2) as a **prior in the task
chain** — before forming the next label state, the internal logit is blended
with `logit(q)` via a learnable, task-specific `α_t = clip(σ(a_t), 0.05, 0.95)`
(initialized at 0.7), keeping noisy pseudo-labels from cascading into the core
regression head.

```python
from ember import EmberHTLNet
model = EmberHTLNet(field_vocab_sizes, n_user_fields, n_item_fields, n_ctx_fields)
out = model(feature_ids, q_click=q_c, q_long_view=q_lv, aug_mask=is_aug)  # training
out = model(feature_ids)                                                  # serving / eval
```

---

## Repository layout

```
ember/                     # cold-aware learner (the paper's contribution)
  routefuse.py             #   RouteFuse — direction-aware fusion (Eq. 2-3)
  calichain.py             #   CaliChain — confidence weighting + prior (Eq. 4-5)
  htlnet.py                #   vendored HTLNet blocks (embedding, tower, LEU, IFU)
  learner.py               #   EmberHTLNet = HTLNet + RouteFuse + CaliChain
  optim.py                 #   HTLNet shared-gradient processor (Algorithm 1)
  data.py                  #   warm + augmented cache dataset
simulation/                # Stage 1: direction-specific LoRA simulation
filter/                    # Stage 2: coupled filter + augmented cache builder
scripts/run_pipeline.sh    # end-to-end driver
train_ember.py             # Stage 3 entry point
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Stage 1 — simulator
python simulation/build_lora_data.py        --focus cold_item   ...
python simulation/train_lora_simulator.py   --focus cold_item   ...
python simulation/simulate.py               --direction cold_item ...

# Stage 2 — coupled filter + augmented cache
python filter/extract_text_embeddings.py    ...
python filter/whiten.py                      ...
python filter/build_kd_labels.py             ...
python filter/train_coupled_filter.py        --gamma_nce 1.0 ...
python filter/build_augmented_cache.py       --out_npz data/ember_cache.npz ...

# Stage 3 — learner (RouteFuse + CaliChain)
python train_ember.py --cache data/ember_cache.npz \
  --n_user_fields 31 --n_item_fields 8 --n_ctx_fields 4 \
  --lambda_a 1.0 --seed 0 --tag ember
```

`scripts/run_pipeline.sh` chains all three stages. Ablations are exposed as
flags: `--no_routefuse`, `--no_confidence_prior`, and `--lambda_a 0` (drop
augmentation).

---

## Configuration (paper §5)

- **Simulator:** Qwen2.5-7B-Instruct, LoRA `r/α/dropout = 16/32/0.1`, max seq.
  1536, greedy decoding, max 48 new tokens, 2 epochs, LR `2e-4`, bf16, 4-bit NF4.
- **Retrieval:** top-`K` = 30 (KuaiRand / Kaggle), 10/30 (Tenrec); filter dim
  512/200; BPR triplets 500K; KD labels 68K.
- **Learner:** batch 1024, 20 epochs; `λ_a = 0.5` (Kaggle) or `1.0`; `α_t`
  learnable; RouteFuse `τ = softplus(η)`.

The full pipeline fits within ~24 A100-GPU-hours and yields ~91K augmented rows.

---

## Datasets

KuaiRand-Pure (click; long-view; clipped watch time), Kaggle
Acquire-Valued-Shoppers (1-month / 1-year repurchase; 1-month spend), and Tenrec
(click; high-intent multi-feedback; normalized engagement). For each, 20% of
items and 20% of users are cold. All LLM-facing components (simulator SFT, filter
BPR triplets, KD teacher judgments) are trained strictly inside the cold-removed
training period; cold entities enter only through metadata, never through
behavioral labels.

> Raw datasets, derived NPZ caches, LoRA adapters, simulated JSONL, and model
> checkpoints are **not** committed (see `.gitignore`); regenerate them with the
> pipeline above.
