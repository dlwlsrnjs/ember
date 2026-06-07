#!/usr/bin/env bash
# End-to-end EMBER pipeline driver (KuaiRand-Pure example).
#
# Stages: (1) direction-specific LoRA simulation, (2) coupled filter +
# augmented cache, (3) cold-aware learner with RouteFuse + CaliChain.
# Paths below are placeholders — point them at your prepared data caches.
set -euo pipefail

DATA=${DATA:-./data}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-0}

# ── Stage 1: simulation (run once per direction) ────────────────────
for DIR in cold_item cold_user; do
  python simulation/build_lora_data.py      --focus "$DIR"
  python simulation/train_lora_simulator.py --focus "$DIR" \
      --base_model Qwen/Qwen2.5-7B-Instruct --lora_r 16 --lora_alpha 32 \
      --lora_dropout 0.1 --epochs 2 --lr 2e-4
  python simulation/simulate.py             --direction "$DIR" \
      --adapter_dir "$DATA/lora_adapter_${DIR}" --out_dir "$DATA"
done

# ── Stage 2: coupled filter + augmented cache ───────────────────────
python filter/extract_text_embeddings.py
python filter/whiten.py
python filter/build_kd_labels.py
python filter/train_coupled_filter.py --alpha_bpr 1.0 --beta_kd 1.0 --gamma_nce 1.0
python filter/build_augmented_cache.py \
    --sim_item_jsonl "$DATA/ember_simulated_cold_item.jsonl" \
    --sim_user_jsonl "$DATA/ember_simulated_cold_user.jsonl" \
    --out_npz "$DATA/ember_cache.npz"

# ── Stage 3: learner (RouteFuse + CaliChain) ────────────────────────
python train_ember.py \
    --cache "$DATA/ember_cache.npz" \
    --n_user_fields 31 --n_item_fields 8 --n_ctx_fields 4 \
    --lambda_a 1.0 --seed "$SEED" --device "$DEVICE" --tag ember
