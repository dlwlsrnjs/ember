"""Whiten + L2-normalize a text-emb NPZ (anisotropy fix for raw LLM hidden states).

LLM mean-pooled last-hidden-states have a well-known anisotropy problem:
all vectors cluster in a narrow cone (random pair cosine ≈ 0.7+), which
collapses BPR/contrastive losses. Centering and L2-normalizing pushes the
distribution back toward isotropic (random cosine ≈ 0).

Input  NPZ keys: ids, text_emb, empty_mask
Output NPZ keys: same, but text_emb is whitened+L2-normalized.
"""
from __future__ import annotations
import argparse, os
import numpy as np


def whiten_normalize(emb: np.ndarray) -> np.ndarray:
    emb = emb - emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    return (emb / norms).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_npz", required=True)
    p.add_argument("--out_npz", required=True)
    args = p.parse_args()

    a = np.load(args.in_npz)
    raw = a["text_emb"]
    print(f"Loaded {raw.shape} from {args.in_npz}", flush=True)
    raw_norms = np.linalg.norm(raw, axis=1)
    print(f"  raw norm: mean={raw_norms.mean():.2f} std={raw_norms.std():.2f}", flush=True)

    out = whiten_normalize(raw)
    norms = np.linalg.norm(out, axis=1)
    print(f"  out norm: mean={norms.mean():.4f} std={norms.std():.4f}", flush=True)

    # diagnose anisotropy improvement
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(raw), 1000), rng.integers(0, len(raw), 1000)
    raw_norm_n = raw / (raw_norms.reshape(-1, 1) + 1e-8)
    cos_raw = (raw_norm_n[i] * raw_norm_n[j]).sum(axis=1)
    cos_out = (out[i] * out[j]).sum(axis=1)
    print(f"  random pair cosine:  raw {cos_raw.mean():.3f}±{cos_raw.std():.3f}  →  "
          f"whitened {cos_out.mean():.3f}±{cos_out.std():.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, ids=a["ids"], text_emb=out,
                         empty_mask=a["empty_mask"])
    print(f"Saved → {args.out_npz} ({os.path.getsize(args.out_npz)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
