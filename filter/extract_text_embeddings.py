"""Extract mean-pooled last-layer hidden states from a (LoRA-adapted) Qwen
model — the EMBER coupled-filter text encoder (paper §3.1).

For each text:
  1. Tokenize with model's tokenizer
  2. Forward pass through the LoRA-merged base model
  3. Take output_hidden_states[-1]   shape (1, T, 4096)
  4. Mean-pool over real tokens (mask out padding)
  5. Output single 4096-d vector

Output NPZ (matches OpenAI emb schema for drop-in replacement):
  ids        : int64 (N,)
  text_emb   : float32 (N, hidden_size)
  empty_mask : bool  (N,)
"""
from __future__ import annotations
import argparse, os, sys, time
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


@torch.no_grad()
def encode_batch(model, tok, texts: List[str], device, max_len: int) -> np.ndarray:
    enc = tok(texts, padding=True, truncation=True, max_length=max_len,
              return_tensors="pt").to(device)
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1]  # (B, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / denom  # (B, H)
    return pooled.cpu().to(torch.float32).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--id_col", required=True)
    p.add_argument("--text_cols", nargs="+", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_dir",
                    default="./llm_cold_item/data/lora_adapter_v3_both_no_reasoning",
                    help="LoRA adapter dir. Pass 'none' to use base model only.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--load_4bit", action="store_true",
                    help="Quantize base model to 4-bit (bnb). Saves VRAM.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    print(f"[A] Loading tokenizer + model …", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16,
                        device_map={"": args.device})
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True)
        model_kwargs["quantization_config"] = bnb
        model_kwargs.pop("torch_dtype", None)

    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter_dir.lower() != "none":
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapter")
        print(f"  loading LoRA adapter: {args.adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    else:
        model = base
    model.eval()
    H = model.config.hidden_size
    print(f"  hidden_size: {H}", flush=True)

    print(f"\n[B] Reading {args.input_csv} …", flush=True)
    df = pd.read_csv(args.input_csv)
    df = df.sort_values(args.id_col).reset_index(drop=True)
    print(f"  {len(df):,} rows (id range [{df[args.id_col].min()}, "
          f"{df[args.id_col].max()}])", flush=True)

    def compose(row):
        parts = []
        for c in args.text_cols:
            v = str(row.get(c, "") or "").strip()
            if v and v.lower() not in ("nan", "none", "null"):
                if len(args.text_cols) > 1:
                    label = c.replace("_en", "").replace("_", " ").strip()
                    parts.append(f"{label.capitalize()}: {v}")
                else:
                    parts.append(v)
        return ". ".join(parts).strip() or "(no description available)"

    texts = [compose(r) for _, r in df.iterrows()]
    empty = np.array([t == "(no description available)" for t in texts], dtype=bool)
    print(f"  empty rows: {int(empty.sum()):,}/{len(df):,}", flush=True)

    print(f"\n[C] Forward pass (bs={args.batch_size}, max_len={args.max_len})", flush=True)
    out = np.zeros((len(texts), H), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        pooled = encode_batch(model, tok, batch, args.device, args.max_len)
        out[i:i + len(batch)] = pooled
        if (i // args.batch_size) % 20 == 0:
            done = i + len(batch); el = time.time() - t0
            print(f"    {done:,}/{len(texts):,} "
                  f"({done/max(el,1e-6):.1f}/s, {el:.0f}s)", flush=True)

    print(f"\n[D] Save → {args.out_npz}", flush=True)
    np.savez_compressed(
        args.out_npz,
        ids=df[args.id_col].values.astype(np.int64),
        text_emb=out,
        empty_mask=empty,
    )
    print(f"  shape={out.shape}  "
          f"({os.path.getsize(args.out_npz)/1e6:.1f} MB)", flush=True)
    print(f"  norm stats: mean={np.linalg.norm(out, axis=1).mean():.3f}, "
          f"std={np.linalg.norm(out, axis=1).std():.3f}", flush=True)


if __name__ == "__main__":
    main()
