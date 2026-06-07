"""LoRA fine-tune of Qwen2.5-7B-Instruct as the EMBER direction-specific simulator.

Faithful to paper §4.3.1 (LoRA on LLaMA-7B). We use Qwen2.5-7B-Instruct
(non-gated, comparable size) + 4-bit NF4 quantization + LoRA adapters on
attention + FFN projections.

Inputs (ChatML JSONL produced by build_lora_train_data.py):
  llm_cold_item/data/lora_sft_train.jsonl
  llm_cold_item/data/lora_sft_val.jsonl

Output:
  llm_cold_item/data/lora_adapter/   (PEFT adapter + tokenizer + config)
  + training logs to stdout.

Run inside fp-mix-exp-llm docker on GPU 0 or 1.
"""
from __future__ import annotations
import argparse, json, os, sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_jsonl", default="./llm_cold_item/data/lora_v3/both/no_reasoning/train.jsonl")
    p.add_argument("--val_jsonl",   default="./llm_cold_item/data/lora_v3/both/no_reasoning/val.jsonl")
    p.add_argument("--output_dir",  default="./llm_cold_item/data/lora_adapter_v3_both_no_reasoning")
    p.add_argument("--n_epochs", type=int, default=2)
    p.add_argument("--per_device_bs", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=1536)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--bf16", action="store_true", default=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}, "
          f"device count: {torch.cuda.device_count()}", flush=True)

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # ── Tokenizer ──
    print(f"\n[A] Load tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (4-bit) ──
    print(f"\n[B] Load 4-bit base model: {args.base_model}", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    print(f"  base model loaded, params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B",
          flush=True)

    # ── LoRA ──
    print(f"\n[C] Apply LoRA (r={args.lora_r}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout})", flush=True)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Datasets ──
    print(f"\n[D] Load datasets", flush=True)
    ds = load_dataset("json", data_files={
        "train": args.train_jsonl, "validation": args.val_jsonl,
    })
    print(f"  train: {len(ds['train'])}, val: {len(ds['validation'])}", flush=True)
    print(f"  example: {ds['train'][0]['messages'][1]['content'][:120]}...", flush=True)

    # ── Training ──
    print(f"\n[E] Set up SFTTrainer", flush=True)
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=args.bf16,
        max_seq_length=args.max_seq_len,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
    )

    # Completion-only loss via DataCollatorForCompletionOnlyLM
    # Qwen2.5 chat template marks the assistant turn with the header
    # "<|im_start|>assistant\n" — loss is computed only on tokens after this.
    from trl import DataCollatorForCompletionOnlyLM
    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )
    print(f"  starting training...", flush=True)
    trainer.train()

    # ── Save adapter ──
    print(f"\n[F] Save adapter to {args.output_dir}", flush=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DONE.", flush=True)


if __name__ == "__main__":
    main()
