"""Command line entry points.

  python -m ember2.cli pipeline --tag full
  python -m ember2.cli pipeline --sim hard_label --tag hard_label
  python -m ember2.cli train --cache runs/cache.npz --no-routefuse
  python -m ember2.cli ablate --seeds 0 1 2 --epochs 25
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from .config import LearnerConfig
from .dataset import Cache
from .pipeline import PipelineConfig, run_pipeline
from .train import train


def _learner_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lambda-a", type=float, default=1.0, help="weight on the A_LLM block (Eq. 4)")
    p.add_argument("--no-routefuse", action="store_true")
    p.add_argument("--no-confidence-prior", action="store_true", help="drop Eq. 5")
    p.add_argument("--no-confidence-weighting", action="store_true", help="drop Eq. 4")
    p.add_argument("--unsigned-prior", action="store_true",
                   help="blend Eq. 5 toward |confidence| as written, not toward the "
                        "simulator's posterior")
    p.add_argument("--tau-schedule", default="progress", choices=["progress", "steps"],
                   help="LEU temperature anneal; `steps` is the paper's step-count decay")
    p.add_argument("--no-grad-processor", action="store_true")
    # Beyond the paper.
    p.add_argument("--id-dropout", type=float, default=0.0,
                   help="fraction of training rows that lose one side's ID fields")
    p.add_argument("--reliability", default="norm", choices=["norm", "support"],
                   help="RouteFuse reliability signal: embedding norm (paper) or "
                        "warm training support count")
    p.add_argument("--cold-id-synthesis", action="store_true",
                   help="blend each ID embedding with one generated from its content, "
                        "weighted by support (implies --reliability support)")
    p.add_argument("--pretend-cold", type=float, default=0.5,
                   help="synthesis: fraction of training rows told their support is 0")
    p.add_argument("--soft-targets", action="store_true",
                   help="fit the simulator's posterior on augmented rows, not its label")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-epochs", type=int, default=10,
                   help="no early stopping before this epoch (warm-up plateau)")
    p.add_argument("--emb-dim", type=int, default=10)


def _learner_cfg(a: argparse.Namespace) -> LearnerConfig:
    return LearnerConfig(
        emb_dim=a.emb_dim,
        lambda_a=a.lambda_a,
        use_routefuse=not a.no_routefuse,
        use_confidence_prior=not a.no_confidence_prior,
        use_confidence_weighting=not a.no_confidence_weighting,
        signed_confidence_prior=not a.unsigned_prior,
        tau_schedule=a.tau_schedule,
        use_grad_processor=not a.no_grad_processor,
        id_dropout=a.id_dropout,
        # Synthesis reads the same support counts the gate does; asking for one
        # without the other would leave the gate on the signal the synthesis was
        # introduced to replace.
        reliability="support" if a.cold_id_synthesis else a.reliability,
        cold_id_synthesis=a.cold_id_synthesis,
        synth_pretend_cold=a.pretend_cold,
        aug_soft_targets=a.soft_targets,
        n_epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        patience=a.patience,
        min_epochs=a.min_epochs,
    )


def _save(result: Dict, out_dir: str, tag: str, seed: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    return path


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ember2")
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── pipeline ────────────────────────────────────────────────────
    pp = sub.add_parser("pipeline", help="run all three stages end to end")
    pp.add_argument("--tag", default="full")
    pp.add_argument("--sim", default="directional",
                    choices=["directional", "hard_label", "filter_only",
                             "model_teacher", "none"],
                    help="who labels the retrieved pairs; `model_teacher` is the "
                         "non-LLM control (the warm-only learner labels its own "
                         "cold pairs)")
    pp.add_argument("--pairs", default="filter", choices=["filter", "random"])
    pp.add_argument("--negatives", default="balanced", choices=["balanced", "none", "all"],
                    help="what to do with simulated negatives; `none` is the paper "
                         "(filter only), `all` is the paper's measured failure case")
    pp.add_argument("--top-k", type=int, default=30)
    pp.add_argument("--sim-noise", type=float, default=0.9)
    pp.add_argument("--teacher-epochs", type=int, default=10,
                    help="--sim model_teacher: epochs for the non-LLM teacher")
    pp.add_argument("--with-reasoning", action="store_true", help="reasoning-SFT variant")
    pp.add_argument("--save-cache", default=None)
    _learner_args(pp)

    # ── train ───────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Stage 3 only, from a saved cache")
    tp.add_argument("--cache", required=True)
    tp.add_argument("--tag", default="train")
    _learner_args(tp)

    # ── ablate ──────────────────────────────────────────────────────
    ap = sub.add_parser("ablate", help="reproduce the Table 2 ablation ladder")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--emb-dim", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    a = p.parse_args(argv)

    if a.cmd == "pipeline":
        result = run_pipeline(
            PipelineConfig(
                seed=a.seed, device=a.device, sim=a.sim, pairs=a.pairs,
                negatives=a.negatives, top_k=a.top_k, sim_noise=a.sim_noise,
                teacher_epochs=a.teacher_epochs,
                with_reasoning=a.with_reasoning,
            ),
            _learner_cfg(a),
            cache_path=a.save_cache,
        )
        print(f"\nSaved -> {_save(result, a.out_dir, a.tag, a.seed)}")

    elif a.cmd == "train":
        result = train(Cache.load(a.cache), _learner_cfg(a), device=a.device, seed=a.seed)
        print(f"\nSaved -> {_save(result, a.out_dir, a.tag, a.seed)}")

    elif a.cmd == "ablate":
        from .ablation import run_ablation

        run_ablation(out_dir=a.out_dir, seeds=a.seeds, device=a.device,
                     epochs=a.epochs, emb_dim=a.emb_dim)


if __name__ == "__main__":
    main()
