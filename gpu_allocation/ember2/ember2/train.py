"""Stage 3 -- training and evaluation of the cold-aware learner.

Optimizes ``L(f_Theta; D_warm u A_LLM)`` (Eq. 1) with CaliChain's
confidence-weighted objective, and evaluates under the paper's protocol:
Recall@20 / NDCG@20 on overall / warm / cold splits ranked by the primary intent
logit, plus per-task AUC and NRMSE on the normalized core target.

Evaluation never passes simulator confidence to the model -- serving sees no
simulator, so neither does eval.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calichain import hybrid_losses
from .config import LearnerConfig
from .dataset import Cache, EvalDataset, EvalSplit, TrainDataset
from .gradproc import SharedGradientProcessor
from .learner import EmberLearner
from .metrics import case_groups, evaluate_pointwise, evaluate_ranking, split_groups


def warm_support(cache: Cache) -> tuple:
    """Warm training rows per user / item ID code (``reliability='support'``).

    Counted over ``D_warm`` only. Augmented rows must not count: the whole point
    of the signal is "how much logged behaviour does this ID have", and letting
    the augmentation inflate it reproduces exactly the collapse the embedding
    norm suffers.
    """
    fields = cache.fields
    warm = cache.train_is_aug <= 0
    ids = cache.train_feature_ids[warm]
    n_u = max(int(fields.vocab_sizes[f]) for f in fields.user_id_fields) + 1
    n_i = max(int(fields.vocab_sizes[f]) for f in fields.item_id_fields) + 1
    su = np.bincount(ids[:, list(fields.user_id_fields)].ravel(), minlength=n_u)
    si = np.bincount(ids[:, list(fields.item_id_fields)].ravel(), minlength=n_i)
    su[0] = 0                      # padding code: no support, by definition
    si[0] = 0
    return torch.from_numpy(su.astype(np.float32)), torch.from_numpy(si.astype(np.float32))


@dataclass
class EvalResult:
    ranking: Dict[str, Dict[str, float]]
    pointwise: Dict[str, Dict[str, float]]
    gates: Dict[str, float] = field(default_factory=dict)
    by_case: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def score(self, k: int = 20) -> float:
        """Model-selection scalar: overall Recall@k."""
        return self.ranking.get("overall", {}).get(f"recall@{k}", float("nan"))

    def brief(self, k: int = 20) -> str:
        bits = []
        for split in ("overall", "warm", "cold"):
            r = self.ranking.get(split)
            if r and r.get("n_groups"):
                bits.append(f"{split} R@{k}={r[f'recall@{k}']:.4f} N@{k}={r[f'ndcg@{k}']:.4f}")
        return " | ".join(bits)


@torch.no_grad()
def evaluate(
    model: EmberLearner,
    split: EvalSplit,
    cache: Cache,
    device: torch.device,
    k: int = 20,
    batch_size: int = 4096,
    extra_k: Sequence[int] = (10, 20, 50),
) -> EvalResult:
    model.eval()
    loader = DataLoader(EvalDataset(split), batch_size=batch_size, shuffle=False)

    probs: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    cores: List[np.ndarray] = []
    gate_u2i: List[np.ndarray] = []
    gate_i2u: List[np.ndarray] = []

    for fids in loader:
        out = model(fids.to(device))                    # q=None: no simulator at serving
        logits.append(out["binary_logits"].cpu().numpy())
        probs.append(out["binary_probs"].cpu().numpy())
        cores.append(out["core_pred"].cpu().numpy())
        if model.routefuse is not None:
            g = model.routefuse.last_gates()
            gate_u2i.append(g["g_u_to_i"].cpu().numpy())
            gate_i2u.append(g["g_i_to_u"].cpu().numpy())

    logits = np.concatenate(logits)
    probs = np.concatenate(probs)
    core_pred = np.concatenate(cores)

    # Rank by the primary intent logit (§4.1).
    scores = logits[:, 0]
    labels = split.binary[:, 0]
    ranking = evaluate_ranking(
        scores, labels, split.group_id, split_groups(split.group_id, split.is_cold, labels), k
    )
    # The same lists, cut by cold-start case and at several cutoffs. One "cold"
    # number averages three different problems, and a single k hides whether a
    # gain is real ordering or a shuffle inside the top-20.
    by_case: Dict[str, Dict[str, float]] = {}
    if split.case is not None:
        cases = case_groups(split.group_id, split.case_names())
        for kk in extra_k or (k,):
            for name, block in evaluate_ranking(
                scores, labels, split.group_id, cases, kk
            ).items():
                by_case.setdefault(name, {}).update(
                    {m: v for m, v in block.items() if m != "n_groups"}
                )
                by_case[name]["n_groups"] = block["n_groups"]

    pointwise = evaluate_pointwise(
        binary_probs=probs,
        core_pred=core_pred,
        binary_targets=split.binary,
        core_target=cache.normalize_core(split.core),
        binary_names=cache.tasks.binary,
        core_name=cache.tasks.core,
        masks={"warm": ~split.is_cold, "cold": split.is_cold},
        is_observed=split.is_observed,
    )

    gates: Dict[str, float] = {}
    if gate_u2i:
        gu, gi = np.concatenate(gate_u2i), np.concatenate(gate_i2u)
        for name, m in (("warm", ~split.is_cold), ("cold", split.is_cold)):
            if m.any():
                gates[f"g_u_to_i/{name}"] = float(gu[m].mean())
                gates[f"g_i_to_u/{name}"] = float(gi[m].mean())
    return EvalResult(ranking=ranking, pointwise=pointwise, gates=gates, by_case=by_case)


GATE_CANDIDATES = ("norm", "support")


def train(
    cache: Cache,
    cfg: Optional[LearnerConfig] = None,
    device: str = "cuda:0",
    seed: int = 0,
    k: int = 20,
    verbose: bool = True,
    return_model: bool = False,
) -> Dict:
    """Train the learner, resolving ``reliability="auto"`` first.

    The fusion gate reads either the ID embedding norm or the warm support
    count, and which one works is not a constant: augmentation trains the cold
    IDs, closes the norm gap, and collapses the norm gate, while support is
    unaffected. Fixing the gate by hand therefore couples the gate to the arm
    it was chosen on, which is the confound an ablation is trying to remove.

    Under ``"auto"`` each candidate is trained on the same cache with the same
    seed and the winner is the one with the higher **validation** cold
    Recall@k. Test never participates in the choice. The losing candidate's
    validation score and the selected gate are returned under ``gate_selection``
    so the decision is auditable.
    """
    cfg = cfg or LearnerConfig()
    if cfg.reliability not in GATE_CANDIDATES + ("auto",):
        raise ValueError(f"unknown reliability signal {cfg.reliability!r}")
    if cfg.reliability != "auto":
        return _train_impl(cache, cfg, device, seed, k, verbose, return_model)

    from dataclasses import replace as _replace

    trials = {}
    for gate in GATE_CANDIDATES:
        if verbose:
            print(f"[auto-gate] trying reliability={gate}", flush=True)
        res = _train_impl(cache, _replace(cfg, reliability=gate), device, seed,
                          k, verbose, return_model)
        val = (res.get("val") or {}).get("ranking", {})
        trials[gate] = (res, val.get("cold", {}).get(f"recall@{k}", float("nan")))

    # nan-safe: a gate whose validation score is undefined never wins.
    best = max(GATE_CANDIDATES,
               key=lambda g: (trials[g][1] if trials[g][1] == trials[g][1] else -1.0))
    out = trials[best][0]
    out["gate_selection"] = {
        "chosen": best,
        "criterion": f"val cold recall@{k}",
        "scores": {g: trials[g][1] for g in GATE_CANDIDATES},
    }
    if verbose:
        print(f"[auto-gate] chose {best} "
              f"({', '.join(f'{g}={trials[g][1]:.4f}' for g in GATE_CANDIDATES)})",
              flush=True)
    return out


def _train_impl(
    cache: Cache,
    cfg: LearnerConfig,
    device: str = "cuda:0",
    seed: int = 0,
    k: int = 20,
    verbose: bool = True,
    return_model: bool = False,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    model = EmberLearner(cache.fields, cache.tasks, cfg).to(dev)
    if model.needs_support:
        su, si = warm_support(cache)
        model.set_support(su.to(dev), si.to(dev))
    shared, task = model.shared_parameters(), model.task_parameters()
    opt = torch.optim.Adam(
        [
            {"params": shared, "lr": cfg.lr, "weight_decay": cfg.l2_shared},
            {"params": task, "lr": cfg.lr, "weight_decay": cfg.l2_task},
        ]
    )
    task_names = [f"binary_{t}" for t in range(cache.tasks.n_binary)]
    grad_proc = (
        SharedGradientProcessor(
            shared, "core", task_names, cfg.pcgrad_alpha, cfg.balance_gamma, cfg.balance_clip
        )
        if cfg.use_grad_processor
        else None
    )

    loader = DataLoader(
        TrainDataset(cache),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(dev.type == "cuda"),
        drop_last=False,
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"EmberLearner: {n_params / 1e6:.2f}M params | {cache.summary()}", flush=True)

    best_score, best_state, best_eval, bad, step = -float("inf"), None, None, 0, 0
    history: List[Dict] = []

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0

        if cfg.tau_schedule == "progress":
            # Geometric anneal tau_init -> tau_min over the run, so the label
            # state actually sharpens whatever the dataset size.
            frac = 0.0 if cfg.n_epochs <= 1 else (epoch - 1) / (cfg.n_epochs - 1)
            model.set_tau(cfg.tau_init * (cfg.tau_min / cfg.tau_init) ** frac)

        for fids, binary, core, q, is_aug in loader:
            fids, binary = fids.to(dev), binary.to(dev)
            core, q, is_aug = core.to(dev), q.to(dev), is_aug.to(dev)
            aug_mask = is_aug > 0

            out = model(fids, q=q, aug_mask=aug_mask, y_binary=binary)
            losses = hybrid_losses(
                binary_logits=out["binary_logits"],
                core_pred=out["core_pred"],
                binary_targets=binary,
                core_target=core,
                q=q,
                is_aug=is_aug,
                lambda_a=cfg.lambda_a,
                use_confidence_weighting=cfg.use_confidence_weighting,
                aug_soft_targets=cfg.aug_soft_targets,
            )
            total = sum(losses.values())

            opt.zero_grad(set_to_none=False)
            total.backward(retain_graph=grad_proc is not None)
            if grad_proc is not None:
                # Replaces only the shared block's gradient; task-specific
                # parameters keep what backward() just gave them.
                grad_proc.process(losses)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            step += 1
            if (cfg.tau_schedule == "steps" and cfg.tau_decay_every
                    and step % cfg.tau_decay_every == 0):
                model.set_tau(model.tau * cfg.tau_decay_rate)
            running += float(total.detach())
            nb += 1

        res = evaluate(model, cache.val, cache, dev, k)
        score = res.score(k)
        history.append(
            {
                "epoch": epoch,
                "loss": running / max(nb, 1),
                "val": res.ranking,
                "diagnostics": model.diagnostics(),
            }
        )
        if verbose:
            print(
                f"epoch {epoch:2d}: loss={running / max(nb, 1):.4f} "
                f"({time.time() - t0:.1f}s) | {res.brief(k)} | {model.diagnostics()}",
                flush=True,
            )

        if not np.isnan(score) and score > best_score:
            best_score, best_eval, bad = score, res, 0
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience and epoch >= cfg.min_epochs:
                if verbose:
                    print(f"early stop @ epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out: Dict = {
        "val": {
            "ranking": best_eval.ranking,
            "pointwise": best_eval.pointwise,
            "gates": best_eval.gates,
            "by_case": best_eval.by_case,
        }
        if best_eval
        else None,
        "diagnostics": model.diagnostics(),
        "history": history,
    }
    if cache.test is not None:
        test_res = evaluate(model, cache.test, cache, dev, k)
        out["test"] = {
            "ranking": test_res.ranking,
            "pointwise": test_res.pointwise,
            "gates": test_res.gates,
            "by_case": test_res.by_case,
        }
        if verbose:
            print(f"TEST: {test_res.brief(k)} | gates={test_res.gates}", flush=True)
            for name in ("cold_item", "cold_user", "cold_both"):
                blk = test_res.by_case.get(name)
                if blk and blk.get("n_groups"):
                    cuts = " ".join(
                        f"R@{kk}={blk[f'recall@{kk}']:.4f}" for kk in (10, 20, 50)
                        if f"recall@{kk}" in blk
                    )
                    print(f"  {name:<10} n={blk['n_groups']:>5} {cuts}", flush=True)
    if return_model:
        # Deliberately not part of the serialized result -- `_save` would choke
        # on a module. Only `_warm_model_teacher` asks for it.
        out["model"] = model
    return out
