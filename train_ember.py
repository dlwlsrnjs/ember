"""Train the EMBER cold-aware learner (Stage 3).

Trains :class:`ember.learner.EmberHTLNet` on ``D_warm union A_LLM`` with:
  * RouteFuse direction-aware fusion,
  * CaliChain confidence weighting (Eq. 4) on augmented-row losses, and
  * CaliChain confidence prior (Eq. 5) in the task-dependence chain.

Reports Recall@20 / NDCG@20 on overall / warm / cold splits, ranking by the
primary intent logit (ColdLLM-style layout). When the cache does not carry
ranking groups, the script falls back to per-task AUC / LogLoss / NRMSE so it
still runs on a plain HTL cache.

Example:
  python train_ember.py \
    --cache ./data/kuairand_pure_ember_cache.npz \
    --n_user_fields 31 --n_item_fields 8 --n_ctx_fields 4 \
    --lambda_a 1.0 --seed 0 --tag ember --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ember import EmberHTLNet, EmberHTLDataset, load_cache
from ember.calichain import confidence_weighted_bce, confidence_weighted_mse
from ember.optim import HTLGradientProcessor, split_params

try:
    from sklearn.metrics import roc_auc_score, log_loss
except Exception:  # pragma: no cover - sklearn optional for ranking-only eval
    roc_auc_score = log_loss = None


# ─── Ranking evaluation (Recall@20 / NDCG@20) ───────────────────────
def _recall_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 20):
    """Single ranking list: ``scores`` and binary ``labels`` aligned.

    Recall@k = (#relevant in top-k) / (#relevant); NDCG@k with binary gains.
    """
    n_rel = int(labels.sum())
    if n_rel == 0:
        return None, None
    order = np.argsort(-scores)
    topk = order[:k]
    hits = labels[topk]
    recall = float(hits.sum()) / n_rel
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts).sum())
    ideal = float(discounts[: min(n_rel, k)].sum())
    ndcg = dcg / ideal if ideal > 0 else 0.0
    return recall, ndcg


def evaluate_ranking(scores, labels, groups, split_mask=None, k=20):
    """Mean Recall@k / NDCG@k over per-user candidate ``groups``.

    ``groups`` is a list/array of index arrays (one candidate list per user).
    ``split_mask`` (per-group bool) restricts to warm/cold groups.
    """
    recalls, ndcgs = [], []
    for gi, idx in enumerate(groups):
        if split_mask is not None and not split_mask[gi]:
            continue
        r, n = _recall_ndcg_at_k(scores[idx], labels[idx], k)
        if r is not None:
            recalls.append(r)
            ndcgs.append(n)
    if not recalls:
        return {"n": 0}
    return {"n": len(recalls), "recall@20": float(np.mean(recalls)), "ndcg@20": float(np.mean(ndcgs))}


# ─── Per-task fallback metrics ──────────────────────────────────────
def _safe_auc(y, p):
    if roc_auc_score is None or len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def evaluate_multitask(model, loader, device, wt_mean, wt_std, seg_masks=None):
    model.eval()
    cps, lvps, wtps, cys, lvys, wtys = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            fids = batch[0].to(device)
            out = model(fids)
            cps.append(out["click_prob"].cpu().numpy())
            lvps.append(out["long_view_prob"].cpu().numpy())
            wtps.append(out["watch_time_pred"].cpu().numpy() * wt_std + wt_mean)
            cys.append(batch[1].numpy()); lvys.append(batch[2].numpy()); wtys.append(batch[3].numpy())
    c_p, lv_p, wt_p = map(np.concatenate, (cps, lvps, wtps))
    c_y, lv_y, wt_y = map(np.concatenate, (cys, lvys, wtys))

    def _m(idx):
        out = {"n": int(idx.sum())}
        if out["n"] == 0:
            return out
        cp = np.clip(c_p[idx], 1e-7, 1 - 1e-7)
        lvp = np.clip(lv_p[idx], 1e-7, 1 - 1e-7)
        out["click_auc"] = _safe_auc(c_y[idx], cp)
        out["long_view_auc"] = _safe_auc(lv_y[idx], lvp)
        rmse = float(np.sqrt(((wt_p[idx] - wt_y[idx]) ** 2).mean()))
        out["watch_time_nrmse"] = rmse / (float(wt_y[idx].std()) + 1e-8)
        return out

    results = {"all": _m(np.ones(len(c_p), dtype=bool))}
    if seg_masks:
        for k, mask in seg_masks.items():
            results[k] = _m(mask)
    return results


# ─── Training ───────────────────────────────────────────────────────
def train(args, data):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    vocab = data["field_vocab_sizes"].tolist()
    model = EmberHTLNet(
        field_vocab_sizes=vocab,
        n_user_fields=args.n_user_fields,
        n_item_fields=args.n_item_fields,
        n_ctx_fields=args.n_ctx_fields,
        shared_emb_dim=args.shared_emb_dim,
        label_emb_dim=args.label_emb_dim,
        dropout=args.dropout,
        n_heads=args.n_heads,
        use_routefuse=not args.no_routefuse,
        use_confidence_prior=not args.no_confidence_prior,
    ).to(device)
    print(f"EmberHTLNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    shared_params, task_params = split_params(model, ["shared_emb.embs"])
    opt = torch.optim.Adam(
        [
            {"params": shared_params, "lr": args.lr, "weight_decay": args.l2_shared},
            {"params": task_params, "lr": args.lr, "weight_decay": args.l2_task},
        ]
    )
    grad_proc = HTLGradientProcessor(
        shared_params, core_task="watch_time", preceding_tasks=("click", "long_view"),
        alpha=args.alpha, gamma=args.gamma, clip_c=args.clip_c,
    )

    def _opt(key):
        return data.get(key)

    train_ds = EmberHTLDataset(
        data["feature_ids_train"], data["click_train"], data["long_view_train"], data["watch_time_train"],
        q_click=_opt("q_click_train"), q_long_view=_opt("q_long_view_train"),
        q_watch_time=_opt("q_watch_time_train"), is_aug=_opt("is_aug_train"), lambda_a=args.lambda_a,
    )
    val_ds = EmberHTLDataset(
        data["feature_ids_val"], data["click_val"], data["long_view_val"], data["watch_time_val"],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    wt_mean, wt_std = float(data.get("wt_mean", 0.0)), float(data.get("wt_std", 1.0))
    best_val = -float("inf")
    best = None
    bad = 0
    step = 0

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0
        for fids, c, lv, wt, qc, qlv, qwt, w, is_aug in train_loader:
            fids = fids.to(device); c = c.to(device); lv = lv.to(device); wt = wt.to(device)
            qc = qc.to(device); qlv = qlv.to(device); qwt = qwt.to(device); w = w.to(device)
            is_aug = is_aug.to(device)
            wt_norm = (wt - wt_mean) / wt_std

            # Confidence prior (Eq. 5) only consumes confidence on augmented
            # rows. Warm rows are filled with 0.5 (logit 0) so the masked-out
            # branch never produces a non-finite gradient.
            aug_mask = is_aug > 0
            qc_in = torch.where(aug_mask, qc, torch.full_like(qc, 0.5))
            qlv_in = torch.where(aug_mask, qlv, torch.full_like(qlv, 0.5))
            out = model(fids, q_click=qc_in, q_long_view=qlv_in, aug_mask=aug_mask)

            # CaliChain confidence weighting (Eq. 4): warm rows use q=1, weight=1;
            # augmented rows use simulator q and lambda_a source weight.
            qc_w = torch.where(is_aug > 0, qc, torch.ones_like(qc))
            qlv_w = torch.where(is_aug > 0, qlv, torch.ones_like(qlv))
            qwt_w = torch.where(is_aug > 0, qwt, torch.ones_like(qwt))
            L_c = confidence_weighted_bce(out["click_logit"], c, qc_w, w)
            L_lv = confidence_weighted_bce(out["long_view_logit"], lv, qlv_w, w)
            L_wt = confidence_weighted_mse(out["watch_time_pred"], wt_norm, qwt_w, w)

            opt.zero_grad()
            (L_c + L_lv + L_wt).backward(retain_graph=True)
            snap = {id(p): (p.grad.detach().clone() if p.grad is not None else None) for p in task_params}
            grad_proc.process_step({"click": L_c, "long_view": L_lv, "watch_time": L_wt})
            for p in task_params:
                if snap[id(p)] is not None:
                    p.grad = snap[id(p)]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            step += 1
            if step % args.tau_decay_step == 0:
                model.set_temperature(float(model.leu_click.temperature.item()) * 0.9)
            running += float((L_c + L_lv + L_wt).item())
            nb += 1

        val_metrics = evaluate_multitask(model, val_loader, device, wt_mean, wt_std)
        val_score = val_metrics["all"].get("click_auc", float("nan"))
        diag = model.diagnostics()
        print(
            f"epoch {epoch}: loss={running/max(nb,1):.4f} ({time.time()-t0:.1f}s) "
            f"val_click_AUC={val_score:.4f} | diag={diag}",
            flush=True,
        )
        if not np.isnan(val_score) and val_score > best_val:
            best_val = val_score
            best = val_metrics
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {epoch}", flush=True)
                break

    return best, model.diagnostics()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="combined warm+augmented HTL cache NPZ")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--n_user_fields", type=int, required=True)
    p.add_argument("--n_item_fields", type=int, required=True)
    p.add_argument("--n_ctx_fields", type=int, required=True)
    p.add_argument("--lambda_a", type=float, default=1.0, help="augmented-row loss weight (Eq. 4)")
    p.add_argument("--no_routefuse", action="store_true")
    p.add_argument("--no_confidence_prior", action="store_true")
    p.add_argument("--shared_emb_dim", type=int, default=10)
    p.add_argument("--label_emb_dim", type=int, default=10)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2_shared", type=float, default=1e-5)
    p.add_argument("--l2_task", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--clip_c", type=float, default=10.0)
    p.add_argument("--tau_decay_step", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="ember")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Args: {vars(args)}", flush=True)
    data = load_cache(args.cache)
    best, diag = train(args, data)

    print("\n=== BEST VAL METRICS ===", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    out = os.path.join(args.output_dir, f"ember_{args.tag}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump({"args": vars(args), "metrics": best, "diagnostics": diag}, f, indent=2)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
