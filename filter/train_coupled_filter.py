"""Train the EMBER coupled filter: BPR + KD + InfoNCE (paper §3.1).

Architecture:
  F_U : MLP(1536 → 256 → 128)    [user mapping]
  F_V : MLP(1536 → 256 → 128)    [video mapping]
  score(u, v) = F_U(e_u) · F_V(e_v)              [dot product]

Two losses:
  L_BPR = -Σ log σ(score(u, v+) - score(u, v-))     [recommendation perf]
  L_coupled = BCE(σ(score(u, v)), Ẑ_ui)             [knowledge distillation
                                                       from LLM teacher]
  total = α · L_BPR + β · L_coupled                  [α=1.0, β=1.0 default]

Inputs:
  text_emb_user_combined_pure.npz, text_emb_video_pure.npz       [embeddings]
  log_random_4_22_to_5_08_pure.csv                                [for BPR triplets]
  filter_kd_labels.jsonl                                          [KD soft labels from gpt-4o-mini]

Output:
  llm_cold_item/data/coupled_filter.pt    [F_U, F_V weights + config]
"""
from __future__ import annotations
import argparse, json, os, time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x): return self.net(x)


class CoupledFilter(nn.Module):
    def __init__(self, emb_dim: int = 1536, hidden: int = 256, out: int = 128,
                 dropout: float = 0.1,
                 user_emb_dim: int = None, video_emb_dim: int = None) -> None:
        """Allow asymmetric user/video emb dims (paper §4.3.2 uses CF emb on user
        side, LLM_emb on video side — different dims). Falls back to symmetric
        emb_dim if either is None."""
        super().__init__()
        u_dim = user_emb_dim if user_emb_dim is not None else emb_dim
        v_dim = video_emb_dim if video_emb_dim is not None else emb_dim
        self.F_U = MappingMLP(u_dim, hidden, out, dropout)
        self.F_V = MappingMLP(v_dim, hidden, out, dropout)
        self.emb_dim = emb_dim        # kept for backward compat
        self.user_emb_dim = u_dim
        self.video_emb_dim = v_dim
        self.out_dim = out

    def score(self, u_emb: torch.Tensor, v_emb: torch.Tensor) -> torch.Tensor:
        """Returns dot-product score (B,) for paired (u, v)."""
        u = self.F_U(u_emb)
        v = self.F_V(v_emb)
        return (u * v).sum(-1)

    def score_all(self, u_emb: torch.Tensor, v_pool: torch.Tensor) -> torch.Tensor:
        """Score one user against many videos: returns (B_v,)."""
        u = self.F_U(u_emb)  # (D,)
        v = self.F_V(v_pool) # (N, D)
        return v @ u


def load_emb(path):
    d = np.load(path)
    arr = np.zeros((int(d["ids"].max()) + 1, d["text_emb"].shape[1]), dtype=np.float32)
    arr[d["ids"]] = d["text_emb"]
    return torch.from_numpy(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_csv",
                    default="./data/kuairand_pure/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv")
    p.add_argument("--user_emb_npz",  default="./llm_cold_item/data/text_emb_user_combined_pure.npz")
    p.add_argument("--video_emb_npz", default="./llm_cold_item/data/text_emb_video_pure.npz")
    p.add_argument("--user_backbone", choices=["text", "mf", "lightgcn", "ngcf"], default="text",
                    help="Source of user emb for F_U. Paper §4.3.2 uses CF backbone (mf/lightgcn/ngcf).")
    p.add_argument("--cf_user_emb_npy",
                    default="./llm_cold_item/data/cf_backbones/{backbone}_user_emb.npy",
                    help="Template path for CF user embedding (.npy). {backbone} replaced.")
    p.add_argument("--kd_jsonl",      default="./llm_cold_item/data/filter_kd_labels.jsonl")
    p.add_argument("--out_path",      default="./llm_cold_item/data/coupled_filter.pt",
                    help="Output path. If --user_backbone != text, will append _{backbone} unless overridden.")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_bpr_triplets", type=int, default=100000)
    p.add_argument("--alpha_bpr", type=float, default=1.0)
    p.add_argument("--beta_kd",   type=float, default=1.0)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard_negative", action="store_true",
                    help="Sample negatives proportional to item popularity (harder triplets).")
    p.add_argument("--hard_neg_alpha", type=float, default=0.75,
                    help="Popularity power. 0=uniform, 1=fully proportional. Default 0.75.")
    p.add_argument("--inbatch_neg", action="store_true",
                    help="Add InfoNCE-style in-batch negative loss (other users' positives as negatives).")
    p.add_argument("--gamma_nce", type=float, default=1.0,
                    help="Weight on in-batch contrastive loss.")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Args: {vars(args)}", flush=True)

    print(f"\n[A] Load embeddings (user_backbone={args.user_backbone})", flush=True)
    v_emb = load_emb(args.video_emb_npz).to(device)
    if args.user_backbone == "text":
        u_emb = load_emb(args.user_emb_npz).to(device)
    else:
        cf_path = args.cf_user_emb_npy.format(backbone=args.user_backbone)
        u_emb_np = np.load(cf_path)
        u_emb = torch.from_numpy(u_emb_np).float().to(device)
        # Adjust output path to include backbone suffix
        if args.out_path == "./llm_cold_item/data/coupled_filter.pt":
            args.out_path = f"./llm_cold_item/data/coupled_filter_{args.user_backbone}.pt"
    print(f"  user: {u_emb.shape}, video: {v_emb.shape}, out_path={args.out_path}",
          flush=True)

    print(f"\n[B] Build BPR triplets from real log", flush=True)
    log = pd.read_csv(args.log_csv,
                       usecols=["user_id", "video_id", "time_ms", "is_click"],
                       dtype={"user_id": np.int64, "video_id": np.int64,
                              "time_ms": np.int64, "is_click": np.int64}) \
        .sort_values("time_ms").reset_index(drop=True)
    train_log = log.iloc[:int(len(log) * args.train_ratio)]
    pos = train_log[train_log["is_click"] == 1][["user_id", "video_id"]].values
    print(f"  train positives: {len(pos):,}", flush=True)

    rng = np.random.default_rng(args.seed)
    # Build user → clicked-video set for fast negative sampling
    user_clicked = {}
    for u, v in pos:
        user_clicked.setdefault(int(u), set()).add(int(v))
    all_videos = np.array(sorted(set(train_log["video_id"].unique().tolist()) &
                                    set(range(len(v_emb)))), dtype=np.int64)

    # Build popularity distribution for hard-negative sampling
    if args.hard_negative:
        from collections import Counter
        vid_counts = Counter(int(v) for _, v in pos)
        pop_weights = np.array([vid_counts.get(int(v), 0) for v in all_videos], dtype=np.float64)
        # Smooth with power (lower = more uniform, higher = harder)
        pop_weights = np.power(pop_weights + 1, args.hard_neg_alpha)
        pop_weights /= pop_weights.sum()
        print(f"  hard_negative ON (popularity^{args.hard_neg_alpha}, top-5 prob: "
              f"{pop_weights[np.argsort(-pop_weights)[:5]].sum()*100:.1f}%)", flush=True)

    # Sample triplets
    idx = rng.choice(len(pos), min(args.n_bpr_triplets, len(pos) * 5), replace=True)
    triplets = []
    for i in idx:
        u, vpos = int(pos[i, 0]), int(pos[i, 1])
        if u >= len(u_emb): continue
        # negative sampling (random or popularity-weighted)
        for _ in range(8):
            if args.hard_negative:
                vneg = int(rng.choice(all_videos, p=pop_weights))
            else:
                vneg = int(rng.choice(all_videos))
            if vneg not in user_clicked.get(u, set()) and vneg < len(v_emb):
                triplets.append((u, vpos, vneg)); break
        if len(triplets) >= args.n_bpr_triplets: break
    triplets = np.array(triplets, dtype=np.int64)
    print(f"  built {len(triplets):,} BPR triplets "
          f"({'hard-negative' if args.hard_negative else 'random-negative'})", flush=True)

    print(f"\n[C] Load KD labels (gpt-4o-mini soft labels)", flush=True)
    kd_rows = []
    with open(args.kd_jsonl) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("llm_yes", -1) in (0, 1):
                    kd_rows.append((int(r["user_raw_id"]), int(r["video_raw_id"]),
                                     int(r["llm_yes"])))
            except Exception: pass
    kd_arr = np.array(kd_rows, dtype=np.int64)
    print(f"  loaded {len(kd_arr):,} KD pairs "
          f"(yes_rate={kd_arr[:,2].mean()*100:.1f}%)", flush=True)

    # ── Train/Val split on triplets + KD pairs ──
    n_bpr = len(triplets)
    n_kd = len(kd_arr)
    perm_bpr = rng.permutation(n_bpr)
    perm_kd = rng.permutation(n_kd)
    tr_bpr = triplets[perm_bpr[:int(n_bpr * 0.9)]]
    val_bpr = triplets[perm_bpr[int(n_bpr * 0.9):]]
    tr_kd  = kd_arr[perm_kd[:int(n_kd * 0.9)]]
    val_kd = kd_arr[perm_kd[int(n_kd * 0.9):]]
    print(f"  BPR train/val: {len(tr_bpr):,}/{len(val_bpr):,}, "
          f"KD train/val: {len(tr_kd):,}/{len(val_kd):,}", flush=True)

    print(f"\n[D] Init CoupledFilter (user {u_emb.shape[1]} → {args.hidden_dim} → {args.out_dim}, "
          f"video {v_emb.shape[1]} → {args.hidden_dim} → {args.out_dim})", flush=True)
    model = CoupledFilter(emb_dim=v_emb.shape[1], hidden=args.hidden_dim,
                            out=args.out_dim, dropout=args.dropout,
                            user_emb_dim=u_emb.shape[1],
                            video_emb_dim=v_emb.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def to_tensor(arr): return torch.from_numpy(arr).long().to(device)
    tr_bpr_t = to_tensor(tr_bpr); val_bpr_t = to_tensor(val_bpr)
    tr_kd_t  = to_tensor(tr_kd);  val_kd_t  = to_tensor(val_kd)

    print(f"\n[E] Train {args.n_epochs} epochs (α={args.alpha_bpr}, β={args.beta_kd})",
          flush=True)
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        # shuffle batches
        bpr_perm = torch.randperm(len(tr_bpr_t), device=device)
        kd_perm  = torch.randperm(len(tr_kd_t), device=device)
        n_steps = max(len(tr_bpr_t), len(tr_kd_t)) // args.batch_size
        loss_b_sum = loss_k_sum = 0.0
        for step in range(n_steps):
            # BPR batch
            b_idx = bpr_perm[step * args.batch_size:(step + 1) * args.batch_size]
            if len(b_idx) == 0: break
            b_triplet = tr_bpr_t[b_idx]
            u_e = u_emb[b_triplet[:, 0]]; vp_e = v_emb[b_triplet[:, 1]]; vn_e = v_emb[b_triplet[:, 2]]
            s_pos = model.score(u_e, vp_e)
            s_neg = model.score(u_e, vn_e)
            loss_bpr = -F.logsigmoid(s_pos - s_neg).mean()

            # Optional in-batch negative (InfoNCE-style symmetric contrastive)
            if args.inbatch_neg:
                # F_U(u), F_V(v_pos) — diagonal of (B,B) score matrix is positive,
                # off-diagonal acts as in-batch negatives (other users' positives).
                u_f = model.F_U(u_e)             # (B, D_out)
                v_f = model.F_V(vp_e)            # (B, D_out)
                logits = u_f @ v_f.t()           # (B, B)
                labels = torch.arange(len(u_f), device=device)
                # Symmetric loss (paper-faithful contrastive)
                loss_nce = (F.cross_entropy(logits, labels)
                             + F.cross_entropy(logits.t(), labels)) / 2
                loss_bpr = loss_bpr + args.gamma_nce * loss_nce

            # KD batch
            k_idx = kd_perm[(step % (len(tr_kd_t)//args.batch_size + 1)) * args.batch_size:
                              (step % (len(tr_kd_t)//args.batch_size + 1) + 1) * args.batch_size]
            if len(k_idx) > 0:
                k_pair = tr_kd_t[k_idx]
                u_e_k = u_emb[k_pair[:, 0]]; v_e_k = v_emb[k_pair[:, 1]]
                lbl   = k_pair[:, 2].float()
                s_kd = model.score(u_e_k, v_e_k)
                loss_kd = F.binary_cross_entropy_with_logits(s_kd, lbl)
            else:
                loss_kd = torch.tensor(0.0, device=device)

            total = args.alpha_bpr * loss_bpr + args.beta_kd * loss_kd
            opt.zero_grad(); total.backward(); opt.step()
            loss_b_sum += float(loss_bpr.item()); loss_k_sum += float(loss_kd.item())

        # Validate
        model.eval()
        with torch.no_grad():
            # val BPR
            u_v = u_emb[val_bpr_t[:, 0]]; vp_v = v_emb[val_bpr_t[:, 1]]; vn_v = v_emb[val_bpr_t[:, 2]]
            sp = model.score(u_v, vp_v); sn = model.score(u_v, vn_v)
            val_bpr_loss = float(-F.logsigmoid(sp - sn).mean().item())
            val_bpr_acc  = float((sp > sn).float().mean().item())
            # val KD
            u_k = u_emb[val_kd_t[:, 0]]; v_k = v_emb[val_kd_t[:, 1]]
            lbl_k = val_kd_t[:, 2].float()
            sk = model.score(u_k, v_k)
            val_kd_loss = float(F.binary_cross_entropy_with_logits(sk, lbl_k).item())
            val_kd_acc  = float(((torch.sigmoid(sk) >= 0.5).long() == lbl_k.long())
                                  .float().mean().item())
        print(f"  epoch {epoch:2d}: train_bpr={loss_b_sum/n_steps:.4f} "
              f"train_kd={loss_k_sum/n_steps:.4f} | "
              f"val_bpr_loss={val_bpr_loss:.4f} val_bpr_acc={val_bpr_acc:.4f} | "
              f"val_kd_loss={val_kd_loss:.4f} val_kd_acc={val_kd_acc:.4f}",
              flush=True)

    print(f"\n[F] Save → {args.out_path}", flush=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {"emb_dim": v_emb.shape[1], "hidden": args.hidden_dim,
                    "out": args.out_dim, "dropout": args.dropout,
                    "user_emb_dim": u_emb.shape[1], "video_emb_dim": v_emb.shape[1],
                    "user_backbone": args.user_backbone},
    }, args.out_path)
    print(f"DONE.", flush=True)


if __name__ == "__main__":
    main()
