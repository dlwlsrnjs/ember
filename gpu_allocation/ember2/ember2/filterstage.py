"""Stage 2 -- coupled filter and candidate retrieval (EMBER §3.1).

Querying the simulator for every cold pair is infeasible: it costs
``|U|*|I_c| + |U_c|*|I|`` generations. A coupled filter retrieves plausible
candidates first, bringing that down to ``K(|I_c| + |U_c|)``.

  1. Encode user / item text with the LLM (mean-pooled last hidden states).
  2. **Whiten + L2-normalize.** Mean-pooled LLM states are badly anisotropic --
     they occupy a narrow cone where random pairs already have cosine ~0.7,
     which flattens every ranking loss. Whitening restores an isotropic
     distribution; this implementation does full covariance whitening (the
     released code only centers, which fixes the mean offset but leaves the
     directional collapse in place).
  3. Train mappers ``F_U``, ``F_I`` with ``L_BPR + lambda_kd * L_KD +
     lambda_nce * L_InfoNCE``, where ``L_KD`` distills a teacher LLM's yes/no
     plausibility judgment.
  4. Retrieve top-``K`` partners per cold entity, per direction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FilterConfig
from .rawdata import RawDataset
from .simulate import COLD_ITEM, COLD_USER


# ── text encoders ───────────────────────────────────────────────────
class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class HashingTextEncoder:
    """Deterministic bag-of-tokens embedding, standing in for the LLM encoder.

    Tokens are hashed into a fixed space and randomly projected. It carries real
    content signal (the text spells out the entity's attributes), so the filter
    genuinely has to learn a text -> behaviour mapping rather than being handed
    one. It is a fixture, not a language model.
    """

    def __init__(self, dim: int = 256, n_buckets: int = 4096, seed: int = 0) -> None:
        self.dim, self.n_buckets = dim, n_buckets
        self.proj = np.random.default_rng(seed).normal(
            0, 1 / np.sqrt(dim), size=(n_buckets, dim)
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for n, t in enumerate(texts):
            for tok in t.replace(",", " ").split():
                # Built-in ``hash`` is salted independently for each process;
                # using it here changed retrieval neighborhoods across runs.
                digest = hashlib.blake2b(
                    tok.encode("utf-8"), digest_size=8, person=b"EMBERtxt"
                ).digest()
                bucket = int.from_bytes(digest, "little") % self.n_buckets
                out[n] += self.proj[bucket]
        return out


class HFTextEncoder:
    """Mean-pooled last-layer hidden states from a (LoRA-adapted) causal LM."""

    def __init__(
        self,
        base_model: str,
        adapter_dir: Optional[str] = None,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_len: int = 512,
        load_4bit: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device, self.batch_size, self.max_len = device, batch_size, max_len
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs: Dict = {"torch_dtype": torch.bfloat16}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        chunks = []
        for s in range(0, len(texts), self.batch_size):
            enc = self.tok(
                list(texts[s : s + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            ).to(self.device)
            h = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states[-1]
            m = enc["attention_mask"].unsqueeze(-1).float()
            chunks.append(((h * m).sum(1) / m.sum(1).clamp(min=1.0)).float().cpu().numpy())
        return np.concatenate(chunks, 0)


def whiten(emb: np.ndarray, out_dim: Optional[int] = None, eps: float = 1e-6) -> np.ndarray:
    """Center, decorrelate to unit covariance, then L2-normalize.

    ``out_dim`` keeps the leading principal directions, which also drops the
    near-null directions whose tiny eigenvalues would otherwise be amplified by
    ``1/sqrt(lambda)`` into pure noise.
    """
    x = emb - emb.mean(0, keepdims=True)
    cov = np.cov(x, rowvar=False) + eps * np.eye(x.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    if out_dim is not None:
        vals, vecs = vals[:out_dim], vecs[:, :out_dim]
    z = x @ vecs / np.sqrt(np.maximum(vals, eps))
    return (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


# ── KD teacher ──────────────────────────────────────────────────────
class KDTeacher(Protocol):
    def judge(self, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        """Yes/no plausibility in ``{0, 1}`` per pair."""


class MockKDTeacher:
    """Oracle-based yes/no judgments with noise. A fixture, not a teacher LLM."""

    def __init__(self, oracle: Dict[str, np.ndarray], noise: float = 1.0, seed: int = 0) -> None:
        self.oracle, self.noise = oracle, float(noise)
        self.rng = np.random.default_rng(seed)

    def judge(self, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
        from .rawdata import oracle_score

        u = np.array([p[0] for p in pairs])
        i = np.array([p[1] for p in pairs])
        s = oracle_score(self.oracle, u, i)
        return (s + self.rng.normal(0, self.noise, len(s)) > 0).astype(np.float32)


# ── coupled filter ──────────────────────────────────────────────────
class CoupledFilter(nn.Module):
    """Mappers ``F_U``, ``F_I`` into a shared space; ``s(u,i) = z_u . z_i``."""

    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        def mlp(d_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        self.f_u, self.f_i = mlp(user_dim), mlp(item_dim)
        # Affine calibration for the KD head: the dot product lives on its own
        # scale, and forcing it directly through a sigmoid would couple the
        # ranking geometry to the distillation target.
        self.kd_scale = nn.Parameter(torch.ones(()))
        self.kd_bias = nn.Parameter(torch.zeros(()))

    def z_user(self, e: torch.Tensor) -> torch.Tensor:
        return self.f_u(e)

    def z_item(self, e: torch.Tensor) -> torch.Tensor:
        return self.f_i(e)

    def score(self, eu: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        return (self.z_user(eu) * self.z_item(ei)).sum(-1)

    def kd_logit(self, eu: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        return self.kd_scale * self.score(eu, ei) + self.kd_bias


@dataclass
class FilterTrainingData:
    bpr: np.ndarray          # (N, 3) user, pos item, neg item
    kd_pairs: np.ndarray     # (M, 2) user, item
    kd_labels: np.ndarray    # (M,)


def build_filter_training_data(
    raw: RawDataset,
    teacher: KDTeacher,
    n_bpr: int,
    n_kd: int,
    rng: np.random.Generator,
) -> FilterTrainingData:
    """BPR triplets and KD judgments, both drawn only from ``D_warm`` (§4.1)."""
    warm_rows = np.flatnonzero(raw.warm_training_mask())
    pos_rows = warm_rows[raw.binary[warm_rows, 0] > 0]
    if len(pos_rows) == 0:
        raise ValueError("no warm positives to build BPR triplets from")

    take = rng.choice(len(pos_rows), min(n_bpr, len(pos_rows)), replace=len(pos_rows) < n_bpr)
    rows = pos_rows[take]
    warm_items = np.setdiff1d(np.arange(raw.n_items), raw.cold_items)
    neg = rng.choice(warm_items, len(rows))
    bpr = np.stack([raw.user[rows], raw.item[rows], neg], 1)

    n_kd_pos = n_kd // 2
    kd_pos = pos_rows[rng.choice(len(pos_rows), min(n_kd_pos, len(pos_rows)), replace=True)]
    warm_users = np.setdiff1d(np.arange(raw.n_users), raw.cold_users)
    kd_rand_u = rng.choice(warm_users, n_kd - len(kd_pos))
    kd_rand_i = rng.choice(warm_items, n_kd - len(kd_pos))
    kd_pairs = np.concatenate(
        [np.stack([raw.user[kd_pos], raw.item[kd_pos]], 1), np.stack([kd_rand_u, kd_rand_i], 1)]
    )
    return FilterTrainingData(
        bpr=bpr.astype(np.int64),
        kd_pairs=kd_pairs.astype(np.int64),
        kd_labels=teacher.judge([tuple(p) for p in kd_pairs]),
    )


def train_coupled_filter(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    data: FilterTrainingData,
    cfg: FilterConfig,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> CoupledFilter:
    """``L = L_BPR + lambda_kd * L_KD + lambda_nce * L_InfoNCE``."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    U = torch.as_tensor(user_emb, dtype=torch.float32, device=dev)
    I = torch.as_tensor(item_emb, dtype=torch.float32, device=dev)

    model = CoupledFilter(U.shape[1], I.shape[1], cfg.hidden_dim, cfg.out_dim, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    bpr = torch.as_tensor(data.bpr, device=dev)
    kd_pairs = torch.as_tensor(data.kd_pairs, device=dev)
    kd_labels = torch.as_tensor(data.kd_labels, dtype=torch.float32, device=dev)

    n = len(bpr)
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        totals = {"bpr": 0.0, "kd": 0.0, "nce": 0.0}
        n_batches = 0
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            u, ip, ineg = bpr[idx, 0], bpr[idx, 1], bpr[idx, 2]

            zu, zp, zn = model.z_user(U[u]), model.z_item(I[ip]), model.z_item(I[ineg])
            l_bpr = -F.logsigmoid((zu * zp).sum(-1) - (zu * zn).sum(-1)).mean()

            # InfoNCE over in-batch negatives.
            logits = (zu @ zp.t()) / cfg.nce_temperature
            l_nce = F.cross_entropy(logits, torch.arange(len(idx), device=dev))

            k = torch.randint(0, len(kd_pairs), (min(cfg.batch_size, len(kd_pairs)),), device=dev)
            l_kd = F.binary_cross_entropy_with_logits(
                model.kd_logit(U[kd_pairs[k, 0]], I[kd_pairs[k, 1]]), kd_labels[k]
            )

            loss = l_bpr + cfg.lambda_kd * l_kd + cfg.lambda_nce * l_nce
            opt.zero_grad()
            loss.backward()
            opt.step()

            totals["bpr"] += float(l_bpr)
            totals["kd"] += float(l_kd)
            totals["nce"] += float(l_nce)
            n_batches += 1

        if verbose:
            avg = {k: v / max(n_batches, 1) for k, v in totals.items()}
            print(
                f"  filter epoch {epoch}: "
                f"bpr={avg['bpr']:.4f} kd={avg['kd']:.4f} nce={avg['nce']:.4f}",
                flush=True,
            )
    return model.eval()


@torch.no_grad()
def retrieve_candidates(
    model: CoupledFilter,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    raw: RawDataset,
    top_k: int,
    device: str = "cpu",
) -> Dict[str, List[Tuple[int, int]]]:
    """Top-``K`` partners per cold entity, one candidate list per direction.

    Cold items are matched against warm users and vice versa: an augmented row
    must have exactly one cold side, otherwise neither side carries the warm
    evidence the simulator is supposed to reason from.
    """
    dev = torch.device(device)
    U = torch.as_tensor(user_emb, dtype=torch.float32, device=dev)
    I = torch.as_tensor(item_emb, dtype=torch.float32, device=dev)
    zu_all, zi_all = model.z_user(U), model.z_item(I)

    warm_users = torch.as_tensor(
        np.setdiff1d(np.arange(raw.n_users), raw.cold_users), device=dev
    )
    warm_items = torch.as_tensor(
        np.setdiff1d(np.arange(raw.n_items), raw.cold_items), device=dev
    )
    out: Dict[str, List[Tuple[int, int]]] = {COLD_ITEM: [], COLD_USER: []}

    if len(raw.cold_items):
        cold_i = torch.as_tensor(raw.cold_items, device=dev)
        scores = zi_all[cold_i] @ zu_all[warm_users].t()            # (n_cold_i, n_warm_u)
        top = scores.topk(min(top_k, scores.shape[1]), dim=1).indices
        for r, i in enumerate(raw.cold_items):
            out[COLD_ITEM] += [(int(warm_users[c]), int(i)) for c in top[r]]

    if len(raw.cold_users):
        cold_u = torch.as_tensor(raw.cold_users, device=dev)
        scores = zu_all[cold_u] @ zi_all[warm_items].t()            # (n_cold_u, n_warm_i)
        top = scores.topk(min(top_k, scores.shape[1]), dim=1).indices
        for r, u in enumerate(raw.cold_users):
            out[COLD_USER] += [(int(u), int(warm_items[c])) for c in top[r]]

    return out
