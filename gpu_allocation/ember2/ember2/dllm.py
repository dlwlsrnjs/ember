"""Diffusion-LM simulator: the hybrid tuple as constrained slot denoising.

EMBER's simulator emits a *fixed-shape* object -- ``T`` binary labels and one
bucketed continuous outcome -- and then needs a per-task confidence for each
field. An autoregressive LM is a poor structural fit for that job in three
specific ways, and a masked-diffusion LM (LLaDA-style) fixes each one natively:

1. **Generation -> slot filling.** The AR simulator free-generates up to 48
   tokens of JSON and hopes it parses; §5's confidence needs a second
   teacher-forced pass to score the label tokens. Here the entire answer
   scaffold is *given* and only the label positions carry the mask token. One
   forward pass produces logits at every slot simultaneously; there is nothing
   to parse and nothing that can be malformed.

2. **Left-to-right -> mutual conditioning.** The paper's own premise (§3.3) is
   that the tasks form a dependence chain -- yet AR decoding fixes an arbitrary
   serialization order: ``watch_time`` can condition on ``click``, never the
   reverse. Confidence-ordered denoising commits the field the model is most
   sure of first and re-predicts the rest *conditioned on it*, whatever its
   position -- the tuple's internal consistency is enforced in both directions.

3. **Post-hoc logprob -> commit-time confidence.** ``q^(t)`` is read off the
   constrained softmax at the step the slot is committed, i.e. the model's
   belief *given every already-committed field*. The remask threshold makes the
   estimator honest: a slot whose belief degrades after later commits is
   reopened rather than kept with stale confidence.

Cost: ``T+1`` slots commit in ``T+1`` batched forwards, followed by up to two
honesty passes and one projection read-out (three to six total here), versus
free-form AR decoding plus a scoring pass.

The backend is model-agnostic: anything callable as ``model(input_ids).logits``
with a mask-token id works -- LLaDA-8B via transformers, or the test stub in
``tests/``. No SFT is required for the mechanism (zero-/few-shot prompting);
LoRA SFT of the diffusion model is orthogonal and slots in where
``train_lora_simulator`` does for the AR path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .simulate import SimPrompt, SimulatedRow

LLADA_MASK_ID = 126336          # LLaDA's [MASK]; overridable per model


@dataclass
class SlotSpec:
    """One answer slot: a name and the token ids it is allowed to resolve to."""

    name: str
    allowed_token_ids: List[int]
    values: List[float]          # numeric value of each allowed token


class DiffusionSlotSimulator:
    """Constrained slot denoising over a fixed answer scaffold."""

    def __init__(
        self,
        model,
        tokenizer,
        slots: Sequence[SlotSpec],
        scaffold: str,
        core_bucket_values: Sequence[float],
        mask_id: int = LLADA_MASK_ID,
        remask_margin: float = 0.05,
        honesty_iters: int = 2,
        constraint=None,
        device: str = "cuda:0",
        batch_size: int = 32,
        max_len: int = 1024,
        prompt_add_special: bool = True,
        cfg_scale: float = 0.0,
    ) -> None:
        self.model, self.tok = model, tokenizer
        self.slots = list(slots)
        self.scaffold = scaffold
        self.core_bucket_values = list(core_bucket_values)
        self.mask_id = int(mask_id)
        self.cfg_scale = float(cfg_scale)
        self.remask_margin = float(remask_margin)
        self.honesty_iters = int(honesty_iters)
        # constraint(values_tuple) -> bool. When set, the final readout projects
        # each row onto the *jointly most probable consistent* combination --
        # cross-slot contradictions become impossible by construction rather
        # than merely discouraged. Slot spaces are tiny (products of 2s and a
        # digit), so exhaustive enumeration is cheap.
        self.constraint = constraint
        self.device, self.batch_size, self.max_len = device, int(batch_size), int(max_len)
        # False when prompts arrive pre-templated (chat template adds its own
        # special tokens; adding them twice shifts every slot position).
        self.prompt_add_special = bool(prompt_add_special)

    # ── scaffold assembly ──
    def _encode_one(self, prompt_text: str) -> Tuple[List[int], List[int]]:
        """Token ids for ``prompt + scaffold`` and the index of each slot."""
        parts = self.scaffold.split("{}")
        if len(parts) != len(self.slots) + 1:
            raise ValueError("scaffold placeholder count != number of slots")
        ids = list(self.tok(prompt_text, add_special_tokens=self.prompt_add_special)["input_ids"])
        slot_pos: List[int] = []
        for k, part in enumerate(parts):
            ids += list(self.tok(part, add_special_tokens=False)["input_ids"])
            if k < len(self.slots):
                slot_pos.append(len(ids))
                ids.append(self.mask_id)
        if len(ids) > self.max_len:
            # Trim the *front* of the prompt: the scaffold and the direction
            # marker live at the end and must survive.
            excess = len(ids) - self.max_len
            ids = ids[:1] + ids[1 + excess:]
            slot_pos = [p - excess for p in slot_pos]
        return ids, slot_pos

    # ── denoising ──
    def _restricted(self, logits, positions, allowed) -> Tuple[np.ndarray, np.ndarray]:
        """Softmax over the allowed ids only -> (prob of argmax, argmax index)."""
        import torch

        sub = logits[torch.arange(len(positions)), positions][:, allowed]   # (B, A)
        p = torch.softmax(sub.float(), dim=-1)
        conf, idx = p.max(dim=-1)
        return conf.cpu().numpy(), idx.cpu().numpy()

    def _forward(self, canvas, pre_end=None, attention_mask=None):
        """Model logits, with unsupervised classifier-free guidance when set.

        LLaDA's unconditional branch masks the prompt and keeps the answer
        region, giving guided logits ``un + (w+1) * (cond - un)``. Here the
        "prompt" is everything before the answer scaffold, which is exactly the
        user and item profile the simulator is supposed to condition on.
        """
        import torch

        if self.cfg_scale <= 0 or pre_end is None:
            return self.model(canvas, attention_mask=attention_mask).logits
        un = canvas.clone()
        for b, end in enumerate(pre_end):
            un[b, :end] = self.mask_id
        mask = None if attention_mask is None else torch.cat(
            [attention_mask, attention_mask], dim=0
        )
        logits = self.model(
            torch.cat([canvas, un], dim=0), attention_mask=mask
        ).logits
        cond, uncond = torch.chunk(logits, 2, dim=0)
        return uncond + (self.cfg_scale + 1.0) * (cond - uncond)

    def _denoise_batch(self, ids_batch, pos_batch) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence-ordered commit loop for one padded batch.

        Returns ``values (B, S)`` and ``q (B, S)``.
        """
        import torch

        B, S = len(ids_batch), len(self.slots)
        L = max(len(x) for x in ids_batch)
        pad = getattr(self.tok, "pad_token_id", None) or 0
        canvas = torch.full((B, L), pad, dtype=torch.long)
        attention = torch.zeros((B, L), dtype=torch.long)
        for b, ids in enumerate(ids_batch):
            canvas[b, : len(ids)] = torch.tensor(ids)
            attention[b, : len(ids)] = 1
        canvas = canvas.to(self.device)
        attention = attention.to(self.device)
        pos = torch.tensor(pos_batch, device=self.device)                    # (B, S)
        # Everything before the scaffold's first literal is the profile prompt.
        pre = len(self.tok(self.scaffold.split("{}")[0],
                           add_special_tokens=False)["input_ids"])
        pre_end = [p[0] - pre for p in pos_batch]

        committed = np.zeros((B, S), bool)
        val_idx = np.zeros((B, S), np.int64)
        q = np.zeros((B, S), np.float64)

        with torch.no_grad():
            for _ in range(S):
                logits = self._forward(canvas, pre_end, attention)
                # For every still-open slot, best restricted belief now.
                best = np.full(B, -1.0)
                best_s = np.zeros(B, np.int64)
                cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                for s, spec in enumerate(self.slots):
                    conf, idx = self._restricted(logits, pos[:, s], spec.allowed_token_ids)
                    cache[s] = (conf, idx)
                    open_and_better = (~committed[:, s]) & (conf > best)
                    best[open_and_better] = conf[open_and_better]
                    best_s[open_and_better] = s
                # Commit each row's single most-confident open slot -- the rest
                # stay masked and get re-predicted conditioned on this commit.
                for b in range(B):
                    s = int(best_s[b])
                    if committed[b, s]:
                        continue
                    conf, idx = cache[s]
                    committed[b, s] = True
                    val_idx[b, s] = idx[b]
                    q[b, s] = conf[b]
                    canvas[b, pos[b, s]] = self.slots[s].allowed_token_ids[idx[b]]

            # Honesty passes: re-score committed slots under the full tuple; a
            # belief that collapsed after later commits is re-resolved. Iterated
            # because one re-resolution can invalidate another slot's belief.
            for _ in range(self.honesty_iters):
                logits = self._forward(canvas, pre_end, attention)
                any_redo = False
                for s, spec in enumerate(self.slots):
                    conf, idx = self._restricted(logits, pos[:, s], spec.allowed_token_ids)
                    worse = conf < q[:, s] - self.remask_margin
                    changed = idx != val_idx[:, s]
                    redo = worse & changed
                    if redo.any():
                        any_redo = True
                        val_idx[redo, s] = idx[redo]
                        for b in np.flatnonzero(redo):
                            canvas[b, pos[b, s]] = spec.allowed_token_ids[idx[b]]
                    q[:, s] = np.where(worse, conf, q[:, s])
                if not any_redo:
                    break

            # Consistency projection: among combinations satisfying the task's
            # own label constraints, take the joint MAP under the final
            # conditionals. The soft machinery above reduces contradictions;
            # this eliminates them.
            if self.constraint is not None:
                import itertools

                probs = []                                  # per slot: (B, A)
                logits = self._forward(canvas, pre_end, attention)
                for s, spec in enumerate(self.slots):
                    sub = logits[torch.arange(B), pos[:, s]][:, spec.allowed_token_ids]
                    probs.append(torch.softmax(sub.float(), dim=-1).cpu().numpy())
                combos = [c for c in itertools.product(
                    *[range(len(sp.allowed_token_ids)) for sp in self.slots])
                    if self.constraint(tuple(self.slots[s].values[c_s]
                                            for s, c_s in enumerate(c)))]
                for b in range(B):
                    cur = tuple(int(v) for v in val_idx[b])
                    if self.constraint(tuple(self.slots[s].values[cur[s]]
                                             for s in range(S))):
                        continue
                    best_c, best_p = None, -1.0
                    for c in combos:
                        pr = 1.0
                        for s, c_s in enumerate(c):
                            pr *= probs[s][b, c_s]
                        if pr > best_p:
                            best_c, best_p = c, pr
                    for s, c_s in enumerate(best_c):
                        val_idx[b, s] = c_s
                        q[b, s] = float(probs[s][b, c_s])

        values = np.stack(
            [np.asarray(self.slots[s].values)[val_idx[:, s]] for s in range(S)], axis=1
        )
        return values, q

    # ── backend interface (same as MockSimulator / HFSimulator) ──
    def run(self, prompts: Sequence[SimPrompt]) -> List[SimulatedRow]:
        out: List[SimulatedRow] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            encoded = [self._encode_one(p.text) for p in chunk]
            values, q = self._denoise_batch([e[0] for e in encoded], [e[1] for e in encoded])
            T = len(self.slots) - 1
            for k, p in enumerate(chunk):
                bucket = int(values[k, T])
                out.append(
                    SimulatedRow(
                        user=p.user,
                        item=p.item,
                        direction=p.direction,
                        binary=values[k, :T].astype(np.float32),
                        core=float(self.core_bucket_values[bucket]),
                        q=q[k].astype(np.float32),
                    )
                )
        return out


def binary_slot(name: str, tokenizer) -> SlotSpec:
    ids = [_single_token_id(tokenizer, "0"), _single_token_id(tokenizer, "1")]
    return SlotSpec(name=name, allowed_token_ids=ids, values=[0.0, 1.0])


def bucket_slot(name: str, tokenizer, n: int = 10) -> SlotSpec:
    ids = [_single_token_id(tokenizer, str(d)) for d in range(n)]
    return SlotSpec(name=name, allowed_token_ids=ids, values=[float(d) for d in range(n)])


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token for this tokenizer")
    return int(ids[0])


def core_buckets(raw, n: int = 10) -> List[float]:
    """Quantile midpoints of the warm-period core target -- the digit->seconds map."""
    warm = raw.core[raw.warm_training_mask()]
    edges = np.quantile(warm, np.linspace(0, 1, n + 1))
    return [float((edges[i] + edges[i + 1]) / 2) for i in range(n)]


def load_llada(
    repo: str = "GSAI-ML/LLaDA-8B-Instruct",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
):
    """Load the diffusion LM once; returns (model, tokenizer, mask_id)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        repo, trust_remote_code=True, torch_dtype=getattr(torch, dtype)
    ).to(device).eval()
    return model, tok, LLADA_MASK_ID
