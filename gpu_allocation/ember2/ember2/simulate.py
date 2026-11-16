"""Stage 1 -- direction-specific LLM behavior simulation (EMBER §3.1).

The simulator maps a structured user-item context to the **full hybrid label
tuple** with **per-task confidence**, rather than the single hard yes/no label
prior cold-start simulators emit.

Two things make it *direction-specific*:

  * A separate LoRA adapter is trained per direction (``cold_item`` /
    ``cold_user``).
  * On the cold side, the behavior block is replaced by an explicit
    ``insufficient_history`` marker rather than left blank -- so the simulator is
    trained under exactly the missingness pattern it meets at inference. A cold
    item does not have *few* item-side behavioral examples; it has none.

Confidence estimators (§3.3), in the paper's order of preference:

  ``logprob``      teacher-forced log-probability of the emitted label tokens.
                   The default.
  ``self_report``  the confidence the model writes into its own JSON. Free, but
                   only as calibrated as the model's introspection.
  ``filter_score`` the coupled filter's retrieval score, squashed. Near-zero
                   cost; the paper's practical fallback when teacher-forced
                   log-probabilities are unavailable. Not implemented here --
                   it needs the Stage-2 score threaded through to Stage 1, and
                   the paper reports it only as a fallback (0.0989 vs 0.1006).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from .config import TaskSpec
from .rawdata import TRAIN, RawDataset

COLD_ITEM = "cold_item"
COLD_USER = "cold_user"
DIRECTIONS = (COLD_ITEM, COLD_USER)

MISSING = "insufficient_history"


# ── data carriers ───────────────────────────────────────────────────
@dataclass
class SimPrompt:
    user: int
    item: int
    direction: str
    text: str


@dataclass
class SimulatedRow:
    """One row of ``A_LLM``."""

    user: int
    item: int
    direction: str
    binary: np.ndarray     # (T,) simulated conversion labels
    core: float            # simulated continuous core target
    q: np.ndarray          # (T+1,) per-task confidence


# ── behavior index ──────────────────────────────────────────────────
class BehaviorIndex:
    """Per-entity behavior summaries drawn strictly from ``D_warm``.

    Leakage control (§4.1): every LLM-facing component is built from the
    cold-removed training period only, so no cold entity's behaviour and no
    evaluation-period row can reach the simulator.
    """

    def __init__(self, raw: RawDataset) -> None:
        self.raw = raw
        mask = raw.warm_training_mask()
        rows = np.flatnonzero(mask)
        self.user_hist: Dict[int, List[int]] = {}
        self.item_hist: Dict[int, List[int]] = {}
        for r in rows:
            self.user_hist.setdefault(int(raw.user[r]), []).append(int(r))
            self.item_hist.setdefault(int(raw.item[r]), []).append(int(r))
        self.rows = rows

    def user_summary(self, u: int) -> str:
        idx = self.user_hist.get(u)
        if not idx:
            return MISSING
        b = self.raw.binary[idx]
        c = self.raw.core[idx]
        names = self.raw.tasks.binary
        parts = [f"{len(idx)} interactions"] + [
            f"{n} rate {b[:, t].mean():.2f}" for t, n in enumerate(names)
        ]
        parts.append(f"mean {self.raw.tasks.core} {c.mean():.1f}")
        return "; ".join(parts)

    def item_summary(self, i: int) -> str:
        idx = self.item_hist.get(i)
        if not idx:
            return MISSING
        b = self.raw.binary[idx]
        names = self.raw.tasks.binary
        parts = [f"{len(idx)} exposures"] + [
            f"{n} rate {b[:, t].mean():.2f}" for t, n in enumerate(names)
        ]
        parts.append(f"mean {self.raw.tasks.core} {self.raw.core[idx].mean():.1f}")
        return "; ".join(parts)

    def top_similar_history(
        self,
        u: int,
        target_item: int,
        item_emb: Optional[np.ndarray],
        top_l: int = 5,
    ) -> List[int]:
        """The user's ``L`` history items most text-similar to the target item."""
        idx = self.user_hist.get(u)
        if not idx:
            return []
        items = np.array([int(self.raw.item[r]) for r in idx])
        if item_emb is None:
            return items[:top_l].tolist()
        sim = item_emb[items] @ item_emb[target_item]
        return items[np.argsort(-sim)[:top_l]].tolist()

    def top_similar_audience(
        self,
        i: int,
        target_user: int,
        user_emb: Optional[np.ndarray],
        top_l: int = 5,
    ) -> List[int]:
        """The item's ``L`` past viewers most text-similar to the target user."""
        idx = self.item_hist.get(i)
        if not idx:
            return []
        users = np.array([int(self.raw.user[r]) for r in idx])
        if user_emb is None:
            return users[:top_l].tolist()
        sim = user_emb[users] @ user_emb[target_user]
        return users[np.argsort(-sim)[:top_l]].tolist()


# ── prompt construction ─────────────────────────────────────────────
class PromptBuilder:
    """Symmetric user / item blocks with direction-conditional masking (§5).

    Every query follows the same template: label schema, user block, item block,
    a direction marker, and an instruction to emit one JSON object. Only the
    *content* of the cold side's behavior block changes between directions --
    the structure is identical, so the model cannot infer the direction from
    layout alone and must read the marker.
    """

    def __init__(
        self,
        raw: RawDataset,
        index: BehaviorIndex,
        user_emb: Optional[np.ndarray] = None,
        item_emb: Optional[np.ndarray] = None,
        top_l: int = 5,
    ) -> None:
        self.raw, self.index = raw, index
        self.user_emb, self.item_emb = user_emb, item_emb
        self.top_l = top_l

    def schema_block(self) -> str:
        t = self.raw.tasks
        binaries = ", ".join(f'"{n}": 0 or 1' for n in t.binary)
        confs = ", ".join(f'"q_{n}": 0.0-1.0' for n in t.all_names)
        return (
            "Label schema (emit exactly one JSON object, no prose):\n"
            f"{{{binaries}, \"{t.core}\": non-negative number, {confs}}}\n"
            f"The q_* fields are your own confidence in each corresponding label."
        )

    def _user_block(self, u: int, masked: bool) -> str:
        profile = self.raw.user_text[u]
        behavior = MISSING if masked else self.index.user_summary(u)
        return f"[USER]\nprofile: {profile}\nbehavior: {behavior}"

    def _item_block(self, i: int, masked: bool) -> str:
        meta = self.raw.item_text[i]
        behavior = MISSING if masked else self.index.item_summary(i)
        return f"[ITEM]\nmetadata: {meta}\nbehavior: {behavior}"

    def _context_block(self, u: int, i: int, direction: str) -> str:
        """Top-L similar evidence, always taken from the *warm* side."""
        if direction == COLD_ITEM:
            hist = self.index.top_similar_history(u, i, self.item_emb, self.top_l)
            if not hist:
                return f"[SIMILAR HISTORY]\n{MISSING}"
            lines = [f"- {self.raw.item_text[h]}" for h in hist]
            return "[SIMILAR HISTORY] items this user engaged with, most similar first:\n" + "\n".join(lines)
        aud = self.index.top_similar_audience(i, u, self.user_emb, self.top_l)
        if not aud:
            return f"[SIMILAR AUDIENCE]\n{MISSING}"
        lines = [f"- {self.raw.user_text[a]}" for a in aud]
        return "[SIMILAR AUDIENCE] users who engaged with this item, most similar first:\n" + "\n".join(lines)

    def build(self, u: int, i: int, direction: str) -> SimPrompt:
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        text = "\n\n".join(
            [
                "Predict how this user would respond to this item.",
                self.schema_block(),
                self._user_block(u, masked=direction == COLD_USER),
                self._item_block(i, masked=direction == COLD_ITEM),
                self._context_block(u, i, direction),
                f"[DIRECTION] {direction}",
                "JSON:",
            ]
        )
        return SimPrompt(user=u, item=i, direction=direction, text=text)


# ── SFT data ────────────────────────────────────────────────────────
def build_sft_examples(
    raw: RawDataset,
    builder: PromptBuilder,
    direction: str,
    n_positive: int,
    n_unobserved: int,
    rng: np.random.Generator,
    with_reasoning: bool = False,
) -> List[Dict[str, str]]:
    """Direction-masked SFT pairs, drawn only from ``D_warm``.

    Observed positives carry their real hybrid labels; randomly sampled
    unobserved pairs carry zero labels, which is what calibrates rejection --
    without them the simulator learns to say yes to everything.

    ``with_reasoning`` prepends a one-line rationale to the JSON (the paper's
    reasoning-SFT ablation). Downstream training still consumes only the parsed
    labels and confidences.
    """
    warm_rows = np.flatnonzero(raw.warm_training_mask())
    pos_rows = warm_rows[raw.binary[warm_rows, 0] > 0]
    if len(pos_rows) > n_positive:
        pos_rows = pos_rows[rng.choice(len(pos_rows), n_positive, replace=False)]

    warm_users = np.setdiff1d(np.arange(raw.n_users), raw.cold_users)
    warm_items = np.setdiff1d(np.arange(raw.n_items), raw.cold_items)
    observed = set(zip(raw.user[warm_rows].tolist(), raw.item[warm_rows].tolist()))

    examples: List[Dict[str, str]] = []

    def _completion(binary: Sequence[float], core: float, confident: bool) -> str:
        t = raw.tasks
        obj: Dict[str, float] = {n: int(binary[k]) for k, n in enumerate(t.binary)}
        obj[t.core] = round(float(core), 2)
        # Supervised confidence targets: observed rows are ground truth, so the
        # target is high; the estimator used downstream is token log-probability,
        # and this only teaches the model to emit the field at all.
        for n in t.all_names:
            obj[f"q_{n}"] = 0.9 if confident else 0.6
        body = json.dumps(obj)
        if not with_reasoning:
            return body
        side = "item" if direction == COLD_ITEM else "user"
        verdict = "engages" if binary[0] > 0 else "does not engage"
        return f"Reasoning: the {side} side has no behavioral history; judging from content overlap the user {verdict}.\n{body}"

    for r in pos_rows:
        u, i = int(raw.user[r]), int(raw.item[r])
        examples.append(
            {
                "prompt": builder.build(u, i, direction).text,
                "completion": _completion(raw.binary[r], float(raw.core[r]), True),
                "direction": direction,
            }
        )

    n_zero = 0
    while n_zero < n_unobserved:
        u = int(rng.choice(warm_users))
        i = int(rng.choice(warm_items))
        if (u, i) in observed:
            continue
        examples.append(
            {
                "prompt": builder.build(u, i, direction).text,
                "completion": _completion(np.zeros(raw.tasks.n_binary), 0.0, False),
                "direction": direction,
            }
        )
        n_zero += 1

    rng.shuffle(examples)
    return examples


# ── backends ────────────────────────────────────────────────────────
class SimulatorBackend(Protocol):
    """Emits a hybrid label tuple with per-task confidence for each prompt."""

    def run(self, prompts: Sequence[SimPrompt]) -> List[SimulatedRow]:
        ...


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_label_json(text: str, tasks: TaskSpec) -> Optional[Dict[str, float]]:
    """Pull the label object out of a generation, tolerating surrounding prose."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(n in obj for n in tasks.binary) or tasks.core not in obj:
        return None
    return obj


class MockSimulator:
    """Test double standing in for the fine-tuned LLM.

    It reads the synthetic dataset's generative score, corrupts it with
    heteroscedastic noise, and reports the confidence *implied by its own
    corrupted belief* -- ``max(p, 1-p)`` for a binary label, and a decreasing
    function of the sampled noise scale for the continuous target.

    That last property is the point: confidence is derived from the simulator's
    own uncertainty, exactly as a token-log-probability estimator would be, so it
    correlates with correctness without anything hand-tuning it to. This is what
    makes an end-to-end run a real test of CaliChain rather than a rehearsal.

    It is a fixture. It is not a language model and proves nothing about one.
    """

    def __init__(
        self,
        raw: RawDataset,
        oracle: Dict[str, np.ndarray],
        noise: float = 0.9,
        core_sigma: float = 3.0,
        seed: int = 0,
    ) -> None:
        self.raw, self.oracle = raw, oracle
        self.noise, self.core_sigma = float(noise), float(core_sigma)
        self.rng = np.random.default_rng(seed)

    def _score(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        from .rawdata import oracle_score

        return oracle_score(self.oracle, u, i)

    def run(self, prompts: Sequence[SimPrompt]) -> List[SimulatedRow]:
        if not prompts:
            return []
        u = np.array([p.user for p in prompts])
        i = np.array([p.item for p in prompts])
        T = self.raw.tasks.n_binary

        belief = self._score(u, i) + self.rng.normal(0, self.noise, len(u))
        p_click = 1.0 / (1.0 + np.exp(-belief))
        p_lv = 1.0 / (1.0 + np.exp(-(0.8 * belief - 0.6)))
        probs = np.stack([p_click, p_lv][:T], 1)

        labels = (probs > 0.5).astype(np.float32)
        q_bin = np.maximum(probs, 1 - probs).astype(np.float32)

        sigma = self.core_sigma * np.exp(self.rng.normal(0, 0.5, len(u)))
        core_mu = np.clip(np.log1p(np.exp(belief)) * 12.0, 0, None) * labels[:, 0]
        core = np.clip(core_mu + self.rng.normal(0, sigma), 0, None).astype(np.float32)
        q_core = (self.core_sigma / (self.core_sigma + sigma)).astype(np.float32)

        q = np.concatenate([q_bin, q_core[:, None]], 1)
        return [
            SimulatedRow(
                user=int(u[k]),
                item=int(i[k]),
                direction=prompts[k].direction,
                binary=labels[k],
                core=float(core[k]),
                q=q[k],
            )
            for k in range(len(prompts))
        ]


class HFSimulator:
    """The real simulator: a LoRA-adapted causal LM served through transformers.

    Greedy decoding (``T = 0``, max 48 new tokens per §5), then confidence from
    the teacher-forced log-probability of the tokens that spell out each label
    value. The generated text is re-scored in one forward pass and each task's
    character span in the JSON is mapped to token positions via the tokenizer's
    offset mapping, so ``q^(t)`` is the mean probability of exactly the tokens
    that produced label ``t`` -- not of the whole completion.

    Requires ``transformers`` (and ``peft`` for an adapter); imported lazily so
    the rest of the package runs without them.
    """

    def __init__(
        self,
        base_model: str,
        tasks: TaskSpec,
        adapter_dir: Optional[str] = None,
        device: str = "cuda:0",
        max_new_tokens: int = 48,
        batch_size: int = 16,
        load_4bit: bool = True,
        confidence: str = "logprob",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tasks, self.device = tasks, device
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.confidence = confidence

        if confidence not in ("logprob", "self_report"):
            raise ValueError(f"unknown confidence estimator {confidence!r}")

        self.tok = AutoTokenizer.from_pretrained(base_model)
        if confidence == "logprob" and not self.tok.is_fast:
            # Span-level confidence needs offset mapping, which slow tokenizers
            # do not provide. Failing here beats silently scoring the wrong span.
            raise ValueError(
                f"{base_model} has no fast tokenizer; "
                "'logprob' confidence needs offset mapping (use confidence='self_report')"
            )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        kwargs: Dict = {"torch_dtype": torch.bfloat16}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.to(device).eval()

    def _token_confidence(self, prompt: str, completion: str) -> Dict[str, float]:
        """Mean token probability over each label value's character span."""
        import torch

        enc = self.tok(prompt + completion, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        ids = enc["input_ids"][0][1:]
        tok_lp = logprobs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # aligned to offsets[1:]

        base = len(prompt)
        out: Dict[str, float] = {}
        for name in self.tasks.all_names:
            # Locate `"name": <value>` inside the completion and score <value>.
            m = re.search(rf'"{re.escape(name)}"\s*:\s*([^,}}]+)', completion)
            if not m:
                out[name] = float("nan")
                continue
            lo, hi = base + m.start(1), base + m.end(1)
            sel = [
                k for k in range(len(offsets) - 1)
                if offsets[k + 1][0] < hi and offsets[k + 1][1] > lo
            ]
            out[name] = (
                float(tok_lp[sel].mean().exp()) if sel else float("nan")
            )
        return out

    def run(self, prompts: Sequence[SimPrompt]) -> List[SimulatedRow]:
        import torch

        rows: List[SimulatedRow] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            enc = self.tok([p.text for p in chunk], return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            completions = self.tok.batch_decode(
                gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for p, comp in zip(chunk, completions):
                obj = parse_label_json(comp, self.tasks)
                if obj is None:
                    continue  # unparseable generation -> drop, do not guess
                binary = np.array([float(obj[n]) for n in self.tasks.binary], np.float32)
                core = float(obj[self.tasks.core])

                if self.confidence == "logprob":
                    cq = self._token_confidence(p.text, comp)
                    q = np.array([cq[n] for n in self.tasks.all_names], np.float32)
                    q = np.nan_to_num(q, nan=0.5)
                else:  # self_report -- validated in __init__
                    q = np.array(
                        [float(obj.get(f"q_{n}", 0.5)) for n in self.tasks.all_names], np.float32
                    )

                rows.append(
                    SimulatedRow(
                        user=p.user,
                        item=p.item,
                        direction=p.direction,
                        binary=binary,
                        core=core,
                        q=np.clip(q, 1e-3, 1.0),
                    )
                )
        return rows


class ConstrainedARSimulator:
    """Autoregressive labeler for the same fixed three-slot record as dLLM.

    The prompt supplies the evidence and instruction, while this decoder supplies
    the JSON literals and generates only one admissible token per slot.  The KV
    cache is reused across slots.  Confidence is the constrained softmax at the
    decision point, matching the estimator used by slot denoising.

    ``core_token_values`` are the printable one-token choices (for example
    ``range(10)`` for watch-time buckets); ``core_output_values`` maps those
    choices back to the continuous values consumed by the recommender.
    """

    def __init__(
        self,
        base_model: str,
        tasks: TaskSpec,
        core_token_values: Sequence[int],
        core_output_values: Sequence[float],
        constraint=None,
        device: str = "cuda:0",
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if len(tasks.binary) != 2:
            raise ValueError("ConstrainedARSimulator currently expects two binary slots")
        if len(core_token_values) != len(core_output_values):
            raise ValueError("core token choices and output values must align")

        self.tasks = tasks
        self.device = device
        self.batch_size = int(batch_size)
        self.core_token_values = [int(v) for v in core_token_values]
        self.core_output_values = [float(v) for v in core_output_values]
        self.constraint = constraint
        self.tok = AutoTokenizer.from_pretrained(base_model)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        ).to(device).eval()

        b0, b1 = tasks.binary
        self.pieces = [
            f' {{"{b0}": ',
            f', "{b1}": ',
            f', "{tasks.core}": ',
            "}",
        ]

        def one_token(value: int) -> int:
            ids = self.tok(str(value), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"{value!r} is not one token for {base_model}: {ids}")
            return int(ids[0])

        choices = [[0, 1], [0, 1], self.core_token_values]
        self.allowed_values = [[float(v) for v in vs] for vs in choices]
        self.allowed_ids = [[one_token(v) for v in vs] for vs in choices]

    def run(self, prompts: Sequence[SimPrompt]) -> List[SimulatedRow]:
        import torch

        rows: List[SimulatedRow] = []
        allowed_t = [torch.as_tensor(ids, device=self.device) for ids in self.allowed_ids]
        literal_t = [
            self.tok(p, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
                self.device
            )
            for p in self.pieces
        ]

        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start : start + self.batch_size]
                texts = [p.text + self.pieces[0] for p in chunk]
                enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
                attention = enc["attention_mask"]
                out = self.model(**enc, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                chosen: List[np.ndarray] = []
                confidence: List[np.ndarray] = []

                for slot, ids_t in enumerate(allowed_t):
                    restricted = logits.index_select(1, ids_t).float()
                    if self.constraint is not None:
                        import itertools

                        valid = torch.zeros_like(restricted, dtype=torch.bool)
                        for b in range(len(chunk)):
                            prefix = [
                                self.allowed_values[s][int(chosen[s][b])]
                                for s in range(slot)
                            ]
                            for candidate in range(len(self.allowed_values[slot])):
                                head = prefix + [self.allowed_values[slot][candidate]]
                                tails = itertools.product(
                                    *self.allowed_values[slot + 1:]
                                )
                                if any(self.constraint(tuple(head + list(t))) for t in tails):
                                    valid[b, candidate] = True
                        if not valid.all():
                            restricted = restricted.masked_fill(~valid, float("-inf"))
                    prob = torch.softmax(restricted, dim=-1)
                    conf, idx = prob.max(dim=-1)
                    nxt = ids_t[idx].unsqueeze(1)
                    chosen.append(idx.cpu().numpy())
                    confidence.append(conf.cpu().numpy())
                    if slot == len(allowed_t) - 1:
                        break

                    step = torch.cat(
                        [nxt, literal_t[slot + 1].expand(len(chunk), -1)], dim=1
                    )
                    attention = torch.cat(
                        [
                            attention,
                            torch.ones(
                                (len(chunk), step.shape[1]),
                                dtype=attention.dtype,
                                device=self.device,
                            ),
                        ],
                        dim=1,
                    )
                    out = self.model(
                        input_ids=step,
                        attention_mask=attention,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]

                for k, prompt in enumerate(chunk):
                    b = np.asarray(
                        [self.allowed_values[s][chosen[s][k]] for s in range(2)],
                        np.float32,
                    )
                    core_idx = int(chosen[2][k])
                    q = np.asarray([confidence[s][k] for s in range(3)], np.float32)
                    rows.append(
                        SimulatedRow(
                            user=prompt.user,
                            item=prompt.item,
                            direction=prompt.direction,
                            binary=b,
                            core=self.core_output_values[core_idx],
                            q=q,
                        )
                    )
        return rows


def train_lora_simulator(
    base_model: str,
    examples: Sequence[Dict[str, str]],
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    epochs: int = 2,
    lr: float = 2e-4,
    max_seq_len: int = 1536,
    batch_size: int = 4,
    load_4bit: bool = True,
) -> str:
    """LoRA SFT of the simulator for ONE direction (§5 configuration).

    Train once per direction and keep the adapters separate -- a shared adapter
    would have to serve both missingness patterns at once, which is the failure
    mode direction-specific simulation exists to avoid.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: Dict = {"torch_dtype": torch.bfloat16}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if load_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    def _encode(ex):
        prompt_ids = tok(ex["prompt"], truncation=True, max_length=max_seq_len).input_ids
        full_ids = tok(
            ex["prompt"] + ex["completion"] + tok.eos_token,
            truncation=True,
            max_length=max_seq_len,
        ).input_ids
        labels = list(full_ids)
        # Loss on the completion only; the prompt is context, not a target.
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    ds = Dataset.from_list([{"prompt": e["prompt"], "completion": e["completion"]} for e in examples])
    ds = ds.map(_encode, remove_columns=ds.column_names)

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            bf16=True,
            logging_steps=50,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    ).train()

    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir


def simulate_pairs(
    pairs: Sequence[tuple],
    builder: PromptBuilder,
    backend: SimulatorBackend,
    direction: str,
) -> List[SimulatedRow]:
    """Run the simulator over ``(user, item)`` candidate pairs for one direction."""
    prompts = [builder.build(int(u), int(i), direction) for u, i in pairs]
    return backend.run(prompts)


def keep_positive_rows(rows: Sequence[SimulatedRow]) -> List[SimulatedRow]:
    """Keep only positive primary-behaviour predictions (§3.1, as written).

    Predicted negatives serve as a filter and stop there: the paper reports that
    writing them all into the cache skews class balance and hurts cold Recall@20
    (0.1006 -> 0.0934). See :func:`balance_negatives` for why "all or nothing" is
    a false choice.
    """
    return [r for r in rows if r.binary[0] > 0]


def balance_negatives(
    rows: Sequence[SimulatedRow],
    target_positive_rate: float,
    rng: np.random.Generator,
) -> List[SimulatedRow]:
    """Keep every positive and sample negatives to a target class balance.

    The paper measures that adding *all* predicted negatives hurts, and concludes
    that negatives should only ever act as a filter. But the stated cause is
    class-balance skew, and that argues for matching the balance, not for
    dropping the negatives entirely -- "all" and "none" are not the only options.

    Positives-only augmentation has a specific failure mode. Every row mentioning
    a cold entity carries label 1, so the cold id embedding receives one-signed
    gradient and acquires a uniform score offset rather than any ordering among
    cold entities. Under a protocol that ranks cold items against other cold
    items, a uniform offset cancels exactly and the augmentation buys nothing.
    Sampled negatives restore the contrast that makes the offset informative.

    ``target_positive_rate`` should be ``D_warm``'s own positive rate, so the
    augmented block carries the same prior as the logged data it is merged into.
    """
    pos = [r for r in rows if r.binary[0] > 0]
    neg = [r for r in rows if r.binary[0] <= 0]
    if not pos or target_positive_rate <= 0 or target_positive_rate >= 1:
        return list(pos)

    n_target = int(round(len(pos) * (1.0 - target_positive_rate) / target_positive_rate))
    if n_target >= len(neg):
        keep = neg
    else:
        # Uniform sampling: taking the most confident negatives would bias the
        # block toward easy ones and defeat the point of adding contrast.
        keep = [neg[i] for i in rng.choice(len(neg), n_target, replace=False)]
    out = list(pos) + list(keep)
    rng.shuffle(out)
    return out


class WarmModelTeacher:
    """A non-LLM soft teacher: the warm-only learner labeling its own cold pairs.

    Reviewer question this exists to answer: how much of the gain is *the LLM*,
    and how much is any soft teacher plus the augmentation machinery? A standard
    self-distillation setup would take the model trained on ``D_warm``, run it
    over the retrieved cold pairs, and write its predictions back as training
    rows with a confidence. That is exactly this class.

    It emits the same hybrid tuple the LLM simulator does -- ``T`` binary labels
    plus the continuous core, with per-task confidence ``max(p, 1-p)`` -- so the
    only thing that differs between the two arms is who produced the tuple.

    What it cannot do is the thing the paper claims for the LLM: it has never
    seen the cold entity's ID, so its prediction is a function of content and
    context alone. If it matches the simulator, the augmentation mechanism is
    doing the work; if it falls short, the gap is what the teacher's outside
    knowledge is worth.
    """

    def __init__(self, model, cache, raw, device: str = "cpu", batch_size: int = 4096) -> None:
        self.model, self.cache, self.raw = model, cache, raw
        self.device, self.batch_size = device, int(batch_size)

    def run(self, prompts: Sequence[SimPrompt]) -> List[SimulatedRow]:
        if not prompts:
            return []
        import torch

        u = np.array([p.user for p in prompts], np.int64)
        i = np.array([p.item for p in prompts], np.int64)
        # A simulated pair carries no context of its own; sample one from the
        # warm rows, the same marginalization `build_cache` performs.
        warm_rows = np.flatnonzero(self.raw.warm_training_mask())
        rng = np.random.default_rng(0)
        ctx = self.raw.ctx[warm_rows[rng.integers(0, len(warm_rows), len(u))]]
        feats = torch.from_numpy(self.raw.compose_rows(u, i, ctx))

        probs, cores = [], []
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(feats), self.batch_size):
                out = self.model(feats[s : s + self.batch_size].to(self.device))
                probs.append(out["binary_probs"].cpu().numpy())
                cores.append(out["core_pred"].cpu().numpy())
        probs = np.concatenate(probs)
        core_pred = np.concatenate(cores)

        labels = (probs > 0.5).astype(np.float32)
        q_bin = np.maximum(probs, 1 - probs).astype(np.float32)
        # The learner predicts the *normalized* core target; A_LLM rows are
        # written in raw units, so undo the normalization the cache applied.
        core = (core_pred * self.cache.core_std + self.cache.core_mean).astype(np.float32)
        core = np.clip(core, 0, None) * labels[:, 0]
        # No uncertainty estimate for a point regression: report the primary
        # task's confidence rather than inventing one.
        q = np.concatenate([q_bin, q_bin[:, :1]], 1)

        return [
            SimulatedRow(
                user=int(u[k]),
                item=int(i[k]),
                direction=prompts[k].direction,
                binary=labels[k],
                core=float(core[k]),
                q=q[k],
            )
            for k in range(len(prompts))
        ]
