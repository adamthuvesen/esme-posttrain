"""Decoding pre-check: quantify rambling/repetition before spending on DPO.

Degenerate repetition in small models is partly a *decoding* artifact, not an
objective failure (Holtzman et al., nucleus sampling, arXiv:1904.09751). Before
attributing Esme's rambling to the SFT objective, this measures n-gram repetition
rate and response-length distribution on fixed multi-turn prompts under two
decoders:

- greedy (the default ``DenseBackbone.generate`` argmax path), and
- nucleus sampling (top-p) + a repetition penalty.

``DenseBackbone.generate`` only does argmax/temperature, so this module carries a
small standalone top-p + repetition-penalty sampler rather than expanding the
model's generate API. The result is recorded as the pre-DPO decoding baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from esme_posttrain.modeling import DenseBackbone, soft_cap_logits

F = torch.nn.functional


@dataclass(frozen=True)
class DecodingConfig:
    name: str
    strategy: str  # "greedy" or "nucleus"
    max_new_tokens: int = 64
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 214

    def __post_init__(self) -> None:
        if self.strategy not in {"greedy", "nucleus"}:
            raise ValueError("strategy must be greedy or nucleus")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


GREEDY = DecodingConfig(name="greedy", strategy="greedy")
NUCLEUS_REP_PENALTY = DecodingConfig(
    name="nucleus_p0.95_rep1.3",
    strategy="nucleus",
    top_p=0.95,
    temperature=1.0,
    repetition_penalty=1.3,
)


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: Sequence[int], penalty: float
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    unique = torch.tensor(sorted(set(generated_ids)), dtype=torch.long, device=logits.device)
    selected = logits.index_select(0, unique)
    # CTRL-style penalty: divide positive logits, multiply negative ones.
    penalized = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits.index_copy(0, unique, penalized)


def generate_with_decoding(
    model: DenseBackbone,
    prompt_ids: Sequence[int],
    config: DecodingConfig,
    *,
    eos_token_id: int | None = None,
) -> list[int]:
    """Generate continuation token ids (excluding the prompt) for one decoder."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    generated = list(prompt_ids)
    new_ids: list[int] = []
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            window = generated[-model.config.context_length :]
            input_tensor = torch.tensor([window], dtype=torch.long, device=device)
            next_logits = soft_cap_logits(
                model(input_tensor)[0, -1, :], model.config.logit_soft_cap
            )
            next_logits = _apply_repetition_penalty(next_logits, new_ids, config.repetition_penalty)
            if config.strategy == "greedy":
                next_id = int(next_logits.argmax().item())
            else:
                next_id = _nucleus_sample(next_logits, config, generator)
            generated.append(next_id)
            new_ids.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                new_ids = new_ids[:-1]
                break
    if was_training:
        model.train()
    return new_ids


def _nucleus_sample(
    logits: torch.Tensor, config: DecodingConfig, generator: torch.Generator
) -> int:
    probs = F.softmax(logits.float() / config.temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    keep = _nucleus_keep_mask(sorted_probs, config.top_p)
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum()
    choice = int(torch.multinomial(filtered.cpu(), num_samples=1, generator=generator).item())
    return int(sorted_idx[choice].item())


def _nucleus_keep_mask(sorted_probs: torch.Tensor, top_p: float) -> torch.Tensor:
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumulative - sorted_probs) < top_p
    keep[0] = True
    return keep


def ngram_repetition_rate(token_ids: Sequence[int], n: int = 3) -> float:
    """Fraction of n-grams that are repeats: 1 - (distinct n-grams / total n-grams).

    0.0 means no repetition; values near 1.0 mean heavy looping. Sequences shorter
    than ``n`` have no n-grams and return 0.0.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


@dataclass(frozen=True)
class PromptDecodingResult:
    prompt_name: str
    decoding: str
    response_length: int
    repetition_rate_3gram: float
    repetition_rate_2gram: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_name": self.prompt_name,
            "decoding": self.decoding,
            "response_length": self.response_length,
            "repetition_rate_3gram": self.repetition_rate_3gram,
            "repetition_rate_2gram": self.repetition_rate_2gram,
            "text": self.text,
        }


@dataclass(frozen=True)
class DecodingPrecheckReport:
    is_real_checkpoint: bool
    note: str
    per_decoding: dict[str, dict[str, Any]]
    per_prompt: tuple[PromptDecodingResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_real_checkpoint": self.is_real_checkpoint,
            "note": self.note,
            "per_decoding_summary": self.per_decoding,
            "per_prompt": [result.to_dict() for result in self.per_prompt],
        }


def run_decoding_precheck(
    model: DenseBackbone,
    tokenizer: Any,
    prompts: Sequence[tuple[str, str]],
    *,
    decodings: Sequence[DecodingConfig] = (GREEDY, NUCLEUS_REP_PENALTY),
    is_real_checkpoint: bool,
    note: str,
    eos_token: str = "<eos>",
) -> DecodingPrecheckReport:
    """Run greedy vs nucleus+rep-penalty over fixed prompts and summarize.

    ``prompts`` is a sequence of ``(name, rendered_prompt_text)``. Each summary
    reports mean response length + mean 3-gram repetition rate per decoder, the
    direct proxies the research brief flags as the primary signal.
    """
    eos_id = tokenizer.token_to_id(eos_token)
    per_prompt: list[PromptDecodingResult] = []
    for name, prompt_text in prompts:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
        for decoding in decodings:
            new_ids = generate_with_decoding(model, prompt_ids, decoding, eos_token_id=eos_id)
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            per_prompt.append(
                PromptDecodingResult(
                    prompt_name=name,
                    decoding=decoding.name,
                    response_length=len(new_ids),
                    repetition_rate_3gram=ngram_repetition_rate(new_ids, n=3),
                    repetition_rate_2gram=ngram_repetition_rate(new_ids, n=2),
                    text=text,
                )
            )
    per_decoding: dict[str, dict[str, Any]] = {}
    for decoding in decodings:
        rows = [r for r in per_prompt if r.decoding == decoding.name]
        lengths = [r.response_length for r in rows]
        rep3 = [r.repetition_rate_3gram for r in rows]
        per_decoding[decoding.name] = {
            "strategy": decoding.strategy,
            "top_p": decoding.top_p,
            "repetition_penalty": decoding.repetition_penalty,
            "prompts": len(rows),
            "mean_response_length": sum(lengths) / max(1, len(lengths)),
            "min_response_length": min(lengths, default=0),
            "max_response_length": max(lengths, default=0),
            "mean_repetition_rate_3gram": sum(rep3) / max(1, len(rep3)),
            "max_repetition_rate_3gram": max(rep3, default=0.0),
        }
    return DecodingPrecheckReport(
        is_real_checkpoint=is_real_checkpoint,
        note=note,
        per_decoding=per_decoding,
        per_prompt=tuple(per_prompt),
    )
