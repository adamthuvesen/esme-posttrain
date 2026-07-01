"""Fair side-by-side chat-quality eval: SFT reference vs DPO checkpoint.

The end-of-DPO-run ``chat-samples.md`` used hard UltraFeedback *eval-task* prompts
(translation / NLP), which are unfair to a 214M model, and no judge was wired -- so
there was no fair conversational readout. This module answers the real question:
does the DPO model chat better / ramble less than the SFT model on CONVERSATIONAL
prompts?

It reuses ``run_decoding_precheck`` (greedy + nucleus(p=0.95, rep-penalty=1.3),
3-gram repetition rate, response-length distribution) on BOTH checkpoints over the
shared ``FIXED_MULTI_TURN_PROMPTS`` plus a handful of simple conversational prompts,
writes the actual generations SIDE BY SIDE (SFT vs DPO) per prompt to markdown, and
flags obvious degeneration (loops, truncation-to-cap, empty). No external judge --
the comparison is for eyeballing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.dpo.decoding_precheck import (
    GREEDY,
    NUCLEUS_REP_PENALTY,
    DecodingConfig,
    DecodingPrecheckReport,
    run_decoding_precheck,
)
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.checkpointing import load_training_checkpoint
from esme_posttrain.sft.multiturn_judge import FIXED_MULTI_TURN_PROMPTS

# Simple conversational prompts a 214M chat model can fairly attempt, rendered with
# the repo chat template. Deliberately easy/open -- the point is chat feel and
# rambling, not capability. The two-turn prompt tests context carry-over.
CONVERSATIONAL_PROMPTS: tuple[tuple[str, str], ...] = (
    ("capital_of_france", "user\nWhat is the capital of France?\nassistant\n"),
    (
        "photosynthesis_one_sentence",
        "user\nExplain photosynthesis in one sentence.\nassistant\n",
    ),
    (
        "thank_you_note",
        "user\nWrite a short thank-you note to a coworker who helped me.\nassistant\n",
    ),
    (
        "weekend_followup",
        "user\nI went hiking this weekend.\nassistant\nThat sounds lovely! How was it?\n"
        "user\nIt was great, we saw a waterfall.\nassistant\n",
    ),
    ("tell_me_about_yourself", "user\nTell me about yourself.\nassistant\n"),
)


def chat_eval_prompts() -> tuple[tuple[str, str], ...]:
    """The shared prompt set: the fixed multi-turn prompts plus conversational ones."""
    fixed = tuple((p.name, p.prompt) for p in FIXED_MULTI_TURN_PROMPTS)
    return fixed + CONVERSATIONAL_PROMPTS


@dataclass(frozen=True)
class ModelChatEval:
    label: str
    checkpoint_path: str
    report: DecodingPrecheckReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "checkpoint_path": self.checkpoint_path,
            "report": self.report.to_dict(),
        }


@dataclass(frozen=True)
class ChatEvalComparison:
    sft: ModelChatEval
    dpo: ModelChatEval
    prompts: tuple[tuple[str, str], ...]
    decodings: tuple[str, ...]
    degeneration_flags: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "dpo_chat_eval",
            "decodings": list(self.decodings),
            "prompt_names": [name for name, _ in self.prompts],
            "sft": self.sft.to_dict(),
            "dpo": self.dpo.to_dict(),
            "summary_comparison": self._summary_comparison(),
            "degeneration_flags": list(self.degeneration_flags),
        }

    def _summary_comparison(self) -> dict[str, Any]:
        comparison: dict[str, Any] = {}
        for decoding in self.decodings:
            sft = self.sft.report.per_decoding[decoding]
            dpo = self.dpo.report.per_decoding[decoding]
            comparison[decoding] = {
                "sft_mean_response_length": sft["mean_response_length"],
                "dpo_mean_response_length": dpo["mean_response_length"],
                "sft_mean_repetition_rate_3gram": sft["mean_repetition_rate_3gram"],
                "dpo_mean_repetition_rate_3gram": dpo["mean_repetition_rate_3gram"],
                "sft_max_repetition_rate_3gram": sft["max_repetition_rate_3gram"],
                "dpo_max_repetition_rate_3gram": dpo["max_repetition_rate_3gram"],
            }
        return comparison


def run_chat_eval(
    sft_checkpoint_path: Path,
    dpo_checkpoint_path: Path,
    tokenizer: Tokenizer,
    *,
    device: torch.device,
    max_new_tokens: int = 96,
    prompts: tuple[tuple[str, str], ...] | None = None,
    decodings: tuple[DecodingConfig, ...] = (GREEDY, NUCLEUS_REP_PENALTY),
) -> ChatEvalComparison:
    """Generate with both checkpoints over the shared prompt set and compare.

    Both models use the SAME tokenizer (the DPO policy is a continuation of the SFT
    model, same vocab) and the SAME prompts/decoders, so the comparison is fair.
    """
    prompt_set = prompts if prompts is not None else chat_eval_prompts()
    sized_decodings = tuple(
        DecodingConfig(
            name=d.name,
            strategy=d.strategy,
            max_new_tokens=max_new_tokens,
            top_p=d.top_p,
            temperature=d.temperature,
            repetition_penalty=d.repetition_penalty,
            seed=d.seed,
        )
        for d in decodings
    )

    sft_model = _load_backbone(sft_checkpoint_path, device=device)
    sft_report = run_decoding_precheck(
        sft_model,
        tokenizer,
        prompt_set,
        decodings=sized_decodings,
        is_real_checkpoint=True,
        note="SFT reference (Esme-214M-Instruct) on conversational prompts",
    )
    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dpo_model = _load_backbone(dpo_checkpoint_path, device=device)
    dpo_report = run_decoding_precheck(
        dpo_model,
        tokenizer,
        prompt_set,
        decodings=sized_decodings,
        is_real_checkpoint=True,
        note="DPO chat checkpoint (Esme-214M-Chat) on conversational prompts",
    )
    del dpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    flags = _degeneration_flags(sft_report, dpo_report, max_new_tokens=max_new_tokens)
    return ChatEvalComparison(
        sft=ModelChatEval("SFT", str(sft_checkpoint_path), sft_report),
        dpo=ModelChatEval("DPO", str(dpo_checkpoint_path), dpo_report),
        prompts=prompt_set,
        decodings=tuple(d.name for d in sized_decodings),
        degeneration_flags=tuple(flags),
    )


def _degeneration_flags(
    sft_report: DecodingPrecheckReport,
    dpo_report: DecodingPrecheckReport,
    *,
    max_new_tokens: int,
    high_repetition: float = 0.5,
) -> list[dict[str, Any]]:
    """Flag obvious degeneration: heavy 3-gram looping, truncation to the token cap,
    or an empty generation. Reported, not fatal."""
    flags: list[dict[str, Any]] = []
    for label, report in (("SFT", sft_report), ("DPO", dpo_report)):
        for result in report.per_prompt:
            reasons: list[str] = []
            if result.repetition_rate_3gram >= high_repetition:
                reasons.append(f"high_3gram_repetition={result.repetition_rate_3gram:.2f}")
            if result.response_length >= max_new_tokens:
                reasons.append("truncated_at_token_cap")
            if result.response_length == 0:
                reasons.append("empty_generation")
            if reasons:
                flags.append(
                    {
                        "model": label,
                        "prompt_name": result.prompt_name,
                        "decoding": result.decoding,
                        "response_length": result.response_length,
                        "repetition_rate_3gram": result.repetition_rate_3gram,
                        "reasons": reasons,
                    }
                )
    return flags


def write_chat_eval_markdown(comparison: ChatEvalComparison, path: Path) -> None:
    lines: list[str] = [
        "# DPO vs SFT chat-quality comparison",
        "",
        f"- SFT checkpoint: `{comparison.sft.checkpoint_path}`",
        f"- DPO checkpoint: `{comparison.dpo.checkpoint_path}`",
        "- Decoders: " + ", ".join(comparison.decodings),
        "",
        "## Summary (lower repetition is better; length should stay sane)",
        "",
        "| Decoder | Metric | SFT | DPO |",
        "| --- | --- | --- | --- |",
    ]
    summary = comparison._summary_comparison()
    for decoding, row in summary.items():
        lines.append(
            f"| {decoding} | mean response length | "
            f"{row['sft_mean_response_length']:.1f} | {row['dpo_mean_response_length']:.1f} |"
        )
        lines.append(
            f"| {decoding} | mean 3-gram repetition | "
            f"{row['sft_mean_repetition_rate_3gram']:.3f} | "
            f"{row['dpo_mean_repetition_rate_3gram']:.3f} |"
        )
        lines.append(
            f"| {decoding} | max 3-gram repetition | "
            f"{row['sft_max_repetition_rate_3gram']:.3f} | "
            f"{row['dpo_max_repetition_rate_3gram']:.3f} |"
        )
    lines.extend(["", "## Degeneration flags", ""])
    if comparison.degeneration_flags:
        for flag in comparison.degeneration_flags:
            lines.append(
                f"- **{flag['model']}** / {flag['prompt_name']} / {flag['decoding']}: "
                f"{', '.join(flag['reasons'])} (len={flag['response_length']})"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Side-by-side generations (SFT vs DPO)", ""])

    sft_by_key = {(r.prompt_name, r.decoding): r for r in comparison.sft.report.per_prompt}
    dpo_by_key = {(r.prompt_name, r.decoding): r for r in comparison.dpo.report.per_prompt}
    prompt_text = {name: text for name, text in comparison.prompts}
    for name, _text in comparison.prompts:
        lines.extend([f"### Prompt: {name}", "", "Prompt:", "", *_fenced(prompt_text[name]), ""])
        for decoding in comparison.decodings:
            sft = sft_by_key.get((name, decoding))
            dpo = dpo_by_key.get((name, decoding))
            lines.extend([f"#### {decoding}", ""])
            lines.append(_caption("SFT", sft))
            lines.extend(["", *_fenced(sft.text if sft else ""), ""])
            lines.append(_caption("DPO", dpo))
            lines.extend(["", *_fenced(dpo.text if dpo else ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _caption(label: str, result: Any | None) -> str:
    if result is None:
        return f"{label}: n/a"
    return f"{label} (len={result.response_length}, rep3={result.repetition_rate_3gram:.2f}):"


def _fenced(text: str) -> list[str]:
    fence = "```"
    safe = text if text.strip() else "(empty)"
    return [f"{fence}text", safe, fence]


def write_chat_eval_json(comparison: ChatEvalComparison, path: Path) -> None:
    path.write_text(json.dumps(comparison.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _load_backbone(checkpoint_path: Path, *, device: torch.device) -> DenseBackbone:
    loaded = load_training_checkpoint(checkpoint_path, map_location=device)
    return loaded.model.to(device)
