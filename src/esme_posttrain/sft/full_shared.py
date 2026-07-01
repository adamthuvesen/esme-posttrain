from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from esme_posttrain.run_artifacts import write_selected_row_manifest


class SFTFullRunError(RuntimeError):
    pass


def write_eval_suite_manifests(
    output_dir: Path, matched_eval_reports: dict[str, Any], no_robots_report: Any
) -> None:
    for name, report in matched_eval_reports.items():
        write_selected_row_manifest(output_dir / f"eval-{name}-manifest.jsonl", report.examples)
    write_selected_row_manifest(
        output_dir / "eval-no_robots-manifest.jsonl", no_robots_report.examples
    )


def assert_full_run_data_safe(
    budgets: dict[str, Any], train_report: dict[str, Any], eval_report: dict[str, Any]
) -> None:
    if int(train_report["selected_samples"]) > int(budgets["max_train_samples"]):
        raise SFTFullRunError("train sample cap exceeded")
    if int(train_report["selected_tokens"]) > int(budgets["max_train_tokens"]):
        raise SFTFullRunError("train token cap exceeded")
    if int(eval_report["selected_tokens"]) > int(budgets["max_eval_tokens"]):
        raise SFTFullRunError("eval token cap exceeded")
    if train_report["shortfalls"]:
        raise SFTFullRunError("training data shortfall: " + "; ".join(train_report["shortfalls"]))
    if eval_report["shortfalls"]:
        raise SFTFullRunError("eval data shortfall: " + "; ".join(eval_report["shortfalls"]))
    if "no_robots" in set(train_report["counts_by_source"]):
        raise SFTFullRunError("HuggingFaceH4/no_robots must never be used for training")


def assert_required_artifacts(output_dir: Path, expected_artifacts: tuple[str, ...]) -> None:
    missing = [name for name in expected_artifacts if not (output_dir / name).is_file()]
    if missing:
        raise SFTFullRunError("missing required full-run artifacts: " + ", ".join(missing))


def steps_for_target_tokens(
    examples: tuple[Any, ...],
    *,
    target_train_tokens: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
) -> int:
    if target_train_tokens <= 0:
        raise SFTFullRunError("budgets.target_train_tokens must be positive")
    if not examples:
        raise SFTFullRunError("cannot compute train steps without selected examples")
    trained_tokens = 0
    for step in range(1, max_steps + 1):
        trained_tokens += tokens_for_step(
            examples,
            step=step,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if trained_tokens >= target_train_tokens:
            return step
    return max_steps


def trained_tokens_for_steps(
    examples: tuple[Any, ...],
    *,
    steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    return sum(
        tokens_for_step(
            examples,
            step=step,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        for step in range(1, steps + 1)
    )


def tokens_for_step(
    examples: tuple[Any, ...],
    *,
    step: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    tokens = 0
    for accumulation_index in range(gradient_accumulation_steps):
        batch_index = (step - 1) * gradient_accumulation_steps + accumulation_index
        start = batch_index * micro_batch_size
        tokens += sum(
            len(examples[(start + offset) % len(examples)].input_ids)
            for offset in range(micro_batch_size)
        )
    return tokens


def select_full_run_device(*, require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise SFTFullRunError(
            "Modal full SFT requires CUDA, but torch.cuda.is_available() is false"
        )
    return torch.device("cpu")
