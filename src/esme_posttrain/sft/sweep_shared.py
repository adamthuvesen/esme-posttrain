from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.run_artifacts import write_json, write_selected_row_manifest


class SFTSweepError(RuntimeError):
    pass


@dataclass(frozen=True)
class SFTSweepArm:
    name: str
    learning_rate: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    eval_interval: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 60

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def planned_token_upper_bound(self, *, max_sequence_tokens: int) -> int:
        return self.max_steps * self.effective_batch_size * max_sequence_tokens

    def to_dict(self, *, max_sequence_tokens: int) -> dict[str, Any]:
        return {
            "arm_name": self.name,
            "learning_rate": self.learning_rate,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "planned_token_upper_bound": self.planned_token_upper_bound(
                max_sequence_tokens=max_sequence_tokens
            ),
        }


def select_sweep_device(*, require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise SFTSweepError(
            "Modal interval sweep requires CUDA, but torch.cuda.is_available() is false"
        )
    return torch.device("cpu")


def interval_eval_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_rows = [row for row in rows if row.get("event") == "eval"]
    if not eval_rows:
        raise SFTSweepError(f"{metrics_path} has no eval metrics")
    for row in eval_rows:
        value = row.get("eval/matched/response_loss")
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise SFTSweepError(f"{metrics_path} has non-finite eval/matched/response_loss")
    return eval_rows


def train_sanity(metrics_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [row for row in rows if row.get("event") == "train"]
    finite_loss = bool(train_rows) and all(
        isinstance(row.get("train/loss"), int | float) and math.isfinite(float(row["train/loss"]))
        for row in train_rows
    )
    return {
        "finite_loss": finite_loss,
        "train_metric_rows": len(train_rows),
        "first_train_loss": train_rows[0]["train/loss"] if train_rows else None,
        "final_train_loss": train_rows[-1]["train/loss"] if train_rows else None,
        "final_token_accuracy": train_rows[-1].get("train/token_accuracy") if train_rows else None,
    }


def step0_eval(eval_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    for row in eval_metrics:
        if int(row["step"]) == 0:
            return row
    raise SFTSweepError("sweep metrics are missing step-0 eval")


def assert_sweep_data_safe(
    train_report: dict[str, Any],
    eval_report: dict[str, Any],
    *,
    train_sample_cap: int,
    train_token_cap: int,
    eval_sample_cap: int,
    eval_token_cap: int,
) -> None:
    if int(train_report["selected_samples"]) > train_sample_cap:
        raise SFTSweepError("sweep train sample cap exceeded")
    if int(train_report["selected_tokens"]) > train_token_cap:
        raise SFTSweepError("sweep train token cap exceeded")
    if int(eval_report["selected_samples"]) > eval_sample_cap:
        raise SFTSweepError("sweep eval sample cap exceeded")
    if int(eval_report["selected_tokens"]) > eval_token_cap:
        raise SFTSweepError("sweep eval token cap exceeded")
    if train_report["shortfalls"]:
        raise SFTSweepError("training data shortfall: " + "; ".join(train_report["shortfalls"]))
    if eval_report["shortfalls"] and int(eval_report["selected_samples"]) == 0:
        raise SFTSweepError("eval data shortfall: " + "; ".join(eval_report["shortfalls"]))
    if "no_robots" in set(train_report["counts_by_source"]):
        raise SFTSweepError("HuggingFaceH4/no_robots must never be used for training")


def fresh_launch_id(output_root: Path, arms: tuple[SFTSweepArm, ...]) -> str:
    base = time.strftime("sweep-%Y%m%dT%H%M%SZ", time.gmtime())
    for suffix in ("", *[f"-{index}" for index in range(1, 100)]):
        candidate = f"{base}{suffix}"
        paths = [output_root / f"{candidate}-evidence"] + [
            output_root / f"{candidate}-{arm.name}" for arm in arms
        ]
        if all(not path.exists() for path in paths):
            return candidate
    raise SFTSweepError(f"could not find an isolated sweep launch id under {output_root}")


def arm_failure_payload(
    arm: SFTSweepArm,
    *,
    arm_id: str,
    output_dir: Path,
    error: Exception,
    max_sequence_tokens: int,
) -> dict[str, Any]:
    payload = {
        "status": "failed",
        "arm_id": arm_id,
        "arm": arm.to_dict(max_sequence_tokens=max_sequence_tokens),
        "output_dir": str(output_dir),
        "error": str(error),
        "train_sanity": {"finite_loss": False},
    }
    write_json(output_dir / "arm-summary.json", payload)
    return payload


def write_eval_suite_manifests(
    output_dir: Path, matched_eval_reports: dict[str, Any], no_robots_examples: tuple[Any, ...]
) -> None:
    for name, report in matched_eval_reports.items():
        write_selected_row_manifest(output_dir / f"eval-{name}-manifest.jsonl", report.examples)
    write_selected_row_manifest(output_dir / "eval-no_robots-manifest.jsonl", no_robots_examples)
