from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from esme_posttrain.launch.config_guards import LaunchError, positive_int, require_keys, str_field
from esme_posttrain.sft.data import DatasetSource


def prepare_evidence_dir(default_output_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        evidence_dir = (Path.cwd() / default_output_dir.parent / "local-cpu-fixture").resolve()
        if evidence_dir.exists():
            shutil.rmtree(evidence_dir)
        evidence_dir.mkdir(parents=True)
        return evidence_dir

    evidence_dir = output_dir.expanduser().resolve()
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise ValueError(f"custom output_dir must be empty or absent: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def validate_eval_source(
    payload: dict[str, Any], expected_dataset: dict[str, Any]
) -> DatasetSource:
    require_keys(
        payload,
        {"name", "source", "revision", "license", "split", "role", "train_allowed", "usage"},
        "datasets.eval_holdout",
    )
    for key, value in expected_dataset.items():
        if payload[key] != value:
            raise LaunchError(f"datasets.eval_holdout.{key} must be {value}")
    if payload["role"] != "eval":
        raise LaunchError("datasets.eval_holdout.role must be eval")
    if payload["usage"] != "eval_only":
        raise LaunchError("datasets.eval_holdout.usage must be eval_only")
    return DatasetSource(
        name=expected_dataset["name"],
        source=expected_dataset["source"],
        revision=expected_dataset["revision"],
        license=expected_dataset["license"],
        split=expected_dataset["split"],
        role="eval",
        train_allowed=False,
    )


def validate_sft_budgets(
    payload: dict[str, Any],
    *,
    max_train_samples_cap: int,
    max_train_tokens_cap: int,
    require_context_1024: bool = False,
) -> dict[str, Any]:
    require_keys(
        payload,
        {
            "max_train_samples",
            "max_train_tokens",
            "target_train_tokens",
            "max_eval_samples",
            "max_eval_tokens",
            "matched_eval_samples_per_source",
            "matched_eval_tokens_per_source",
            "max_sequence_tokens",
            "smoke_train_samples",
            "smoke_train_tokens",
            "smoke_eval_samples",
        },
        "budgets",
    )
    max_train_samples = positive_int(payload["max_train_samples"], "budgets.max_train_samples")
    max_train_tokens = positive_int(payload["max_train_tokens"], "budgets.max_train_tokens")
    target_train_tokens = positive_int(
        payload["target_train_tokens"], "budgets.target_train_tokens"
    )
    if max_train_samples > max_train_samples_cap:
        raise LaunchError(f"budgets.max_train_samples must be <= {max_train_samples_cap}")
    if max_train_tokens > max_train_tokens_cap:
        raise LaunchError(f"budgets.max_train_tokens must be <= {max_train_tokens_cap}")
    if target_train_tokens > max_train_tokens:
        raise LaunchError("budgets.target_train_tokens must be <= budgets.max_train_tokens")
    if require_context_1024 and int(payload["max_sequence_tokens"]) != 1024:
        raise LaunchError(
            "budgets.max_sequence_tokens must be 1024 to match the Esme-214M-Base context"
        )
    for key in (
        "max_eval_samples",
        "max_eval_tokens",
        "matched_eval_samples_per_source",
        "matched_eval_tokens_per_source",
        "max_sequence_tokens",
        "smoke_train_samples",
        "smoke_train_tokens",
        "smoke_eval_samples",
    ):
        positive_int(payload[key], f"budgets.{key}")
    if payload["smoke_train_samples"] > max_train_samples:
        raise LaunchError("budgets.smoke_train_samples must be <= budgets.max_train_samples")
    if payload["smoke_train_tokens"] > max_train_tokens:
        raise LaunchError("budgets.smoke_train_tokens must be <= budgets.max_train_tokens")
    return payload


def validate_sft_monitoring(payload: dict[str, Any], *, require_judge: bool = False) -> None:
    keys = {
        "log_interval",
        "eval_interval",
        "checkpoint_interval",
        "retain_last_checkpoints",
        "early_stopping_patience",
        "no_robots_catastrophic_regression_multiplier",
        "sample_new_tokens",
        "wandb_project",
        "wandb_required_for_modal",
    }
    if require_judge:
        keys.add("judge_repeat_passes")
    require_keys(payload, keys, "monitoring")
    positive_int(payload["log_interval"], "monitoring.log_interval")
    positive_int(payload["eval_interval"], "monitoring.eval_interval")
    positive_int(payload["checkpoint_interval"], "monitoring.checkpoint_interval")
    positive_int(payload["retain_last_checkpoints"], "monitoring.retain_last_checkpoints")
    positive_int(payload["early_stopping_patience"], "monitoring.early_stopping_patience")
    if (
        not isinstance(payload["no_robots_catastrophic_regression_multiplier"], int | float)
        or payload["no_robots_catastrophic_regression_multiplier"] <= 1.0
    ):
        raise LaunchError("monitoring.no_robots_catastrophic_regression_multiplier must be > 1")
    positive_int(payload["sample_new_tokens"], "monitoring.sample_new_tokens")
    str_field(payload["wandb_project"], "monitoring.wandb_project")
    if payload["wandb_required_for_modal"] is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")
    if require_judge:
        judge_passes = positive_int(
            payload["judge_repeat_passes"], "monitoring.judge_repeat_passes"
        )
        if judge_passes < 5:
            raise LaunchError("monitoring.judge_repeat_passes must be >= 5")
