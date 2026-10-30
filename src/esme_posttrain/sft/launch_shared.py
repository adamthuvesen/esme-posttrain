from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from esme_posttrain.launch.config_guards import LaunchError, positive_int, str_field
from esme_posttrain.launch.config_models import FrozenJSON
from esme_posttrain.sft.config import (
    EvalSource,
    FullTuning,
    SFTBudgets,
    SFTLoss,
    SFTMonitoring,
    SFTOptimizer,
    UnpackedSequence,
)
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
    payload: EvalSource, expected_dataset: Mapping[str, str | bool]
) -> DatasetSource:
    for key, actual in (
        ("name", payload.name),
        ("source", payload.source),
        ("revision", payload.revision),
        ("license", payload.license),
        ("split", payload.split),
        ("train_allowed", payload.train_allowed),
    ):
        expected = expected_dataset[key]
        if actual != expected:
            raise LaunchError(f"datasets.eval_holdout.{key} must be {expected}")
    if payload.role != "eval":
        raise LaunchError("datasets.eval_holdout.role must be eval")
    if payload.usage != "eval_only":
        raise LaunchError("datasets.eval_holdout.usage must be eval_only")
    return DatasetSource(
        name=payload.name,
        source=payload.source,
        revision=payload.revision,
        license=payload.license,
        split=payload.split,
        role="eval",
        train_allowed=False,
    )


def validate_sft_budgets(
    payload: SFTBudgets,
    *,
    max_train_samples_cap: int,
    max_train_tokens_cap: int,
    require_context_1024: bool = False,
) -> SFTBudgets:
    max_train_samples = positive_int(payload.max_train_samples, "budgets.max_train_samples")
    max_train_tokens = positive_int(payload.max_train_tokens, "budgets.max_train_tokens")
    target_train_tokens = positive_int(payload.target_train_tokens, "budgets.target_train_tokens")
    if max_train_samples > max_train_samples_cap:
        raise LaunchError(f"budgets.max_train_samples must be <= {max_train_samples_cap}")
    if max_train_tokens > max_train_tokens_cap:
        raise LaunchError(f"budgets.max_train_tokens must be <= {max_train_tokens_cap}")
    if target_train_tokens > max_train_tokens:
        raise LaunchError("budgets.target_train_tokens must be <= budgets.max_train_tokens")
    if require_context_1024 and payload.max_sequence_tokens != 1024:
        raise LaunchError(
            "budgets.max_sequence_tokens must be 1024 to match the Esme-214M-Base context"
        )
    for key, value in (
        ("max_eval_samples", payload.max_eval_samples),
        ("max_eval_tokens", payload.max_eval_tokens),
        ("matched_eval_samples_per_source", payload.matched_eval_samples_per_source),
        ("matched_eval_tokens_per_source", payload.matched_eval_tokens_per_source),
        ("max_sequence_tokens", payload.max_sequence_tokens),
        ("smoke_train_samples", payload.smoke_train_samples),
        ("smoke_train_tokens", payload.smoke_train_tokens),
        ("smoke_eval_samples", payload.smoke_eval_samples),
    ):
        positive_int(value, f"budgets.{key}")
    if payload.smoke_train_samples > max_train_samples:
        raise LaunchError("budgets.smoke_train_samples must be <= budgets.max_train_samples")
    if payload.smoke_train_tokens > max_train_tokens:
        raise LaunchError("budgets.smoke_train_tokens must be <= budgets.max_train_tokens")
    return payload


def validate_sft_monitoring(payload: SFTMonitoring) -> None:
    positive_int(payload.log_interval, "monitoring.log_interval")
    positive_int(payload.eval_interval, "monitoring.eval_interval")
    positive_int(payload.checkpoint_interval, "monitoring.checkpoint_interval")
    positive_int(payload.retain_last_checkpoints, "monitoring.retain_last_checkpoints")
    positive_int(payload.early_stopping_patience, "monitoring.early_stopping_patience")
    if payload.no_robots_catastrophic_regression_multiplier <= 1.0:
        raise LaunchError("monitoring.no_robots_catastrophic_regression_multiplier must be > 1")
    positive_int(payload.sample_new_tokens, "monitoring.sample_new_tokens")
    str_field(payload.wandb_project, "monitoring.wandb_project")
    if payload.wandb_required_for_modal is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")


def validate_sft_loss(payload: SFTLoss) -> None:
    if payload.assistant_only_loss is not True:
        raise LaunchError("loss.assistant_only_loss must be true")
    if payload.completion_only_loss is not True:
        raise LaunchError("loss.completion_only_loss must be true")
    if payload.ignore_index != -100:
        raise LaunchError("loss.ignore_index must be -100")


def validate_adamw_optimizer(
    payload: SFTOptimizer, *, effective_batch_size: int | None = None
) -> SFTOptimizer:
    if payload.name != "AdamW":
        raise LaunchError("optimizer.name must be AdamW")
    if payload.learning_rate <= 0:
        raise LaunchError("optimizer.learning_rate must be positive")
    for key, value in (
        ("micro_batch_size", payload.micro_batch_size),
        ("gradient_accumulation_steps", payload.gradient_accumulation_steps),
        ("effective_batch_size", payload.effective_batch_size),
        ("max_steps", payload.max_steps),
        ("smoke_max_steps", payload.smoke_max_steps),
        ("seed", payload.seed),
    ):
        positive_int(value, f"optimizer.{key}")
    if (
        payload.micro_batch_size * payload.gradient_accumulation_steps
        != payload.effective_batch_size
    ):
        raise LaunchError(
            "optimizer.effective_batch_size must equal micro_batch_size * "
            "gradient_accumulation_steps"
        )
    if effective_batch_size is not None and payload.effective_batch_size != effective_batch_size:
        raise LaunchError(
            f"optimizer.effective_batch_size must be {effective_batch_size} for this recipe"
        )
    if payload.smoke_max_steps > payload.max_steps:
        raise LaunchError("optimizer.smoke_max_steps must be <= optimizer.max_steps")
    if payload.grad_clip <= 0:
        raise LaunchError("optimizer.grad_clip must be positive")
    if payload.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError(
            "optimizer.scheduler must be constant, linear_warmup_decay, or cosine_decay"
        )
    if payload.warmup_steps < 0:
        raise LaunchError("optimizer.warmup_steps must be a non-negative integer")
    if payload.warmup_steps > payload.max_steps:
        raise LaunchError("optimizer.warmup_steps must be <= optimizer.max_steps")
    if payload.weight_decay < 0:
        raise LaunchError("optimizer.weight_decay must be non-negative")
    return payload


def validate_full_tuning(payload: FullTuning) -> None:
    if payload.mode != "full":
        raise LaunchError("only tuning.mode='full' is supported")


def validate_unpacked_sequence(payload: UnpackedSequence) -> None:
    if payload.sequence_packing is not False:
        raise LaunchError("sequence.sequence_packing must be false until packing is implemented")
    value = payload.pad_to_multiple_of
    if value is not None and value <= 0:
        raise LaunchError("sequence.pad_to_multiple_of must be a positive integer or null")
    str_field(payload.no_packing_rationale, "sequence.no_packing_rationale")


def matched_interval_eval_sweep_blockers(evidence: Mapping[str, FrozenJSON]) -> list[str]:
    label = "learning_gate.evidence.bounded_matched_interval_eval_sweep"
    missing = sorted(
        {
            "kind",
            "eval_metric",
            "baseline_step",
            "step0_response_loss",
            "best_response_loss",
            "interval_eval_steps",
            "evidence_uri",
        }
        - set(evidence)
    )
    if missing:
        return [f"{label} is missing required keys: " + ", ".join(missing)]
    blockers: list[str] = []
    if evidence["kind"] != "bounded_matched_interval_eval_sweep":
        blockers.append(f"{label}.kind must be bounded_matched_interval_eval_sweep")
    if evidence["eval_metric"] != "eval/matched/response_loss":
        blockers.append(f"{label}.eval_metric must be eval/matched/response_loss")
    if evidence["baseline_step"] != 0:
        blockers.append(f"{label}.baseline_step must be 0")
    step0 = evidence["step0_response_loss"]
    best = evidence["best_response_loss"]
    if (
        isinstance(step0, bool)
        or isinstance(best, bool)
        or not isinstance(step0, int | float)
        or not isinstance(best, int | float)
        or step0 <= 0
        or best <= 0
    ):
        blockers.append(f"{label} response losses must be positive numbers")
    elif best >= step0:
        blockers.append(f"{label} must show best_response_loss < step0_response_loss")
    steps = evidence["interval_eval_steps"]
    if (
        not isinstance(steps, tuple)
        or not steps
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
    ):
        blockers.append(f"{label}.interval_eval_steps must list positive steps")
    if not isinstance(evidence["evidence_uri"], str) or not evidence["evidence_uri"]:
        blockers.append(f"{label}.evidence_uri must be a non-empty string")
    return blockers
