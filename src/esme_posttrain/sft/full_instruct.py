from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.run_artifacts import (
    RuntimeSpendTracker,
    refresh_manifest_files,
    write_environment,
    write_json,
    write_selected_row_manifest,
)
from esme_posttrain.sft.data import (
    build_eval_set,
    build_matched_eval_sets,
    build_training_mix,
    sequence_efficiency_report,
)
from esme_posttrain.sft.full_shared import (
    SFTFullRunError,
)
from esme_posttrain.sft.full_shared import (
    assert_full_run_data_safe as _assert_full_run_data_safe,
)
from esme_posttrain.sft.full_shared import (
    assert_required_artifacts as _assert_required_artifacts,
)
from esme_posttrain.sft.full_shared import (
    select_full_run_device as _select_full_run_device,
)
from esme_posttrain.sft.full_shared import (
    steps_for_target_tokens as _steps_for_target_tokens,
)
from esme_posttrain.sft.full_shared import (
    trained_tokens_for_steps as _trained_tokens_for_steps,
)
from esme_posttrain.sft.full_shared import (
    write_eval_suite_manifests as _write_eval_suite_manifests,
)
from esme_posttrain.sft.launch_instruct import EXPECTED_ARTIFACTS, SFTLaunchConfig
from esme_posttrain.sft.trainer import EvalSplit, SFTTrainerConfig, WandbConfig, run_sft_training
from esme_posttrain.training.checkpointing import latest_checkpoint_path


def run_full_instruct_sft(
    config: SFTLaunchConfig,
    *,
    output_dir: Path,
    base_bundle_path: Path | None = None,
    allow_remote_download: bool,
    require_cuda: bool,
    wandb_enabled: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
    resume_from_latest: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from_latest:
        raise SFTFullRunError(f"full-run output_dir must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = latest_checkpoint_path(output_dir) if resume_from_latest else None
    if resume_from_latest and resume_checkpoint is None:
        raise SFTFullRunError(f"--resume requested but no checkpoint exists in {output_dir}")

    profile = config.selected_gpu_profile
    spend_tracker = RuntimeSpendTracker(
        started=started or time.perf_counter(),
        usd_per_hour=float(profile["usd_per_hour"]),
        stop_usd=float(config.runtime["full_run_runtime_spend_stop_usd"]),
        output_dir=output_dir,
    )
    check_spend = partial(
        spend_tracker.check_cap,
        label="full SFT",
        error_type=SFTFullRunError,
    )
    device = _select_full_run_device(require_cuda=require_cuda)
    if config.estimated_full_cost_usd > float(config.runtime["full_run_max_cost_usd"]):
        raise SFTFullRunError("projected full-run cost exceeds runtime.full_run_max_cost_usd")

    bundle_path = (base_bundle_path or config.base_bundle_path).expanduser().resolve()
    loaded = load_dense_backbone_bundle(bundle_path, map_location="cpu")

    budgets = config.budgets
    train_report = build_training_mix(
        config.train_sources,
        loaded.tokenizer,
        max_samples=int(budgets["max_train_samples"]),
        max_tokens=int(budgets["max_train_tokens"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    eval_report = build_eval_set(
        config.eval_source,
        loaded.tokenizer,
        max_samples=int(budgets["max_eval_samples"]),
        max_tokens=int(budgets["max_eval_tokens"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    matched_eval_reports = build_matched_eval_sets(
        config.train_sources,
        loaded.tokenizer,
        skip_selected_by_source={
            name: counts.selected for name, counts in train_report.counts_by_source.items()
        },
        max_samples_per_source=int(budgets["matched_eval_samples_per_source"]),
        max_tokens_per_source=int(budgets["matched_eval_tokens_per_source"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    _assert_full_run_data_safe(config.budgets, train_report.to_dict(), eval_report.to_dict())
    optimizer_config = config.payload["optimizer"]
    monitoring_config = config.payload["monitoring"]
    sequence_config = config.payload["sequence"]
    train_steps = _steps_for_target_tokens(
        train_report.examples,
        target_train_tokens=int(budgets["target_train_tokens"]),
        micro_batch_size=int(optimizer_config["micro_batch_size"]),
        gradient_accumulation_steps=int(optimizer_config["gradient_accumulation_steps"]),
        max_steps=int(optimizer_config["max_steps"]),
    )

    write_json(
        output_dir / "config.json",
        _config_evidence(
            config,
            commit=commit,
            dirty=dirty,
            resume_from_latest=resume_from_latest,
            resume_checkpoint=resume_checkpoint,
        ),
    )
    write_selected_row_manifest(output_dir / "selected-row-manifest.jsonl", train_report.examples)
    _write_eval_suite_manifests(output_dir, matched_eval_reports, eval_report)
    write_json(
        output_dir / "data-report.json",
        {
            "mode": "full_instruct_sft",
            "remote_dataset_download": allow_remote_download,
            "train": train_report.to_dict(),
            "eval": {
                "matched": {
                    name: report.to_dict() for name, report in matched_eval_reports.items()
                },
                "no_robots": eval_report.to_dict(),
            },
            "train_sources": [source.__dict__ for source in config.train_sources],
            "eval_source": config.eval_source.__dict__,
            "token_accounting": {
                "max_train_tokens": budgets["max_train_tokens"],
                "target_train_tokens": budgets["target_train_tokens"],
                "selected_train_tokens": train_report.selected_tokens,
                "selected_supervised_tokens": train_report.selected_supervised_tokens,
                "selected_examples": len(train_report.examples),
                "used_optimizer_steps": train_steps,
                "unused_examples": train_report.to_dict()["unused_examples"],
                "eval_examples": len(eval_report.examples),
                "matched_eval_examples": {
                    name: len(report.examples) for name, report in matched_eval_reports.items()
                },
                "effective_epochs_estimate": _trained_tokens_for_steps(
                    train_report.examples,
                    steps=train_steps,
                    micro_batch_size=int(optimizer_config["micro_batch_size"]),
                    gradient_accumulation_steps=int(
                        optimizer_config["gradient_accumulation_steps"]
                    ),
                )
                / max(1, train_report.selected_tokens),
            },
            "sequence_efficiency": {
                "train": sequence_efficiency_report(
                    train_report.examples,
                    max_sequence_tokens=int(budgets["max_sequence_tokens"]),
                    micro_batch_size=int(optimizer_config["micro_batch_size"]),
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
                "eval": sequence_efficiency_report(
                    eval_report.examples,
                    max_sequence_tokens=int(budgets["max_sequence_tokens"]),
                    micro_batch_size=int(optimizer_config["micro_batch_size"]),
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
                "matched_eval": {
                    name: sequence_efficiency_report(
                        report.examples,
                        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
                        micro_batch_size=int(optimizer_config["micro_batch_size"]),
                        sequence_packing=bool(sequence_config["sequence_packing"]),
                        pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                        no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                    )
                    for name, report in matched_eval_reports.items()
                },
            },
            "prompt_masking_asserted": all(
                all(label == -100 for label in example.labels[: example.prompt_tokens])
                for example in train_report.examples
            ),
            "no_robots_training": False,
            "no_robots_eval_only_reason": (
                "HuggingFaceH4/no_robots is CC-BY-NC-4.0 and this run card does not approve "
                "non-commercial training data."
            ),
        },
    )
    write_environment(output_dir / "environment.txt", device=device)

    result = run_sft_training(
        loaded.model,
        loaded.tokenizer,
        train_report.examples,
        eval_report.examples,
        SFTTrainerConfig(
            max_steps=train_steps,
            micro_batch_size=int(optimizer_config["micro_batch_size"]),
            gradient_accumulation_steps=int(optimizer_config["gradient_accumulation_steps"]),
            learning_rate=float(optimizer_config["learning_rate"]),
            scheduler=str(optimizer_config["scheduler"]),
            warmup_steps=int(optimizer_config["warmup_steps"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            precision=str(config.runtime["precision"]),
            pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
            seed=int(optimizer_config["seed"]),
            output_dir=output_dir,
            assistant_only_loss=bool(config.payload["loss"]["assistant_only_loss"]),
            completion_only_loss=bool(config.payload["loss"]["completion_only_loss"]),
            tuning_mode=str(config.payload["tuning"]["mode"]),
            grad_clip=float(optimizer_config["grad_clip"]),
            log_interval=int(monitoring_config["log_interval"]),
            eval_interval=int(monitoring_config["eval_interval"]),
            checkpoint_interval=int(monitoring_config["checkpoint_interval"]),
            retain_last_checkpoints=int(monitoring_config["retain_last_checkpoints"]),
            early_stopping_patience=int(monitoring_config["early_stopping_patience"]),
            no_robots_catastrophic_regression_multiplier=float(
                monitoring_config["no_robots_catastrophic_regression_multiplier"]
            ),
            sample_new_tokens=int(monitoring_config["sample_new_tokens"]),
            device=device.type,
            resume_from_latest=resume_from_latest,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(monitoring_config["wandb_project"]),
                run_name=f"{config.run_id}-{'resume' if resume_from_latest else 'full-sft'}",
                tags=(
                    "Esme-214M-Instruct",
                    "sft",
                    "resume" if resume_from_latest else "full",
                    "smol-smoltalk",
                    "tulu-personas",
                    "no-robots-eval",
                    config.runtime["selected_gpu"],
                    str(config.payload["tuning"]["mode"]),
                ),
                group=config.run_id,
                job_type="full-sft",
                notes="Full supervised cold-start SFT; no downstream DPO/RLVR, LoRA, or QLoRA.",
                extra_config={
                    "model": config.artifact_name,
                    "stage": "sft_cold_start",
                    "run_type": "full",
                    "dataset_mix": {
                        source.name: source.mix_ratio for source in config.train_sources
                    },
                    "eval_holdout": config.eval_source.source,
                    "eval_holdout_revision": config.eval_source.revision,
                    "gpu": config.runtime["selected_gpu"],
                    "tuning_mode": config.payload["tuning"]["mode"],
                    "resume": resume_from_latest,
                    "resume_checkpoint": str(resume_checkpoint)
                    if resume_checkpoint is not None
                    else None,
                    "precision": config.runtime["precision"],
                    "scheduler": optimizer_config["scheduler"],
                },
            ),
        ),
        eval_splits=_eval_splits(matched_eval_reports, eval_report),
        base_bundle_manifest=loaded.bundle.manifest,
        step_callback=check_spend,
    )
    if not result.instruct_beats_base:
        raise SFTFullRunError("weighted matched response loss did not beat Base")

    write_json(output_dir / "eval-report.json", result.to_dict())
    cost = spend_tracker.write_cost(step=result.steps_completed, status="complete")
    refresh_manifest_files(output_dir, EXPECTED_ARTIFACTS)
    _assert_required_artifacts(output_dir, EXPECTED_ARTIFACTS)

    return {
        "status": "modal_full_sft_complete",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "output_dir": str(output_dir),
        "volume": config.runtime["modal_volume"],
        "commit": commit,
        "dirty": dirty,
        "device": device.type,
        "paid_compute": True,
        "training_mode": result.training_mode,
        "resumed_from_checkpoint": result.resumed_from_checkpoint,
        "cost": cost,
        "wandb_run": result.wandb_run_url,
        "result": result.to_dict(),
        "required_artifacts_present": {
            name: (output_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


def mirror_launch_evidence(target_dir: Path, payload: dict[str, Any]) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "launch-receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _eval_splits(
    matched_eval_reports: dict[str, Any], no_robots_report: Any
) -> tuple[EvalSplit, ...]:
    return (
        EvalSplit(
            "smol-smoltalk",
            matched_eval_reports["smol-smoltalk"].examples,
            selector_weight=0.8,
        ),
        EvalSplit(
            "tulu-3-personas",
            matched_eval_reports["tulu-3-personas"].examples,
            selector_weight=0.2,
        ),
        EvalSplit("no_robots", no_robots_report.examples),
    )


def _config_evidence(
    config: SFTLaunchConfig,
    *,
    commit: str,
    dirty: bool,
    resume_from_latest: bool,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    return {
        "mode": "full_instruct_sft",
        "training_mode": "resumed" if resume_from_latest else "fresh",
        "resume_from_latest": resume_from_latest,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "source_config": config.payload,
        "commit": commit,
        "dirty": dirty,
        "projected_full_cost_usd": config.estimated_full_cost_usd,
        "target_train_tokens": config.budgets["target_train_tokens"],
        "max_train_tokens": config.budgets["max_train_tokens"],
        "max_train_steps": config.train_steps,
        "approval": {
            "approved_by": "Adam",
            "approved_on": "2026-06-27",
            "condition": "spend approval only after the real full-run path is safe",
        },
    }
