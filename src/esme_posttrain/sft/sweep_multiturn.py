"""Bounded matched-eval LR sweep for the multi-turn SFT foundation.

Mirrors ``sft_sweep`` but trains on real multi-turn conversations at the recipe's
1024 sequence length and evaluates the weighted matched held-out (multi-turn
smol-smoltalk 0.85 + single-turn tulu-3-personas 0.15, with no_robots as an OOD
guardrail). It anchors learning rates around the config's small-model ``1e-4``
and emits the bounded-matched learning-gate evidence the full-run launcher
requires (``eval/matched/response_loss`` lower than step 0).
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    estimate_cost_usd,
)
from esme_posttrain.run_artifacts import (
    RuntimeSpendTracker,
    write_environment,
    write_eval_suite_manifests,
    write_json,
    write_selected_row_manifest,
)
from esme_posttrain.sft.data import sequence_efficiency_report
from esme_posttrain.sft.launch_multiturn import MultiTurnLaunchConfig
from esme_posttrain.sft.multiturn_data import (
    build_multi_turn_eval_set,
    build_multi_turn_matched_eval_sets,
    build_multi_turn_mix,
)
from esme_posttrain.sft.sweep_shared import (
    SFTSweepArm,
    SFTSweepError,
)
from esme_posttrain.sft.sweep_shared import (
    arm_failure_payload as _arm_failure_payload,
)
from esme_posttrain.sft.sweep_shared import (
    assert_sweep_data_safe as _assert_sweep_data_safe,
)
from esme_posttrain.sft.sweep_shared import (
    fresh_launch_id as _fresh_launch_id,
)
from esme_posttrain.sft.sweep_shared import (
    interval_eval_metrics as _interval_eval_metrics,
)
from esme_posttrain.sft.sweep_shared import (
    select_sweep_device as _select_sweep_device,
)
from esme_posttrain.sft.sweep_shared import (
    step0_eval as _step0_eval,
)
from esme_posttrain.sft.sweep_shared import (
    train_sanity as _train_sanity,
)
from esme_posttrain.sft.trainer import EvalSplit, SFTTrainerConfig, run_sft_training
from esme_posttrain.training.wandb_init import WandbConfig

SWEEP_OUTPUT_STEM = "esme-multiturn-sft-interval-sweep"
SWEEP_GROUP = "esme_214m_sft_multiturn_interval_sweep"
SWEEP_SPEND_CAP_USD = 8.0
SWEEP_TIMEOUT_HOURS = 3
SWEEP_TRAIN_SAMPLE_CAP = 512
SWEEP_TRAIN_TOKEN_CAP = 1_572_864
SWEEP_EVAL_SAMPLE_CAP = 96
SWEEP_EVAL_TOKEN_CAP = 393_216
SWEEP_MATCHED_EVAL_SAMPLE_CAP = 64
SWEEP_MATCHED_EVAL_TOKEN_CAP = 262_144
# Matches the recipe's max_sequence_tokens (the Esme-214M-Base 1024 context); used
# only to annotate an arm's planned-token upper bound in a pre-training failure record.
RECIPE_MAX_SEQUENCE_TOKENS = 1024
DEFAULT_MODAL_SWEEP_ROOT = Path("/posttrain") / SWEEP_OUTPUT_STEM


# Anchored around the config's small-model 1e-4; SmolLM2-class models tolerate a
# higher LR than the 2e-5 large-model default. Effective batch 16 = the recipe's
# microbatch 2 x grad-accum 8, kept identical so the sweep matches the full run.
SWEEP_ARMS: tuple[SFTSweepArm, ...] = (
    SFTSweepArm(
        name="lr5e-5-mb2-ga8-eb16",
        learning_rate=5e-5,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=120,
        warmup_steps=12,
        checkpoint_interval=60,
    ),
    SFTSweepArm(
        name="lr1e-4-mb2-ga8-eb16",
        learning_rate=1e-4,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=120,
        warmup_steps=12,
        checkpoint_interval=60,
    ),
    SFTSweepArm(
        name="lr2e-4-mb2-ga8-eb16",
        learning_rate=2e-4,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=120,
        warmup_steps=12,
        checkpoint_interval=60,
    ),
    SFTSweepArm(
        name="lr3e-4-mb2-ga8-eb16",
        learning_rate=3e-4,
        micro_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=120,
        warmup_steps=12,
        checkpoint_interval=60,
    ),
)


def build_multi_turn_sweep_preflight(
    config: MultiTurnLaunchConfig,
    *,
    timeout_hours: int = SWEEP_TIMEOUT_HOURS,
    modal_gpu: str | None = None,
) -> dict[str, Any]:
    selected_profile = config.selected_gpu_profile
    max_sequence_tokens = int(config.budgets["max_sequence_tokens"])
    projected_tokens = sum(
        arm.planned_token_upper_bound(max_sequence_tokens=max_sequence_tokens) for arm in SWEEP_ARMS
    )
    projected_cost = estimate_cost_usd(
        tokens=projected_tokens,
        projected_tokens_per_second=float(selected_profile["projected_tokens_per_second"]),
        usd_per_hour=float(selected_profile["usd_per_hour"]),
    )
    blockers = multi_turn_sweep_blockers(
        config,
        timeout_hours=timeout_hours,
        modal_gpu=modal_gpu or str(config.runtime["selected_gpu"]),
    )
    return {
        "status": "ready_for_modal_sweep" if not blockers else "blocked_by_launch_safety",
        "mode": "multi_turn_interval_eval_sweep",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "starts_from": config.payload["starts_from"],
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "will_start_modal_job": False,
        "will_download_data": False,
        "modal_run_will_download_real_data": True,
        "uses_cpu_fixture_path": False,
        "uses_full_run_output_dir": False,
        "volume": config.runtime["modal_volume"],
        "volume_output_root": str(DEFAULT_MODAL_SWEEP_ROOT),
        "arm_id_pattern": "sweep-<UTC>-<arm-name>",
        "arms": [arm.to_dict(max_sequence_tokens=max_sequence_tokens) for arm in SWEEP_ARMS],
        "datasets": config.payload["datasets"],
        "data_caps": {
            "train_samples": SWEEP_TRAIN_SAMPLE_CAP,
            "train_tokens": SWEEP_TRAIN_TOKEN_CAP,
            "eval_samples": SWEEP_EVAL_SAMPLE_CAP,
            "eval_tokens": SWEEP_EVAL_TOKEN_CAP,
            "matched_eval_samples_per_source": SWEEP_MATCHED_EVAL_SAMPLE_CAP,
            "matched_eval_tokens_per_source": SWEEP_MATCHED_EVAL_TOKEN_CAP,
            "max_sequence_tokens": max_sequence_tokens,
            "no_robots_train_allowed": False,
        },
        "runtime": {
            "provider": "modal",
            "selected_gpu": config.runtime["selected_gpu"],
            "modal_gpu": modal_gpu or config.runtime["selected_gpu"],
            "precision": config.runtime["precision"],
            "timeout_hours": timeout_hours,
            "sweep_spend_cap_usd": SWEEP_SPEND_CAP_USD,
            "timeout_cost_ceiling_usd": float(selected_profile["usd_per_hour"]) * timeout_hours,
            "projected_train_token_upper_bound": projected_tokens,
            "projected_cost_usd": round(projected_cost, 4),
        },
        "monitoring": {
            "wandb_project": config.payload["monitoring"]["wandb_project"],
            "job_type": "sweep",
            "group": SWEEP_GROUP,
            "required_eval_metrics": [
                "eval/response_loss",
                "eval/matched/response_loss",
                "eval/smol-smoltalk/response_loss",
                "eval/tulu-3-personas/response_loss",
                "eval/no_robots/response_loss",
                "eval/perplexity",
                "eval/supervised_tokens",
                "eval/examples",
            ],
            "required_train_metrics": [
                "train/loss",
                "train/learning_rate",
                "train/grad_norm",
                "train/tokens",
                "train/supervised_tokens",
                "train/token_accuracy",
            ],
        },
        "acceptance": {
            "gate": "matched held-out response loss must move down versus step 0 for one sane arm",
            "metric": "eval/matched/response_loss",
            "baseline_step": 0,
        },
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION},
        "launch_blockers": blockers,
        "modal_sweep_command": multi_turn_sweep_command(
            config.config_path, timeout_hours=timeout_hours, gpu=str(config.runtime["selected_gpu"])
        ),
    }


def multi_turn_sweep_blockers(
    config: MultiTurnLaunchConfig, *, timeout_hours: int, modal_gpu: str
) -> list[str]:
    blockers: list[str] = []
    runtime = config.runtime
    if timeout_hours <= 0 or timeout_hours > 24:
        blockers.append("SFT_SWEEP_TIMEOUT_HOURS must be between 1 and 24")
    if modal_gpu != runtime["selected_gpu"]:
        blockers.append("SFT_MODAL_GPU must match runtime.selected_gpu for sweep cost accounting")
    selected_profile = config.selected_gpu_profile
    timeout_cost = timeout_hours * float(selected_profile["usd_per_hour"])
    if timeout_cost > SWEEP_SPEND_CAP_USD:
        blockers.append("sweep timeout cost ceiling exceeds the approved $8 sweep spend cap")
    max_sequence_tokens = int(config.budgets["max_sequence_tokens"])
    projected_tokens = sum(
        arm.planned_token_upper_bound(max_sequence_tokens=max_sequence_tokens) for arm in SWEEP_ARMS
    )
    projected_cost = estimate_cost_usd(
        tokens=projected_tokens,
        projected_tokens_per_second=float(selected_profile["projected_tokens_per_second"]),
        usd_per_hour=float(selected_profile["usd_per_hour"]),
    )
    if projected_cost > SWEEP_SPEND_CAP_USD:
        blockers.append("projected interval sweep cost exceeds the approved $8 sweep spend cap")
    if "-excellence" in str(DEFAULT_MODAL_SWEEP_ROOT):
        blockers.append("sweep output root must not use an excellence public name")
    for arm in SWEEP_ARMS:
        if arm.eval_interval <= 0:
            blockers.append(f"{arm.name} must enable interval eval")
        if arm.max_steps > 200:
            blockers.append(f"{arm.name} exceeds the bounded sweep max_steps cap")
        if arm.effective_batch_size > 16:
            blockers.append(f"{arm.name} exceeds the bounded effective batch cap")
    return blockers


def multi_turn_sweep_command(config_path: Path, *, timeout_hours: int, gpu: str) -> str:
    return (
        f"SFT_MODAL_GPU='{gpu}' SFT_SWEEP_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} modal run "
        f"scripts/modal_chat_sft.py --config {config_path.as_posix()} "
        f"--modal-sweep --approved --json"
    )


def run_multi_turn_interval_eval_sweep(
    config: MultiTurnLaunchConfig,
    *,
    output_root: Path,
    base_bundle_path: Path | None = None,
    allow_remote_download: bool,
    require_cuda: bool,
    wandb_enabled: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if output_root.name != SWEEP_OUTPUT_STEM:
        raise SFTSweepError(f"sweep output root must end with {SWEEP_OUTPUT_STEM}")
    if "-excellence" in str(output_root):
        raise SFTSweepError("sweep output root must not use an excellence public name")
    output_root.mkdir(parents=True, exist_ok=True)

    started = started or time.perf_counter()
    device = _select_sweep_device(require_cuda=require_cuda)
    profile = config.selected_gpu_profile
    sweep_spend = RuntimeSpendTracker(
        started=started,
        usd_per_hour=float(profile["usd_per_hour"]),
        stop_usd=SWEEP_SPEND_CAP_USD,
        output_dir=output_root,
    )
    bundle_path = (base_bundle_path or config.base_bundle_path).expanduser().resolve()
    loaded = load_dense_backbone_bundle(bundle_path, map_location="cpu")
    budgets = config.budgets
    max_sequence_tokens = int(budgets["max_sequence_tokens"])
    train_report = build_multi_turn_mix(
        config.train_sources,
        loaded.tokenizer,
        max_samples=SWEEP_TRAIN_SAMPLE_CAP,
        max_tokens=SWEEP_TRAIN_TOKEN_CAP,
        max_sequence_tokens=max_sequence_tokens,
        allow_remote_download=allow_remote_download,
    )
    eval_report = build_multi_turn_eval_set(
        config.eval_source,
        loaded.tokenizer,
        max_samples=SWEEP_EVAL_SAMPLE_CAP,
        max_tokens=SWEEP_EVAL_TOKEN_CAP,
        max_sequence_tokens=max_sequence_tokens,
        allow_remote_download=allow_remote_download,
    )
    matched_eval_reports = build_multi_turn_matched_eval_sets(
        config.train_sources,
        loaded.tokenizer,
        skip_selected_by_source={
            name: counts.selected for name, counts in train_report.counts_by_source.items()
        },
        max_samples_per_source=SWEEP_MATCHED_EVAL_SAMPLE_CAP,
        max_tokens_per_source=SWEEP_MATCHED_EVAL_TOKEN_CAP,
        max_sequence_tokens=max_sequence_tokens,
        allow_remote_download=allow_remote_download,
    )
    _assert_sweep_data_safe(
        train_report.to_dict(),
        eval_report.to_dict(),
        train_sample_cap=SWEEP_TRAIN_SAMPLE_CAP,
        train_token_cap=SWEEP_TRAIN_TOKEN_CAP,
        eval_sample_cap=SWEEP_EVAL_SAMPLE_CAP,
        eval_token_cap=SWEEP_EVAL_TOKEN_CAP,
    )
    launch_id = _fresh_launch_id(output_root, SWEEP_ARMS)
    evidence_dir = output_root / f"{launch_id}-evidence"
    evidence_dir.mkdir()
    write_json(
        evidence_dir / "data-report.json",
        {
            "mode": "bounded_matched_interval_eval_sweep",
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
            "no_robots_training": False,
            "caps": {
                "train_samples": SWEEP_TRAIN_SAMPLE_CAP,
                "train_tokens": SWEEP_TRAIN_TOKEN_CAP,
                "eval_samples": SWEEP_EVAL_SAMPLE_CAP,
                "eval_tokens": SWEEP_EVAL_TOKEN_CAP,
                "matched_eval_samples_per_source": SWEEP_MATCHED_EVAL_SAMPLE_CAP,
                "matched_eval_tokens_per_source": SWEEP_MATCHED_EVAL_TOKEN_CAP,
                "spend_cap_usd": SWEEP_SPEND_CAP_USD,
            },
        },
    )

    arm_payloads: list[dict[str, Any]] = []
    for arm in SWEEP_ARMS:
        arm_id = f"{launch_id}-{arm.name}"
        arm_output_dir = output_root / arm_id
        arm_output_dir.mkdir()
        try:
            arm_payloads.append(
                _run_sweep_arm(
                    config,
                    arm,
                    arm_id=arm_id,
                    output_dir=arm_output_dir,
                    base_bundle_path=bundle_path,
                    train_examples=train_report.examples,
                    eval_examples=eval_report.examples,
                    matched_eval_reports=matched_eval_reports,
                    train_report=train_report.to_dict(),
                    eval_report=eval_report.to_dict(),
                    device=device,
                    sweep_spend=sweep_spend,
                    wandb_enabled=wandb_enabled,
                    commit=commit,
                    dirty=dirty,
                )
            )
        except SFTSweepError as error:
            arm_payloads.append(
                _arm_failure_payload(
                    arm,
                    arm_id=arm_id,
                    output_dir=arm_output_dir,
                    error=error,
                    max_sequence_tokens=RECIPE_MAX_SEQUENCE_TOKENS,
                )
            )
            break
        except Exception as error:  # noqa: BLE001 - record the failure and continue to next arm.
            arm_payloads.append(
                _arm_failure_payload(
                    arm,
                    arm_id=arm_id,
                    output_dir=arm_output_dir,
                    error=error,
                    max_sequence_tokens=RECIPE_MAX_SEQUENCE_TOKENS,
                )
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed = [
        arm
        for arm in arm_payloads
        if arm["status"] == "complete" and arm["train_sanity"]["finite_loss"]
    ]
    improving = [
        arm
        for arm in completed
        if arm["best_eval"]["eval/matched/response_loss"]
        < arm["step0_eval"]["eval/matched/response_loss"]
    ]
    best_arm = min(
        improving, key=lambda arm: arm["best_eval"]["eval/matched/response_loss"], default=None
    )
    status = "interval_eval_sweep_passed" if best_arm is not None else "interval_eval_sweep_failed"
    cost = sweep_spend.write_cost(
        step=max((int(arm.get("steps_completed", 0)) for arm in arm_payloads), default=0),
        status=status,
    )
    learning_gate = _learning_gate_payload(
        status=status,
        best_arm=best_arm,
        evidence_uri=str(evidence_dir / "interval-eval-sweep.json"),
    )
    payload = {
        "status": status,
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "mode": "bounded_matched_interval_eval_sweep",
        "launch_id": launch_id,
        "output_root": str(output_root),
        "evidence_dir": str(evidence_dir),
        "interval_eval_sweep_path": str(evidence_dir / "interval-eval-sweep.json"),
        "learning_gate_path": str(evidence_dir / "learning-gate.json"),
        "volume": config.runtime["modal_volume"],
        "commit": commit,
        "dirty": dirty,
        "device": device.type,
        "paid_compute": True,
        "cost": cost,
        "spend_cap_usd": SWEEP_SPEND_CAP_USD,
        "arms": arm_payloads,
        "selected_best_arm": best_arm["arm_id"] if best_arm is not None else None,
        "selected_best_learning_rate": best_arm["arm"]["learning_rate"]
        if best_arm is not None
        else None,
        "learning_gate": learning_gate,
    }
    write_json(evidence_dir / "interval-eval-sweep.json", payload)
    write_json(evidence_dir / "learning-gate.json", learning_gate)
    return payload


def _run_sweep_arm(
    config: MultiTurnLaunchConfig,
    arm: SFTSweepArm,
    *,
    arm_id: str,
    output_dir: Path,
    base_bundle_path: Path,
    train_examples: tuple[Any, ...],
    eval_examples: tuple[Any, ...],
    matched_eval_reports: dict[str, Any],
    train_report: dict[str, Any],
    eval_report: dict[str, Any],
    device: torch.device,
    sweep_spend: RuntimeSpendTracker,
    wandb_enabled: bool,
    commit: str,
    dirty: bool,
) -> dict[str, Any]:
    loaded = load_dense_backbone_bundle(base_bundle_path, map_location="cpu")
    optimizer_config = config.payload["optimizer"]
    monitoring_config = config.payload["monitoring"]
    sequence_config = config.payload["sequence"]
    max_sequence_tokens = int(config.budgets["max_sequence_tokens"])
    write_json(
        output_dir / "config.json",
        {
            "mode": "bounded_matched_interval_eval_sweep_arm",
            "arm": arm.to_dict(max_sequence_tokens=max_sequence_tokens),
            "source_config": config.payload,
            "commit": commit,
            "dirty": dirty,
            "output_dir": str(output_dir),
            "approval": {
                "approved_by": "Adam",
                "approved_on": "2026-06-27",
                "condition": "bounded real-Esme multi-turn matched-eval SFT sweep only; not full",
            },
        },
    )
    write_selected_row_manifest(output_dir / "selected-row-manifest.jsonl", train_examples)
    write_eval_suite_manifests(output_dir, matched_eval_reports, eval_examples)
    write_environment(output_dir / "environment.txt", device=device)

    arm_started_cost = sweep_spend.estimated_cost_usd()
    result = run_sft_training(
        loaded.model,
        loaded.tokenizer,
        train_examples,
        eval_examples,
        SFTTrainerConfig(
            max_steps=arm.max_steps,
            micro_batch_size=arm.micro_batch_size,
            gradient_accumulation_steps=arm.gradient_accumulation_steps,
            learning_rate=arm.learning_rate,
            scheduler=str(optimizer_config["scheduler"]),
            warmup_steps=arm.warmup_steps,
            weight_decay=float(optimizer_config["weight_decay"]),
            precision=str(config.runtime["precision"]),
            pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
            seed=int(optimizer_config["seed"]),
            output_dir=output_dir,
            artifact_name=config.artifact_name,
            assistant_only_loss=bool(config.payload["loss"]["assistant_only_loss"]),
            completion_only_loss=bool(config.payload["loss"]["completion_only_loss"]),
            tuning_mode=str(config.payload["tuning"]["mode"]),
            grad_clip=float(optimizer_config["grad_clip"]),
            log_interval=arm.log_interval,
            eval_interval=arm.eval_interval,
            checkpoint_interval=arm.checkpoint_interval,
            retain_last_checkpoints=1,
            early_stopping_patience=int(monitoring_config["early_stopping_patience"]),
            no_robots_catastrophic_regression_multiplier=float(
                monitoring_config["no_robots_catastrophic_regression_multiplier"]
            ),
            sample_new_tokens=min(40, int(monitoring_config["sample_new_tokens"])),
            device=device.type,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(monitoring_config["wandb_project"]),
                run_name=f"{SWEEP_GROUP}-{arm_id}",
                tags=(
                    f"model:{config.artifact_name}",
                    "stage:sft",
                    "run-type:multi-turn-interval-sweep",
                    "dataset-mix:smol-smoltalk-85-tulu-personas-15",
                    "eval-holdout:no-robots",
                    f"gpu:{config.runtime['selected_gpu']}",
                    f"tuning-mode:{config.payload['tuning']['mode']}",
                    f"arm:{arm_id}",
                ),
                group=SWEEP_GROUP,
                job_type="sweep",
                notes=(
                    "Bounded real-Esme multi-turn matched-eval SFT sweep; no full launch, "
                    "no DPO, no RL."
                ),
                extra_config={
                    "model": config.artifact_name,
                    "stage": "sft",
                    "run_type": "multi_turn_interval_sweep",
                    "dataset_mix": {
                        source.name: source.mix_ratio for source in config.train_sources
                    },
                    "eval_holdout": config.eval_source.source,
                    "eval_holdout_revision": config.eval_source.revision,
                    "gpu": config.runtime["selected_gpu"],
                    "precision": config.runtime["precision"],
                    "tuning_mode": config.payload["tuning"]["mode"],
                    "max_sequence_tokens": max_sequence_tokens,
                    "arm_id": arm_id,
                    "sweep_group": SWEEP_GROUP,
                    "sweep_spend_cap_usd": SWEEP_SPEND_CAP_USD,
                },
            ),
        ),
        eval_splits=_eval_splits(config, matched_eval_reports, eval_examples),
        base_bundle_manifest=loaded.bundle.manifest,
        step_callback=lambda step: sweep_spend.check_cap(
            step,
            label="multi-turn interval sweep",
            error_type=SFTSweepError,
        ),
    )
    write_json(
        output_dir / "data-report.json",
        {
            "mode": "bounded_matched_interval_eval_sweep_arm",
            "arm_id": arm_id,
            "train": train_report,
            "eval": {
                "matched": {
                    name: report.to_dict() for name, report in matched_eval_reports.items()
                },
                "no_robots": eval_report,
            },
            "token_accounting": {
                "selected_train_tokens": train_report["selected_tokens"],
                "selected_supervised_tokens": train_report["selected_supervised_tokens"],
                "selected_examples": train_report["selected_samples"],
                "eval_examples": eval_report["selected_samples"],
                "matched_eval_examples": {
                    name: len(report.examples) for name, report in matched_eval_reports.items()
                },
            },
            "sequence_efficiency": {
                "train": sequence_efficiency_report(
                    train_examples,
                    max_sequence_tokens=max_sequence_tokens,
                    micro_batch_size=arm.micro_batch_size,
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
            },
            "no_robots_training": False,
        },
    )

    cost = sweep_spend.write_cost(step=result.steps_completed, status="arm_complete")
    estimated_arm_cost = max(0.0, cost["estimated_cost_usd"] - arm_started_cost)
    eval_metrics = _interval_eval_metrics(result.metrics_path)
    train_sanity = _train_sanity(result.metrics_path)
    step0_eval = _step0_eval(eval_metrics)
    post_step_evals = [row for row in eval_metrics if int(row["step"]) > 0]
    best_eval = min(post_step_evals, key=lambda row: row["eval/matched/response_loss"])
    final_eval = post_step_evals[-1]
    payload = {
        "status": "complete",
        "arm_id": arm_id,
        "arm": arm.to_dict(max_sequence_tokens=max_sequence_tokens),
        "output_dir": str(output_dir),
        "metrics_path": str(result.metrics_path),
        "wandb_run": result.wandb_run_url,
        "steps_completed": result.steps_completed,
        "trained_tokens": result.trained_tokens,
        "supervised_tokens": result.supervised_tokens,
        "selected_examples": result.selected_examples,
        "eval_examples": result.eval_examples,
        "step0_eval": step0_eval,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "interval_eval_steps": sorted({int(row["step"]) for row in post_step_evals}),
        "response_loss_decreased": best_eval["eval/matched/response_loss"]
        < step0_eval["eval/matched/response_loss"],
        "train_sanity": train_sanity,
        "cost": {**cost, "estimated_arm_cost_usd": estimated_arm_cost},
    }
    write_json(output_dir / "arm-summary.json", payload)
    return payload


def _eval_splits(
    config: MultiTurnLaunchConfig,
    matched_eval_reports: dict[str, Any],
    no_robots_examples: tuple[Any, ...],
) -> tuple[EvalSplit, ...]:
    splits: list[EvalSplit] = []
    for source in config.train_sources:
        report = matched_eval_reports[source.name]
        splits.append(EvalSplit(source.name, report.examples, selector_weight=source.mix_ratio))
    splits.append(EvalSplit("no_robots", no_robots_examples))
    return tuple(splits)


def _learning_gate_payload(
    *, status: str, best_arm: dict[str, Any] | None, evidence_uri: str
) -> dict[str, Any]:
    if best_arm is None:
        return {
            "status": "fail",
            "kind": "bounded_matched_interval_eval_sweep",
            "eval_metric": "eval/matched/response_loss",
            "baseline_step": 0,
            "evidence_uri": evidence_uri,
            "blocker": (
                "matched held-out response loss did not move down versus step 0 for any "
                "completed sane sweep arm"
            ),
        }
    return {
        "status": "pass" if status == "interval_eval_sweep_passed" else "fail",
        "bounded_matched_interval_eval_sweep": {
            "kind": "bounded_matched_interval_eval_sweep",
            "eval_metric": "eval/matched/response_loss",
            "baseline_step": 0,
            "step0_response_loss": best_arm["step0_eval"]["eval/matched/response_loss"],
            "best_response_loss": best_arm["best_eval"]["eval/matched/response_loss"],
            "best_arm_id": best_arm["arm_id"],
            "best_learning_rate": best_arm["arm"]["learning_rate"],
            "interval_eval_steps": best_arm["interval_eval_steps"],
            "evidence_uri": evidence_uri,
        },
    }
