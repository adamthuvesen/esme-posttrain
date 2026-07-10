"""Spend-guarded Countdown-Lite GRPO run body."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.bundle import file_sha256, load_dense_backbone_bundle
from esme_posttrain.rl.countdown_lite import load_countdown_lite_rows
from esme_posttrain.rl.countdown_lite_baseline import (
    PASS_AT_KS,
    CountdownBaselineRequest,
    run_countdown_lite_baseline,
)
from esme_posttrain.rl.grpo import CountdownGRPOTrainerConfig, run_countdown_lite_grpo
from esme_posttrain.rl.launch import (
    FULL_EVAL_PROFILE,
    PIPELINE_SMOKE_PROFILE,
    RLVRLaunchConfig,
    build_eval_profile,
    pipeline_smoke_grpo_settings,
)
from esme_posttrain.run_artifacts import (
    RuntimeSpendTracker,
    refresh_manifest_files,
    write_environment,
)
from esme_posttrain.training.wandb_init import WandbConfig, start_wandb

_WANDB_LIFECYCLE_INDEX_BY_RUN: dict[int, int] = {}


class CountdownGRPOFullRunError(RuntimeError):
    pass


def run_countdown_lite_grpo_job(
    config: RLVRLaunchConfig,
    *,
    output_dir: Path,
    input_bundle_path: Path | None = None,
    manifest_path: Path | None = None,
    require_cuda: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
    milestone_callback: Callable[[str, dict[str, Any]], None] | None = None,
    wandb_enabled: bool = False,
    wandb_mode: str | None = None,
    before_eval_only: bool = False,
    pipeline_smoke: bool = False,
    paid_compute: bool = True,
    skip_acceptance_eval: bool = False,
) -> dict[str, Any]:
    if skip_acceptance_eval and (before_eval_only or pipeline_smoke):
        raise CountdownGRPOFullRunError(
            "skip_acceptance_eval is incompatible with before_eval_only / pipeline_smoke"
        )
    run_profile = PIPELINE_SMOKE_PROFILE if pipeline_smoke else None
    _emit_milestone(
        "job_start",
        milestone_callback,
        output_dir=str(output_dir),
        run_profile=run_profile,
        pipeline_smoke=pipeline_smoke,
    )
    wandb_run: Any | None = None
    output_dir = output_dir.expanduser().resolve()
    try:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise CountdownGRPOFullRunError(
                f"GRPO output_dir must be empty or absent: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        _emit_milestone("output_dir_ready", milestone_callback, output_dir=str(output_dir))

        if config.estimated_full_cost_usd > float(config.runtime["full_run_max_cost_usd"]):
            raise CountdownGRPOFullRunError(
                "projected GRPO cost exceeds runtime.full_run_max_cost_usd"
            )
        device = _select_device(require_cuda=require_cuda)
        _emit_milestone("cuda_selected", milestone_callback, device=device.type)
        profile = config.selected_gpu_profile
        spend_tracker = RuntimeSpendTracker(
            started=started or time.perf_counter(),
            usd_per_hour=float(profile["usd_per_hour"]) if paid_compute else 0.0,
            stop_usd=float(config.runtime["full_run_runtime_spend_stop_usd"]),
            output_dir=output_dir,
            paid_compute=paid_compute,
        )
        check_spend = partial(
            spend_tracker.check_cap,
            label="GRPO",
            error_type=CountdownGRPOFullRunError,
        )

        bundle_path = (input_bundle_path or config.input_bundle_path).expanduser().resolve()
        dataset_manifest = (manifest_path or config.dataset_manifest_path).expanduser().resolve()
        _emit_milestone(
            "bundle_load_start",
            milestone_callback,
            bundle_path=str(bundle_path),
            dataset_manifest=str(dataset_manifest),
        )
        loaded = load_dense_backbone_bundle(bundle_path, map_location="cpu")
        config_hash = _config_hash(config)
        model_id = _model_id(loaded.bundle.manifest, fallback=config.artifact_name)
        wandb_run = _start_rlvr_wandb(
            config,
            loaded.bundle.manifest,
            enabled=wandb_enabled,
            mode=wandb_mode,
            before_eval_only=before_eval_only,
            pipeline_smoke=pipeline_smoke,
        )
        _configure_rlvr_wandb_metrics(wandb_run)
        _emit_milestone(
            "bundle_loaded",
            milestone_callback,
            wandb_run=wandb_run,
            wandb_enabled=wandb_run is not None,
            wandb_mode=wandb_mode,
        )
        policy = loaded.model
        reference = copy.deepcopy(policy)

        dataset = config.payload["dataset"]
        budgets = config.budgets
        required_artifacts = tuple(config.payload["artifacts"]["required_files"])
        if skip_acceptance_eval:
            # The placebo decomposition run needs only the trained bundle; its completions
            # come from the emitter and checkpoint selection uses train/reward_mean, not the
            # acceptance eval. Drop the eval-derived artifacts from the required set.
            eval_artifacts = {"eval-before.json", "eval-after.json", "eval-after-final.json"}
            required_artifacts = tuple(
                name for name in required_artifacts if name not in eval_artifacts
            )
        final_eval_required = "eval-after-final.json" in required_artifacts
        precision = str(config.runtime["precision"])
        if pipeline_smoke and device.type != "cuda" and precision == "bf16":
            # The no-spend lifecycle gate runs on CPU where bf16 autocast is
            # unsupported; downgrade loudly instead of refusing the smoke.
            precision = "fp32"
            _emit_milestone(
                "pipeline_smoke_precision_downgraded",
                milestone_callback,
                wandb_run=wandb_run,
                configured_precision="bf16",
                effective_precision=precision,
            )
        grpo_settings = pipeline_smoke_grpo_settings(config) if pipeline_smoke else config.grpo
        train_task_budget = (
            int(config.pipeline_smoke["train_task_budget"])
            if pipeline_smoke
            else int(budgets["train_task_budget"])
        )
        monitoring = config.payload["monitoring"]
        train_rows = tuple(
            list(load_countdown_lite_rows(dataset_manifest, split=str(dataset["train_split"])))[
                :train_task_budget
            ]
        )
        _emit_milestone(
            "data_loaded",
            milestone_callback,
            wandb_run=wandb_run,
            train_rows=len(train_rows),
        )
        eval_settings = build_eval_profile(
            config, before_eval_only=before_eval_only, pipeline_smoke=pipeline_smoke
        )

        before: dict[str, Any] | None = None
        if skip_acceptance_eval:
            _emit_milestone(
                "before_eval_skipped",
                milestone_callback,
                wandb_run=wandb_run,
                reason="skip_acceptance_eval",
            )
        else:
            _emit_milestone(
                "before_eval_start",
                milestone_callback,
                wandb_run=wandb_run,
                eval_tasks=eval_settings["tasks"],
                samples_per_task=eval_settings["samples_per_task"],
                eval_profile=eval_settings["profile"],
                before_eval_only=before_eval_only,
                pipeline_smoke=pipeline_smoke,
            )
            before = _run_progress_reporting_eval(
                config,
                manifest_path=dataset_manifest,
                bundle_path=bundle_path,
                output_dir=output_dir / "eval-before",
                device=device.type,
                label="before_eval",
                settings=eval_settings,
                milestone_callback=milestone_callback,
                wandb_run=wandb_run,
                config_hash=config_hash,
                model_id=model_id,
            )
            shutil.copy2(
                output_dir / "eval-before" / "baseline-report.json",
                output_dir / "eval-before.json",
            )
            _log_eval_summary(wandb_run, "before", before, step=0)
            _emit_milestone("before_eval_complete", milestone_callback, wandb_run=wandb_run)
        check_spend(0)

        if before_eval_only:
            cost = spend_tracker.write_cost(step=0, status="before_eval_probe_complete")
            write_environment(output_dir / "environment.txt", device=device)
            _emit_milestone("return_serialization", milestone_callback, wandb_run=wandb_run)
            return {
                "status": "modal_rlvr_before_eval_probe_complete",
                "run_id": config.run_id,
                "artifact_name": config.artifact_name,
                "eval_profile": eval_settings["profile"],
                "output_dir": str(output_dir),
                "bundle_dir": None,
                "volume": config.runtime["modal_volume"],
                "commit": commit,
                "dirty": dirty,
                "device": device.type,
                "paid_compute": paid_compute,
                "cost": cost,
                "projected_cost_usd": config.estimated_full_cost_usd,
                "grpo_result": "before-eval-probe",
                "trainer": None,
                "before": _selected_eval_metrics(before, eval_profile=eval_settings["profile"]),
                "after": None,
                "before_report": str(output_dir / "eval-before.json"),
                "after_report": None,
                "wandb_run": _wandb_url(wandb_run),
                "gsm8k_lite": {
                    "status": "not_run",
                    "reason": "before-eval debug probe does not run transfer eval",
                },
                "required_artifacts_present": {
                    "eval-before.json": (output_dir / "eval-before.json").is_file(),
                    "cost.json": (output_dir / "cost.json").is_file(),
                    "environment.txt": (output_dir / "environment.txt").is_file(),
                },
            }

        _emit_milestone(
            "trainer_start",
            milestone_callback,
            wandb_run=wandb_run,
            max_steps=int(grpo_settings["max_steps"]),
            prompts_per_step=int(grpo_settings["prompts_per_step"]),
            group_size=int(grpo_settings["group_size"]),
            eval_profile=eval_settings["profile"],
            pipeline_smoke=pipeline_smoke,
        )
        result = run_countdown_lite_grpo(
            policy,
            reference,
            loaded.tokenizer,
            train_rows,
            CountdownGRPOTrainerConfig(
                max_steps=int(grpo_settings["max_steps"]),
                prompts_per_step=int(grpo_settings["prompts_per_step"]),
                group_size=int(grpo_settings["group_size"]),
                max_new_tokens=int(grpo_settings["max_new_tokens"]),
                temperature=float(grpo_settings["temperature"]),
                kl_beta=float(grpo_settings["kl_beta"]),
                learning_rate=float(grpo_settings["learning_rate"]),
                weight_decay=float(grpo_settings["weight_decay"]),
                warmup_steps=int(grpo_settings["warmup_steps"]),
                scheduler=str(grpo_settings["scheduler"]),
                grad_clip=float(grpo_settings["grad_clip"]),
                seed=int(grpo_settings["seed"]),
                output_dir=output_dir,
                max_rollout_tokens=int(grpo_settings["max_rollout_tokens"])
                if pipeline_smoke
                else int(budgets["max_rollout_tokens"]),
                exact_solve_reward=float(config.payload["reward_policy"]["exact_solve_reward"]),
                valid_expression_reward=float(
                    config.payload["reward_policy"]["valid_expression_reward"]
                ),
                invalid_reward=float(config.payload["reward_policy"]["invalid_reward"]),
                format_expression_reward=float(
                    config.payload["reward_policy"].get("format_expression_reward", 0.0)
                ),
                closeness_weight=float(
                    config.payload["reward_policy"].get("closeness_weight", 0.0)
                ),
                reward_mode=str(grpo_settings.get("reward_mode", "verifier")),
                random_reward_seed=int(grpo_settings.get("random_reward_seed", 0)),
                zero_variance_max_resamples=int(
                    grpo_settings.get("zero_variance_max_resamples", 0)
                ),
                replay_buffer_max_age_steps=int(
                    grpo_settings.get("replay_buffer_max_age_steps", 0)
                ),
                stratified_difficulty_sampling=bool(
                    grpo_settings.get("stratified_difficulty_sampling", False)
                ),
                write_final_bundle=final_eval_required,
                precision=precision,
                device=device.type,
                log_interval=int(monitoring["log_interval"]),
                checkpoint_interval=int(monitoring["checkpoint_interval"]),
                artifact_name=config.artifact_name,
                reference_artifact_name=str(config.payload["starts_from"]),
                source_manifest=loaded.bundle.manifest,
                source_checkpoint=str(loaded.bundle.weights_path),
                source_checkpoint_sha256=file_sha256(loaded.bundle.weights_path),
            ),
            step_callback=check_spend,
            wandb_run=_RLVRWandbTrainLogger(wandb_run) if wandb_run is not None else None,
        )
        _emit_milestone(
            "trainer_complete",
            milestone_callback,
            wandb_run=wandb_run,
            steps_completed=result.steps_completed,
        )
        check_spend(result.steps_completed)

        after: dict[str, Any] | None = None
        if skip_acceptance_eval:
            _emit_milestone(
                "after_eval_skipped",
                milestone_callback,
                wandb_run=wandb_run,
                reason="skip_acceptance_eval",
            )
        else:
            _emit_milestone(
                "after_eval_start",
                milestone_callback,
                wandb_run=wandb_run,
                eval_tasks=eval_settings["tasks"],
                samples_per_task=eval_settings["samples_per_task"],
                eval_profile=eval_settings["profile"],
                pipeline_smoke=pipeline_smoke,
            )
            after = _run_progress_reporting_eval(
                config,
                manifest_path=dataset_manifest,
                bundle_path=result.bundle_dir,
                output_dir=output_dir / "eval-after",
                device=device.type,
                label="after_eval",
                settings=eval_settings,
                milestone_callback=milestone_callback,
                wandb_run=wandb_run,
                config_hash=config_hash,
                model_id=model_id,
            )
            shutil.copy2(
                output_dir / "eval-after" / "baseline-report.json",
                output_dir / "eval-after.json",
            )
            _log_eval_summary(wandb_run, "after", after, step=result.steps_completed)
            _emit_milestone("after_eval_complete", milestone_callback, wandb_run=wandb_run)

        after_final = None
        if final_eval_required:
            if result.bundle_final_dir is None:
                raise CountdownGRPOFullRunError(
                    "eval-after-final.json is required but the trainer wrote no final bundle"
                )
            _emit_milestone(
                "after_final_eval_start",
                milestone_callback,
                wandb_run=wandb_run,
                eval_tasks=eval_settings["tasks"],
                samples_per_task=eval_settings["samples_per_task"],
                eval_profile=eval_settings["profile"],
            )
            after_final = _run_progress_reporting_eval(
                config,
                manifest_path=dataset_manifest,
                bundle_path=Path(result.bundle_final_dir),
                output_dir=output_dir / "eval-after-final",
                device=device.type,
                label="after_final_eval",
                settings=eval_settings,
                milestone_callback=milestone_callback,
                wandb_run=wandb_run,
                config_hash=config_hash,
                model_id=model_id,
            )
            shutil.copy2(
                output_dir / "eval-after-final" / "baseline-report.json",
                output_dir / "eval-after-final.json",
            )
            _log_eval_summary(wandb_run, "after_final", after_final, step=result.steps_completed)
            _emit_milestone("after_final_eval_complete", milestone_callback, wandb_run=wandb_run)

        cost = spend_tracker.write_cost(step=result.steps_completed, status="complete")
        if float(cost["estimated_cost_usd"]) > float(config.runtime["full_run_max_cost_usd"]):
            raise CountdownGRPOFullRunError("GRPO run exceeded runtime.full_run_max_cost_usd")

        write_environment(output_dir / "environment.txt", device=device)
        refresh_manifest_files(output_dir, required_artifacts)
        _assert_required_artifacts(output_dir, required_artifacts)
        _emit_milestone("return_serialization", milestone_callback, wandb_run=wandb_run)
        return {
            "status": "pipeline_smoke_complete"
            if pipeline_smoke
            else "modal_full_countdown_lite_grpo_complete",
            "run_id": config.run_id,
            "artifact_name": config.artifact_name,
            "eval_profile": eval_settings["profile"],
            "output_dir": str(output_dir),
            "bundle_dir": str(result.bundle_dir),
            "volume": config.runtime["modal_volume"],
            "commit": commit,
            "dirty": dirty,
            "device": device.type,
            "paid_compute": paid_compute,
            "cost": cost,
            "projected_cost_usd": config.estimated_full_cost_usd,
            "grpo_result": _grpo_result(
                pipeline_smoke=pipeline_smoke,
                skip_acceptance_eval=skip_acceptance_eval,
                before=before,
                after=after,
                eval_profile=str(eval_settings["profile"]),
            ),
            "skip_acceptance_eval": skip_acceptance_eval,
            "trainer": result.to_dict(),
            "before": (
                _selected_eval_metrics(before, eval_profile=eval_settings["profile"])
                if before is not None
                else None
            ),
            "after": (
                _selected_eval_metrics(after, eval_profile=eval_settings["profile"])
                if after is not None
                else None
            ),
            "after_final": (
                _selected_eval_metrics(after_final, eval_profile=eval_settings["profile"])
                if after_final is not None
                else None
            ),
            "before_report": (str(output_dir / "eval-before.json") if before is not None else None),
            "after_report": (str(output_dir / "eval-after.json") if after is not None else None),
            "after_final_report": (
                str(output_dir / "eval-after-final.json") if after_final is not None else None
            ),
            "wandb_run": _wandb_url(wandb_run),
            "gsm8k_lite": {
                "status": "not_run",
                "reason": "no checked-in GSM8K-lite fixture or evaluator exists in esme-posttrain",
            },
            "required_artifacts_present": {
                name: (output_dir / name).is_file() for name in required_artifacts
            },
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_progress_reporting_eval(
    config: RLVRLaunchConfig,
    *,
    manifest_path: Path,
    bundle_path: Path,
    output_dir: Path,
    device: str,
    label: str,
    settings: dict[str, Any],
    milestone_callback: Callable[[str, dict[str, Any]], None] | None,
    wandb_run: Any | None,
    config_hash: str,
    model_id: str,
) -> dict[str, Any]:
    progress_path = output_dir.parent / "eval-progress.jsonl"

    def progress_callback(stage: str, fields: dict[str, Any]) -> None:
        _append_eval_progress(progress_path, stage, fields)
        _emit_milestone(stage, milestone_callback, wandb_run=wandb_run, **fields)

    return run_countdown_lite_baseline(
        CountdownBaselineRequest(
            manifest_path=manifest_path,
            bundle_path=bundle_path,
            output_dir=output_dir,
            split=str(config.payload["dataset"]["eval_split"]),
            samples_per_task=int(settings["samples_per_task"]),
            max_tasks=int(settings["tasks"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            seed=int(config.grpo["seed"]),
            device=device,
            progress_label=label,
            progress_callback=progress_callback,
            progress_interval_tasks=int(settings["progress_interval_tasks"]),
            progress_interval_samples=int(settings["progress_interval_samples"]),
            sample_batch_size=int(settings["sample_batch_size"]),
            wall_timeout_seconds=float(settings["wall_timeout_seconds"]),
            no_progress_timeout_seconds=float(settings["no_progress_timeout_seconds"]),
            resume_from_partial=True,
            eval_profile=str(settings["profile"]),
            config_hash=config_hash,
            model_id=model_id,
        )
    )


def _append_eval_progress(path: Path, stage: str, fields: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"event": "rlvr_eval_progress", "stage": stage, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _start_rlvr_wandb(
    config: RLVRLaunchConfig,
    base_bundle_manifest: dict[str, Any] | None,
    *,
    enabled: bool,
    mode: str | None,
    before_eval_only: bool,
    pipeline_smoke: bool,
) -> Any | None:
    monitoring = config.payload["monitoring"]
    if pipeline_smoke:
        eval_profile = build_eval_profile(config, pipeline_smoke=True)
        run_kind = PIPELINE_SMOKE_PROFILE
    elif before_eval_only:
        eval_profile = build_eval_profile(config, before_eval_only=True)
        run_kind = "before-eval-debug"
    else:
        eval_profile = build_eval_profile(config)
        run_kind = "full-grpo"
    wandb_config = WandbConfig(
        enabled=enabled,
        project=str(monitoring["wandb_project"]),
        run_name=f"{config.run_id}-{run_kind}",
        mode=mode,
        tags=tuple(monitoring["wandb_tags"]) + (run_kind, str(eval_profile["profile"])),
        group=config.run_id,
        job_type=run_kind,
        notes="Countdown-Lite RLVR GRPO with verifiable rewards only.",
        extra_config={
            "model": config.artifact_name,
            "stage": "rlvr",
            "run_type": run_kind,
            "eval_profile": eval_profile["profile"],
            "selected_gpu": config.runtime["selected_gpu"],
            "eval_task_budget": eval_profile["tasks"],
            "samples_per_eval_task": eval_profile["samples_per_task"],
            "eval_rollouts": eval_profile["total_samples"],
            "debug_eval_task_budget": monitoring["debug_eval_task_budget"],
            "debug_samples_per_eval_task": monitoring["debug_samples_per_eval_task"],
        },
    )
    return start_wandb(
        wandb_config,
        run_config={
            "artifact_name": config.artifact_name,
            "max_steps": int(config.grpo["max_steps"]),
            "prompts_per_step": int(config.grpo["prompts_per_step"]),
            "group_size": int(config.grpo["group_size"]),
            "learning_rate": float(config.grpo["learning_rate"]),
            "scheduler": str(config.grpo["scheduler"]),
            "warmup_steps": int(config.grpo["warmup_steps"]),
            "weight_decay": float(config.grpo["weight_decay"]),
            "precision": str(config.runtime["precision"]),
            "seed": int(config.grpo["seed"]),
        },
        base_bundle_manifest=base_bundle_manifest,
    )


def _configure_rlvr_wandb_metrics(wandb_run: Any | None) -> None:
    define_metric = getattr(wandb_run, "define_metric", None)
    if not callable(define_metric):
        return
    define_metric("train/*", step_metric="train/step")
    define_metric("eval/*", step_metric="eval/step")
    define_metric("lifecycle/*", step_metric="lifecycle/index")


def _log_eval_summary(
    wandb_run: Any | None, label: str, report: dict[str, Any], *, step: int
) -> None:
    if wandb_run is None:
        return
    prefix = f"eval/{label}"
    wandb_run.log(
        {
            "event": "eval",
            "phase": label,
            "eval/step": step,
            **{
                f"{prefix}/{key}": report[key]
                for key in (f"pass@{k}" for k in PASS_AT_KS)
                if key in report
            },
            f"{prefix}/valid_expression_rate": report["valid_expression_rate"],
            f"{prefix}/exact_solve_rate": report["exact_solve_rate"],
            f"{prefix}/tasks": report["task_count"],
            f"{prefix}/samples_per_task": report["samples_per_task"],
        }
    )


def _wandb_url(wandb_run: Any | None) -> str | None:
    value = getattr(wandb_run, "url", None) if wandb_run is not None else None
    return str(value) if value else None


class _RLVRWandbTrainLogger:
    def __init__(self, wandb_run: Any) -> None:
        self._wandb_run = wandb_run

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        log_payload = dict(payload)
        if step is not None:
            log_payload.setdefault("train/step", step)
        elif isinstance(log_payload.get("step"), int):
            log_payload.setdefault("train/step", log_payload["step"])
        self._wandb_run.log(log_payload)


def _select_device(*, require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise CountdownGRPOFullRunError(
            "Modal GRPO requires CUDA, but torch.cuda.is_available() is false"
        )
    return torch.device("cpu")


def _selected_eval_metrics(report: dict[str, Any], *, eval_profile: str) -> dict[str, Any]:
    return {
        "eval_profile": eval_profile,
        **{key: report[key] for key in (f"pass@{k}" for k in PASS_AT_KS) if key in report},
        "valid_expression_rate": report["valid_expression_rate"],
        "exact_solve_rate": report["exact_solve_rate"],
        "task_count": report["task_count"],
        "samples_per_task": report["samples_per_task"],
        "difficulty_breakdown": report["difficulty_breakdown"],
        "json_path": report["json_path"],
        "markdown_path": report["markdown_path"],
        "partial_path": report.get("partial_path"),
    }


def _config_hash(config: RLVRLaunchConfig) -> str:
    payload = json.dumps(config.payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_id(manifest: dict[str, Any] | None, *, fallback: str) -> str:
    if isinstance(manifest, dict):
        value = manifest.get("model_id") or manifest.get("model")
        if isinstance(value, str) and value:
            return value
        provenance = manifest.get("provenance")
        if isinstance(provenance, dict):
            value = provenance.get("model_id") or provenance.get("model")
            if isinstance(value, str) and value:
                return value
    return fallback


def _grpo_result(
    *,
    pipeline_smoke: bool,
    skip_acceptance_eval: bool,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    eval_profile: str,
) -> str:
    if pipeline_smoke:
        return "pipeline_smoke_passed"
    if skip_acceptance_eval or before is None or after is None:
        # No acceptance eval ran, so there is no before/after to judge — the run is a
        # training-only artifact (the placebo decomposition arm), not acceptance evidence.
        return "training-only"
    return _result_label(before, after, eval_profile=eval_profile)


def _result_label(before: dict[str, Any], after: dict[str, Any], *, eval_profile: str) -> str:
    shared_pass_key = _largest_shared_pass_at_key(before, after)
    if eval_profile != FULL_EVAL_PROFILE or shared_pass_key is None:
        # A reduced eval profile (or missing pass@k evidence) cannot back an
        # "RLVR-improved" acceptance claim.
        return "not-acceptance-evidence"
    if float(after["valid_expression_rate"]) > float(before["valid_expression_rate"]):
        return "RLVR-improved"
    if float(after["exact_solve_rate"]) > float(before["exact_solve_rate"]):
        return "RLVR-improved"
    if float(after[shared_pass_key]) > float(before[shared_pass_key]):
        return "RLVR-improved"
    return "no-improvement"


def _largest_shared_pass_at_key(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    for k in sorted(PASS_AT_KS, reverse=True):
        key = f"pass@{k}"
        if key in before and key in after:
            return key
    return None


def _assert_required_artifacts(output_dir: Path, required_artifacts: tuple[str, ...]) -> None:
    missing = [name for name in required_artifacts if not (output_dir / name).is_file()]
    if missing:
        raise CountdownGRPOFullRunError("missing required GRPO artifacts: " + ", ".join(missing))


def _emit_milestone(
    stage: str,
    callback: Callable[[str, dict[str, Any]], None] | None,
    *,
    wandb_run: Any | None = None,
    **fields: Any,
) -> None:
    payload = {"event": "rlvr_modal_milestone", "stage": stage, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)
    if wandb_run is not None:
        run_key = id(wandb_run)
        lifecycle_index = _WANDB_LIFECYCLE_INDEX_BY_RUN.get(run_key, 0) + 1
        _WANDB_LIFECYCLE_INDEX_BY_RUN[run_key] = lifecycle_index
        wandb_run.log(
            {
                "event": "lifecycle",
                "lifecycle/index": lifecycle_index,
                "lifecycle/stage": stage,
                **fields,
            }
        )
    if callback is not None:
        callback(stage, fields)
