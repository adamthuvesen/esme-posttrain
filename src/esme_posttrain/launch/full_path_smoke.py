"""Validated inputs for the no-spend full post-training CPU smoke."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.bundle import (
    BundleError,
    ValidatedBundle,
    file_sha256,
    validate_bundle_contract,
)
from esme_posttrain.dpo.launch import DPOLaunchConfig, load_dpo_config
from esme_posttrain.dpo.smoke import run_dpo_cpu_fixture
from esme_posttrain.export.dense_bundle import ExportRequest, export_dense_bundle
from esme_posttrain.launch.config_guards import (
    LaunchError,
    load_json_object,
    object_field,
    positive_int,
    require_keys,
    str_field,
)
from esme_posttrain.rl.countdown_lite_baseline import (
    CountdownBaselineRequest,
    run_countdown_lite_baseline,
)
from esme_posttrain.rl.launch import RLVRLaunchConfig, load_rlvr_config
from esme_posttrain.rl.pipeline_smoke import run_rlvr_pipeline_smoke
from esme_posttrain.sft.launch_multiturn import MultiTurnLaunchConfig, load_multi_turn_config
from esme_posttrain.sft.smoke_multiturn import run_multi_turn_cpu_fixture
from esme_posttrain.training.checkpointing import load_training_checkpoint

RUN_ID = "full_path_cpu_smoke"
FIXTURE_STEPS = 2
FIXTURE_INTERRUPT_AFTER_STEP = 1
FIXTURE_EVAL_BUDGET = 1
FIXTURE_RLVR_TOKEN_BUDGET = 512


@dataclass(frozen=True)
class FullPathSmokeConfig:
    config_path: Path
    repo_root: Path
    base_bundle_path: Path
    sft_config: MultiTurnLaunchConfig
    dpo_config: DPOLaunchConfig
    rlvr_config: RLVRLaunchConfig
    task_manifest_path: Path
    sft_steps: int
    dpo_steps: int
    dpo_interrupt_after_step: int
    rlvr_steps: int
    rlvr_max_rollout_tokens: int
    eval_task_budget: int
    samples_per_task: int


def run_full_path_cpu_smoke(
    config: FullPathSmokeConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the six-step local stage chain and record every artifact handoff."""
    started = time.monotonic()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LaunchError(f"full-path output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_checked = validate_bundle_contract(config.base_bundle_path)
    sft_dir = output_dir / "sft"
    sft = run_multi_turn_cpu_fixture(
        config.sft_config,
        output_dir=sft_dir,
        input_bundle_path=config.base_bundle_path,
        max_steps=config.sft_steps,
    )
    sft_checkpoint = sft_dir / "best-checkpoint.pt"

    dpo_resumed_dir = output_dir / "dpo-resumed"
    dpo_resumed = run_dpo_cpu_fixture(
        config.dpo_config,
        output_dir=dpo_resumed_dir,
        reference_checkpoint_path=sft_checkpoint,
        max_steps=config.dpo_steps,
        interrupt_after_step=config.dpo_interrupt_after_step,
    )
    dpo_control_dir = output_dir / "dpo-uninterrupted"
    run_dpo_cpu_fixture(
        config.dpo_config,
        output_dir=dpo_control_dir,
        reference_checkpoint_path=sft_checkpoint,
        max_steps=config.dpo_steps,
    )
    _require_dpo_resume_equivalence(dpo_resumed_dir, dpo_control_dir)

    dpo_bundle_dir = output_dir / "dpo-bundle"
    export_dense_bundle(
        ExportRequest(
            artifact_dir=dpo_resumed_dir,
            output_dir=dpo_bundle_dir,
            model_id="full-path-dpo-fixture",
            source_volume="local-cpu",
            source_path="dpo-resumed",
            wandb_run="disabled",
            dpo_step=config.dpo_steps,
            max_new_tokens=2,
        )
    )
    dpo_bundle = validate_bundle_contract(dpo_bundle_dir)

    rlvr_config = _write_and_load_rlvr_config(config, output_dir, dpo_bundle_dir)
    rlvr_dir = output_dir / "rlvr"
    rlvr = run_rlvr_pipeline_smoke(
        rlvr_config,
        output_dir=rlvr_dir,
        report_path=output_dir / "rlvr-report.json",
        doc_path=output_dir / "rlvr-report.md",
        repo_root=config.repo_root,
        emit_milestones_to_stdout=False,
    )
    trainer = rlvr.get("trainer")
    if not isinstance(trainer, dict) or trainer.get("steps_completed") != config.rlvr_steps:
        raise LaunchError("RLVR full-path smoke did not complete the declared two steps")
    final_bundle_value = trainer.get("bundle_final_dir")
    if not isinstance(final_bundle_value, str):
        raise LaunchError("RLVR full-path smoke did not produce a final-step bundle")
    final_bundle_dir = Path(final_bundle_value).resolve()
    final_bundle = validate_bundle_contract(final_bundle_dir)

    downstream_dir = output_dir / "downstream-score"
    downstream = run_countdown_lite_baseline(
        CountdownBaselineRequest(
            manifest_path=config.task_manifest_path,
            bundle_path=final_bundle_dir,
            output_dir=downstream_dir,
            split="eval",
            samples_per_task=config.samples_per_task,
            max_tasks=config.eval_task_budget,
            max_new_tokens=2,
            seed=int(config.rlvr_config.grpo["seed"]),
            device="cpu",
            eval_profile="full_path_cpu_smoke",
            config_hash=file_sha256(final_bundle.config_path),
            model_id=str(final_bundle.manifest["model"]["id"]),
        )
    )
    tasks = downstream.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise LaunchError("downstream score must contain exactly one typed task result")

    report = {
        "schema_version": 1,
        "status": "full_path_cpu_smoke_passed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "paid_compute": False,
        "remote_dataset_download": False,
        "wandb_enabled": False,
        "steps": {
            "sft": int(sft["result"]["steps_completed"]),
            "dpo": int(dpo_resumed["result"]["steps_completed"]),
            "rlvr": int(trainer["steps_completed"]),
        },
        "dpo_resume": {
            "interrupted_after_step": dpo_resumed["interrupted_after_step"],
            "resume_checkpoint": dpo_resumed["resume_checkpoint"],
            "equivalent_to_uninterrupted": True,
        },
        "bundles": {
            "base": _checked_bundle_payload(base_checked),
            "dpo": _checked_bundle_payload(dpo_bundle),
            "rlvr_final": _checked_bundle_payload(final_bundle),
        },
        "lineage": {
            "sft_parent_manifest": str(sft_dir / "manifest.json"),
            "dpo_parent_manifest": str(dpo_resumed_dir / "manifest.json"),
            "rlvr_parent_manifest": str(dpo_bundle_dir / "manifest.json"),
        },
        "downstream_score": tasks[0],
    }
    report_path = output_dir / "full-path-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report_path": str(report_path), "output_dir": str(output_dir)}


def load_full_path_smoke_config(
    config_path: Path, *, repo_root: Path | None = None
) -> FullPathSmokeConfig:
    config_path, payload = load_json_object(config_path)
    root = (repo_root or _find_repo_root(config_path)).expanduser().resolve()
    if not root.is_dir():
        raise LaunchError(f"repo_root does not exist: {root}")
    require_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "base_bundle_path",
            "sft",
            "dpo",
            "rlvr",
            "evaluation",
            "execution",
        },
        "full_path_cpu_smoke",
    )
    if payload["schema_version"] != 1:
        raise LaunchError("full_path_cpu_smoke.schema_version must be 1")
    if payload["run_id"] != RUN_ID:
        raise LaunchError(f"full_path_cpu_smoke.run_id must be {RUN_ID}")

    config_dir = config_path.parent
    base_bundle_path = _existing_repo_path(
        payload["base_bundle_path"], config_dir, root, "base_bundle_path", directory=True
    )
    try:
        validate_bundle_contract(base_bundle_path)
    except BundleError as error:
        raise LaunchError(f"base_bundle_path is invalid: {error}") from error
    sft_payload = object_field(payload["sft"], "sft")
    require_keys(sft_payload, {"config_path", "max_steps"}, "sft")
    sft_steps = _two_steps(sft_payload["max_steps"], "sft.max_steps")
    sft_config_path = _existing_repo_path(
        sft_payload["config_path"], config_dir, root, "sft.config_path"
    )

    dpo_payload = object_field(payload["dpo"], "dpo")
    require_keys(
        dpo_payload,
        {"config_path", "max_steps", "interrupt_after_step"},
        "dpo",
    )
    dpo_steps = _two_steps(dpo_payload["max_steps"], "dpo.max_steps")
    interrupt_after_step = positive_int(
        dpo_payload["interrupt_after_step"], "dpo.interrupt_after_step"
    )
    if interrupt_after_step >= dpo_steps:
        raise LaunchError("dpo.interrupt_after_step must be less than dpo.max_steps")
    if interrupt_after_step != FIXTURE_INTERRUPT_AFTER_STEP:
        raise LaunchError(
            f"dpo.interrupt_after_step must be {FIXTURE_INTERRUPT_AFTER_STEP} for the fixture"
        )
    dpo_config_path = _existing_repo_path(
        dpo_payload["config_path"], config_dir, root, "dpo.config_path"
    )

    rlvr_payload = object_field(payload["rlvr"], "rlvr")
    require_keys(rlvr_payload, {"config_path", "max_steps", "max_rollout_tokens"}, "rlvr")
    rlvr_steps = _two_steps(rlvr_payload["max_steps"], "rlvr.max_steps")
    rlvr_max_rollout_tokens = positive_int(
        rlvr_payload["max_rollout_tokens"], "rlvr.max_rollout_tokens"
    )
    if rlvr_max_rollout_tokens != FIXTURE_RLVR_TOKEN_BUDGET:
        raise LaunchError(
            f"rlvr.max_rollout_tokens must be {FIXTURE_RLVR_TOKEN_BUDGET} for the fixture"
        )
    rlvr_config_path = _existing_repo_path(
        rlvr_payload["config_path"], config_dir, root, "rlvr.config_path"
    )

    evaluation = object_field(payload["evaluation"], "evaluation")
    require_keys(
        evaluation,
        {"manifest_path", "task_budget", "samples_per_task"},
        "evaluation",
    )
    task_manifest_path = _existing_repo_path(
        evaluation["manifest_path"], config_dir, root, "evaluation.manifest_path"
    )
    eval_task_budget = positive_int(evaluation["task_budget"], "evaluation.task_budget")
    samples_per_task = positive_int(evaluation["samples_per_task"], "evaluation.samples_per_task")
    if eval_task_budget != FIXTURE_EVAL_BUDGET or samples_per_task != FIXTURE_EVAL_BUDGET:
        raise LaunchError("full-path fixture evaluation budgets must both be 1")

    execution = object_field(payload["execution"], "execution")
    require_keys(
        execution,
        {"paid_compute", "remote_dataset_download", "wandb_enabled"},
        "execution",
    )
    for key in ("paid_compute", "remote_dataset_download", "wandb_enabled"):
        if execution[key] is not False:
            raise LaunchError(f"execution.{key} must be false")

    sft_config = load_multi_turn_config(sft_config_path)
    dpo_config = load_dpo_config(dpo_config_path)
    rlvr_config = load_rlvr_config(rlvr_config_path)
    if task_manifest_path != rlvr_config.dataset_manifest_path:
        raise LaunchError("evaluation.manifest_path must match the validated RLVR dataset manifest")
    return FullPathSmokeConfig(
        config_path=config_path,
        repo_root=root,
        base_bundle_path=base_bundle_path,
        sft_config=sft_config,
        dpo_config=dpo_config,
        rlvr_config=rlvr_config,
        task_manifest_path=task_manifest_path,
        sft_steps=sft_steps,
        dpo_steps=dpo_steps,
        dpo_interrupt_after_step=interrupt_after_step,
        rlvr_steps=rlvr_steps,
        rlvr_max_rollout_tokens=rlvr_max_rollout_tokens,
        eval_task_budget=eval_task_budget,
        samples_per_task=samples_per_task,
    )


def _write_and_load_rlvr_config(
    config: FullPathSmokeConfig,
    output_dir: Path,
    input_bundle_path: Path,
) -> RLVRLaunchConfig:
    payload = deepcopy(config.rlvr_config.payload)
    payload["input_bundle"]["path"] = str(input_bundle_path)
    payload["dataset"]["manifest_path"] = str(config.task_manifest_path)
    payload["grpo"]["max_steps"] = config.rlvr_steps
    payload["budgets"]["max_rollout_tokens"] = config.rlvr_max_rollout_tokens
    payload["pipeline_smoke"]["max_steps"] = config.rlvr_steps
    payload["pipeline_smoke"]["max_rollout_tokens"] = config.rlvr_max_rollout_tokens
    payload["pipeline_smoke"]["output_dir"] = "runs/full-path-cpu-smoke-rlvr"
    payload["pipeline_smoke"]["report_path"] = "artifacts/full-path-cpu-smoke-rlvr.json"
    payload["pipeline_smoke"]["doc_path"] = "docs/full-path-cpu-smoke-rlvr.md"
    config_path = output_dir / "rlvr-config.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loaded = load_rlvr_config(config_path)
    if loaded.pipeline_smoke["max_steps"] != config.rlvr_steps:
        raise LaunchError("revalidated RLVR smoke config did not preserve the step budget")
    return loaded


def _require_dpo_resume_equivalence(resumed_dir: Path, control_dir: Path) -> None:
    resumed = load_training_checkpoint(resumed_dir / "checkpoint.pt")
    control = load_training_checkpoint(control_dir / "checkpoint.pt")
    if resumed.step != control.step or resumed.data_position != control.data_position:
        raise LaunchError("resumed DPO checkpoint position differs from uninterrupted control")
    _require_nested_equal(resumed.model.state_dict(), control.model.state_dict(), "model")
    _require_nested_equal(resumed.optimizer_state, control.optimizer_state, "optimizer")
    _require_nested_equal(resumed.scheduler_state, control.scheduler_state, "scheduler")


def _require_nested_equal(left: Any, right: Any, label: str) -> None:
    if type(left) is not type(right):
        raise LaunchError(f"resumed DPO {label} state type differs from control")
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise LaunchError(f"resumed DPO {label} tensor differs from control")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise LaunchError(f"resumed DPO {label} state keys differ from control")
        for key in left:
            _require_nested_equal(left[key], right[key], f"{label}.{key}")
    elif isinstance(left, list | tuple):
        if len(left) != len(right):
            raise LaunchError(f"resumed DPO {label} state length differs from control")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _require_nested_equal(left_item, right_item, f"{label}[{index}]")
    elif left != right:
        raise LaunchError(f"resumed DPO {label} state differs from control")


def _checked_bundle_payload(bundle: ValidatedBundle) -> dict[str, Any]:
    return {
        "bundle_dir": str(bundle.bundle_dir),
        "manifest_path": str(bundle.manifest_path),
        "schema_version": bundle.manifest["schema_version"],
        "format": bundle.manifest["format"],
        "file_set": sorted(path.name for path in bundle.bundle_dir.iterdir()),
        "files": {
            "config": str(bundle.config_path),
            "tokenizer": str(bundle.tokenizer_path),
            "weights": str(bundle.weights_path),
        },
    }


def _two_steps(value: Any, label: str) -> int:
    steps = positive_int(value, label)
    if steps != FIXTURE_STEPS:
        raise LaunchError(f"{label} must be {FIXTURE_STEPS} for the fixture")
    return steps


def _existing_repo_path(
    value: Any,
    config_dir: Path,
    repo_root: Path,
    label: str,
    *,
    directory: bool = False,
) -> Path:
    raw_path = Path(str_field(value, label))
    path = (config_dir / raw_path).resolve()
    if not path.is_relative_to(repo_root):
        raise LaunchError(f"{label} escapes the repository: {path}")
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise LaunchError(f"{label} {kind} does not exist: {path}")
    return path


def _find_repo_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise LaunchError(
        f"could not find repository root above {config_path}; pass repo_root explicitly"
    )
