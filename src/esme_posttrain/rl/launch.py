"""Bounded launch validation for the Esme-214M-RL Countdown-Lite GRPO tranche."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.bundle import BundleError, validate_base_bundle
from esme_posttrain.launch.config_guards import (
    IMAGE_PACKAGE_PINS,
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    LaunchError,
    build_modal_launch_command,
    estimate_cost_usd,
    load_json_object,
    object_field,
    positive_int,
    require_keys,
    str_field,
    validate_modal_runtime,
)
from esme_posttrain.launch.config_guards import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.rl.countdown_lite import load_countdown_lite_rows, render_chat_prompt

RUN_ID = "esme_214m_rlvr_countdown_lite_grpo"
RUN_CARD = "run_cards/esme-214m-rl.md"
ARTIFACT_NAME = "Esme-214M-RL"
STAGE = "rlvr"
STARTS_FROM = "Esme-214M-Chat"
MANIFEST_TYPE = "rl_tasks"
COUNTDOWN_LITE_DATASET = "esme_214m_rl_countdown_lite"
METHOD = "grpo"
MODAL_VOLUME = "esme-posttrain-esme-rlvr-countdown"
FULL_RUN_SPEND_CAP_USD = 25.0
FULL_EVAL_PROFILE = "full_acceptance_30x32"
BEFORE_EVAL_DEBUG_PROFILE = "before_eval_debug"
PIPELINE_SMOKE_PROFILE = "pipeline_smoke"

REQUIRED_BLOCKED_ACTIONS = frozenset(
    {"dataset_download", "modal", "gpu", "paid_api", "full_training"}
)
DISALLOWED_REWARD_TERMS = frozenset({"style", "sharpness", "friendliness", "naturalness", "tone"})
VERIFIABLE_REWARD_TYPES = frozenset(
    {"exact_match", "numeric_match", "regex_match", "unit_test", "execution_check"}
)
EXPECTED_ARTIFACTS = (
    "config.json",
    "data-report.json",
    "rollouts.jsonl",
    "metrics.jsonl",
    "checkpoint.pt",
    "best-checkpoint.pt",
    "best-checkpoint.json",
    "tokenizer.json",
    "eval-before.json",
    "eval-after.json",
    "cost.json",
    "environment.txt",
    "manifest.json",
)


@dataclass(frozen=True)
class DatasetSummary:
    manifest_type: str
    name: str
    record_count: int
    split_counts: dict[str, int]
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class RLVRLaunchConfig:
    payload: dict[str, Any]
    config_path: Path
    input_bundle_path: Path
    dataset_manifest_path: Path
    dataset: DatasetSummary
    output_dir: Path
    report_path: Path
    doc_path: Path
    pipeline_smoke_output_dir: Path
    pipeline_smoke_report_path: Path
    pipeline_smoke_doc_path: Path
    estimated_full_cost_usd: float
    estimated_train_tokens: int
    full_launch_command: str

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def artifact_name(self) -> str:
        return str(self.payload["artifact_name"])

    @property
    def budgets(self) -> dict[str, Any]:
        return dict(self.payload["budgets"])

    @property
    def grpo(self) -> dict[str, Any]:
        return dict(self.payload["grpo"])

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.payload["runtime"])

    @property
    def pipeline_smoke(self) -> dict[str, Any]:
        return dict(self.payload["pipeline_smoke"])

    @property
    def selected_gpu_profile(self) -> dict[str, Any]:
        runtime = self.runtime
        return dict(runtime["gpu_profiles"][runtime["selected_gpu"]])


def load_rlvr_config(config_path: Path) -> RLVRLaunchConfig:
    config_path, payload = load_json_object(config_path)
    return validate_rlvr_payload(payload, config_path)


def build_rlvr_dry_run(config_path: Path) -> dict[str, object]:
    config = load_rlvr_config(config_path)
    return build_grpo_dry_run(config)


def build_grpo_dry_run(
    config: RLVRLaunchConfig,
    *,
    full_run_approved: bool = False,
    full_run_modal_gpu: str | None = None,
) -> dict[str, Any]:
    full_blockers = full_launch_blockers(
        config, approved=full_run_approved, modal_gpu=full_run_modal_gpu
    )
    status = "ready_for_grpo_launch" if full_run_approved and not full_blockers else "dry_run"
    full_eval = build_eval_profile(config)
    debug_eval = build_eval_profile(config, before_eval_only=True)
    smoke_eval = build_eval_profile(config, pipeline_smoke=True)
    return {
        "status": status,
        "command": "rlvr-dry-run",
        "run_id": config.run_id,
        "artifact_name": ARTIFACT_NAME,
        "stage": STAGE,
        "starts_from": STARTS_FROM,
        "method": METHOD,
        "input_bundle_path": str(config.input_bundle_path),
        "dataset_manifest_path": str(config.dataset_manifest_path),
        "dataset_name": config.dataset.name,
        "dataset_type": config.dataset.manifest_type,
        "dataset_records_declared": config.dataset.record_count,
        "dataset_split_counts": config.dataset.split_counts,
        "dataset_details": list(config.dataset.details),
        "sample_budget": config.payload["budgets"]["dataset_sample_budget"],
        "train_task_budget": config.payload["budgets"]["train_task_budget"],
        "rollout_group_size": config.payload["grpo"]["group_size"],
        "prompts_per_step": config.payload["grpo"]["prompts_per_step"],
        "max_steps": config.payload["grpo"]["max_steps"],
        "max_new_tokens": config.payload["grpo"]["max_new_tokens"],
        "token_budget": config.payload["budgets"]["max_rollout_tokens"],
        "estimated_train_tokens": config.estimated_train_tokens,
        "hardware": config.runtime["selected_gpu"],
        "modal_gpu": config.selected_gpu_profile["modal_gpu"],
        "expected_duration_minutes": _expected_duration_minutes(config),
        "projected_cost_usd": round(config.estimated_full_cost_usd, 4),
        "spend_cap_usd": FULL_RUN_SPEND_CAP_USD,
        "runtime_spend_stop_usd": config.runtime["full_run_runtime_spend_stop_usd"],
        "timeout_hours": config.runtime["timeout_hours"],
        "timeout_cost_ceiling_usd": round(_timeout_cost_ceiling_usd(config.runtime), 4),
        "eval_profile": full_eval["profile"],
        "eval_rollouts": full_eval["total_samples"],
        "eval_task_budget": full_eval["tasks"],
        "samples_per_eval_task": full_eval["samples_per_task"],
        "eval_progress_interval_tasks": config.payload["monitoring"][
            "eval_progress_interval_tasks"
        ],
        "eval_progress_interval_samples": config.payload["monitoring"][
            "eval_progress_interval_samples"
        ],
        "eval_sample_batch_size": config.payload["monitoring"]["eval_sample_batch_size"],
        "eval_wall_timeout_seconds": full_eval["wall_timeout_seconds"],
        "eval_no_progress_timeout_seconds": full_eval["no_progress_timeout_seconds"],
        "eval_timeout_basis": full_eval["timeout_basis"],
        "acceptance_preflight": _acceptance_preflight(config, full_eval),
        "debug_eval_task_budget": config.payload["monitoring"]["debug_eval_task_budget"],
        "debug_samples_per_eval_task": config.payload["monitoring"]["debug_samples_per_eval_task"],
        "debug_eval_profile": debug_eval["profile"],
        "debug_eval_rollouts": debug_eval["total_samples"],
        "pipeline_smoke_profile": smoke_eval["profile"],
        "pipeline_smoke_eval_rollouts": smoke_eval["total_samples"],
        "pipeline_smoke_command": _pipeline_smoke_command(config.config_path),
        "modal_smoke_command": _modal_pipeline_smoke_command(config.config_path, config.runtime),
        "wandb_project": config.payload["monitoring"]["wandb_project"],
        "wandb_required_for_modal": config.payload["monitoring"]["wandb_required_for_modal"],
        "wandb_tags": config.payload["monitoring"]["wandb_tags"],
        "output_dir": str(config.output_dir),
        "report_path": str(config.report_path),
        "doc_path": str(config.doc_path),
        "acceptance": config.payload["acceptance"],
        "approval_required": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "approval_state": "not approved for launch",
        "blocked_actions": list(config.payload["approval_gate"]["blocked_actions"]),
        "full_launch_blockers": full_blockers,
        "full_launch_command": config.full_launch_command,
        "resume_command": config.full_launch_command,
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION, **IMAGE_PACKAGE_PINS},
        "training_started": False,
        "modal_gpu_or_paid_work_started": False,
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def format_rlvr_dry_run(payload: dict[str, object]) -> str:
    blockers = payload.get("full_launch_blockers")
    blockers_text = ", ".join(str(item) for item in blockers) if isinstance(blockers, list) else ""
    return "\n".join(
        [
            f"dry_run: {payload.get('command')}",
            f"artifact: {payload.get('artifact_name')}",
            f"stage: {payload.get('stage')}",
            f"method: {payload.get('method')}",
            f"starts_from: {payload.get('starts_from')}",
            f"input_bundle: {payload.get('input_bundle_path')}",
            f"dataset_manifest: {payload.get('dataset_manifest_path')}",
            f"dataset_name: {payload.get('dataset_name')}",
            f"dataset_records_declared: {payload.get('dataset_records_declared')}",
            f"sample_budget: {payload.get('sample_budget')}",
            f"token_budget: {payload.get('token_budget')}",
            f"estimated_train_tokens: {payload.get('estimated_train_tokens')}",
            f"hardware: {payload.get('hardware')}",
            f"expected_duration_minutes: {payload.get('expected_duration_minutes')}",
            f"projected_cost_usd: {payload.get('projected_cost_usd')}",
            f"runtime_spend_stop_usd: {payload.get('runtime_spend_stop_usd')}",
            f"timeout_cost_ceiling_usd: {payload.get('timeout_cost_ceiling_usd')}",
            f"eval_profile: {payload.get('eval_profile')}",
            f"eval_rollouts: {payload.get('eval_rollouts')}",
            f"eval_wall_timeout_seconds: {payload.get('eval_wall_timeout_seconds')}",
            f"eval_no_progress_timeout_seconds: {payload.get('eval_no_progress_timeout_seconds')}",
            f"debug_eval_task_budget: {payload.get('debug_eval_task_budget')}",
            f"debug_samples_per_eval_task: {payload.get('debug_samples_per_eval_task')}",
            f"pipeline_smoke_profile: {payload.get('pipeline_smoke_profile')}",
            f"pipeline_smoke_command: {payload.get('pipeline_smoke_command')}",
            f"modal_smoke_command: {payload.get('modal_smoke_command')}",
            f"wandb_project: {payload.get('wandb_project')}",
            f"wandb_required_for_modal: {payload.get('wandb_required_for_modal')}",
            f"output_dir: {payload.get('output_dir')}",
            f"report_path: {payload.get('report_path')}",
            f"doc_path: {payload.get('doc_path')}",
            "approval_required: yes",
            "approval_state: not approved for launch",
            f"full_launch_blockers: {blockers_text or 'none'}",
            "training_started: no",
            "modal_gpu_or_paid_work_started: no",
            f"full_launch_command: {payload.get('full_launch_command')}",
        ]
    )


def validate_rlvr_payload(payload: dict[str, Any], config_path: Path) -> RLVRLaunchConfig:
    require_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "run_card",
            "requires_approval",
            "artifact_name",
            "stage",
            "starts_from",
            "input_bundle",
            "dataset",
            "approval_gate",
            "budgets",
            "grpo",
            "reward_policy",
            "runtime",
            "monitoring",
            "pipeline_smoke",
            "artifacts",
            "acceptance",
            "abort_rules",
        },
        "config",
    )
    if payload["schema_version"] != 1:
        raise LaunchError("schema_version must be 1")
    if payload["run_id"] != RUN_ID:
        raise LaunchError(f"run_id must be {RUN_ID}")
    if payload["run_card"] != RUN_CARD:
        raise LaunchError(f"run_card must be {RUN_CARD}")
    if payload["requires_approval"] is not True:
        raise LaunchError("requires_approval must be true")
    if payload["artifact_name"] != ARTIFACT_NAME:
        raise LaunchError(f"artifact_name must be {ARTIFACT_NAME}")
    if payload["stage"] != STAGE:
        raise LaunchError(f"stage must be {STAGE}")
    if payload["starts_from"] != STARTS_FROM:
        raise LaunchError(f"starts_from must be {STARTS_FROM}")

    config_dir = config_path.parent
    input_bundle_path = _validate_input_bundle(
        object_field(payload["input_bundle"], "input_bundle"), config_dir
    )
    dataset_manifest_path, dataset = _validate_dataset(
        object_field(payload["dataset"], "dataset"), config_dir
    )
    _validate_approval_gate(object_field(payload["approval_gate"], "approval_gate"))
    budgets = _validate_budgets(object_field(payload["budgets"], "budgets"), dataset)
    grpo = _validate_grpo(object_field(payload["grpo"], "grpo"))
    _validate_reward_policy(object_field(payload["reward_policy"], "reward_policy"))
    runtime = validate_modal_runtime(
        object_field(payload["runtime"], "runtime"),
        full_run_spend_cap_usd=FULL_RUN_SPEND_CAP_USD,
        full_run_cap_label=f"{FULL_RUN_SPEND_CAP_USD:.0f}",
        modal_volume=MODAL_VOLUME,
        require_smoke_profile_metrics=False,
    )
    _validate_timeout_cost_ceiling(runtime)
    _validate_monitoring(object_field(payload["monitoring"], "monitoring"))
    _validate_eval_monitoring_bounds(
        object_field(payload["monitoring"], "monitoring"),
        budgets,
    )
    pipeline_smoke_output_dir, pipeline_smoke_report_path, pipeline_smoke_doc_path = (
        _validate_pipeline_smoke(
            object_field(payload["pipeline_smoke"], "pipeline_smoke"),
            budgets=budgets,
            grpo=grpo,
            monitoring=object_field(payload["monitoring"], "monitoring"),
            config_dir=config_dir,
        )
    )
    output_dir, report_path, doc_path = _validate_artifacts(
        object_field(payload["artifacts"], "artifacts"), config_dir
    )
    _validate_acceptance(object_field(payload["acceptance"], "acceptance"))
    _validate_abort_rules(payload["abort_rules"])

    estimated_train_tokens = _estimate_train_tokens(
        input_bundle_path=input_bundle_path,
        manifest_path=dataset_manifest_path,
        train_split=str(payload["dataset"]["train_split"]),
        max_train_tasks=int(budgets["train_task_budget"]),
        prompts_per_step=int(grpo["prompts_per_step"]),
        group_size=int(grpo["group_size"]),
        max_steps=int(grpo["max_steps"]),
        max_new_tokens=int(grpo["max_new_tokens"]),
    )
    if estimated_train_tokens > int(budgets["max_rollout_tokens"]):
        raise LaunchError(
            "estimated GRPO rollout tokens exceed budgets.max_rollout_tokens: "
            f"{estimated_train_tokens} > {budgets['max_rollout_tokens']}"
        )

    selected_profile = runtime["gpu_profiles"][runtime["selected_gpu"]]
    estimated_full_cost = estimate_cost_usd(
        tokens=estimated_train_tokens,
        projected_tokens_per_second=float(selected_profile["projected_tokens_per_second"]),
        usd_per_hour=float(selected_profile["usd_per_hour"]),
    )
    if estimated_full_cost > float(runtime["full_run_max_cost_usd"]):
        raise LaunchError("projected GRPO cost exceeds runtime.full_run_max_cost_usd")

    return RLVRLaunchConfig(
        payload=payload,
        config_path=config_path,
        input_bundle_path=input_bundle_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset=dataset,
        output_dir=output_dir,
        report_path=report_path,
        doc_path=doc_path,
        pipeline_smoke_output_dir=pipeline_smoke_output_dir,
        pipeline_smoke_report_path=pipeline_smoke_report_path,
        pipeline_smoke_doc_path=pipeline_smoke_doc_path,
        estimated_full_cost_usd=estimated_full_cost,
        estimated_train_tokens=estimated_train_tokens,
        full_launch_command=_launch_command(config_path, runtime),
    )


def _validate_timeout_cost_ceiling(runtime: dict[str, Any]) -> None:
    timeout_cost_ceiling = _timeout_cost_ceiling_usd(runtime)
    hard_stop = float(runtime["full_run_runtime_spend_stop_usd"])
    if timeout_cost_ceiling > hard_stop:
        raise LaunchError(
            "runtime timeout cost ceiling exceeds runtime.full_run_runtime_spend_stop_usd: "
            f"{timeout_cost_ceiling:.4f} > {hard_stop:.4f}"
        )


def _timeout_cost_ceiling_usd(runtime: dict[str, Any]) -> float:
    selected = str(runtime["selected_gpu"])
    selected_profile = object_field(
        runtime["gpu_profiles"][selected], f"runtime.gpu_profiles.{selected}"
    )
    return int(runtime["timeout_hours"]) * float(selected_profile["usd_per_hour"])


def validate_rl_task_manifest(manifest_path: Path) -> DatasetSummary:
    manifest_path = manifest_path.expanduser().resolve()
    _, manifest = load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise LaunchError("dataset manifest.schema_version must be 1")
    if manifest.get("manifest_type") != MANIFEST_TYPE:
        raise LaunchError(f"dataset manifest.manifest_type must be {MANIFEST_TYPE}")
    return _validate_rl_manifest(manifest_path, manifest)


def full_launch_blockers(
    config: RLVRLaunchConfig, *, approved: bool = False, modal_gpu: str | None = None
) -> list[str]:
    blockers = _common_full_launch_blockers(
        runtime=config.runtime,
        estimated_full_cost_usd=config.estimated_full_cost_usd,
        approved=approved,
        modal_gpu=modal_gpu,
        approval_message="full Esme-214M-RL GRPO launch requires --approved",
        modal_gpu_env_var="RLVR_MODAL_GPU",
        full_run_cap_usd=FULL_RUN_SPEND_CAP_USD,
        cap_label=f"${FULL_RUN_SPEND_CAP_USD:.0f} cap",
    )
    if config.estimated_train_tokens > int(config.budgets["max_rollout_tokens"]):
        blockers.append("estimated rollout tokens exceed budgets.max_rollout_tokens")
    return blockers


def build_eval_profile(
    config: RLVRLaunchConfig,
    *,
    before_eval_only: bool = False,
    pipeline_smoke: bool = False,
) -> dict[str, Any]:
    if before_eval_only and pipeline_smoke:
        raise LaunchError("before_eval_only and pipeline_smoke eval profiles are separate")
    monitoring = config.payload["monitoring"]
    budgets = config.budgets
    if pipeline_smoke:
        smoke = config.pipeline_smoke
        tasks = int(smoke["eval_task_budget"])
        samples_per_task = int(smoke["samples_per_eval_task"])
        total_samples = tasks * samples_per_task
        return {
            "profile": PIPELINE_SMOKE_PROFILE,
            "tasks": tasks,
            "samples_per_task": samples_per_task,
            "total_samples": total_samples,
            "max_new_tokens": int(smoke["eval_max_new_tokens"]),
            "progress_interval_tasks": int(smoke["eval_progress_interval_tasks"]),
            "progress_interval_samples": int(smoke["eval_progress_interval_samples"]),
            "sample_batch_size": int(smoke["eval_sample_batch_size"]),
            "wall_timeout_seconds": float(smoke["eval_wall_timeout_seconds"]),
            "no_progress_timeout_seconds": float(smoke["eval_no_progress_timeout_seconds"]),
            "timeout_basis": (
                "pipeline_smoke explicit toy eval guard: "
                f"{tasks} tasks x {samples_per_task} samples"
            ),
        }

    tasks = int(budgets["eval_task_budget"])
    samples_per_task = int(monitoring["samples_per_eval_task"])
    profile = (
        FULL_EVAL_PROFILE
        if tasks == 30 and samples_per_task == 32
        else f"full_eval_{tasks}x{samples_per_task}"
    )
    if before_eval_only:
        tasks = min(tasks, int(monitoring["debug_eval_task_budget"]))
        samples_per_task = min(samples_per_task, int(monitoring["debug_samples_per_eval_task"]))
        profile = BEFORE_EVAL_DEBUG_PROFILE
    total_samples = tasks * samples_per_task
    wall_timeout = _scaled_eval_timeout(
        floor_seconds=float(monitoring["eval_wall_timeout_seconds"]),
        seconds_per_sample=float(monitoring["eval_wall_timeout_seconds_per_sample"]),
        total_samples=total_samples,
    )
    no_progress_timeout = _scaled_eval_timeout(
        floor_seconds=float(monitoring["eval_no_progress_timeout_seconds"]),
        seconds_per_sample=float(monitoring["eval_no_progress_timeout_seconds_per_sample"]),
        total_samples=total_samples,
    )
    return {
        "profile": profile,
        "tasks": tasks,
        "samples_per_task": samples_per_task,
        "total_samples": total_samples,
        "max_new_tokens": int(monitoring["eval_max_new_tokens"]),
        "progress_interval_tasks": int(monitoring["eval_progress_interval_tasks"]),
        "progress_interval_samples": int(monitoring["eval_progress_interval_samples"]),
        "sample_batch_size": int(monitoring["eval_sample_batch_size"]),
        "wall_timeout_seconds": wall_timeout,
        "no_progress_timeout_seconds": no_progress_timeout,
        "timeout_basis": (
            "max(configured floor, eval samples x seconds_per_sample): "
            f"{tasks} tasks x {samples_per_task} samples = {total_samples} samples"
        ),
    }


def pipeline_smoke_grpo_settings(config: RLVRLaunchConfig) -> dict[str, Any]:
    smoke = config.pipeline_smoke
    grpo = config.grpo
    return {
        **grpo,
        "max_steps": int(smoke["max_steps"]),
        "warmup_steps": int(smoke["warmup_steps"]),
        "prompts_per_step": int(smoke["prompts_per_step"]),
        "group_size": int(smoke["group_size"]),
        "max_new_tokens": int(smoke["max_new_tokens"]),
        "max_rollout_tokens": int(smoke["max_rollout_tokens"]),
    }


def _scaled_eval_timeout(
    *,
    floor_seconds: float,
    seconds_per_sample: float,
    total_samples: int,
) -> float:
    return max(floor_seconds, float(total_samples) * seconds_per_sample)


def _acceptance_preflight(config: RLVRLaunchConfig, full_eval: dict[str, Any]) -> dict[str, Any]:
    profile = str(full_eval["profile"])
    total_samples = int(full_eval["total_samples"])
    wall_timeout = float(full_eval["wall_timeout_seconds"])
    estimated_seconds = total_samples * float(
        config.payload["monitoring"]["eval_wall_timeout_seconds_per_sample"]
    )
    return {
        "decision": "ready_for_visible_modal_decision",
        "eval_profile": profile,
        "requires_pipeline_smoke_review": True,
        "full_acceptance_preserved": profile == FULL_EVAL_PROFILE,
        "total_samples": total_samples,
        "estimated_eval_seconds_from_budget": estimated_seconds,
        "wall_timeout_seconds": wall_timeout,
        "timeout_margin_seconds": wall_timeout - estimated_seconds,
        "full_launch_command": config.full_launch_command,
        "resume_command": config.full_launch_command,
    }


def _validate_input_bundle(payload: dict[str, Any], config_dir: Path) -> Path:
    require_keys(payload, {"path", "format", "model_family", "read_only"}, "input_bundle")
    if payload["format"] != "llm_pretrain_dense_v1":
        raise LaunchError("input_bundle.format must be llm_pretrain_dense_v1")
    if payload["model_family"] != "DenseBackbone":
        raise LaunchError("input_bundle.model_family must be DenseBackbone")
    if payload["read_only"] is not True:
        raise LaunchError("input_bundle.read_only must be true")
    path = (config_dir / str_field(payload["path"], "input_bundle.path")).resolve()
    try:
        validate_base_bundle(path)
    except BundleError as error:
        raise LaunchError(str(error)) from error
    return path


def _validate_dataset(payload: dict[str, Any], config_dir: Path) -> tuple[Path, DatasetSummary]:
    require_keys(
        payload,
        {
            "name",
            "manifest_path",
            "train_split",
            "eval_split",
            "secondary_transfer_eval",
            "remote_download_allowed",
            "train_on_gsm8k_lite",
        },
        "dataset",
    )
    if payload["name"] != COUNTDOWN_LITE_DATASET:
        raise LaunchError(f"dataset.name must be {COUNTDOWN_LITE_DATASET}")
    if payload["train_split"] != "train":
        raise LaunchError("dataset.train_split must be train")
    if payload["eval_split"] != "eval":
        raise LaunchError("dataset.eval_split must be eval")
    if payload["secondary_transfer_eval"] != "GSM8K-lite":
        raise LaunchError("dataset.secondary_transfer_eval must be GSM8K-lite")
    if payload["remote_download_allowed"] is not False:
        raise LaunchError("dataset.remote_download_allowed must stay false")
    if payload["train_on_gsm8k_lite"] is not False:
        raise LaunchError("dataset.train_on_gsm8k_lite must be false")
    manifest_path = (
        config_dir / str_field(payload["manifest_path"], "dataset.manifest_path")
    ).resolve()
    summary = validate_rl_task_manifest(manifest_path)
    if summary.name != COUNTDOWN_LITE_DATASET:
        raise LaunchError(f"dataset manifest.name must be {COUNTDOWN_LITE_DATASET}")
    return manifest_path, summary


def _validate_rl_manifest(manifest_path: Path, manifest: dict[str, Any]) -> DatasetSummary:
    rewards = manifest.get("reward_definitions")
    if not isinstance(rewards, list) or not rewards:
        raise LaunchError("rl task manifest.reward_definitions must not be empty")

    reward_names: set[str] = set()
    for index, raw_reward in enumerate(rewards):
        reward = object_field(raw_reward, f"rl task manifest.reward_definitions[{index}]")
        name = str_field(reward.get("name"), f"rl task manifest.reward_definitions[{index}].name")
        reward_type = str_field(
            reward.get("reward_type"), f"rl task manifest.reward_definitions[{index}].reward_type"
        )
        verifier = str_field(
            reward.get("verifier"), f"rl task manifest.reward_definitions[{index}].verifier"
        )
        str_field(
            reward.get("pass_condition"),
            f"rl task manifest.reward_definitions[{index}].pass_condition",
        )
        if reward.get("verifiable") is not True:
            raise LaunchError(f"rl reward '{name}' must be verifiable")
        if not verifier:
            raise LaunchError(f"rl reward '{name}' must name a verifier")
        if reward_type not in VERIFIABLE_REWARD_TYPES:
            raise LaunchError(f"rl reward '{name}' has unsupported reward_type: {reward_type}")
        reward_text = f"{name} {reward_type}".lower()
        blocked_terms = sorted(term for term in DISALLOWED_REWARD_TERMS if term in reward_text)
        if blocked_terms:
            terms = ", ".join(blocked_terms)
            raise LaunchError(
                f"rl reward '{name}' uses eval-observation terms, not rewards: {terms}"
            )
        if name in reward_names:
            raise LaunchError(
                f"rl task manifest.reward_definitions has duplicate reward name: {name}"
            )
        reward_names.add(name)

    data_files = manifest.get("data_files")
    if not isinstance(data_files, list) or not data_files:
        raise LaunchError("dataset manifest.data_files must not be empty")
    split_counts: dict[str, int] = {}
    for index, raw_data_file in enumerate(data_files):
        data_file = object_field(raw_data_file, f"dataset manifest.data_files[{index}]")
        if data_file.get("format") != "jsonl":
            raise LaunchError(f"dataset manifest.data_files[{index}].format must be jsonl")
        positive_int(data_file.get("records"), f"dataset manifest.data_files[{index}].records")
        raw_path = str_field(data_file.get("path"), f"dataset manifest.data_files[{index}].path")
        data_path = (manifest_path.parent / raw_path).resolve()
        data_root = manifest_path.parent.parent
        if not data_path.is_relative_to(data_root):
            raise LaunchError(f"dataset manifest.data_files[{index}].path escapes data root")
        if not data_path.is_file():
            raise LaunchError(f"missing dataset manifest.data_files[{index}].path: {data_path}")
    rows = load_countdown_lite_rows(manifest_path, split=None)
    for row in rows:
        reward_name = row.get("reward_name")
        if reward_name not in reward_names:
            raise LaunchError(f"rl row reward_name is not declared: {reward_name}")
        split = str(row.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
    total = sum(split_counts.values())
    declared_total = sum(
        positive_int(
            object_field(raw, f"dataset manifest.data_files[{index}]").get("records"),
            f"dataset manifest.data_files[{index}].records",
        )
        for index, raw in enumerate(data_files)
    )
    if total != declared_total:
        raise LaunchError(f"dataset manifest declares {declared_total} records but parsed {total}")
    return DatasetSummary(
        manifest_type=str(manifest["manifest_type"]),
        name=str(manifest["name"]),
        record_count=total,
        split_counts=split_counts,
        details=(f"reward_definitions={len(reward_names)}",),
    )


def _validate_approval_gate(payload: dict[str, Any]) -> None:
    require_keys(
        payload, {"requires_adam_approval", "approved", "blocked_actions"}, "approval_gate"
    )
    if payload["requires_adam_approval"] is not True:
        raise LaunchError("approval_gate.requires_adam_approval must be true")
    if payload["approved"] is not True:
        raise LaunchError("approval_gate.approved must be true for this approved mission config")
    blocked_actions = payload["blocked_actions"]
    if not isinstance(blocked_actions, list):
        raise LaunchError("approval_gate.blocked_actions must be a list")
    missing = REQUIRED_BLOCKED_ACTIONS - {str(item) for item in blocked_actions}
    if missing:
        names = ", ".join(sorted(missing))
        raise LaunchError(f"approval_gate.blocked_actions is missing: {names}")


def _validate_budgets(payload: dict[str, Any], dataset: DatasetSummary) -> dict[str, Any]:
    require_keys(
        payload,
        {
            "dataset_sample_budget",
            "train_task_budget",
            "eval_task_budget",
            "max_rollout_tokens",
            "max_eval_rollouts",
        },
        "budgets",
    )
    dataset_budget = positive_int(payload["dataset_sample_budget"], "budgets.dataset_sample_budget")
    if dataset_budget != dataset.record_count:
        raise LaunchError("budgets.dataset_sample_budget must equal the manifest record count")
    train_budget = positive_int(payload["train_task_budget"], "budgets.train_task_budget")
    eval_budget = positive_int(payload["eval_task_budget"], "budgets.eval_task_budget")
    if train_budget > dataset.split_counts.get("train", 0):
        raise LaunchError("budgets.train_task_budget exceeds train split records")
    if eval_budget > dataset.split_counts.get("eval", 0):
        raise LaunchError("budgets.eval_task_budget exceeds eval split records")
    positive_int(payload["max_rollout_tokens"], "budgets.max_rollout_tokens")
    positive_int(payload["max_eval_rollouts"], "budgets.max_eval_rollouts")
    return payload


def _validate_grpo(payload: dict[str, Any]) -> dict[str, Any]:
    require_keys(
        payload,
        {
            "method",
            "max_steps",
            "prompts_per_step",
            "group_size",
            "max_new_tokens",
            "temperature",
            "kl_beta",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "scheduler",
            "grad_clip",
            "seed",
        },
        "grpo",
    )
    if payload["method"] != METHOD:
        raise LaunchError(f"grpo.method must be {METHOD}")
    for key in ("max_steps", "prompts_per_step", "group_size", "max_new_tokens", "seed"):
        positive_int(payload[key], f"grpo.{key}")
    for key in ("temperature", "learning_rate", "grad_clip"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise LaunchError(f"grpo.{key} must be positive")
    kl_beta = payload["kl_beta"]
    if isinstance(kl_beta, bool) or not isinstance(kl_beta, int | float) or kl_beta < 0:
        raise LaunchError("grpo.kl_beta must be non-negative")
    if payload["scheduler"] not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError("grpo.scheduler must be constant, linear_warmup_decay, or cosine_decay")
    warmup_steps = payload["warmup_steps"]
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise LaunchError("grpo.warmup_steps must be a non-negative integer")
    if warmup_steps > int(payload["max_steps"]):
        raise LaunchError("grpo.warmup_steps must be <= grpo.max_steps")
    weight_decay = payload["weight_decay"]
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, int | float)
        or weight_decay < 0
    ):
        raise LaunchError("grpo.weight_decay must be non-negative")
    return payload


def _validate_reward_policy(payload: dict[str, Any]) -> None:
    require_keys(
        payload,
        {
            "verifiable_only",
            "exact_solve_reward",
            "valid_expression_reward",
            "invalid_reward",
            "disallowed_observation_terms",
        },
        "reward_policy",
    )
    if payload["verifiable_only"] is not True:
        raise LaunchError("reward_policy.verifiable_only must be true")
    exact = payload["exact_solve_reward"]
    valid = payload["valid_expression_reward"]
    invalid = payload["invalid_reward"]
    for key, value in (
        ("exact_solve_reward", exact),
        ("valid_expression_reward", valid),
        ("invalid_reward", invalid),
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise LaunchError(f"reward_policy.{key} must be numeric")
    if not (float(exact) > float(valid) >= float(invalid)):
        raise LaunchError(
            "reward_policy must satisfy exact_solve_reward > valid_expression_reward >= "
            "invalid_reward"
        )
    blocked_terms = payload["disallowed_observation_terms"]
    if not isinstance(blocked_terms, list) or not DISALLOWED_REWARD_TERMS.issubset(
        {str(item) for item in blocked_terms}
    ):
        raise LaunchError("reward_policy.disallowed_observation_terms is missing required terms")


def _validate_monitoring(payload: dict[str, Any]) -> None:
    require_keys(
        payload,
        {
            "log_interval",
            "checkpoint_interval",
            "samples_per_eval_task",
            "eval_max_new_tokens",
            "eval_progress_interval_tasks",
            "eval_progress_interval_samples",
            "eval_sample_batch_size",
            "eval_wall_timeout_seconds",
            "eval_wall_timeout_seconds_per_sample",
            "eval_no_progress_timeout_seconds",
            "eval_no_progress_timeout_seconds_per_sample",
            "debug_eval_task_budget",
            "debug_samples_per_eval_task",
            "wandb_project",
            "wandb_required_for_modal",
            "wandb_tags",
        },
        "monitoring",
    )
    for key in (
        "log_interval",
        "checkpoint_interval",
        "samples_per_eval_task",
        "eval_max_new_tokens",
        "eval_progress_interval_tasks",
        "eval_progress_interval_samples",
        "eval_sample_batch_size",
        "debug_eval_task_budget",
        "debug_samples_per_eval_task",
    ):
        positive_int(payload[key], f"monitoring.{key}")
    for key in (
        "eval_wall_timeout_seconds",
        "eval_wall_timeout_seconds_per_sample",
        "eval_no_progress_timeout_seconds",
        "eval_no_progress_timeout_seconds_per_sample",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise LaunchError(f"monitoring.{key} must be positive")
    if int(payload["eval_sample_batch_size"]) > int(payload["samples_per_eval_task"]):
        raise LaunchError("monitoring.eval_sample_batch_size must be <= samples_per_eval_task")
    if int(payload["debug_samples_per_eval_task"]) > int(payload["samples_per_eval_task"]):
        raise LaunchError("monitoring.debug_samples_per_eval_task must be <= samples_per_eval_task")
    if int(payload["eval_progress_interval_samples"]) > int(payload["samples_per_eval_task"]):
        raise LaunchError(
            "monitoring.eval_progress_interval_samples must be <= samples_per_eval_task"
        )
    str_field(payload["wandb_project"], "monitoring.wandb_project")
    if payload["wandb_required_for_modal"] is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")
    tags = payload["wandb_tags"]
    if not isinstance(tags, list) or "stage=rlvr" not in tags:
        raise LaunchError("monitoring.wandb_tags must include stage=rlvr")


def _validate_eval_monitoring_bounds(monitoring: dict[str, Any], budgets: dict[str, Any]) -> None:
    eval_rollouts = int(budgets["eval_task_budget"]) * int(monitoring["samples_per_eval_task"])
    if eval_rollouts > int(budgets["max_eval_rollouts"]):
        raise LaunchError(
            "eval_task_budget * monitoring.samples_per_eval_task exceeds budgets.max_eval_rollouts"
        )
    if int(monitoring["debug_eval_task_budget"]) > int(budgets["eval_task_budget"]):
        raise LaunchError("monitoring.debug_eval_task_budget must be <= budgets.eval_task_budget")


def _validate_pipeline_smoke(
    payload: dict[str, Any],
    *,
    budgets: dict[str, Any],
    grpo: dict[str, Any],
    monitoring: dict[str, Any],
    config_dir: Path,
) -> tuple[Path, Path, Path]:
    require_keys(
        payload,
        {
            "profile",
            "enabled",
            "train_task_budget",
            "max_steps",
            "warmup_steps",
            "prompts_per_step",
            "group_size",
            "max_new_tokens",
            "max_rollout_tokens",
            "eval_task_budget",
            "samples_per_eval_task",
            "eval_max_new_tokens",
            "eval_progress_interval_tasks",
            "eval_progress_interval_samples",
            "eval_sample_batch_size",
            "eval_wall_timeout_seconds",
            "eval_no_progress_timeout_seconds",
            "output_dir",
            "report_path",
            "doc_path",
        },
        "pipeline_smoke",
    )
    if payload["profile"] != PIPELINE_SMOKE_PROFILE:
        raise LaunchError(f"pipeline_smoke.profile must be {PIPELINE_SMOKE_PROFILE}")
    if payload["enabled"] is not True:
        raise LaunchError("pipeline_smoke.enabled must be true")
    positive_checks = (
        "train_task_budget",
        "max_steps",
        "prompts_per_step",
        "group_size",
        "max_new_tokens",
        "max_rollout_tokens",
        "eval_task_budget",
        "samples_per_eval_task",
        "eval_max_new_tokens",
        "eval_progress_interval_tasks",
        "eval_progress_interval_samples",
        "eval_sample_batch_size",
    )
    for key in positive_checks:
        positive_int(payload[key], f"pipeline_smoke.{key}")
    warmup_steps = payload["warmup_steps"]
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise LaunchError("pipeline_smoke.warmup_steps must be a non-negative integer")
    for key in ("eval_wall_timeout_seconds", "eval_no_progress_timeout_seconds"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            raise LaunchError(f"pipeline_smoke.{key} must be positive")
    if int(payload["train_task_budget"]) > int(budgets["train_task_budget"]):
        raise LaunchError("pipeline_smoke.train_task_budget must be <= budgets.train_task_budget")
    if int(payload["eval_task_budget"]) > int(budgets["eval_task_budget"]):
        raise LaunchError("pipeline_smoke.eval_task_budget must be <= budgets.eval_task_budget")
    if int(payload["max_steps"]) > int(grpo["max_steps"]):
        raise LaunchError("pipeline_smoke.max_steps must be <= grpo.max_steps")
    if int(payload["warmup_steps"]) > int(payload["max_steps"]):
        raise LaunchError("pipeline_smoke.warmup_steps must be <= pipeline_smoke.max_steps")
    if int(payload["prompts_per_step"]) > int(grpo["prompts_per_step"]):
        raise LaunchError("pipeline_smoke.prompts_per_step must be <= grpo.prompts_per_step")
    if int(payload["group_size"]) > int(grpo["group_size"]):
        raise LaunchError("pipeline_smoke.group_size must be <= grpo.group_size")
    if int(payload["samples_per_eval_task"]) > int(monitoring["samples_per_eval_task"]):
        raise LaunchError(
            "pipeline_smoke.samples_per_eval_task must be <= monitoring.samples_per_eval_task"
        )
    if int(payload["eval_sample_batch_size"]) > int(payload["samples_per_eval_task"]):
        raise LaunchError(
            "pipeline_smoke.eval_sample_batch_size must be <= pipeline_smoke.samples_per_eval_task"
        )

    output_dir = Path(str_field(payload["output_dir"], "pipeline_smoke.output_dir"))
    if output_dir.is_absolute() or ".." in output_dir.parts or output_dir.parts[:1] != ("runs",):
        raise LaunchError("pipeline_smoke.output_dir must be a relative path under runs/")
    report_path = Path(str_field(payload["report_path"], "pipeline_smoke.report_path"))
    if report_path.is_absolute() or ".." in report_path.parts:
        raise LaunchError("pipeline_smoke.report_path must stay inside the repository")
    doc_path = Path(str_field(payload["doc_path"], "pipeline_smoke.doc_path"))
    if doc_path.is_absolute() or ".." in doc_path.parts or doc_path.parts[:1] != ("docs",):
        raise LaunchError("pipeline_smoke.doc_path must be a relative path under docs/")
    repo_root = config_dir.parent
    return (
        (repo_root / output_dir).resolve(),
        (repo_root / report_path).resolve(),
        (repo_root / doc_path).resolve(),
    )


def _validate_artifacts(payload: dict[str, Any], config_dir: Path) -> tuple[Path, Path, Path]:
    require_keys(
        payload,
        {"output_dir", "report_path", "doc_path", "required_files"},
        "artifacts",
    )
    output_dir = Path(str_field(payload["output_dir"], "artifacts.output_dir"))
    if output_dir.is_absolute() or ".." in output_dir.parts or output_dir.parts[:1] != ("runs",):
        raise LaunchError("artifacts.output_dir must be a relative path under runs/")
    report_path = Path(str_field(payload["report_path"], "artifacts.report_path"))
    if report_path.is_absolute() or ".." in report_path.parts:
        raise LaunchError("artifacts.report_path must stay inside the repository")
    doc_path = Path(str_field(payload["doc_path"], "artifacts.doc_path"))
    if doc_path.is_absolute() or ".." in doc_path.parts or doc_path.parts[:1] != ("docs",):
        raise LaunchError("artifacts.doc_path must be a relative path under docs/")
    required = payload["required_files"]
    if not isinstance(required, list) or tuple(required) != EXPECTED_ARTIFACTS:
        raise LaunchError("artifacts.required_files must match the GRPO evidence manifest")
    repo_root = config_dir.parent
    return (
        (repo_root / output_dir).resolve(),
        (repo_root / report_path).resolve(),
        (repo_root / doc_path).resolve(),
    )


def _validate_acceptance(payload: dict[str, Any]) -> None:
    require_keys(
        payload,
        {
            "primary_metric",
            "baseline_valid_expression_rate",
            "baseline_exact_solve_rate",
            "baseline_pass_at_1",
            "baseline_pass_at_8",
            "baseline_pass_at_32",
            "record_no_improvement",
        },
        "acceptance",
    )
    if payload["primary_metric"] != "valid_expression_rate":
        raise LaunchError("acceptance.primary_metric must be valid_expression_rate")
    if payload["record_no_improvement"] is not True:
        raise LaunchError("acceptance.record_no_improvement must be true")
    for key in (
        "baseline_valid_expression_rate",
        "baseline_exact_solve_rate",
        "baseline_pass_at_1",
        "baseline_pass_at_8",
        "baseline_pass_at_32",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
            raise LaunchError(f"acceptance.{key} must be a rate in [0, 1]")


def _validate_abort_rules(value: Any) -> None:
    if not isinstance(value, list) or len(value) < 6:
        raise LaunchError("abort_rules must list the launch and runtime stop rules")
    joined = " ".join(str(item).lower() for item in value)
    for phrase in ("approved", "$25", "countdown-lite", "gsm8k", "rollout", "modal"):
        if phrase not in joined:
            raise LaunchError(f"abort_rules must include {phrase}")


def _estimate_train_tokens(
    *,
    input_bundle_path: Path,
    manifest_path: Path,
    train_split: str,
    max_train_tasks: int,
    prompts_per_step: int,
    group_size: int,
    max_steps: int,
    max_new_tokens: int,
) -> int:
    try:
        bundle = validate_base_bundle(input_bundle_path)
    except BundleError as error:
        raise LaunchError(str(error)) from error
    tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))
    rows = list(load_countdown_lite_rows(manifest_path, split=train_split))[:max_train_tasks]
    if not rows:
        raise LaunchError(f"no Countdown-Lite rows found for split: {train_split}")
    prompt_lengths = [
        len(tokenizer.encode(render_chat_prompt(str(row["prompt"])), add_special_tokens=False).ids)
        for row in rows
    ]
    max_prompt_tokens = max(prompt_lengths)
    rollout_sequences = prompts_per_step * group_size * max_steps
    return rollout_sequences * (max_prompt_tokens + max_new_tokens)


def _launch_command(config_path: Path, runtime: dict[str, Any]) -> str:
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="RLVR_MODAL_GPU",
        timeout_env_var="RLVR_TIMEOUT_HOURS",
        script_path="scripts/modal_rlvr_grpo.py",
        mode_flag=" --full-run",
    )


def _pipeline_smoke_command(config_path: Path) -> str:
    return f"uv run esme-posttrain rlvr-pipeline-smoke --config {config_path.as_posix()} --json"


def _modal_pipeline_smoke_command(config_path: Path, runtime: dict[str, Any]) -> str:
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="RLVR_MODAL_GPU",
        timeout_env_var="RLVR_TIMEOUT_HOURS",
        script_path="scripts/modal_rlvr_grpo.py",
        mode_flag=" --modal-pipeline-smoke",
    )


def _expected_duration_minutes(config: RLVRLaunchConfig) -> float:
    tokens_per_second = float(config.selected_gpu_profile["projected_tokens_per_second"])
    return round(config.estimated_train_tokens / tokens_per_second / 60.0, 2)
