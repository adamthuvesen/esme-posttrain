"""Bounded launch preparation for the Esme-214M-RL Countdown-Lite GRPO tranche.

``rl.config`` pins the recipe at parse time. This module checks the bundle and task
manifest on disk, projects rollout tokens and cost, and builds launch blockers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.bundle import (
    BundleError,
    validate_base_bundle,
    validate_bundle_compatibility,
)
from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    build_modal_launch_command,
    estimate_cost_usd,
    image_package_pins,
    load_json_object,
    object_field,
)
from esme_posttrain.launch.config_guards import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.launch.config_models import (
    GPUProfile,
    ModalRuntime,
    parse_config,
    positive_int,
    str_field,
)
from esme_posttrain.launch.errors import LaunchError
from esme_posttrain.rl.config import (
    ARTIFACT_NAME,
    COUNTDOWN_LITE_DATASET,
    DISALLOWED_REWARD_TERMS,
    FULL_RUN_SPEND_CAP_USD,
    METHOD,
    PIPELINE_SMOKE_PROFILE,
    STAGE,
    STARTS_FROM,
    CountdownDataset,
    GRPOSettings,
    PipelineSmoke,
    RLVRBudgets,
    RLVRSettings,
    timeout_cost_ceiling_usd,
)
from esme_posttrain.rl.countdown_lite import load_countdown_lite_rows, render_chat_prompt

MANIFEST_TYPE = "rl_tasks"
FULL_EVAL_PROFILE = "full_acceptance_30x32"
BEFORE_EVAL_DEBUG_PROFILE = "before_eval_debug"
VERIFIABLE_REWARD_TYPES = frozenset(
    {"exact_match", "numeric_match", "regex_match", "unit_test", "execution_check"}
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
    settings: RLVRSettings
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
        return self.settings.run_id

    @property
    def artifact_name(self) -> str:
        return self.settings.artifact_name

    @property
    def budgets(self) -> RLVRBudgets:
        return self.settings.budgets

    @property
    def grpo(self) -> GRPOSettings:
        return self.settings.grpo

    @property
    def runtime(self) -> ModalRuntime:
        return self.settings.runtime

    @property
    def pipeline_smoke(self) -> PipelineSmoke:
        return self.settings.pipeline_smoke

    @property
    def skip_acceptance_eval(self) -> bool:
        return self.settings.skip_acceptance_eval

    @property
    def selected_gpu_profile(self) -> GPUProfile:
        return self.runtime.gpu_profiles[self.runtime.selected_gpu]


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
        "sample_budget": config.settings.budgets.dataset_sample_budget,
        "train_task_budget": config.settings.budgets.train_task_budget,
        "rollout_group_size": config.settings.grpo.group_size,
        "prompts_per_step": config.settings.grpo.prompts_per_step,
        "max_steps": config.settings.grpo.max_steps,
        "max_new_tokens": config.settings.grpo.max_new_tokens,
        "token_budget": config.settings.budgets.max_rollout_tokens,
        "estimated_train_tokens": config.estimated_train_tokens,
        "hardware": config.runtime.selected_gpu,
        "modal_gpu": config.selected_gpu_profile.modal_gpu,
        "expected_duration_minutes": _expected_duration_minutes(config),
        "projected_cost_usd": round(config.estimated_full_cost_usd, 4),
        "spend_cap_usd": FULL_RUN_SPEND_CAP_USD,
        "runtime_spend_stop_usd": config.runtime.full_run_runtime_spend_stop_usd,
        "timeout_hours": config.runtime.timeout_hours,
        "timeout_cost_ceiling_usd": round(timeout_cost_ceiling_usd(config.runtime), 4),
        "eval_profile": full_eval["profile"],
        "eval_rollouts": full_eval["total_samples"],
        "eval_task_budget": full_eval["tasks"],
        "samples_per_eval_task": full_eval["samples_per_task"],
        "eval_progress_interval_tasks": config.settings.monitoring.eval_progress_interval_tasks,
        "eval_progress_interval_samples": config.settings.monitoring.eval_progress_interval_samples,
        "eval_sample_batch_size": config.settings.monitoring.eval_sample_batch_size,
        "eval_wall_timeout_seconds": full_eval["wall_timeout_seconds"],
        "eval_no_progress_timeout_seconds": full_eval["no_progress_timeout_seconds"],
        "eval_timeout_basis": full_eval["timeout_basis"],
        "acceptance_preflight": _acceptance_preflight(config, full_eval),
        "debug_eval_task_budget": config.settings.monitoring.debug_eval_task_budget,
        "debug_samples_per_eval_task": config.settings.monitoring.debug_samples_per_eval_task,
        "debug_eval_profile": debug_eval["profile"],
        "debug_eval_rollouts": debug_eval["total_samples"],
        "pipeline_smoke_profile": smoke_eval["profile"],
        "pipeline_smoke_eval_rollouts": smoke_eval["total_samples"],
        "pipeline_smoke_command": _pipeline_smoke_command(config.config_path),
        "modal_smoke_command": _modal_pipeline_smoke_command(config.config_path, config.runtime),
        "wandb_project": config.settings.monitoring.wandb_project,
        "wandb_required_for_modal": config.settings.monitoring.wandb_required_for_modal,
        "wandb_tags": list(config.settings.monitoring.wandb_tags),
        "output_dir": str(config.output_dir),
        "report_path": str(config.report_path),
        "doc_path": str(config.doc_path),
        "acceptance": config.settings.acceptance.to_dict(),
        "approval_required": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "approval_state": "not approved for launch",
        "full_launch_blockers": full_blockers,
        "full_launch_command": config.full_launch_command,
        "resume_command": config.full_launch_command,
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION, **image_package_pins()},
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


def validate_rlvr_payload(payload: object, config_path: Path) -> RLVRLaunchConfig:
    settings = parse_config(RLVRSettings, payload, source=config_path)
    config_dir = config_path.parent
    repo_root = config_dir.parent
    input_bundle_path = (config_dir / settings.input_bundle.path).resolve()
    try:
        validate_bundle_compatibility(input_bundle_path)
    except BundleError as error:
        raise LaunchError(str(error)) from error
    dataset_manifest_path, dataset = _validate_dataset(settings.dataset, config_dir)
    budgets = settings.budgets
    _validate_budgets_against_dataset(budgets, dataset)
    grpo = settings.grpo
    runtime = settings.runtime
    smoke = settings.pipeline_smoke
    artifacts = settings.artifacts

    estimated_train_tokens = _estimate_train_tokens(
        input_bundle_path=input_bundle_path,
        manifest_path=dataset_manifest_path,
        train_split=settings.dataset.train_split,
        max_train_tasks=int(budgets.train_task_budget),
        prompts_per_step=int(grpo.prompts_per_step),
        group_size=int(grpo.group_size),
        max_steps=int(grpo.max_steps),
        max_new_tokens=int(grpo.max_new_tokens),
        zero_variance_max_resamples=int(grpo.zero_variance_max_resamples),
    )
    if estimated_train_tokens > int(budgets.max_rollout_tokens):
        raise LaunchError(
            "estimated GRPO rollout tokens exceed budgets.max_rollout_tokens: "
            f"{estimated_train_tokens} > {budgets.max_rollout_tokens}"
        )

    selected_profile = runtime.gpu_profiles[runtime.selected_gpu]
    estimated_full_cost = estimate_cost_usd(
        tokens=estimated_train_tokens,
        projected_tokens_per_second=float(selected_profile.projected_tokens_per_second),
        usd_per_hour=float(selected_profile.usd_per_hour),
    )
    if estimated_full_cost > float(runtime.full_run_max_cost_usd):
        raise LaunchError("projected GRPO cost exceeds runtime.full_run_max_cost_usd")

    return RLVRLaunchConfig(
        settings=settings,
        config_path=config_path,
        input_bundle_path=input_bundle_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset=dataset,
        output_dir=(repo_root / artifacts.output_dir).resolve(),
        report_path=(repo_root / artifacts.report_path).resolve(),
        doc_path=(repo_root / artifacts.doc_path).resolve(),
        pipeline_smoke_output_dir=(repo_root / smoke.output_dir).resolve(),
        pipeline_smoke_report_path=(repo_root / smoke.report_path).resolve(),
        pipeline_smoke_doc_path=(repo_root / smoke.doc_path).resolve(),
        estimated_full_cost_usd=estimated_full_cost,
        estimated_train_tokens=estimated_train_tokens,
        full_launch_command=_launch_command(config_path, runtime),
    )


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
    )
    if config.estimated_train_tokens > config.budgets.max_rollout_tokens:
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
    monitoring = config.settings.monitoring
    budgets = config.budgets
    if pipeline_smoke:
        smoke = config.pipeline_smoke
        tasks = smoke.eval_task_budget
        samples_per_task = smoke.samples_per_eval_task
        total_samples = tasks * samples_per_task
        return {
            "profile": PIPELINE_SMOKE_PROFILE,
            "tasks": tasks,
            "samples_per_task": samples_per_task,
            "total_samples": total_samples,
            "max_new_tokens": smoke.eval_max_new_tokens,
            "progress_interval_tasks": smoke.eval_progress_interval_tasks,
            "progress_interval_samples": smoke.eval_progress_interval_samples,
            "sample_batch_size": smoke.eval_sample_batch_size,
            "wall_timeout_seconds": float(smoke.eval_wall_timeout_seconds),
            "no_progress_timeout_seconds": float(smoke.eval_no_progress_timeout_seconds),
            "timeout_basis": (
                "pipeline_smoke explicit toy eval guard: "
                f"{tasks} tasks x {samples_per_task} samples"
            ),
        }

    tasks = budgets.eval_task_budget
    samples_per_task = monitoring.samples_per_eval_task
    profile = (
        FULL_EVAL_PROFILE
        if tasks == 30 and samples_per_task == 32
        else f"full_eval_{tasks}x{samples_per_task}"
    )
    if before_eval_only:
        tasks = min(tasks, monitoring.debug_eval_task_budget)
        samples_per_task = min(samples_per_task, monitoring.debug_samples_per_eval_task)
        profile = BEFORE_EVAL_DEBUG_PROFILE
    total_samples = tasks * samples_per_task
    wall_timeout = _scaled_eval_timeout(
        floor_seconds=monitoring.eval_wall_timeout_seconds,
        seconds_per_sample=monitoring.eval_wall_timeout_seconds_per_sample,
        total_samples=total_samples,
    )
    no_progress_timeout = _scaled_eval_timeout(
        floor_seconds=monitoring.eval_no_progress_timeout_seconds,
        seconds_per_sample=monitoring.eval_no_progress_timeout_seconds_per_sample,
        total_samples=total_samples,
    )
    return {
        "profile": profile,
        "tasks": tasks,
        "samples_per_task": samples_per_task,
        "total_samples": total_samples,
        "max_new_tokens": monitoring.eval_max_new_tokens,
        "progress_interval_tasks": monitoring.eval_progress_interval_tasks,
        "progress_interval_samples": monitoring.eval_progress_interval_samples,
        "sample_batch_size": monitoring.eval_sample_batch_size,
        "wall_timeout_seconds": wall_timeout,
        "no_progress_timeout_seconds": no_progress_timeout,
        "timeout_basis": (
            "max(configured floor, eval samples x seconds_per_sample): "
            f"{tasks} tasks x {samples_per_task} samples = {total_samples} samples"
        ),
    }


def pipeline_smoke_grpo_settings(config: RLVRLaunchConfig) -> GRPOSettings:
    smoke = config.pipeline_smoke
    return config.grpo.model_copy(
        update={
            "max_steps": smoke.max_steps,
            "warmup_steps": smoke.warmup_steps,
            "prompts_per_step": smoke.prompts_per_step,
            "group_size": smoke.group_size,
            "max_new_tokens": smoke.max_new_tokens,
        }
    )


def _scaled_eval_timeout(
    *,
    floor_seconds: float,
    seconds_per_sample: float,
    total_samples: int,
) -> float:
    return float(max(floor_seconds, total_samples * seconds_per_sample))


def _acceptance_preflight(config: RLVRLaunchConfig, full_eval: dict[str, Any]) -> dict[str, Any]:
    profile = str(full_eval["profile"])
    total_samples = int(full_eval["total_samples"])
    wall_timeout = float(full_eval["wall_timeout_seconds"])
    estimated_seconds = (
        total_samples * config.settings.monitoring.eval_wall_timeout_seconds_per_sample
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


def _validate_dataset(payload: CountdownDataset, config_dir: Path) -> tuple[Path, DatasetSummary]:
    manifest_path = (config_dir / payload.manifest_path).resolve()
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


def _validate_budgets_against_dataset(payload: RLVRBudgets, dataset: DatasetSummary) -> None:
    if payload.dataset_sample_budget != dataset.record_count:
        raise LaunchError("budgets.dataset_sample_budget must equal the manifest record count")
    if payload.train_task_budget > dataset.split_counts.get("train", 0):
        raise LaunchError("budgets.train_task_budget exceeds train split records")
    if payload.eval_task_budget > dataset.split_counts.get("eval", 0):
        raise LaunchError("budgets.eval_task_budget exceeds eval split records")


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
    zero_variance_max_resamples: int = 0,
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
    # Worst case every group is zero-variance and burns every allowed resample,
    # so the spend projection and token budget cover the dynamic-sampling cap.
    resample_factor = 1 + zero_variance_max_resamples
    return rollout_sequences * (max_prompt_tokens + max_new_tokens) * resample_factor


def _launch_command(config_path: Path, runtime: ModalRuntime) -> str:
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


def _modal_pipeline_smoke_command(config_path: Path, runtime: ModalRuntime) -> str:
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="RLVR_MODAL_GPU",
        timeout_env_var="RLVR_TIMEOUT_HOURS",
        script_path="scripts/modal_rlvr_grpo.py",
        mode_flag=" --modal-pipeline-smoke",
    )


def _expected_duration_minutes(config: RLVRLaunchConfig) -> float:
    tokens_per_second = config.selected_gpu_profile.projected_tokens_per_second
    return round(config.estimated_train_tokens / tokens_per_second / 60.0, 2)
