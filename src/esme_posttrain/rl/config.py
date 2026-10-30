from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from esme_posttrain.launch.config_models import (
    BundleConfig,
    FrozenConfig,
    ModalRuntime,
    OutputArtifacts,
    check_stage_runtime,
    positive_int,
    str_field,
)
from esme_posttrain.launch.errors import LaunchError

RUN_ID = "esme_214m_rlvr_countdown_lite_grpo"
RUN_CARD = "run_cards/esme-214m-rl.md"
ARTIFACT_NAME = "Esme-214M-RL"
STAGE = "rlvr"
STARTS_FROM = "Esme-214M-Chat"
COUNTDOWN_LITE_DATASET = "esme_214m_rl_countdown_lite"
METHOD = "grpo"
MODAL_VOLUME = "esme-posttrain-esme-rlvr-countdown"
FULL_RUN_SPEND_CAP_USD = 25.0
PIPELINE_SMOKE_PROFILE = "pipeline_smoke"
DISALLOWED_REWARD_TERMS = frozenset({"style", "sharpness", "friendliness", "naturalness", "tone"})
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
# Opting into a final-checkpoint acceptance eval adds one artifact; the base
# manifest stays valid for provenance configs that predate it.
EXPECTED_ARTIFACTS_WITH_FINAL_EVAL = (*EXPECTED_ARTIFACTS, "eval-after-final.json")
EXACT_SOLVE_REWARD_MARGIN = 0.4


class CountdownDataset(FrozenConfig):
    name: StrictStr
    manifest_path: StrictStr
    train_split: StrictStr
    eval_split: StrictStr
    secondary_transfer_eval: StrictStr
    remote_download_allowed: StrictBool
    train_on_gsm8k_lite: StrictBool


class RLVRBudgets(FrozenConfig):
    dataset_sample_budget: StrictInt
    train_task_budget: StrictInt
    eval_task_budget: StrictInt
    max_rollout_tokens: StrictInt
    max_eval_rollouts: StrictInt


class GRPOSettings(FrozenConfig):
    method: StrictStr
    max_steps: StrictInt
    prompts_per_step: StrictInt
    group_size: StrictInt
    max_new_tokens: StrictInt
    temperature: StrictFloat | StrictInt
    kl_beta: StrictFloat | StrictInt
    learning_rate: StrictFloat | StrictInt
    weight_decay: StrictFloat | StrictInt
    warmup_steps: StrictInt
    scheduler: StrictStr
    grad_clip: StrictFloat | StrictInt
    seed: StrictInt
    zero_variance_max_resamples: StrictInt = 0
    replay_buffer_max_age_steps: StrictInt = 0
    stratified_difficulty_sampling: StrictBool = False
    reward_mode: Literal["verifier", "random"] = "verifier"
    random_reward_seed: StrictInt = 0
    advantage_normalization: Literal["mean", "std"] = "mean"


class RewardPolicy(FrozenConfig):
    verifiable_only: StrictBool
    exact_solve_reward: StrictFloat | StrictInt
    valid_expression_reward: StrictFloat | StrictInt
    invalid_reward: StrictFloat | StrictInt
    disallowed_observation_terms: tuple[StrictStr, ...]
    format_expression_reward: StrictFloat | StrictInt = 0.0
    closeness_weight: StrictFloat | StrictInt = 0.0


class RLVRMonitoring(FrozenConfig):
    log_interval: StrictInt
    checkpoint_interval: StrictInt
    samples_per_eval_task: StrictInt
    eval_max_new_tokens: StrictInt
    eval_progress_interval_tasks: StrictInt
    eval_progress_interval_samples: StrictInt
    eval_sample_batch_size: StrictInt
    eval_wall_timeout_seconds: StrictFloat | StrictInt
    eval_wall_timeout_seconds_per_sample: StrictFloat | StrictInt
    eval_no_progress_timeout_seconds: StrictFloat | StrictInt
    eval_no_progress_timeout_seconds_per_sample: StrictFloat | StrictInt
    debug_eval_task_budget: StrictInt
    debug_samples_per_eval_task: StrictInt
    wandb_project: StrictStr
    wandb_required_for_modal: StrictBool
    wandb_tags: tuple[StrictStr, ...]


class PipelineSmoke(FrozenConfig):
    profile: StrictStr
    enabled: StrictBool
    train_task_budget: StrictInt
    max_steps: StrictInt
    warmup_steps: StrictInt
    prompts_per_step: StrictInt
    group_size: StrictInt
    max_new_tokens: StrictInt
    max_rollout_tokens: StrictInt
    eval_task_budget: StrictInt
    samples_per_eval_task: StrictInt
    eval_max_new_tokens: StrictInt
    eval_progress_interval_tasks: StrictInt
    eval_progress_interval_samples: StrictInt
    eval_sample_batch_size: StrictInt
    eval_wall_timeout_seconds: StrictFloat | StrictInt
    eval_no_progress_timeout_seconds: StrictFloat | StrictInt
    output_dir: StrictStr
    report_path: StrictStr
    doc_path: StrictStr


class Artifacts(OutputArtifacts):
    report_path: StrictStr
    doc_path: StrictStr


class Acceptance(FrozenConfig):
    primary_metric: StrictStr
    baseline_valid_expression_rate: StrictFloat | StrictInt
    baseline_exact_solve_rate: StrictFloat | StrictInt
    baseline_pass_at_1: StrictFloat | StrictInt
    baseline_pass_at_8: StrictFloat | StrictInt
    baseline_pass_at_32: StrictFloat | StrictInt
    record_no_improvement: StrictBool


class RLVRSettings(FrozenConfig):
    schema_version: StrictInt
    run_id: StrictStr
    run_card: StrictStr
    requires_approval: StrictBool
    artifact_name: StrictStr
    stage: StrictStr
    starts_from: StrictStr
    input_bundle: BundleConfig
    dataset: CountdownDataset
    budgets: RLVRBudgets
    grpo: GRPOSettings
    reward_policy: RewardPolicy
    runtime: ModalRuntime
    monitoring: RLVRMonitoring
    pipeline_smoke: PipelineSmoke
    artifacts: Artifacts
    acceptance: Acceptance
    skip_acceptance_eval: StrictBool = False

    @model_validator(mode="after")
    def _check_recipe(self) -> Self:
        if self.schema_version != 1:
            raise LaunchError("schema_version must be 1")
        if self.run_id != RUN_ID:
            raise LaunchError(f"run_id must be {RUN_ID}")
        if self.run_card != RUN_CARD:
            raise LaunchError(f"run_card must be {RUN_CARD}")
        if self.requires_approval is not True:
            raise LaunchError("requires_approval must be true")
        if self.artifact_name != ARTIFACT_NAME:
            raise LaunchError(f"artifact_name must be {ARTIFACT_NAME}")
        if self.stage != STAGE:
            raise LaunchError(f"stage must be {STAGE}")
        if self.starts_from != STARTS_FROM:
            raise LaunchError(f"starts_from must be {STARTS_FROM}")
        _check_dataset(self.dataset)
        _check_budgets(self.budgets)
        _check_grpo(self.grpo)
        _check_reward_policy(self.reward_policy)
        check_stage_runtime(
            self.runtime,
            full_run_spend_cap_usd=FULL_RUN_SPEND_CAP_USD,
            modal_volume=MODAL_VOLUME,
        )
        _check_timeout_cost_ceiling(self.runtime)
        _check_monitoring(self.monitoring)
        _check_eval_monitoring_bounds(self.monitoring, self.budgets)
        _check_pipeline_smoke(self.pipeline_smoke, self)
        _check_artifacts(self.artifacts)
        _check_acceptance(self.acceptance)
        return self


def timeout_cost_ceiling_usd(runtime: ModalRuntime) -> float:
    selected_profile = runtime.gpu_profiles[runtime.selected_gpu]
    return int(runtime.timeout_hours) * float(selected_profile.usd_per_hour)


def _check_timeout_cost_ceiling(runtime: ModalRuntime) -> None:
    timeout_cost_ceiling = timeout_cost_ceiling_usd(runtime)
    hard_stop = float(runtime.full_run_runtime_spend_stop_usd)
    if timeout_cost_ceiling > hard_stop:
        raise LaunchError(
            "runtime timeout cost ceiling exceeds runtime.full_run_runtime_spend_stop_usd: "
            f"{timeout_cost_ceiling:.4f} > {hard_stop:.4f}"
        )


def _check_dataset(payload: CountdownDataset) -> None:
    if payload.name != COUNTDOWN_LITE_DATASET:
        raise LaunchError(f"dataset.name must be {COUNTDOWN_LITE_DATASET}")
    if payload.train_split != "train":
        raise LaunchError("dataset.train_split must be train")
    if payload.eval_split != "eval":
        raise LaunchError("dataset.eval_split must be eval")
    if payload.secondary_transfer_eval != "GSM8K-lite":
        raise LaunchError("dataset.secondary_transfer_eval must be GSM8K-lite")
    if payload.remote_download_allowed is not False:
        raise LaunchError("dataset.remote_download_allowed must stay false")
    if payload.train_on_gsm8k_lite is not False:
        raise LaunchError("dataset.train_on_gsm8k_lite must be false")
    str_field(payload.manifest_path, "dataset.manifest_path")


def _check_budgets(payload: RLVRBudgets) -> None:
    for key, value in (
        ("dataset_sample_budget", payload.dataset_sample_budget),
        ("train_task_budget", payload.train_task_budget),
        ("eval_task_budget", payload.eval_task_budget),
        ("max_rollout_tokens", payload.max_rollout_tokens),
        ("max_eval_rollouts", payload.max_eval_rollouts),
    ):
        positive_int(value, f"budgets.{key}")


def _check_grpo(payload: GRPOSettings) -> None:
    for key, value in (
        ("zero_variance_max_resamples", payload.zero_variance_max_resamples),
        ("replay_buffer_max_age_steps", payload.replay_buffer_max_age_steps),
    ):
        if value < 0:
            raise LaunchError(f"grpo.{key} must be a non-negative integer")
    if "random_reward_seed" in payload.model_fields_set:
        seed_value = payload.random_reward_seed
        if seed_value < 0:
            raise LaunchError("grpo.random_reward_seed must be a non-negative integer")
    if payload.method != METHOD:
        raise LaunchError(f"grpo.method must be {METHOD}")
    for key, value in (
        ("max_steps", payload.max_steps),
        ("prompts_per_step", payload.prompts_per_step),
        ("group_size", payload.group_size),
        ("max_new_tokens", payload.max_new_tokens),
        ("seed", payload.seed),
    ):
        positive_int(value, f"grpo.{key}")
    for key, numeric_value in (
        ("temperature", payload.temperature),
        ("learning_rate", payload.learning_rate),
        ("grad_clip", payload.grad_clip),
    ):
        if numeric_value <= 0:
            raise LaunchError(f"grpo.{key} must be positive")
    kl_beta = payload.kl_beta
    if kl_beta < 0:
        raise LaunchError("grpo.kl_beta must be non-negative")
    if payload.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError("grpo.scheduler must be constant, linear_warmup_decay, or cosine_decay")
    warmup_steps = payload.warmup_steps
    if warmup_steps < 0:
        raise LaunchError("grpo.warmup_steps must be a non-negative integer")
    if warmup_steps > payload.max_steps:
        raise LaunchError("grpo.warmup_steps must be <= grpo.max_steps")
    weight_decay = payload.weight_decay
    if weight_decay < 0:
        raise LaunchError("grpo.weight_decay must be non-negative")


def _check_reward_policy(payload: RewardPolicy) -> None:
    if payload.verifiable_only is not True:
        raise LaunchError("reward_policy.verifiable_only must be true")
    exact = payload.exact_solve_reward
    valid = payload.valid_expression_reward
    invalid = payload.invalid_reward
    format_reward = payload.format_expression_reward
    closeness_weight = payload.closeness_weight
    if not (float(exact) > float(valid) >= float(invalid)):
        raise LaunchError(
            "reward_policy must satisfy exact_solve_reward > valid_expression_reward >= "
            "invalid_reward"
        )
    if "format_expression_reward" in payload.model_fields_set and not (
        float(valid) >= float(format_reward) >= float(invalid)
    ):
        raise LaunchError(
            "reward_policy must satisfy valid_expression_reward >= "
            "format_expression_reward >= invalid_reward"
        )
    if float(closeness_weight) < 0:
        raise LaunchError("reward_policy.closeness_weight must be non-negative")
    if float(exact) < float(valid) + float(closeness_weight) + EXACT_SOLVE_REWARD_MARGIN:
        raise LaunchError(
            "reward_policy.exact_solve_reward must exceed the closeness ceiling "
            "(valid_expression_reward + closeness_weight) by at least "
            f"{EXACT_SOLVE_REWARD_MARGIN}"
        )
    blocked_terms = payload.disallowed_observation_terms
    if not DISALLOWED_REWARD_TERMS.issubset({str(item) for item in blocked_terms}):
        raise LaunchError("reward_policy.disallowed_observation_terms is missing required terms")


def _check_monitoring(payload: RLVRMonitoring) -> None:
    for key, value in (
        ("log_interval", payload.log_interval),
        ("checkpoint_interval", payload.checkpoint_interval),
        ("samples_per_eval_task", payload.samples_per_eval_task),
        ("eval_max_new_tokens", payload.eval_max_new_tokens),
        ("eval_progress_interval_tasks", payload.eval_progress_interval_tasks),
        ("eval_progress_interval_samples", payload.eval_progress_interval_samples),
        ("eval_sample_batch_size", payload.eval_sample_batch_size),
        ("debug_eval_task_budget", payload.debug_eval_task_budget),
        ("debug_samples_per_eval_task", payload.debug_samples_per_eval_task),
    ):
        positive_int(value, f"monitoring.{key}")
    for key, numeric_value in (
        ("eval_wall_timeout_seconds", payload.eval_wall_timeout_seconds),
        ("eval_wall_timeout_seconds_per_sample", payload.eval_wall_timeout_seconds_per_sample),
        ("eval_no_progress_timeout_seconds", payload.eval_no_progress_timeout_seconds),
        (
            "eval_no_progress_timeout_seconds_per_sample",
            payload.eval_no_progress_timeout_seconds_per_sample,
        ),
    ):
        if numeric_value <= 0:
            raise LaunchError(f"monitoring.{key} must be positive")
    if payload.eval_sample_batch_size > payload.samples_per_eval_task:
        raise LaunchError("monitoring.eval_sample_batch_size must be <= samples_per_eval_task")
    if payload.debug_samples_per_eval_task > payload.samples_per_eval_task:
        raise LaunchError("monitoring.debug_samples_per_eval_task must be <= samples_per_eval_task")
    if payload.eval_progress_interval_samples > payload.samples_per_eval_task:
        raise LaunchError(
            "monitoring.eval_progress_interval_samples must be <= samples_per_eval_task"
        )
    str_field(payload.wandb_project, "monitoring.wandb_project")
    if payload.wandb_required_for_modal is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")
    tags = payload.wandb_tags
    if "stage=rlvr" not in tags:
        raise LaunchError("monitoring.wandb_tags must include stage=rlvr")


def _check_eval_monitoring_bounds(monitoring: RLVRMonitoring, budgets: RLVRBudgets) -> None:
    eval_rollouts = int(budgets.eval_task_budget) * int(monitoring.samples_per_eval_task)
    if eval_rollouts > int(budgets.max_eval_rollouts):
        raise LaunchError(
            "eval_task_budget * monitoring.samples_per_eval_task exceeds budgets.max_eval_rollouts"
        )
    if int(monitoring.debug_eval_task_budget) > int(budgets.eval_task_budget):
        raise LaunchError("monitoring.debug_eval_task_budget must be <= budgets.eval_task_budget")


def _check_acceptance(payload: Acceptance) -> None:
    if payload.primary_metric != "valid_expression_rate":
        raise LaunchError("acceptance.primary_metric must be valid_expression_rate")
    if payload.record_no_improvement is not True:
        raise LaunchError("acceptance.record_no_improvement must be true")
    for key, value in (
        ("baseline_valid_expression_rate", payload.baseline_valid_expression_rate),
        ("baseline_exact_solve_rate", payload.baseline_exact_solve_rate),
        ("baseline_pass_at_1", payload.baseline_pass_at_1),
        ("baseline_pass_at_8", payload.baseline_pass_at_8),
        ("baseline_pass_at_32", payload.baseline_pass_at_32),
    ):
        if not 0 <= value <= 1:
            raise LaunchError(f"acceptance.{key} must be a rate in [0, 1]")


def _check_pipeline_smoke(payload: PipelineSmoke, settings: RLVRSettings) -> None:
    budgets, grpo, monitoring = settings.budgets, settings.grpo, settings.monitoring
    if payload.profile != PIPELINE_SMOKE_PROFILE:
        raise LaunchError(f"pipeline_smoke.profile must be {PIPELINE_SMOKE_PROFILE}")
    if payload.enabled is not True:
        raise LaunchError("pipeline_smoke.enabled must be true")
    for key, count in (
        ("train_task_budget", payload.train_task_budget),
        ("max_steps", payload.max_steps),
        ("prompts_per_step", payload.prompts_per_step),
        ("group_size", payload.group_size),
        ("max_new_tokens", payload.max_new_tokens),
        ("max_rollout_tokens", payload.max_rollout_tokens),
        ("eval_task_budget", payload.eval_task_budget),
        ("samples_per_eval_task", payload.samples_per_eval_task),
        ("eval_max_new_tokens", payload.eval_max_new_tokens),
        ("eval_progress_interval_tasks", payload.eval_progress_interval_tasks),
        ("eval_progress_interval_samples", payload.eval_progress_interval_samples),
        ("eval_sample_batch_size", payload.eval_sample_batch_size),
    ):
        positive_int(count, f"pipeline_smoke.{key}")
    warmup_steps = payload.warmup_steps
    if warmup_steps < 0:
        raise LaunchError("pipeline_smoke.warmup_steps must be a non-negative integer")
    for key, value in (
        ("eval_wall_timeout_seconds", payload.eval_wall_timeout_seconds),
        ("eval_no_progress_timeout_seconds", payload.eval_no_progress_timeout_seconds),
    ):
        if value <= 0:
            raise LaunchError(f"pipeline_smoke.{key} must be positive")
    if payload.train_task_budget > int(budgets.train_task_budget):
        raise LaunchError("pipeline_smoke.train_task_budget must be <= budgets.train_task_budget")
    if payload.eval_task_budget > int(budgets.eval_task_budget):
        raise LaunchError("pipeline_smoke.eval_task_budget must be <= budgets.eval_task_budget")
    if payload.max_steps > int(grpo.max_steps):
        raise LaunchError("pipeline_smoke.max_steps must be <= grpo.max_steps")
    if payload.warmup_steps > payload.max_steps:
        raise LaunchError("pipeline_smoke.warmup_steps must be <= pipeline_smoke.max_steps")
    if payload.prompts_per_step > int(grpo.prompts_per_step):
        raise LaunchError("pipeline_smoke.prompts_per_step must be <= grpo.prompts_per_step")
    if payload.group_size > int(grpo.group_size):
        raise LaunchError("pipeline_smoke.group_size must be <= grpo.group_size")
    if payload.samples_per_eval_task > int(monitoring.samples_per_eval_task):
        raise LaunchError(
            "pipeline_smoke.samples_per_eval_task must be <= monitoring.samples_per_eval_task"
        )
    if payload.eval_sample_batch_size > payload.samples_per_eval_task:
        raise LaunchError(
            "pipeline_smoke.eval_sample_batch_size must be <= pipeline_smoke.samples_per_eval_task"
        )
    _check_report_paths(payload, "pipeline_smoke")


def _check_artifacts(payload: Artifacts) -> None:
    _check_report_paths(payload, "artifacts")
    if tuple(payload.required_files) not in (
        EXPECTED_ARTIFACTS,
        EXPECTED_ARTIFACTS_WITH_FINAL_EVAL,
    ):
        raise LaunchError("artifacts.required_files must match the GRPO evidence manifest")


def _check_report_paths(payload: PipelineSmoke | Artifacts, label: str) -> None:
    output_dir = Path(str_field(payload.output_dir, f"{label}.output_dir"))
    if output_dir.is_absolute() or ".." in output_dir.parts or output_dir.parts[:1] != ("runs",):
        raise LaunchError(f"{label}.output_dir must be a relative path under runs/")
    report_path = Path(str_field(payload.report_path, f"{label}.report_path"))
    if report_path.is_absolute() or ".." in report_path.parts:
        raise LaunchError(f"{label}.report_path must stay inside the repository")
    doc_path = Path(str_field(payload.doc_path, f"{label}.doc_path"))
    if doc_path.is_absolute() or ".." in doc_path.parts or doc_path.parts[:1] != ("docs",):
        raise LaunchError(f"{label}.doc_path must be a relative path under docs/")
