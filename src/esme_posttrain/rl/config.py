from __future__ import annotations

from typing import Literal

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

from esme_posttrain.launch.config_models import (
    BundleConfig,
    FrozenConfig,
    ModalRuntime,
    OutputArtifacts,
)


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
