from __future__ import annotations

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

from esme_posttrain.launch.config_models import (
    BundleConfig,
    FrozenConfig,
    JSONEvidence,
    OutputArtifacts,
    PaddedSequence,
    SmokeModalRuntime,
)


class SourceFilters(FrozenConfig):
    multi_turn_included: StrictBool = False
    capacity_filtered_subsets: tuple[StrictStr, ...] = ()
    max_prompt_chars: StrictInt
    max_response_chars: StrictInt
    include_constraints_in_prompt: StrictBool = False


class TrainingSource(FrozenConfig):
    name: StrictStr
    source: StrictStr
    revision: StrictStr
    license: StrictStr
    split: StrictStr
    role: StrictStr
    mix_ratio: StrictFloat | StrictInt
    filters: SourceFilters


class EvalSource(FrozenConfig):
    name: StrictStr
    source: StrictStr
    revision: StrictStr
    license: StrictStr
    split: StrictStr
    role: StrictStr
    usage: StrictStr
    train_allowed: StrictBool


class Datasets(FrozenConfig):
    non_commercial_training_approved: StrictBool
    train_mix: tuple[TrainingSource, ...]
    eval_holdout: EvalSource


class SFTBudgets(FrozenConfig):
    max_train_samples: StrictInt
    max_train_tokens: StrictInt
    target_train_tokens: StrictInt
    max_eval_samples: StrictInt
    max_eval_tokens: StrictInt
    matched_eval_samples_per_source: StrictInt
    matched_eval_tokens_per_source: StrictInt
    max_sequence_tokens: StrictInt
    smoke_train_samples: StrictInt
    smoke_train_tokens: StrictInt
    smoke_eval_samples: StrictInt


class SFTOptimizer(FrozenConfig):
    name: StrictStr
    learning_rate: StrictFloat | StrictInt
    micro_batch_size: StrictInt
    gradient_accumulation_steps: StrictInt
    effective_batch_size: StrictInt
    max_steps: StrictInt
    smoke_max_steps: StrictInt
    seed: StrictInt
    grad_clip: StrictFloat | StrictInt
    scheduler: StrictStr
    warmup_steps: StrictInt
    weight_decay: StrictFloat | StrictInt


class SFTLoss(FrozenConfig):
    assistant_only_loss: StrictBool
    completion_only_loss: StrictBool
    ignore_index: StrictInt


class FullTuning(FrozenConfig):
    mode: StrictStr


class UnpackedSequence(PaddedSequence):
    sequence_packing: StrictBool
    no_packing_rationale: StrictStr


class SFTMonitoring(FrozenConfig):
    log_interval: StrictInt
    eval_interval: StrictInt
    checkpoint_interval: StrictInt
    retain_last_checkpoints: StrictInt
    early_stopping_patience: StrictInt
    no_robots_catastrophic_regression_multiplier: StrictFloat | StrictInt
    sample_new_tokens: StrictInt
    wandb_project: StrictStr
    wandb_required_for_modal: StrictBool


class LearningGate(FrozenConfig):
    full_run_requires_matched_sweep: StrictBool
    evidence: JSONEvidence | None


class Acceptance(FrozenConfig):
    heldout_response_loss_required: StrictBool
    sft_must_beat_base: StrictBool
    eval_dataset: StrictStr


class MultiTurnSettings(FrozenConfig):
    schema_version: StrictInt
    run_id: StrictStr
    run_card: StrictStr
    requires_approval: StrictBool
    artifact_name: StrictStr
    starts_from: StrictStr
    stage: StrictStr
    base_bundle: BundleConfig
    datasets: Datasets
    budgets: SFTBudgets
    optimizer: SFTOptimizer
    loss: SFTLoss
    tuning: FullTuning
    sequence: UnpackedSequence
    runtime: SmokeModalRuntime
    monitoring: SFTMonitoring
    artifacts: OutputArtifacts
    learning_gate: LearningGate
    acceptance: Acceptance
