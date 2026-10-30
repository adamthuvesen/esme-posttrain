from __future__ import annotations

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr

from esme_posttrain.launch.config_models import (
    FrozenConfig,
    JSONEvidence,
    ModalRuntime,
    OutputArtifacts,
    PaddedSequence,
)


class SFTReference(FrozenConfig):
    volume: StrictStr
    checkpoint_path: StrictStr
    tokenizer_path: StrictStr
    format: StrictStr
    model_family: StrictStr
    read_only: StrictBool
    wandb_run: StrictStr
    best_step: StrictInt


class PreferenceSource(FrozenConfig):
    name: StrictStr
    source: StrictStr
    revision: StrictStr
    license: StrictStr
    split: StrictStr
    role: StrictStr


class PairFiltering(FrozenConfig):
    max_prompt_chars: StrictInt
    max_response_chars: StrictInt
    drop_identical_responses: StrictBool


class Datasets(FrozenConfig):
    non_commercial_training_approved: StrictBool
    preference_train: PreferenceSource
    preference_eval: PreferenceSource
    filtering: PairFiltering


class DPOBudgets(FrozenConfig):
    max_train_pairs: StrictInt
    min_train_pairs: StrictInt
    max_train_tokens: StrictInt
    target_train_tokens: StrictInt
    max_eval_pairs: StrictInt
    min_eval_pairs: StrictInt
    max_eval_tokens: StrictInt
    max_length: StrictInt
    max_prompt_length: StrictInt
    smoke_train_pairs: StrictInt
    smoke_train_tokens: StrictInt
    smoke_eval_pairs: StrictInt


class DPOOptimizer(FrozenConfig):
    name: StrictStr
    learning_rate: StrictFloat | StrictInt
    micro_batch_size: StrictInt
    gradient_accumulation_steps: StrictInt
    effective_batch_size: StrictInt
    epochs: StrictInt
    max_steps: StrictInt
    smoke_max_steps: StrictInt
    seed: StrictInt
    grad_clip: StrictFloat | StrictInt
    scheduler: StrictStr
    warmup_ratio: StrictFloat | StrictInt
    weight_decay: StrictFloat | StrictInt


class DPOObjective(FrozenConfig):
    loss_type: StrictStr
    beta: StrictFloat | StrictInt
    length_normalized: StrictBool
    reference_free: StrictBool
    auxiliary_sft_loss: StrictBool


class DPOMonitoring(FrozenConfig):
    log_interval: StrictInt
    eval_interval: StrictInt
    checkpoint_interval: StrictInt
    sample_new_tokens: StrictInt
    wandb_project: StrictStr
    wandb_required_for_modal: StrictBool
    wandb_tags: tuple[StrictStr, ...]
    log_chosen_rejected_logps: StrictBool


class LearningGate(FrozenConfig):
    full_run_requires_beta_sweep: StrictBool
    swept_betas: tuple[StrictFloat, ...]
    evidence: JSONEvidence | None


class Acceptance(FrozenConfig):
    preference_accuracy_must_beat_reference: StrictBool
    no_chosen_logp_collapse: StrictBool
    repetition_or_length_must_improve: StrictBool
    proxies: tuple[StrictStr, ...]


class DPOSettings(FrozenConfig):
    schema_version: StrictInt
    run_id: StrictStr
    run_card: StrictStr
    requires_approval: StrictBool
    artifact_name: StrictStr
    starts_from: StrictStr
    stage: StrictStr
    sft_reference: SFTReference
    datasets: Datasets
    budgets: DPOBudgets
    optimizer: DPOOptimizer
    dpo: DPOObjective
    sequence: PaddedSequence
    runtime: ModalRuntime
    monitoring: DPOMonitoring
    artifacts: OutputArtifacts
    learning_gate: LearningGate
    acceptance: Acceptance
