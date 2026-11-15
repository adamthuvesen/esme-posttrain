from __future__ import annotations

from typing import Self

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from esme_posttrain.launch.config_models import (
    BundleConfig,
    FrozenConfig,
    JSONEvidence,
    OutputArtifacts,
    PaddedSequence,
    SmokeModalRuntime,
    check_output_artifacts,
    check_stage_runtime,
    positive_int,
    str_field,
)
from esme_posttrain.launch.errors import LaunchError

# Hard cap for the approved multi-turn foundation run.
MULTITURN_FULL_RUN_SPEND_CAP_USD = 40.0

RUN_ID = "esme_214m_sft_multiturn"
RUN_CARD = "run_cards/esme-214m-sft-multiturn.md"
ARTIFACT_NAME = "Esme-214M-Instruct"
MODAL_VOLUME = "esme-posttrain-esme-sft-multiturn"
MAX_APPROVED_TRAIN_SAMPLES = 50_000
MAX_APPROVED_TRAIN_TOKENS = 80_000_000
EFFECTIVE_BATCH_SIZE = 16

EXPECTED_TRAIN_DATASETS: dict[str, dict[str, str]] = {
    "smol-smoltalk": {
        "source": "HuggingFaceTB/smol-smoltalk",
        "revision": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "license": "apache-2.0",
        "split": "train",
    },
    "tulu-3-personas": {
        "source": "allenai/tulu-3-sft-personas-instruction-following",
        "revision": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        "license": "odc-by",
        "split": "train",
    },
}
EXPECTED_EVAL_DATASET: dict[str, str | bool] = {
    "name": "no_robots",
    "source": "HuggingFaceH4/no_robots",
    "revision": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
    "license": "cc-by-nc-4.0",
    "split": "test",
    "train_allowed": False,
}
EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "data-report.json",
    "selected-row-manifest.jsonl",
    "eval-smol-smoltalk-manifest.jsonl",
    "eval-tulu-3-personas-manifest.jsonl",
    "eval-no_robots-manifest.jsonl",
    "metrics.jsonl",
    "checkpoint.pt",
    "best-checkpoint.pt",
    "best-checkpoint.json",
    "samples.md",
    "multi-turn-samples.md",
    "tokenizer.json",
    "manifest.json",
    "eval-report.json",
    "cost.json",
    "environment.txt",
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
        if self.starts_from != "Esme-214M-Base":
            raise LaunchError("starts_from must be Esme-214M-Base")
        if self.stage != "sft":
            raise LaunchError("stage must be sft")
        _check_datasets(self.datasets)
        _check_budgets(self.budgets)
        _check_optimizer(self.optimizer)
        _check_loss(self.loss)
        if self.tuning.mode != "full":
            raise LaunchError("only tuning.mode='full' is supported")
        _check_sequence(self.sequence)
        check_stage_runtime(
            self.runtime,
            full_run_spend_cap_usd=MULTITURN_FULL_RUN_SPEND_CAP_USD,
            modal_volume=MODAL_VOLUME,
        )
        _check_monitoring(self.monitoring)
        check_output_artifacts(
            self.artifacts,
            expected_files=EXPECTED_ARTIFACTS,
            manifest_label="multi-turn SFT evidence manifest",
        )
        if self.learning_gate.full_run_requires_matched_sweep is not True:
            raise LaunchError("learning_gate.full_run_requires_matched_sweep must be true")
        _check_acceptance(self.acceptance)
        return self


def _check_datasets(payload: Datasets) -> None:
    if payload.non_commercial_training_approved is not False:
        raise LaunchError("datasets.non_commercial_training_approved must stay false")
    if not 1 <= len(payload.train_mix) <= 2:
        raise LaunchError("datasets.train_mix must contain one or two sources")
    for source in payload.train_mix:
        _check_train_source(source)
    names = {source.name for source in payload.train_mix}
    if "smol-smoltalk" not in names:
        raise LaunchError("datasets.train_mix must include smol-smoltalk")
    ratio_sum = sum(source.mix_ratio for source in payload.train_mix)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise LaunchError("datasets.train_mix ratios must sum to 1.0")
    _check_eval_source(payload.eval_holdout)


def _check_train_source(payload: TrainingSource) -> None:
    name = str_field(payload.name, "datasets.train_mix[].name")
    if name not in EXPECTED_TRAIN_DATASETS:
        raise LaunchError(f"unsupported training dataset: {name}")
    expected = EXPECTED_TRAIN_DATASETS[name]
    for key, value in (
        ("source", payload.source),
        ("revision", payload.revision),
        ("license", payload.license),
        ("split", payload.split),
    ):
        if value != expected[key]:
            raise LaunchError(f"dataset {name}.{key} must be {expected[key]}")
    if payload.role != "train":
        raise LaunchError(f"dataset {name}.role must be train")
    if not 0 < payload.mix_ratio <= 1:
        raise LaunchError(f"dataset {name}.mix_ratio must be in (0, 1]")
    positive_int(payload.filters.max_prompt_chars, f"dataset {name}.filters.max_prompt_chars")
    positive_int(payload.filters.max_response_chars, f"dataset {name}.filters.max_response_chars")


def _check_eval_source(payload: EvalSource) -> None:
    for key, actual in (
        ("name", payload.name),
        ("source", payload.source),
        ("revision", payload.revision),
        ("license", payload.license),
        ("split", payload.split),
        ("train_allowed", payload.train_allowed),
    ):
        expected = EXPECTED_EVAL_DATASET[key]
        if actual != expected:
            raise LaunchError(f"datasets.eval_holdout.{key} must be {expected}")
    if payload.role != "eval":
        raise LaunchError("datasets.eval_holdout.role must be eval")
    if payload.usage != "eval_only":
        raise LaunchError("datasets.eval_holdout.usage must be eval_only")


def _check_budgets(payload: SFTBudgets) -> None:
    max_train_samples = positive_int(payload.max_train_samples, "budgets.max_train_samples")
    max_train_tokens = positive_int(payload.max_train_tokens, "budgets.max_train_tokens")
    target_train_tokens = positive_int(payload.target_train_tokens, "budgets.target_train_tokens")
    if max_train_samples > MAX_APPROVED_TRAIN_SAMPLES:
        raise LaunchError(f"budgets.max_train_samples must be <= {MAX_APPROVED_TRAIN_SAMPLES}")
    if max_train_tokens > MAX_APPROVED_TRAIN_TOKENS:
        raise LaunchError(f"budgets.max_train_tokens must be <= {MAX_APPROVED_TRAIN_TOKENS}")
    if target_train_tokens > max_train_tokens:
        raise LaunchError("budgets.target_train_tokens must be <= budgets.max_train_tokens")
    if payload.max_sequence_tokens != 1024:
        raise LaunchError(
            "budgets.max_sequence_tokens must be 1024 to match the Esme-214M-Base context"
        )
    for key, value in (
        ("max_eval_samples", payload.max_eval_samples),
        ("max_eval_tokens", payload.max_eval_tokens),
        ("matched_eval_samples_per_source", payload.matched_eval_samples_per_source),
        ("matched_eval_tokens_per_source", payload.matched_eval_tokens_per_source),
        ("max_sequence_tokens", payload.max_sequence_tokens),
        ("smoke_train_samples", payload.smoke_train_samples),
        ("smoke_train_tokens", payload.smoke_train_tokens),
        ("smoke_eval_samples", payload.smoke_eval_samples),
    ):
        positive_int(value, f"budgets.{key}")
    if payload.smoke_train_samples > max_train_samples:
        raise LaunchError("budgets.smoke_train_samples must be <= budgets.max_train_samples")
    if payload.smoke_train_tokens > max_train_tokens:
        raise LaunchError("budgets.smoke_train_tokens must be <= budgets.max_train_tokens")


def _check_optimizer(payload: SFTOptimizer) -> None:
    if payload.name != "AdamW":
        raise LaunchError("optimizer.name must be AdamW")
    if payload.learning_rate <= 0:
        raise LaunchError("optimizer.learning_rate must be positive")
    for key, value in (
        ("micro_batch_size", payload.micro_batch_size),
        ("gradient_accumulation_steps", payload.gradient_accumulation_steps),
        ("effective_batch_size", payload.effective_batch_size),
        ("max_steps", payload.max_steps),
        ("smoke_max_steps", payload.smoke_max_steps),
        ("seed", payload.seed),
    ):
        positive_int(value, f"optimizer.{key}")
    if (
        payload.micro_batch_size * payload.gradient_accumulation_steps
        != payload.effective_batch_size
    ):
        raise LaunchError(
            "optimizer.effective_batch_size must equal micro_batch_size * "
            "gradient_accumulation_steps"
        )
    if payload.effective_batch_size != EFFECTIVE_BATCH_SIZE:
        raise LaunchError(
            f"optimizer.effective_batch_size must be {EFFECTIVE_BATCH_SIZE} for this recipe"
        )
    if payload.smoke_max_steps > payload.max_steps:
        raise LaunchError("optimizer.smoke_max_steps must be <= optimizer.max_steps")
    if payload.grad_clip <= 0:
        raise LaunchError("optimizer.grad_clip must be positive")
    if payload.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError(
            "optimizer.scheduler must be constant, linear_warmup_decay, or cosine_decay"
        )
    if payload.warmup_steps < 0:
        raise LaunchError("optimizer.warmup_steps must be a non-negative integer")
    if payload.warmup_steps > payload.max_steps:
        raise LaunchError("optimizer.warmup_steps must be <= optimizer.max_steps")
    if payload.weight_decay < 0:
        raise LaunchError("optimizer.weight_decay must be non-negative")


def _check_loss(payload: SFTLoss) -> None:
    if payload.assistant_only_loss is not True:
        raise LaunchError("loss.assistant_only_loss must be true")
    if payload.completion_only_loss is not True:
        raise LaunchError("loss.completion_only_loss must be true")
    if payload.ignore_index != -100:
        raise LaunchError("loss.ignore_index must be -100")


def _check_sequence(payload: UnpackedSequence) -> None:
    if payload.sequence_packing is not False:
        raise LaunchError("sequence.sequence_packing must be false until packing is implemented")
    value = payload.pad_to_multiple_of
    if value is not None and value <= 0:
        raise LaunchError("sequence.pad_to_multiple_of must be a positive integer or null")
    str_field(payload.no_packing_rationale, "sequence.no_packing_rationale")


def _check_monitoring(payload: SFTMonitoring) -> None:
    positive_int(payload.log_interval, "monitoring.log_interval")
    positive_int(payload.eval_interval, "monitoring.eval_interval")
    positive_int(payload.checkpoint_interval, "monitoring.checkpoint_interval")
    positive_int(payload.retain_last_checkpoints, "monitoring.retain_last_checkpoints")
    positive_int(payload.early_stopping_patience, "monitoring.early_stopping_patience")
    if payload.no_robots_catastrophic_regression_multiplier <= 1.0:
        raise LaunchError("monitoring.no_robots_catastrophic_regression_multiplier must be > 1")
    positive_int(payload.sample_new_tokens, "monitoring.sample_new_tokens")
    str_field(payload.wandb_project, "monitoring.wandb_project")
    if payload.wandb_required_for_modal is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")


def _check_acceptance(payload: Acceptance) -> None:
    if payload.heldout_response_loss_required is not True:
        raise LaunchError("acceptance.heldout_response_loss_required must be true")
    if payload.sft_must_beat_base is not True:
        raise LaunchError("acceptance.sft_must_beat_base must be true")
    str_field(payload.eval_dataset, "acceptance.eval_dataset")
