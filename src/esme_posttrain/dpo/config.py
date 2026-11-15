from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from esme_posttrain.launch.config_models import (
    FrozenConfig,
    JSONEvidence,
    ModalRuntime,
    OutputArtifacts,
    PaddedSequence,
    check_output_artifacts,
    check_stage_runtime,
    positive_int,
    str_field,
)
from esme_posttrain.launch.errors import LaunchError

# DPO is far cheaper than SFT (one bounded pass on UltraFeedback at lr 1e-6); the
# full-run runaway cap is intentionally lower than the SFT $40 cap.
DPO_FULL_RUN_SPEND_CAP_USD = 15.0

RUN_ID = "esme_214m_chat_dpo"
RUN_CARD = "run_cards/esme-214m-chat-dpo.md"
ARTIFACT_NAME = "Esme-214M-Chat"
REFERENCE_ARTIFACT_NAME = "Esme-214M-Instruct"
MODAL_VOLUME = "esme-posttrain-esme-chat-dpo"
SFT_FOUNDATION_VOLUME = "esme-posttrain-esme-sft-multiturn"
MAX_APPROVED_TRAIN_PAIRS = 64_000
MAX_APPROVED_TRAIN_TOKENS = 200_000_000

_ULTRAFEEDBACK_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
EXPECTED_PREFERENCE_DATASET: dict[str, str] = {
    "name": "ultrafeedback-binarized",
    "source": "HuggingFaceH4/ultrafeedback_binarized",
    "revision": _ULTRAFEEDBACK_REVISION,
    "license": "mit",
    "split": "train_prefs",
}
EXPECTED_EVAL_DATASET: dict[str, str] = {
    "name": "ultrafeedback-binarized",
    "source": "HuggingFaceH4/ultrafeedback_binarized",
    "revision": _ULTRAFEEDBACK_REVISION,
    "license": "mit",
    "split": "test_prefs",
}
EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "data-report.json",
    "selected-pair-manifest.jsonl",
    "eval-pair-manifest.jsonl",
    "decoding-precheck.json",
    "metrics.jsonl",
    "checkpoint.pt",
    "best-checkpoint.pt",
    "best-checkpoint.json",
    "chat-samples.md",
    "tokenizer.json",
    "manifest.json",
    "eval-report.json",
    "cost.json",
    "environment.txt",
)
# The bounded beta sweep the launch learning-gate requires.
EXPECTED_SWEEP_BETAS: tuple[float, ...] = (0.1, 0.3, 0.5)


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
        if self.starts_from != REFERENCE_ARTIFACT_NAME:
            raise LaunchError(f"starts_from must be {REFERENCE_ARTIFACT_NAME}")
        if self.stage != "dpo":
            raise LaunchError("stage must be dpo")
        _check_sft_reference(self.sft_reference)
        _check_datasets(self.datasets)
        _check_budgets(self.budgets)
        _check_optimizer(self.optimizer)
        _check_dpo(self.dpo)
        _check_sequence(self.sequence)
        check_stage_runtime(
            self.runtime,
            full_run_spend_cap_usd=DPO_FULL_RUN_SPEND_CAP_USD,
            modal_volume=MODAL_VOLUME,
        )
        _check_monitoring(self.monitoring)
        check_output_artifacts(
            self.artifacts,
            expected_files=EXPECTED_ARTIFACTS,
            manifest_label="DPO evidence manifest",
        )
        _check_learning_gate(self.learning_gate)
        _check_acceptance(self.acceptance)
        return self


def _check_sft_reference(payload: SFTReference) -> None:
    if payload.volume != SFT_FOUNDATION_VOLUME:
        raise LaunchError(f"sft_reference.volume must be {SFT_FOUNDATION_VOLUME}")
    if payload.format != "llm_posttrain_instruct_sft_v1":
        raise LaunchError("sft_reference.format must be llm_posttrain_instruct_sft_v1")
    if payload.model_family != "DenseBackbone":
        raise LaunchError("sft_reference.model_family must be DenseBackbone")
    if payload.read_only is not True:
        raise LaunchError("sft_reference.read_only must be true")
    str_field(payload.checkpoint_path, "sft_reference.checkpoint_path")
    str_field(payload.tokenizer_path, "sft_reference.tokenizer_path")
    str_field(payload.wandb_run, "sft_reference.wandb_run")
    positive_int(payload.best_step, "sft_reference.best_step")


def _check_datasets(payload: Datasets) -> None:
    if payload.non_commercial_training_approved is not False:
        raise LaunchError("datasets.non_commercial_training_approved must stay false")
    _check_preference_source(
        payload.preference_train, expected=EXPECTED_PREFERENCE_DATASET, role="train"
    )
    _check_preference_source(payload.preference_eval, expected=EXPECTED_EVAL_DATASET, role="eval")
    _check_filtering(payload.filtering)


def _check_preference_source(
    payload: PreferenceSource, *, expected: Mapping[str, str], role: str
) -> None:
    for key, value in (
        ("name", payload.name),
        ("source", payload.source),
        ("revision", payload.revision),
        ("license", payload.license),
        ("split", payload.split),
    ):
        if value != expected[key]:
            raise LaunchError(f"datasets.preference_{role}.{key} must be {expected[key]}")
    if payload.role != role:
        raise LaunchError(f"datasets.preference_{role}.role must be {role}")


def _check_filtering(payload: PairFiltering) -> None:
    positive_int(payload.max_prompt_chars, "datasets.filtering.max_prompt_chars")
    positive_int(payload.max_response_chars, "datasets.filtering.max_response_chars")
    if payload.drop_identical_responses is not True:
        raise LaunchError("datasets.filtering.drop_identical_responses must be true")


def _check_budgets(payload: DPOBudgets) -> None:
    max_train_pairs = positive_int(payload.max_train_pairs, "budgets.max_train_pairs")
    max_train_tokens = positive_int(payload.max_train_tokens, "budgets.max_train_tokens")
    target_train_tokens = positive_int(payload.target_train_tokens, "budgets.target_train_tokens")
    if max_train_pairs > MAX_APPROVED_TRAIN_PAIRS:
        raise LaunchError(f"budgets.max_train_pairs must be <= {MAX_APPROVED_TRAIN_PAIRS}")
    if max_train_tokens > MAX_APPROVED_TRAIN_TOKENS:
        raise LaunchError(f"budgets.max_train_tokens must be <= {MAX_APPROVED_TRAIN_TOKENS}")
    if target_train_tokens > max_train_tokens:
        raise LaunchError("budgets.target_train_tokens must be <= budgets.max_train_tokens")
    if payload.max_length != 1024:
        raise LaunchError("budgets.max_length must be 1024 to match the Esme-214M context")
    if payload.max_prompt_length != 512:
        raise LaunchError("budgets.max_prompt_length must be 512 (SmolLM2-360M DPO config)")
    if payload.max_prompt_length >= payload.max_length:
        raise LaunchError("budgets.max_prompt_length must be < budgets.max_length")
    for key, value in (
        ("min_train_pairs", payload.min_train_pairs),
        ("max_eval_pairs", payload.max_eval_pairs),
        ("min_eval_pairs", payload.min_eval_pairs),
        ("max_eval_tokens", payload.max_eval_tokens),
        ("smoke_train_pairs", payload.smoke_train_pairs),
        ("smoke_train_tokens", payload.smoke_train_tokens),
        ("smoke_eval_pairs", payload.smoke_eval_pairs),
    ):
        positive_int(value, f"budgets.{key}")
    # max_*_pairs are caps; min_*_pairs are the sufficiency floors. Selecting fewer
    # than the cap (UltraFeedback is length-filtered at max_length=1024) is normal;
    # the floor is the minimum clean pairs a run needs to be sufficient.
    if payload.min_train_pairs > max_train_pairs:
        raise LaunchError(
            "budgets.min_train_pairs (floor) must be <= budgets.max_train_pairs (cap)"
        )
    if payload.min_eval_pairs > payload.max_eval_pairs:
        raise LaunchError("budgets.min_eval_pairs (floor) must be <= budgets.max_eval_pairs (cap)")
    if payload.smoke_train_pairs > max_train_pairs:
        raise LaunchError("budgets.smoke_train_pairs must be <= budgets.max_train_pairs")
    if payload.smoke_train_tokens > max_train_tokens:
        raise LaunchError("budgets.smoke_train_tokens must be <= budgets.max_train_tokens")


def _check_optimizer(payload: DPOOptimizer) -> None:
    if payload.name != "adamw_torch":
        raise LaunchError("optimizer.name must be adamw_torch")
    lr = payload.learning_rate
    if lr <= 0:
        raise LaunchError("optimizer.learning_rate must be positive")
    if lr > 1e-5:
        raise LaunchError("optimizer.learning_rate must be <= 1e-5 (DPO LR is ~10-100x below SFT)")
    for key, value in (
        ("micro_batch_size", payload.micro_batch_size),
        ("gradient_accumulation_steps", payload.gradient_accumulation_steps),
        ("effective_batch_size", payload.effective_batch_size),
        ("epochs", payload.epochs),
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
            "optimizer.effective_batch_size must equal "
            "micro_batch_size * gradient_accumulation_steps"
        )
    if payload.smoke_max_steps > payload.max_steps:
        raise LaunchError("optimizer.smoke_max_steps must be <= optimizer.max_steps")
    if payload.grad_clip <= 0:
        raise LaunchError("optimizer.grad_clip must be positive")
    if payload.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError(
            "optimizer.scheduler must be constant, linear_warmup_decay, or cosine_decay"
        )
    warmup_ratio = payload.warmup_ratio
    if not 0 <= warmup_ratio < 1:
        raise LaunchError("optimizer.warmup_ratio must be in [0, 1)")
    if payload.weight_decay < 0:
        raise LaunchError("optimizer.weight_decay must be non-negative")


def _check_dpo(payload: DPOObjective) -> None:
    if payload.loss_type != "sigmoid":
        raise LaunchError("dpo.loss_type must be sigmoid (vanilla DPO); SimPO/IPO are out of scope")
    beta = payload.beta
    if beta <= 0:
        raise LaunchError("dpo.beta must be positive")
    if payload.reference_free is not False:
        raise LaunchError("dpo.reference_free must be false (frozen SFT reference is required)")
    if payload.auxiliary_sft_loss is not False:
        raise LaunchError(
            "dpo.auxiliary_sft_loss must be false (measured net-negative; not added by reflex)"
        )


def _check_sequence(payload: PaddedSequence) -> None:
    value = payload.pad_to_multiple_of
    if value is not None and value <= 0:
        raise LaunchError("sequence.pad_to_multiple_of must be a positive integer or null")


def _check_monitoring(payload: DPOMonitoring) -> None:
    positive_int(payload.log_interval, "monitoring.log_interval")
    positive_int(payload.eval_interval, "monitoring.eval_interval")
    positive_int(payload.checkpoint_interval, "monitoring.checkpoint_interval")
    positive_int(payload.sample_new_tokens, "monitoring.sample_new_tokens")
    str_field(payload.wandb_project, "monitoring.wandb_project")
    if payload.wandb_required_for_modal is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")
    tags = payload.wandb_tags
    if "stage=dpo" not in tags:
        raise LaunchError("monitoring.wandb_tags must include stage=dpo")
    if payload.log_chosen_rejected_logps is not True:
        raise LaunchError("monitoring.log_chosen_rejected_logps must be true (displacement watch)")


def _check_learning_gate(payload: LearningGate) -> None:
    if payload.full_run_requires_beta_sweep is not True:
        raise LaunchError("learning_gate.full_run_requires_beta_sweep must be true")
    swept = payload.swept_betas
    if tuple(float(b) for b in swept) != EXPECTED_SWEEP_BETAS:
        raise LaunchError(f"learning_gate.swept_betas must be {list(EXPECTED_SWEEP_BETAS)}")


def _check_acceptance(payload: Acceptance) -> None:
    if payload.preference_accuracy_must_beat_reference is not True:
        raise LaunchError("acceptance.preference_accuracy_must_beat_reference must be true")
    if payload.no_chosen_logp_collapse is not True:
        raise LaunchError("acceptance.no_chosen_logp_collapse must be true")
    if payload.repetition_or_length_must_improve is not True:
        raise LaunchError("acceptance.repetition_or_length_must_improve must be true")
    proxies = payload.proxies
    if len(proxies) < 3:
        raise LaunchError("acceptance.proxies must list the primary cheap proxy signals")
