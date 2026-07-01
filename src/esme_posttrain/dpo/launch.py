"""Approval-gated launch validation for the Esme-214M-Chat DPO polish stage.

Mirrors the multi-turn SFT launch contract (``sft_multiturn_launch``): the dry-run
never starts Modal, smoke spend is capped at $2 with no env bypass, and the full
run refuses without ``--approved`` AND bounded beta-sweep learning-gate evidence.
It pins the DPO recipe: its own run_id, a separate output stem + Volume
(``esme-214m-chat-dpo``), warm-start from the accepted SFT foundation as the
frozen reference, ``loss_type=sigmoid`` vanilla DPO, the SmolLM2-360M config
(beta 0.5 / lr 1e-6 / cosine warmup 0.1 / 2 epochs / max_length 1024 /
max_prompt_length 512), a length-normalization toggle, and W&B ``stage=dpo`` tags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_posttrain.launch.common import (
    IMAGE_PACKAGE_PINS,
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    LaunchError,
    build_modal_launch_command,
    estimate_cost_usd,
    load_json_object,
    validate_modal_runtime,
    validate_output_artifacts,
)
from esme_posttrain.launch.common import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.launch.common import (
    object_field as _object,
)
from esme_posttrain.launch.common import (
    positive_int as _positive_int,
)
from esme_posttrain.launch.common import (
    require_keys as _require_keys,
)
from esme_posttrain.launch.common import (
    smoke_launch_blockers as _common_smoke_launch_blockers,
)
from esme_posttrain.launch.common import (
    str_field as _str,
)
from esme_posttrain.launch.models import RuntimeBlock
from esme_posttrain.sft.data import DatasetSource

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
EXPECTED_PREFERENCE_DATASET: dict[str, Any] = {
    "name": "ultrafeedback-binarized",
    "source": "HuggingFaceH4/ultrafeedback_binarized",
    "revision": _ULTRAFEEDBACK_REVISION,
    "license": "mit",
    "split": "train_prefs",
}
EXPECTED_EVAL_DATASET: dict[str, Any] = {
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
LEARNING_GATE_BLOCKER = (
    "full Esme-214M-Chat DPO launch requires learning_gate.evidence with "
    "bounded_beta_sweep proof where held-out preference accuracy improves versus "
    "the SFT reference and chosen-logp does not collapse"
)
# The bounded beta sweep the launch learning-gate requires.
EXPECTED_SWEEP_BETAS: tuple[float, ...] = (0.1, 0.3, 0.5)


@dataclass(frozen=True)
class DPOLaunchConfig:
    payload: dict[str, Any]
    config_path: Path
    sft_reference_volume_path: Path
    preference_source: DatasetSource
    eval_source: DatasetSource
    output_dir: Path
    estimated_full_cost_usd: float
    estimated_smoke_cost_usd: float
    smoke_launch_command: str
    full_launch_command: str

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def artifact_name(self) -> str:
        return str(self.payload["artifact_name"])

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self.payload["runtime"])

    @property
    def budgets(self) -> dict[str, Any]:
        return dict(self.payload["budgets"])

    @property
    def optimizer(self) -> dict[str, Any]:
        return dict(self.payload["optimizer"])

    @property
    def selected_gpu_profile(self) -> dict[str, Any]:
        runtime = self.runtime
        return dict(runtime["gpu_profiles"][runtime["selected_gpu"]])


def load_dpo_config(config_path: Path) -> DPOLaunchConfig:
    config_path, payload = load_json_object(config_path)
    return validate_dpo_payload(payload, config_path)


def validate_dpo_payload(payload: dict[str, Any], config_path: Path) -> DPOLaunchConfig:
    _require_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "run_card",
            "requires_approval",
            "artifact_name",
            "starts_from",
            "stage",
            "sft_reference",
            "datasets",
            "budgets",
            "optimizer",
            "dpo",
            "sequence",
            "runtime",
            "monitoring",
            "artifacts",
            "learning_gate",
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
    if payload["starts_from"] != REFERENCE_ARTIFACT_NAME:
        raise LaunchError(f"starts_from must be {REFERENCE_ARTIFACT_NAME}")
    if payload["stage"] != "dpo":
        raise LaunchError("stage must be dpo")

    sft_reference_path = _validate_sft_reference(_object(payload["sft_reference"], "sft_reference"))
    preference_source, eval_source = _validate_datasets(_object(payload["datasets"], "datasets"))
    budgets = _validate_budgets(_object(payload["budgets"], "budgets"))
    _validate_optimizer(_object(payload["optimizer"], "optimizer"))
    _validate_dpo(_object(payload["dpo"], "dpo"))
    _validate_sequence(_object(payload["sequence"], "sequence"))
    runtime = validate_modal_runtime(
        _object(payload["runtime"], "runtime"),
        full_run_spend_cap_usd=DPO_FULL_RUN_SPEND_CAP_USD,
        full_run_cap_label=f"{DPO_FULL_RUN_SPEND_CAP_USD:.0f}",
        modal_volume=MODAL_VOLUME,
        require_smoke_profile_metrics=False,
    )
    _validate_monitoring(_object(payload["monitoring"], "monitoring"))
    output_dir = validate_output_artifacts(
        _object(payload["artifacts"], "artifacts"),
        expected_files=EXPECTED_ARTIFACTS,
        manifest_label="DPO evidence manifest",
    )
    _validate_learning_gate(_object(payload["learning_gate"], "learning_gate"))
    _validate_acceptance(_object(payload["acceptance"], "acceptance"))
    _validate_abort_rules(payload["abort_rules"])

    runtime_block = RuntimeBlock.from_validated_payload(runtime)
    estimated_full_cost = estimate_cost_usd(
        tokens=int(budgets["target_train_tokens"]),
        projected_tokens_per_second=runtime_block.selected_profile.projected_tokens_per_second,
        usd_per_hour=runtime_block.selected_profile.usd_per_hour,
    )
    estimated_smoke_cost = estimate_cost_usd(
        tokens=int(budgets["smoke_train_tokens"]),
        projected_tokens_per_second=runtime_block.selected_profile.projected_tokens_per_second,
        usd_per_hour=runtime_block.selected_profile.usd_per_hour,
    )

    return DPOLaunchConfig(
        payload=payload,
        config_path=config_path,
        sft_reference_volume_path=sft_reference_path,
        preference_source=preference_source,
        eval_source=eval_source,
        output_dir=output_dir,
        estimated_full_cost_usd=estimated_full_cost,
        estimated_smoke_cost_usd=estimated_smoke_cost,
        smoke_launch_command=_launch_command(config_path, runtime, mode="modal-smoke"),
        full_launch_command=_launch_command(config_path, runtime, mode="full-run"),
    )


def build_dpo_dry_run(
    config: DPOLaunchConfig,
    *,
    full_run_approved: bool = False,
    full_run_modal_gpu: str | None = None,
) -> dict[str, Any]:
    smoke_blockers = smoke_launch_blockers(config)
    full_blockers = full_launch_blockers(
        config, approved=full_run_approved, modal_gpu=full_run_modal_gpu
    )
    if full_run_modal_gpu is not None:
        status = "ready_for_full_run" if not full_blockers else "blocked_by_launch_safety"
    else:
        status = "ready_for_modal_smoke" if not smoke_blockers else "blocked_by_launch_safety"
    return {
        "status": status,
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "stage": config.payload["stage"],
        "starts_from": config.payload["starts_from"],
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "sft_reference": config.payload["sft_reference"],
        "datasets": config.payload["datasets"],
        "budgets": config.payload["budgets"],
        "optimizer": config.payload["optimizer"],
        "dpo": config.payload["dpo"],
        "sequence": config.payload["sequence"],
        "runtime": {
            **config.payload["runtime"],
            "estimated_full_cost_usd": round(config.estimated_full_cost_usd, 4),
            "estimated_smoke_cost_usd": round(config.estimated_smoke_cost_usd, 4),
        },
        "monitoring": config.payload["monitoring"],
        "artifacts": {
            "output_dir": str(config.output_dir),
            "required_files": config.payload["artifacts"]["required_files"],
        },
        "learning_gate": config.payload["learning_gate"],
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION, **IMAGE_PACKAGE_PINS},
        "acceptance": config.payload["acceptance"],
        "abort_rules": config.payload["abort_rules"],
        "launch_blockers": smoke_blockers,
        "full_launch_blockers": full_blockers,
        "modal_smoke_command": config.smoke_launch_command,
        "full_launch_command": config.full_launch_command,
        "preflight": {
            "will_start_modal_job": False,
            "dataset_revisions": {
                config.preference_source.name: config.preference_source.revision,
                f"{config.eval_source.name}-eval": config.eval_source.revision,
            },
            "sft_reference_volume": config.payload["sft_reference"]["volume"],
            "budgets": config.payload["budgets"],
            "projected_cost_usd": round(config.estimated_full_cost_usd, 4),
            "exact_launch_command": config.full_launch_command,
            "blockers": full_blockers,
        },
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def smoke_launch_blockers(config: DPOLaunchConfig) -> list[str]:
    return _common_smoke_launch_blockers(
        runtime=config.payload["runtime"],
        estimated_smoke_cost_usd=config.estimated_smoke_cost_usd,
    )


def full_launch_blockers(
    config: DPOLaunchConfig, *, approved: bool = False, modal_gpu: str | None = None
) -> list[str]:
    runtime = config.payload["runtime"]
    cap_label = f"${DPO_FULL_RUN_SPEND_CAP_USD:.0f} cap"
    blockers = _common_full_launch_blockers(
        runtime=runtime,
        estimated_full_cost_usd=config.estimated_full_cost_usd,
        approved=approved,
        modal_gpu=modal_gpu,
        approval_message="full Esme-214M-Chat DPO launch requires --approved",
        modal_gpu_env_var="DPO_MODAL_GPU",
        full_run_cap_usd=DPO_FULL_RUN_SPEND_CAP_USD,
        cap_label=cap_label,
    )
    blockers.extend(_learning_gate_blockers(config.payload["learning_gate"]))
    return blockers


# --- section validators -------------------------------------------------------


def _validate_sft_reference(payload: dict[str, Any]) -> Path:
    _require_keys(
        payload,
        {
            "volume",
            "checkpoint_path",
            "tokenizer_path",
            "format",
            "model_family",
            "read_only",
            "wandb_run",
            "best_step",
        },
        "sft_reference",
    )
    if payload["volume"] != SFT_FOUNDATION_VOLUME:
        raise LaunchError(f"sft_reference.volume must be {SFT_FOUNDATION_VOLUME}")
    if payload["format"] != "llm_posttrain_instruct_sft_v1":
        raise LaunchError("sft_reference.format must be llm_posttrain_instruct_sft_v1")
    if payload["model_family"] != "DenseBackbone":
        raise LaunchError("sft_reference.model_family must be DenseBackbone")
    if payload["read_only"] is not True:
        raise LaunchError("sft_reference.read_only must be true")
    checkpoint = _str(payload["checkpoint_path"], "sft_reference.checkpoint_path")
    _str(payload["tokenizer_path"], "sft_reference.tokenizer_path")
    _str(payload["wandb_run"], "sft_reference.wandb_run")
    _positive_int(payload["best_step"], "sft_reference.best_step")
    return Path(checkpoint)


def _validate_datasets(payload: dict[str, Any]) -> tuple[DatasetSource, DatasetSource]:
    _require_keys(
        payload,
        {"preference_train", "preference_eval", "non_commercial_training_approved", "filtering"},
        "datasets",
    )
    if payload["non_commercial_training_approved"] is not False:
        raise LaunchError("datasets.non_commercial_training_approved must stay false")
    train = _validate_preference_source(
        _object(payload["preference_train"], "datasets.preference_train"),
        expected=EXPECTED_PREFERENCE_DATASET,
        role="train",
    )
    eval_source = _validate_preference_source(
        _object(payload["preference_eval"], "datasets.preference_eval"),
        expected=EXPECTED_EVAL_DATASET,
        role="eval",
    )
    _validate_filtering(_object(payload["filtering"], "datasets.filtering"))
    return train, eval_source


def _validate_preference_source(
    payload: dict[str, Any], *, expected: dict[str, Any], role: str
) -> DatasetSource:
    _require_keys(
        payload, {"name", "source", "revision", "license", "split", "role"}, "datasets.preference"
    )
    for key in ("name", "source", "revision", "license", "split"):
        if payload[key] != expected[key]:
            raise LaunchError(f"datasets.preference_{role}.{key} must be {expected[key]}")
    if payload["role"] != role:
        raise LaunchError(f"datasets.preference_{role}.role must be {role}")
    return DatasetSource(
        name=expected["name"],
        source=expected["source"],
        revision=expected["revision"],
        license=expected["license"],
        split=expected["split"],
        role="train" if role == "train" else "eval",
        train_allowed=role == "train",
    )


def _validate_filtering(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {"max_prompt_chars", "max_response_chars", "drop_identical_responses"},
        "datasets.filtering",
    )
    _positive_int(payload["max_prompt_chars"], "datasets.filtering.max_prompt_chars")
    _positive_int(payload["max_response_chars"], "datasets.filtering.max_response_chars")
    if payload["drop_identical_responses"] is not True:
        raise LaunchError("datasets.filtering.drop_identical_responses must be true")


def _validate_budgets(payload: dict[str, Any]) -> dict[str, Any]:
    _require_keys(
        payload,
        {
            "max_train_pairs",
            "min_train_pairs",
            "max_train_tokens",
            "target_train_tokens",
            "max_eval_pairs",
            "min_eval_pairs",
            "max_eval_tokens",
            "max_length",
            "max_prompt_length",
            "smoke_train_pairs",
            "smoke_train_tokens",
            "smoke_eval_pairs",
        },
        "budgets",
    )
    max_train_pairs = _positive_int(payload["max_train_pairs"], "budgets.max_train_pairs")
    max_train_tokens = _positive_int(payload["max_train_tokens"], "budgets.max_train_tokens")
    target_train_tokens = _positive_int(
        payload["target_train_tokens"], "budgets.target_train_tokens"
    )
    if max_train_pairs > MAX_APPROVED_TRAIN_PAIRS:
        raise LaunchError(f"budgets.max_train_pairs must be <= {MAX_APPROVED_TRAIN_PAIRS}")
    if max_train_tokens > MAX_APPROVED_TRAIN_TOKENS:
        raise LaunchError(f"budgets.max_train_tokens must be <= {MAX_APPROVED_TRAIN_TOKENS}")
    if target_train_tokens > max_train_tokens:
        raise LaunchError("budgets.target_train_tokens must be <= budgets.max_train_tokens")
    if int(payload["max_length"]) != 1024:
        raise LaunchError("budgets.max_length must be 1024 to match the Esme-214M context")
    if int(payload["max_prompt_length"]) != 512:
        raise LaunchError("budgets.max_prompt_length must be 512 (SmolLM2-360M DPO config)")
    if int(payload["max_prompt_length"]) >= int(payload["max_length"]):
        raise LaunchError("budgets.max_prompt_length must be < budgets.max_length")
    for key in (
        "min_train_pairs",
        "max_eval_pairs",
        "min_eval_pairs",
        "max_eval_tokens",
        "smoke_train_pairs",
        "smoke_train_tokens",
        "smoke_eval_pairs",
    ):
        _positive_int(payload[key], f"budgets.{key}")
    # max_*_pairs are caps; min_*_pairs are the sufficiency floors. Selecting fewer
    # than the cap (UltraFeedback is length-filtered at max_length=1024) is normal;
    # the floor is the minimum clean pairs a run needs to be sufficient.
    if int(payload["min_train_pairs"]) > max_train_pairs:
        raise LaunchError(
            "budgets.min_train_pairs (floor) must be <= budgets.max_train_pairs (cap)"
        )
    if int(payload["min_eval_pairs"]) > int(payload["max_eval_pairs"]):
        raise LaunchError("budgets.min_eval_pairs (floor) must be <= budgets.max_eval_pairs (cap)")
    if payload["smoke_train_pairs"] > max_train_pairs:
        raise LaunchError("budgets.smoke_train_pairs must be <= budgets.max_train_pairs")
    if payload["smoke_train_tokens"] > max_train_tokens:
        raise LaunchError("budgets.smoke_train_tokens must be <= budgets.max_train_tokens")
    return payload


def _validate_optimizer(payload: dict[str, Any]) -> dict[str, Any]:
    _require_keys(
        payload,
        {
            "name",
            "learning_rate",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "epochs",
            "max_steps",
            "smoke_max_steps",
            "seed",
            "grad_clip",
            "scheduler",
            "warmup_ratio",
            "weight_decay",
        },
        "optimizer",
    )
    if payload["name"] != "adamw_torch":
        raise LaunchError("optimizer.name must be adamw_torch")
    lr = payload["learning_rate"]
    if isinstance(lr, bool) or not isinstance(lr, int | float) or lr <= 0:
        raise LaunchError("optimizer.learning_rate must be positive")
    if lr > 1e-5:
        raise LaunchError("optimizer.learning_rate must be <= 1e-5 (DPO LR is ~10-100x below SFT)")
    for key in (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "epochs",
        "max_steps",
        "smoke_max_steps",
        "seed",
    ):
        _positive_int(payload[key], f"optimizer.{key}")
    if int(payload["micro_batch_size"]) * int(payload["gradient_accumulation_steps"]) != int(
        payload["effective_batch_size"]
    ):
        raise LaunchError(
            "optimizer.effective_batch_size must equal "
            "micro_batch_size * gradient_accumulation_steps"
        )
    if int(payload["smoke_max_steps"]) > int(payload["max_steps"]):
        raise LaunchError("optimizer.smoke_max_steps must be <= optimizer.max_steps")
    if not isinstance(payload["grad_clip"], int | float) or payload["grad_clip"] <= 0:
        raise LaunchError("optimizer.grad_clip must be positive")
    if payload["scheduler"] not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError(
            "optimizer.scheduler must be constant, linear_warmup_decay, or cosine_decay"
        )
    warmup_ratio = payload["warmup_ratio"]
    if (
        isinstance(warmup_ratio, bool)
        or not isinstance(warmup_ratio, int | float)
        or not 0 <= warmup_ratio < 1
    ):
        raise LaunchError("optimizer.warmup_ratio must be in [0, 1)")
    if not isinstance(payload["weight_decay"], int | float) or payload["weight_decay"] < 0:
        raise LaunchError("optimizer.weight_decay must be non-negative")
    return payload


def _validate_dpo(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {"loss_type", "beta", "length_normalized", "reference_free", "auxiliary_sft_loss"},
        "dpo",
    )
    if payload["loss_type"] != "sigmoid":
        raise LaunchError("dpo.loss_type must be sigmoid (vanilla DPO); SimPO/IPO are out of scope")
    beta = payload["beta"]
    if isinstance(beta, bool) or not isinstance(beta, int | float) or beta <= 0:
        raise LaunchError("dpo.beta must be positive")
    if not isinstance(payload["length_normalized"], bool):
        raise LaunchError("dpo.length_normalized must be a boolean toggle")
    if payload["reference_free"] is not False:
        raise LaunchError("dpo.reference_free must be false (frozen SFT reference is required)")
    if payload["auxiliary_sft_loss"] is not False:
        raise LaunchError(
            "dpo.auxiliary_sft_loss must be false (measured net-negative; not added by reflex)"
        )


def _validate_sequence(payload: dict[str, Any]) -> None:
    _require_keys(payload, {"pad_to_multiple_of"}, "sequence")
    value = payload["pad_to_multiple_of"]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise LaunchError("sequence.pad_to_multiple_of must be a positive integer or null")


def _validate_monitoring(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {
            "log_interval",
            "eval_interval",
            "checkpoint_interval",
            "sample_new_tokens",
            "wandb_project",
            "wandb_required_for_modal",
            "wandb_tags",
            "judge_repeat_passes",
            "log_chosen_rejected_logps",
        },
        "monitoring",
    )
    _positive_int(payload["log_interval"], "monitoring.log_interval")
    _positive_int(payload["eval_interval"], "monitoring.eval_interval")
    _positive_int(payload["checkpoint_interval"], "monitoring.checkpoint_interval")
    _positive_int(payload["sample_new_tokens"], "monitoring.sample_new_tokens")
    _str(payload["wandb_project"], "monitoring.wandb_project")
    if payload["wandb_required_for_modal"] is not True:
        raise LaunchError("monitoring.wandb_required_for_modal must be true")
    tags = payload["wandb_tags"]
    if not isinstance(tags, list) or "stage=dpo" not in tags:
        raise LaunchError("monitoring.wandb_tags must include stage=dpo")
    if payload["log_chosen_rejected_logps"] is not True:
        raise LaunchError("monitoring.log_chosen_rejected_logps must be true (displacement watch)")
    judge_passes = _positive_int(payload["judge_repeat_passes"], "monitoring.judge_repeat_passes")
    if judge_passes < 5:
        raise LaunchError("monitoring.judge_repeat_passes must be >= 5")


def _validate_learning_gate(payload: dict[str, Any]) -> None:
    _require_keys(
        payload, {"full_run_requires_beta_sweep", "swept_betas", "evidence"}, "learning_gate"
    )
    if payload["full_run_requires_beta_sweep"] is not True:
        raise LaunchError("learning_gate.full_run_requires_beta_sweep must be true")
    swept = payload["swept_betas"]
    if not isinstance(swept, list) or tuple(float(b) for b in swept) != EXPECTED_SWEEP_BETAS:
        raise LaunchError(f"learning_gate.swept_betas must be {list(EXPECTED_SWEEP_BETAS)}")
    evidence = payload["evidence"]
    if evidence is not None and not isinstance(evidence, dict):
        raise LaunchError("learning_gate.evidence must be an object or null")


def _learning_gate_blockers(payload: dict[str, Any]) -> list[str]:
    evidence = payload["evidence"]
    if evidence is None:
        return [LEARNING_GATE_BLOCKER]
    if not isinstance(evidence, dict):
        return ["learning_gate.evidence must be an object"]
    sweep = evidence.get("bounded_beta_sweep")
    if not isinstance(sweep, dict):
        return ["learning_gate.evidence is missing bounded_beta_sweep evidence"]
    return _bounded_beta_sweep_blockers(sweep)


def _bounded_beta_sweep_blockers(evidence: dict[str, Any]) -> list[str]:
    missing = sorted(
        {
            "kind",
            "selector_metric",
            "swept_betas",
            "best_beta",
            "reference_preference_accuracy",
            "best_preference_accuracy",
            "best_chosen_logp_collapsed",
            "evidence_uri",
        }
        - set(evidence)
    )
    if missing:
        return [
            "learning_gate.evidence.bounded_beta_sweep is missing required keys: "
            + ", ".join(missing)
        ]
    blockers: list[str] = []
    if evidence["kind"] != "bounded_beta_sweep":
        blockers.append("learning_gate.evidence.bounded_beta_sweep.kind must be bounded_beta_sweep")
    if evidence["selector_metric"] != "eval/preference_accuracy":
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep.selector_metric must be "
            "eval/preference_accuracy"
        )
    swept = evidence["swept_betas"]
    if not isinstance(swept, list) or tuple(float(b) for b in swept) != EXPECTED_SWEEP_BETAS:
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep.swept_betas must be "
            f"{list(EXPECTED_SWEEP_BETAS)}"
        )
    best_beta = evidence["best_beta"]
    if (
        isinstance(best_beta, bool)
        or not isinstance(best_beta, int | float)
        or best_beta not in EXPECTED_SWEEP_BETAS
    ):
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep.best_beta must be one of the swept betas"
        )
    ref_acc = evidence["reference_preference_accuracy"]
    best_acc = evidence["best_preference_accuracy"]
    if (
        isinstance(ref_acc, bool)
        or isinstance(best_acc, bool)
        or not isinstance(ref_acc, int | float)
        or not isinstance(best_acc, int | float)
    ):
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep preference accuracies must be numbers"
        )
    elif best_acc <= ref_acc:
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep must show "
            "best_preference_accuracy > reference_preference_accuracy"
        )
    if evidence["best_chosen_logp_collapsed"] is not False:
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep.best_chosen_logp_collapsed must be false "
            "(a chosen-logp drop beyond the relative collapse tolerance is a "
            "likelihood-displacement failure; sub-tolerance jitter does not count)"
        )
    if not isinstance(evidence["evidence_uri"], str) or not evidence["evidence_uri"]:
        blockers.append(
            "learning_gate.evidence.bounded_beta_sweep.evidence_uri must be a non-empty string"
        )
    return blockers


def _validate_acceptance(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {
            "preference_accuracy_must_beat_reference",
            "no_chosen_logp_collapse",
            "repetition_or_length_must_improve",
            "proxies",
        },
        "acceptance",
    )
    if payload["preference_accuracy_must_beat_reference"] is not True:
        raise LaunchError("acceptance.preference_accuracy_must_beat_reference must be true")
    if payload["no_chosen_logp_collapse"] is not True:
        raise LaunchError("acceptance.no_chosen_logp_collapse must be true")
    if payload["repetition_or_length_must_improve"] is not True:
        raise LaunchError("acceptance.repetition_or_length_must_improve must be true")
    proxies = payload["proxies"]
    if not isinstance(proxies, list) or len(proxies) < 3:
        raise LaunchError("acceptance.proxies must list the primary cheap proxy signals")


def _validate_abort_rules(value: Any) -> None:
    if not isinstance(value, list) or len(value) < 6:
        raise LaunchError("abort_rules must list the launch and runtime stop rules")
    joined = " ".join(str(item).lower() for item in value)
    for phrase in ("approved", "$2", "beta", "chosen-logp", "reference", "preference"):
        if phrase not in joined:
            raise LaunchError(f"abort_rules must include {phrase}")


def _launch_command(config_path: Path, runtime: dict[str, Any], *, mode: str) -> str:
    flag = " --full-run" if mode == "full-run" else " --modal-smoke"
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="DPO_MODAL_GPU",
        timeout_env_var="DPO_TIMEOUT_HOURS",
        script_path="scripts/modal_chat_dpo.py",
        mode_flag=flag,
    )
