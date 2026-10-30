"""Approval-gated launch validation for the multi-turn SFT foundation.

Dry-run never starts Modal. Spend/GPU guards and the learning gate pin the
multi-turn run to its run_id, Volume, output stem, sequence length, and train
mix.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_posttrain.launch.config_guards import (
    IMAGE_PACKAGE_PINS,
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    LaunchError,
    build_modal_launch_command,
    estimate_cost_usd,
    load_json_object,
    validate_base_bundle_config,
    validate_modal_runtime,
    validate_output_artifacts,
)
from esme_posttrain.launch.config_guards import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.launch.config_guards import (
    positive_int as _positive_int,
)
from esme_posttrain.launch.config_guards import (
    smoke_launch_blockers as _common_smoke_launch_blockers,
)
from esme_posttrain.launch.config_guards import (
    str_field as _str,
)
from esme_posttrain.launch.config_models import (
    ModalRuntime,
    SmokeGPUProfile,
    SmokeModalRuntime,
    parse_config,
)
from esme_posttrain.sft.config import (
    Acceptance,
    Datasets,
    LearningGate,
    MultiTurnSettings,
    SFTBudgets,
    SFTMonitoring,
    TrainingSource,
)
from esme_posttrain.sft.data import DatasetSource
from esme_posttrain.sft.launch_shared import (
    matched_interval_eval_sweep_blockers,
    validate_adamw_optimizer,
    validate_full_tuning,
    validate_sft_loss,
    validate_unpacked_sequence,
)
from esme_posttrain.sft.launch_shared import (
    validate_eval_source as _validate_eval_source,
)
from esme_posttrain.sft.launch_shared import (
    validate_sft_budgets as _validate_sft_budgets,
)
from esme_posttrain.sft.launch_shared import (
    validate_sft_monitoring as _validate_sft_monitoring,
)

# Hard cap for the approved multi-turn foundation run.
MULTITURN_FULL_RUN_SPEND_CAP_USD = 40.0

RUN_ID = "esme_214m_sft_multiturn"
RUN_CARD = "run_cards/esme-214m-sft-multiturn.md"
ARTIFACT_NAME = "Esme-214M-Instruct"
MODAL_VOLUME = "esme-posttrain-esme-sft-multiturn"
MAX_APPROVED_TRAIN_SAMPLES = 50_000
MAX_APPROVED_TRAIN_TOKENS = 80_000_000

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
LEARNING_GATE_BLOCKER = (
    "full multi-turn SFT launch requires learning_gate.evidence with "
    "bounded_matched_interval_eval_sweep proof where eval/matched/response_loss "
    "is lower than step 0"
)


@dataclass(frozen=True)
class MultiTurnLaunchConfig:
    settings: MultiTurnSettings
    config_path: Path
    base_bundle_path: Path
    train_sources: tuple[DatasetSource, ...]
    eval_source: DatasetSource
    output_dir: Path
    train_steps: int
    tokens_per_step: int
    estimated_full_cost_usd: float
    estimated_smoke_cost_usd: float
    smoke_launch_command: str
    full_launch_command: str

    @property
    def run_id(self) -> str:
        return self.settings.run_id

    @property
    def artifact_name(self) -> str:
        return self.settings.artifact_name

    @property
    def runtime(self) -> SmokeModalRuntime:
        return self.settings.runtime

    @property
    def budgets(self) -> SFTBudgets:
        return self.settings.budgets

    @property
    def selected_gpu_profile(self) -> SmokeGPUProfile:
        return self.runtime.gpu_profiles[self.runtime.selected_gpu]


def load_multi_turn_config(config_path: Path) -> MultiTurnLaunchConfig:
    config_path, payload = load_json_object(config_path)
    return validate_multi_turn_payload(payload, config_path)


def validate_multi_turn_payload(
    payload: object, config_path: Path, *, require_base_bundle_exists: bool = True
) -> MultiTurnLaunchConfig:
    settings = parse_config(MultiTurnSettings, payload, source=config_path)
    if settings.schema_version != 1:
        raise LaunchError("schema_version must be 1")
    if settings.run_id != RUN_ID:
        raise LaunchError(f"run_id must be {RUN_ID}")
    if settings.run_card != RUN_CARD:
        raise LaunchError(f"run_card must be {RUN_CARD}")
    if settings.requires_approval is not True:
        raise LaunchError("requires_approval must be true")
    if settings.artifact_name != ARTIFACT_NAME:
        raise LaunchError(f"artifact_name must be {ARTIFACT_NAME}")
    if settings.starts_from != "Esme-214M-Base":
        raise LaunchError("starts_from must be Esme-214M-Base")
    if settings.stage != "sft":
        raise LaunchError("stage must be sft")

    base_bundle_path = validate_base_bundle_config(
        settings.base_bundle,
        require_exists=require_base_bundle_exists,
    )
    train_sources, eval_source = _validate_datasets(settings.datasets)
    budgets = _validate_budgets(settings.budgets)
    optimizer = validate_adamw_optimizer(
        settings.optimizer,
        effective_batch_size=16,
    )
    validate_sft_loss(settings.loss)
    validate_full_tuning(settings.tuning)
    validate_unpacked_sequence(settings.sequence)
    runtime = validate_modal_runtime(
        settings.runtime,
        full_run_spend_cap_usd=MULTITURN_FULL_RUN_SPEND_CAP_USD,
        full_run_cap_label="40",
        modal_volume=MODAL_VOLUME,
        require_smoke_profile_metrics=True,
    )
    _validate_monitoring(settings.monitoring)
    output_dir = validate_output_artifacts(
        settings.artifacts,
        expected_files=EXPECTED_ARTIFACTS,
        manifest_label="multi-turn SFT evidence manifest",
    )
    _validate_learning_gate(settings.learning_gate)
    _validate_acceptance(settings.acceptance)

    train_steps = int(optimizer.max_steps)
    tokens_per_step = max(1, int(budgets.target_train_tokens) // train_steps)
    selected_profile = runtime.gpu_profiles[runtime.selected_gpu]
    estimated_full_cost = estimate_cost_usd(
        tokens=int(budgets.target_train_tokens),
        projected_tokens_per_second=float(selected_profile.projected_tokens_per_second),
        usd_per_hour=float(selected_profile.usd_per_hour),
    )
    estimated_smoke_cost = estimate_cost_usd(
        tokens=int(budgets.smoke_train_tokens),
        projected_tokens_per_second=float(selected_profile.projected_tokens_per_second),
        usd_per_hour=float(selected_profile.usd_per_hour),
    )

    return MultiTurnLaunchConfig(
        settings=settings,
        config_path=config_path,
        base_bundle_path=base_bundle_path,
        train_sources=train_sources,
        eval_source=eval_source,
        output_dir=output_dir,
        train_steps=train_steps,
        tokens_per_step=tokens_per_step,
        estimated_full_cost_usd=estimated_full_cost,
        estimated_smoke_cost_usd=estimated_smoke_cost,
        smoke_launch_command=_launch_command(config_path, runtime, mode="modal-smoke"),
        full_launch_command=_launch_command(config_path, runtime, mode="full-run"),
    )


def build_multi_turn_dry_run(
    config: MultiTurnLaunchConfig,
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
        "stage": config.settings.stage,
        "starts_from": config.settings.starts_from,
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "base_bundle": config.settings.base_bundle.to_dict(),
        "datasets": config.settings.datasets.to_dict(),
        "budgets": config.settings.budgets.to_dict(),
        "optimizer": config.settings.optimizer.to_dict(),
        "runtime": {
            **config.settings.runtime.to_dict(),
            "train_steps": config.train_steps,
            "tokens_per_step": config.tokens_per_step,
            "projected_train_tokens": config.budgets.target_train_tokens,
            "estimated_full_cost_usd": round(config.estimated_full_cost_usd, 4),
            "estimated_smoke_cost_usd": round(config.estimated_smoke_cost_usd, 4),
        },
        "monitoring": config.settings.monitoring.to_dict(),
        "loss": config.settings.loss.to_dict(),
        "tuning": config.settings.tuning.to_dict(),
        "sequence": config.settings.sequence.to_dict(),
        "artifacts": {
            "output_dir": str(config.output_dir),
            "required_files": list(config.settings.artifacts.required_files),
        },
        "learning_gate": config.settings.learning_gate.to_dict(),
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION, **IMAGE_PACKAGE_PINS},
        "acceptance": config.settings.acceptance.to_dict(),
        "launch_blockers": smoke_blockers,
        "full_launch_blockers": full_blockers,
        "modal_smoke_command": config.smoke_launch_command,
        "full_launch_command": config.full_launch_command,
        "preflight": {
            "will_start_modal_job": False,
            "dataset_revisions": {
                source.name: source.revision
                for source in (*config.train_sources, config.eval_source)
            },
            "budgets": config.settings.budgets.to_dict(),
            "projected_cost_usd": round(config.estimated_full_cost_usd, 4),
            "exact_launch_command": config.full_launch_command,
            "blockers": full_blockers,
        },
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def smoke_launch_blockers(config: MultiTurnLaunchConfig) -> list[str]:
    return _common_smoke_launch_blockers(
        runtime=config.runtime,
        estimated_smoke_cost_usd=config.estimated_smoke_cost_usd,
    )


def full_launch_blockers(
    config: MultiTurnLaunchConfig, *, approved: bool = False, modal_gpu: str | None = None
) -> list[str]:
    runtime = config.runtime
    blockers = _common_full_launch_blockers(
        runtime=runtime,
        estimated_full_cost_usd=config.estimated_full_cost_usd,
        approved=approved,
        modal_gpu=modal_gpu,
        approval_message="full multi-turn SFT launch requires --approved",
        modal_gpu_env_var="SFT_MODAL_GPU",
    )
    blockers.extend(_learning_gate_blockers(config.settings.learning_gate))
    return blockers


def _validate_datasets(
    payload: Datasets,
) -> tuple[tuple[DatasetSource, ...], DatasetSource]:
    if payload.non_commercial_training_approved is not False:
        raise LaunchError("datasets.non_commercial_training_approved must stay false")
    raw_mix = payload.train_mix
    if not 1 <= len(raw_mix) <= 2:
        raise LaunchError("datasets.train_mix must contain one or two sources")
    train_sources = tuple(_validate_train_source(item) for item in raw_mix)
    names = {source.name for source in train_sources}
    if not names <= set(EXPECTED_TRAIN_DATASETS):
        raise LaunchError("datasets.train_mix may only use smol-smoltalk and tulu-3-personas")
    if "smol-smoltalk" not in names:
        raise LaunchError("datasets.train_mix must include smol-smoltalk")
    ratio_sum = sum(source.mix_ratio for source in train_sources)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise LaunchError("datasets.train_mix ratios must sum to 1.0")
    eval_source = _validate_eval_source(
        payload.eval_holdout,
        EXPECTED_EVAL_DATASET,
    )
    return train_sources, eval_source


def _validate_train_source(payload: TrainingSource) -> DatasetSource:
    name = _str(payload.name, "datasets.train_mix[].name")
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
    mix_ratio = payload.mix_ratio
    if not 0 < mix_ratio <= 1:
        raise LaunchError(f"dataset {name}.mix_ratio must be in (0, 1]")
    filters = payload.filters
    max_prompt_chars = _positive_int(
        filters.max_prompt_chars, f"dataset {name}.filters.max_prompt_chars"
    )
    max_response_chars = _positive_int(
        filters.max_response_chars, f"dataset {name}.filters.max_response_chars"
    )
    return DatasetSource(
        name=name,
        source=expected["source"],
        revision=expected["revision"],
        license=expected["license"],
        split=expected["split"],
        role="train",
        mix_ratio=float(mix_ratio),
        max_prompt_chars=max_prompt_chars,
        max_response_chars=max_response_chars,
    )


def _validate_budgets(payload: SFTBudgets) -> SFTBudgets:
    return _validate_sft_budgets(
        payload,
        max_train_samples_cap=MAX_APPROVED_TRAIN_SAMPLES,
        max_train_tokens_cap=MAX_APPROVED_TRAIN_TOKENS,
        require_context_1024=True,
    )


def _validate_monitoring(payload: SFTMonitoring) -> None:
    _validate_sft_monitoring(payload)


def _validate_learning_gate(payload: LearningGate) -> None:
    if payload.full_run_requires_matched_sweep is not True:
        raise LaunchError("learning_gate.full_run_requires_matched_sweep must be true")


def _learning_gate_blockers(payload: LearningGate) -> list[str]:
    evidence = payload.evidence
    if evidence is None:
        return [LEARNING_GATE_BLOCKER]
    sweep = evidence.get("bounded_matched_interval_eval_sweep")
    if not isinstance(sweep, Mapping):
        return ["learning_gate.evidence is missing bounded_matched_interval_eval_sweep evidence"]
    return matched_interval_eval_sweep_blockers(sweep)


def _validate_acceptance(payload: Acceptance) -> None:
    if payload.heldout_response_loss_required is not True:
        raise LaunchError("acceptance.heldout_response_loss_required must be true")
    if payload.sft_must_beat_base is not True:
        raise LaunchError("acceptance.sft_must_beat_base must be true")
    _str(payload.eval_dataset, "acceptance.eval_dataset")


def _launch_command(config_path: Path, runtime: ModalRuntime, *, mode: str) -> str:
    flag = " --full-run" if mode == "full-run" else ""
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="SFT_MODAL_GPU",
        timeout_env_var="SFT_TIMEOUT_HOURS",
        script_path="scripts/modal_chat_sft.py",
        mode_flag=flag,
    )
