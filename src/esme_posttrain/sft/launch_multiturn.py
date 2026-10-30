"""Approval-gated launch preparation for the multi-turn SFT foundation.

Dry-run never starts Modal. ``sft.config`` pins the recipe at parse time; this
module adds the base-bundle check, cost projections, and launch blockers.
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
    build_modal_launch_command,
    estimate_cost_usd,
    load_json_object,
)
from esme_posttrain.launch.config_guards import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.launch.config_guards import (
    smoke_launch_blockers as _common_smoke_launch_blockers,
)
from esme_posttrain.launch.config_models import (
    ModalRuntime,
    SmokeGPUProfile,
    SmokeModalRuntime,
    parse_config,
)
from esme_posttrain.launch.errors import LaunchError
from esme_posttrain.sft.config import (
    Datasets,
    LearningGate,
    MultiTurnSettings,
    SFTBudgets,
)
from esme_posttrain.sft.data_types import DatasetSource
from esme_posttrain.sft.launch_shared import matched_interval_eval_sweep_blockers

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
    base_bundle_path = Path(settings.base_bundle.path).expanduser().resolve()
    if require_base_bundle_exists and not base_bundle_path.is_dir():
        raise LaunchError(f"base_bundle.path does not exist: {base_bundle_path}")
    train_sources, eval_source = _dataset_sources(settings.datasets)
    budgets = settings.budgets
    optimizer = settings.optimizer
    runtime = settings.runtime
    output_dir = Path(settings.artifacts.output_dir)

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


def _dataset_sources(payload: Datasets) -> tuple[tuple[DatasetSource, ...], DatasetSource]:
    train_sources = tuple(
        DatasetSource(
            name=source.name,
            source=source.source,
            revision=source.revision,
            license=source.license,
            split=source.split,
            role="train",
            mix_ratio=float(source.mix_ratio),
            max_prompt_chars=source.filters.max_prompt_chars,
            max_response_chars=source.filters.max_response_chars,
        )
        for source in payload.train_mix
    )
    holdout = payload.eval_holdout
    eval_source = DatasetSource(
        name=holdout.name,
        source=holdout.source,
        revision=holdout.revision,
        license=holdout.license,
        split=holdout.split,
        role="eval",
        train_allowed=False,
    )
    return train_sources, eval_source


def _learning_gate_blockers(payload: LearningGate) -> list[str]:
    evidence = payload.evidence
    if evidence is None:
        return [LEARNING_GATE_BLOCKER]
    sweep = evidence.get("bounded_matched_interval_eval_sweep")
    if not isinstance(sweep, Mapping):
        return ["learning_gate.evidence is missing bounded_matched_interval_eval_sweep evidence"]
    return matched_interval_eval_sweep_blockers(sweep)


def _launch_command(config_path: Path, runtime: ModalRuntime, *, mode: str) -> str:
    flag = " --full-run" if mode == "full-run" else " --modal-smoke"
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="SFT_MODAL_GPU",
        timeout_env_var="SFT_TIMEOUT_HOURS",
        script_path="scripts/modal_chat_sft.py",
        mode_flag=flag,
    )
