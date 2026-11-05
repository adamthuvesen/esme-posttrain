"""Approval-gated launch preparation for the Esme-214M-Chat DPO polish stage.

The dry-run never starts Modal. ``dpo.config`` pins the recipe at parse time; this
module adds cost projections, launch blockers, and the bounded beta-sweep gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from esme_posttrain.dpo.config import (
    EXPECTED_SWEEP_BETAS,
    DPOBudgets,
    DPOOptimizer,
    DPOSettings,
    LearningGate,
    PreferenceSource,
)
from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    build_modal_launch_command,
    estimate_cost_usd,
    image_package_pins,
    load_json_object,
    smoke_timeout_hours,
)
from esme_posttrain.launch.config_guards import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.launch.config_guards import (
    smoke_launch_blockers as _common_smoke_launch_blockers,
)
from esme_posttrain.launch.config_models import (
    FrozenJSON,
    GPUProfile,
    ModalRuntime,
    parse_config,
)
from esme_posttrain.sft.data_types import DatasetSource

LEARNING_GATE_BLOCKER = (
    "full Esme-214M-Chat DPO launch requires learning_gate.evidence with "
    "bounded_beta_sweep proof where held-out preference accuracy improves versus "
    "the SFT reference and chosen-logp does not collapse"
)


@dataclass(frozen=True)
class DPOLaunchConfig:
    settings: DPOSettings
    config_path: Path
    preference_source: DatasetSource
    eval_source: DatasetSource
    output_dir: Path
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
    def runtime(self) -> ModalRuntime:
        return self.settings.runtime

    @property
    def budgets(self) -> DPOBudgets:
        return self.settings.budgets

    @property
    def optimizer(self) -> DPOOptimizer:
        return self.settings.optimizer

    @property
    def selected_gpu_profile(self) -> GPUProfile:
        return self.runtime.gpu_profiles[self.runtime.selected_gpu]


def load_dpo_config(config_path: Path) -> DPOLaunchConfig:
    config_path, payload = load_json_object(config_path)
    return validate_dpo_payload(payload, config_path)


def validate_dpo_payload(payload: object, config_path: Path) -> DPOLaunchConfig:
    settings = parse_config(DPOSettings, payload, source=config_path)
    preference_source = _dataset_source(settings.datasets.preference_train, role="train")
    eval_source = _dataset_source(settings.datasets.preference_eval, role="eval")
    budgets = settings.budgets
    runtime = settings.runtime
    output_dir = Path(settings.artifacts.output_dir)

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

    return DPOLaunchConfig(
        settings=settings,
        config_path=config_path,
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
    smoke_modal_gpu: str | None = None,
    smoke_timeout_hours: float | None = None,
) -> dict[str, Any]:
    smoke_blockers = smoke_launch_blockers(
        config, modal_gpu=smoke_modal_gpu, timeout_hours=smoke_timeout_hours
    )
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
        "sft_reference": config.settings.sft_reference.to_dict(),
        "datasets": config.settings.datasets.to_dict(),
        "budgets": config.settings.budgets.to_dict(),
        "optimizer": config.settings.optimizer.to_dict(),
        "dpo": config.settings.dpo.to_dict(),
        "sequence": config.settings.sequence.to_dict(),
        "runtime": {
            **config.settings.runtime.to_dict(),
            "estimated_full_cost_usd": round(config.estimated_full_cost_usd, 4),
            "estimated_smoke_cost_usd": round(config.estimated_smoke_cost_usd, 4),
        },
        "monitoring": config.settings.monitoring.to_dict(),
        "artifacts": {
            "output_dir": str(config.output_dir),
            "required_files": list(config.settings.artifacts.required_files),
        },
        "learning_gate": config.settings.learning_gate.to_dict(),
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION, **image_package_pins()},
        "acceptance": config.settings.acceptance.to_dict(),
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
            "sft_reference_volume": config.settings.sft_reference.volume,
            "budgets": config.settings.budgets.to_dict(),
            "projected_cost_usd": round(config.estimated_full_cost_usd, 4),
            "exact_launch_command": config.full_launch_command,
            "blockers": full_blockers,
        },
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def smoke_launch_blockers(
    config: DPOLaunchConfig, *, modal_gpu: str | None, timeout_hours: float | None
) -> list[str]:
    return _common_smoke_launch_blockers(
        runtime=config.runtime,
        estimated_smoke_cost_usd=config.estimated_smoke_cost_usd,
        modal_gpu=modal_gpu,
        modal_gpu_env_var="DPO_MODAL_GPU",
        timeout_hours=timeout_hours,
        timeout_env_var="DPO_TIMEOUT_HOURS",
    )


def full_launch_blockers(
    config: DPOLaunchConfig, *, approved: bool = False, modal_gpu: str | None = None
) -> list[str]:
    runtime = config.runtime
    blockers = _common_full_launch_blockers(
        runtime=runtime,
        estimated_full_cost_usd=config.estimated_full_cost_usd,
        approved=approved,
        modal_gpu=modal_gpu,
        approval_message="full Esme-214M-Chat DPO launch requires --approved",
        modal_gpu_env_var="DPO_MODAL_GPU",
    )
    blockers.extend(_learning_gate_blockers(config.settings.learning_gate))
    return blockers


def _dataset_source(payload: PreferenceSource, *, role: Literal["train", "eval"]) -> DatasetSource:
    return DatasetSource(
        name=payload.name,
        source=payload.source,
        revision=payload.revision,
        license=payload.license,
        split=payload.split,
        role=role,
        train_allowed=role == "train",
    )


def _learning_gate_blockers(payload: LearningGate) -> list[str]:
    evidence = payload.evidence
    if evidence is None:
        return [LEARNING_GATE_BLOCKER]
    sweep = evidence.get("bounded_beta_sweep")
    if not isinstance(sweep, Mapping):
        return ["learning_gate.evidence is missing bounded_beta_sweep evidence"]
    return _bounded_beta_sweep_blockers(sweep)


def _bounded_beta_sweep_blockers(evidence: Mapping[str, FrozenJSON]) -> list[str]:
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
    try:
        swept_betas = TypeAdapter(tuple[float, ...]).validate_python(evidence["swept_betas"])
    except ValidationError:
        swept_betas = ()
    if swept_betas != EXPECTED_SWEEP_BETAS:
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


def _launch_command(config_path: Path, runtime: ModalRuntime, *, mode: str) -> str:
    flag = " --full-run" if mode == "full-run" else " --modal-smoke"
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="DPO_MODAL_GPU",
        timeout_env_var="DPO_TIMEOUT_HOURS",
        script_path="scripts/modal_chat_dpo.py",
        mode_flag=flag,
        timeout_hours=None if mode == "full-run" else smoke_timeout_hours(runtime),
    )
