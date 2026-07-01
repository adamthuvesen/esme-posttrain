from __future__ import annotations

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
    matched_interval_eval_sweep_blockers,
    validate_adamw_optimizer,
    validate_base_bundle_config,
    validate_full_tuning,
    validate_modal_runtime,
    validate_output_artifacts,
    validate_sft_loss,
    validate_unpacked_sequence,
)
from esme_posttrain.launch.config_guards import (
    full_launch_blockers as _common_full_launch_blockers,
)
from esme_posttrain.launch.config_guards import (
    object_field as _require_object_field,
)
from esme_posttrain.launch.config_guards import (
    positive_int as _positive_int,
)
from esme_posttrain.launch.config_guards import (
    require_keys as _require_keys,
)
from esme_posttrain.launch.config_guards import (
    smoke_launch_blockers as _common_smoke_launch_blockers,
)
from esme_posttrain.launch.config_guards import (
    str_field as _str,
)
from esme_posttrain.launch.models import RuntimeBlock
from esme_posttrain.sft.data import DatasetSource
from esme_posttrain.sft.launch_shared import (
    validate_eval_source as _validate_eval_source,
)
from esme_posttrain.sft.launch_shared import (
    validate_sft_budgets as _validate_sft_budgets,
)
from esme_posttrain.sft.launch_shared import (
    validate_sft_monitoring as _validate_sft_monitoring,
)

EXPECTED_TRAIN_DATASETS: dict[str, dict[str, Any]] = {
    "smol-smoltalk": {
        "source": "HuggingFaceTB/smol-smoltalk",
        "revision": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "license": "apache-2.0",
        "split": "train",
        "mix_ratio": 0.8,
    },
    "tulu-3-personas": {
        "source": "allenai/tulu-3-sft-personas-instruction-following",
        "revision": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        "license": "odc-by",
        "split": "train",
        "mix_ratio": 0.2,
    },
}
EXPECTED_EVAL_DATASET: dict[str, Any] = {
    "name": "no_robots",
    "source": "HuggingFaceH4/no_robots",
    "revision": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
    "license": "cc-by-nc-4.0",
    "split": "test",
    "train_allowed": False,
}
MAX_APPROVED_TRAIN_SAMPLES = 50_000
MAX_APPROVED_TRAIN_TOKENS = 50_000_000
FULL_RUN_SPEND_CAP_USD = 25.0
LEARNING_GATE_BLOCKER = (
    "full-data Esme-214M-Instruct SFT launch requires learning_gate.evidence "
    "with stopped_run_reconciliation proof and bounded_matched_interval_eval_sweep proof "
    "where eval/matched/response_loss is lower than step 0"
)
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
    "tokenizer.json",
    "manifest.json",
    "eval-report.json",
    "cost.json",
    "environment.txt",
)


@dataclass(frozen=True)
class SFTLaunchConfig:
    payload: dict[str, Any]
    config_path: Path
    base_bundle_path: Path
    train_sources: tuple[DatasetSource, DatasetSource]
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
    def selected_gpu_profile(self) -> dict[str, Any]:
        runtime = self.runtime
        return dict(runtime["gpu_profiles"][runtime["selected_gpu"]])


def load_sft_config(config_path: Path) -> SFTLaunchConfig:
    config_path, payload = load_json_object(config_path)
    return validate_sft_payload(payload, config_path)


def validate_sft_payload(
    payload: dict[str, Any], config_path: Path, *, require_base_bundle_exists: bool = True
) -> SFTLaunchConfig:
    _require_keys(
        payload,
        {
            "schema_version",
            "run_id",
            "run_card",
            "requires_approval",
            "artifact_name",
            "starts_from",
            "base_bundle",
            "datasets",
            "budgets",
            "optimizer",
            "loss",
            "tuning",
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
    if payload["run_id"] != "esme_214m_instruct_sft_pilot":
        raise LaunchError("run_id must be esme_214m_instruct_sft_pilot")
    if payload["run_card"] != "run_cards/esme-214m-instruct.md":
        raise LaunchError("run_card must be run_cards/esme-214m-instruct.md")
    if payload["requires_approval"] is not True:
        raise LaunchError("requires_approval must be true")
    if payload["artifact_name"] != "Esme-214M-Instruct":
        raise LaunchError("artifact_name must be Esme-214M-Instruct")
    if payload["starts_from"] != "Esme-214M-Base":
        raise LaunchError("starts_from must be Esme-214M-Base")

    base_bundle_path = validate_base_bundle_config(
        _require_object_field(payload["base_bundle"], "base_bundle"),
        require_exists=require_base_bundle_exists,
    )
    train_sources, eval_source = _validate_datasets(
        _require_object_field(payload["datasets"], "datasets")
    )
    budgets = _validate_budgets(_require_object_field(payload["budgets"], "budgets"))
    optimizer = validate_adamw_optimizer(_require_object_field(payload["optimizer"], "optimizer"))
    validate_sft_loss(_require_object_field(payload["loss"], "loss"))
    validate_full_tuning(_require_object_field(payload["tuning"], "tuning"))
    validate_unpacked_sequence(_require_object_field(payload["sequence"], "sequence"))
    runtime = validate_modal_runtime(
        _require_object_field(payload["runtime"], "runtime"),
        full_run_spend_cap_usd=FULL_RUN_SPEND_CAP_USD,
        full_run_cap_label="25",
        modal_volume="esme-posttrain-esme-instruct-sft-pilot",
        require_smoke_profile_metrics=True,
    )
    _validate_monitoring(_require_object_field(payload["monitoring"], "monitoring"))
    output_dir = validate_output_artifacts(
        _require_object_field(payload["artifacts"], "artifacts"),
        expected_files=EXPECTED_ARTIFACTS,
        manifest_label="SFT evidence manifest",
    )
    _validate_learning_gate(_require_object_field(payload["learning_gate"], "learning_gate"))
    _validate_acceptance(_require_object_field(payload["acceptance"], "acceptance"))
    _validate_abort_rules(payload["abort_rules"])

    train_steps = int(optimizer["max_steps"])
    tokens_per_step = max(1, int(budgets["target_train_tokens"]) // train_steps)
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

    return SFTLaunchConfig(
        payload=payload,
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


def build_sft_dry_run(
    config: SFTLaunchConfig,
    *,
    full_run_approved: bool = False,
    full_run_modal_gpu: str | None = None,
) -> dict[str, Any]:
    smoke_blockers = smoke_launch_blockers(config)
    full_blockers = full_launch_blockers(
        config,
        approved=full_run_approved,
        modal_gpu=full_run_modal_gpu,
    )
    if full_run_modal_gpu is not None:
        status = "ready_for_full_run" if not full_blockers else "blocked_by_launch_safety"
    else:
        status = "ready_for_modal_smoke" if not smoke_blockers else "blocked_by_launch_safety"
    return {
        "status": status,
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "starts_from": config.payload["starts_from"],
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "base_bundle": config.payload["base_bundle"],
        "datasets": config.payload["datasets"],
        "budgets": config.payload["budgets"],
        "optimizer": config.payload["optimizer"],
        "runtime": {
            **config.payload["runtime"],
            "train_steps": config.train_steps,
            "tokens_per_step": config.tokens_per_step,
            "projected_train_tokens": config.budgets["target_train_tokens"],
            "estimated_full_cost_usd": round(config.estimated_full_cost_usd, 4),
            "estimated_smoke_cost_usd": round(config.estimated_smoke_cost_usd, 4),
        },
        "monitoring": config.payload["monitoring"],
        "loss": config.payload["loss"],
        "tuning": config.payload["tuning"],
        "sequence": config.payload["sequence"],
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
                source.name: source.revision
                for source in (*config.train_sources, config.eval_source)
            },
            "budgets": config.payload["budgets"],
            "projected_cost_usd": round(config.estimated_full_cost_usd, 4),
            "exact_launch_command": config.full_launch_command,
            "blockers": full_blockers,
        },
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def smoke_launch_blockers(config: SFTLaunchConfig) -> list[str]:
    return _common_smoke_launch_blockers(
        runtime=config.payload["runtime"],
        estimated_smoke_cost_usd=config.estimated_smoke_cost_usd,
    )


def full_launch_blockers(
    config: SFTLaunchConfig, *, approved: bool = False, modal_gpu: str | None = None
) -> list[str]:
    runtime = config.payload["runtime"]
    blockers = _common_full_launch_blockers(
        runtime=runtime,
        estimated_full_cost_usd=config.estimated_full_cost_usd,
        approved=approved,
        modal_gpu=modal_gpu,
        approval_message="full Esme-214M-Instruct SFT launch requires --approved",
        modal_gpu_env_var="SFT_MODAL_GPU",
        full_run_cap_usd=FULL_RUN_SPEND_CAP_USD,
        cap_label="$25 runaway cap",
    )
    blockers.extend(_learning_gate_blockers(config.payload["learning_gate"]))
    return blockers


def _validate_datasets(
    payload: dict[str, Any],
) -> tuple[tuple[DatasetSource, DatasetSource], DatasetSource]:
    _require_keys(
        payload,
        {"train_mix", "eval_holdout", "non_commercial_training_approved"},
        "datasets",
    )
    if payload["non_commercial_training_approved"] is not False:
        raise LaunchError("datasets.non_commercial_training_approved must stay false")
    raw_mix = payload["train_mix"]
    if not isinstance(raw_mix, list) or len(raw_mix) != 2:
        raise LaunchError("datasets.train_mix must contain exactly two sources")
    train_sources = tuple(
        _validate_train_source(_require_object_field(item, "datasets.train_mix[]"))
        for item in raw_mix
    )
    if {source.name for source in train_sources} != set(EXPECTED_TRAIN_DATASETS):
        raise LaunchError("datasets.train_mix must be smol-smoltalk and tulu-3-personas")
    ratios = {source.name: source.mix_ratio for source in train_sources}
    if ratios != {"smol-smoltalk": 0.8, "tulu-3-personas": 0.2}:
        raise LaunchError("datasets.train_mix must be exactly 80% smol-smoltalk and 20% tulu")
    eval_source = _validate_eval_source(
        _require_object_field(payload["eval_holdout"], "datasets.eval_holdout"),
        EXPECTED_EVAL_DATASET,
    )
    return (train_sources[0], train_sources[1]), eval_source


def _validate_train_source(payload: dict[str, Any]) -> DatasetSource:
    _require_keys(
        payload,
        {
            "name",
            "source",
            "revision",
            "license",
            "split",
            "role",
            "mix_ratio",
            "filters",
        },
        "datasets.train_mix[]",
    )
    name = _str(payload["name"], "datasets.train_mix[].name")
    if name not in EXPECTED_TRAIN_DATASETS:
        raise LaunchError(f"unsupported training dataset: {name}")
    expected = EXPECTED_TRAIN_DATASETS[name]
    for key in ("source", "revision", "license", "split", "mix_ratio"):
        if payload[key] != expected[key]:
            raise LaunchError(f"dataset {name}.{key} must be {expected[key]}")
    if payload["role"] != "train":
        raise LaunchError(f"dataset {name}.role must be train")
    filters = _require_object_field(payload["filters"], f"dataset {name}.filters")
    max_prompt_chars = _positive_int(
        filters.get("max_prompt_chars"),
        f"dataset {name}.filters.max_prompt_chars",
    )
    max_response_chars = _positive_int(
        filters.get("max_response_chars"), f"dataset {name}.filters.max_response_chars"
    )
    return DatasetSource(
        name=name,
        source=expected["source"],
        revision=expected["revision"],
        license=expected["license"],
        split=expected["split"],
        role="train",
        mix_ratio=float(expected["mix_ratio"]),
        max_prompt_chars=max_prompt_chars,
        max_response_chars=max_response_chars,
    )


def _validate_budgets(payload: dict[str, Any]) -> dict[str, Any]:
    return _validate_sft_budgets(
        payload,
        max_train_samples_cap=MAX_APPROVED_TRAIN_SAMPLES,
        max_train_tokens_cap=MAX_APPROVED_TRAIN_TOKENS,
    )


def _validate_monitoring(payload: dict[str, Any]) -> None:
    _validate_sft_monitoring(payload)


def _validate_learning_gate(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {"full_run_requires_interval_eval_sweep", "evidence"},
        "learning_gate",
    )
    if payload["full_run_requires_interval_eval_sweep"] is not True:
        raise LaunchError("learning_gate.full_run_requires_interval_eval_sweep must be true")
    evidence = payload["evidence"]
    if evidence is not None and not isinstance(evidence, dict):
        raise LaunchError("learning_gate.evidence must be an object or null")


def _learning_gate_blockers(payload: dict[str, Any]) -> list[str]:
    evidence = payload["evidence"]
    if evidence is None:
        return [LEARNING_GATE_BLOCKER]
    if not isinstance(evidence, dict):
        return ["learning_gate.evidence must be an object"]
    if evidence.get("kind") == "bounded_interval_eval_sweep":
        return [
            "learning_gate.evidence is missing stopped_run_reconciliation evidence",
            *matched_interval_eval_sweep_blockers(evidence),
        ]
    blockers = []
    if "stopped_run_reconciliation" not in evidence:
        blockers.append("learning_gate.evidence is missing stopped_run_reconciliation evidence")
    else:
        blockers.extend(
            _stopped_run_reconciliation_blockers(evidence["stopped_run_reconciliation"])
        )
    if "bounded_matched_interval_eval_sweep" not in evidence:
        blockers.append(
            "learning_gate.evidence is missing bounded_matched_interval_eval_sweep evidence"
        )
    else:
        blockers.extend(
            matched_interval_eval_sweep_blockers(evidence["bounded_matched_interval_eval_sweep"])
        )
    return blockers


def _stopped_run_reconciliation_blockers(evidence: Any) -> list[str]:
    if not isinstance(evidence, dict):
        return ["learning_gate.evidence.stopped_run_reconciliation must be an object"]
    missing = sorted(
        {
            "kind",
            "showcase_metrics_uri",
            "older_full_metrics_uri",
            "showcase_eval_rows",
            "showcase_best_step",
            "showcase_latest_step",
            "notes",
        }
        - set(evidence)
    )
    if missing:
        return [
            "learning_gate.evidence.stopped_run_reconciliation is missing required keys: "
            + ", ".join(missing)
        ]
    blockers = []
    if evidence["kind"] != "stopped_run_reconciliation":
        blockers.append(
            "learning_gate.evidence.stopped_run_reconciliation.kind must be "
            "stopped_run_reconciliation"
        )
    for key in ("showcase_metrics_uri", "older_full_metrics_uri", "notes"):
        if not isinstance(evidence[key], str) or not evidence[key]:
            blockers.append(
                f"learning_gate.evidence.stopped_run_reconciliation.{key} must be "
                "a non-empty string"
            )
    if evidence.get("showcase_eval_rows") != 98:
        blockers.append(
            "learning_gate.evidence.stopped_run_reconciliation must record 98 eval rows"
        )
    if evidence.get("showcase_best_step") != 600:
        blockers.append("learning_gate.evidence.stopped_run_reconciliation best step must be 600")
    if evidence.get("showcase_latest_step") != 19400:
        blockers.append(
            "learning_gate.evidence.stopped_run_reconciliation latest step must be 19400"
        )
    return blockers


def _validate_acceptance(payload: dict[str, Any]) -> None:
    _require_keys(
        payload,
        {"heldout_response_loss_required", "instruct_must_beat_base", "eval_dataset"},
        "acceptance",
    )
    if payload["heldout_response_loss_required"] is not True:
        raise LaunchError("acceptance.heldout_response_loss_required must be true")
    if payload["instruct_must_beat_base"] is not True:
        raise LaunchError("acceptance.instruct_must_beat_base must be true")
    expected_eval_dataset = "weighted matched SmolTalk/Tulu with no_robots OOD guardrail"
    if payload["eval_dataset"] != expected_eval_dataset:
        raise LaunchError(
            "acceptance.eval_dataset must be weighted matched SmolTalk/Tulu "
            "with no_robots OOD guardrail"
        )


def _validate_abort_rules(value: Any) -> None:
    if not isinstance(value, list) or len(value) < 7:
        raise LaunchError("abort_rules must list the launch and runtime stop rules")
    joined = " ".join(str(item).lower() for item in value)
    for phrase in ("approved", "$2", "$25", "no_robots", "response loss", "checkpoint"):
        if phrase not in joined:
            raise LaunchError(f"abort_rules must include {phrase}")


def _launch_command(config_path: Path, runtime: dict[str, Any], *, mode: str) -> str:
    mode_flag = " --full-run" if mode == "full-run" else ""
    return build_modal_launch_command(
        config_path=config_path,
        runtime=runtime,
        gpu_env_var="SFT_MODAL_GPU",
        timeout_env_var="SFT_TIMEOUT_HOURS",
        script_path="scripts/modal_instruct_sft.py",
        mode_flag=mode_flag,
    )
