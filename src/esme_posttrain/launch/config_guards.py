from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esme_posttrain.launch.config_models import (
    BundleConfig,
    ModalRuntime,
    OutputArtifacts,
    SmokeGPUProfile,
)
from esme_posttrain.launch.errors import LaunchError

LAUNCH_APPROVAL_FLAG = "--approved"
MODAL_CLIENT_VERSION = "1.5.1"
IMAGE_PACKAGE_PINS: dict[str, str] = {
    "torch": "2.12.1",
    "datasets": "5.0.0",
    "tokenizers": "0.23.1",
    "numpy": "2.5.0",
    "wandb": "0.28.0",
}
SMOKE_SPEND_CAP_USD = 2.0


def load_json_object(config_path: Path) -> tuple[Path, dict[str, Any]]:
    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.exists():
        raise LaunchError(f"config path does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise LaunchError(f"config path must be a file: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LaunchError(f"malformed config JSON at {resolved_path}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise LaunchError("config must be a JSON object")
    return resolved_path, payload


def estimate_cost_usd(
    *, tokens: int, projected_tokens_per_second: float, usd_per_hour: float
) -> float:
    if tokens <= 0:
        raise LaunchError("tokens must be positive")
    if projected_tokens_per_second <= 0:
        raise LaunchError("projected_tokens_per_second must be positive")
    if usd_per_hour <= 0:
        raise LaunchError("usd_per_hour must be positive")
    return tokens / projected_tokens_per_second * usd_per_hour / 3600.0


def smoke_launch_blockers(
    *,
    runtime: ModalRuntime,
    estimated_smoke_cost_usd: float,
) -> list[str]:
    # Cap and timeout bounds are already enforced by validate_modal_runtime,
    # which raises before any blocker check can run. Only the projection —
    # computed after validation — can still block here.
    blockers: list[str] = []
    if estimated_smoke_cost_usd > float(runtime.smoke_max_cost_usd):
        blockers.append("projected Modal smoke cost exceeds runtime.smoke_max_cost_usd")
    return blockers


def full_launch_blockers(
    *,
    runtime: ModalRuntime,
    estimated_full_cost_usd: float,
    approved: bool,
    modal_gpu: str | None,
    approval_message: str,
    modal_gpu_env_var: str,
) -> list[str]:
    # Cap bounds are already enforced by validate_modal_runtime, which raises
    # before any blocker check can run. Blockers cover only launch-time state:
    # approval, the GPU env var, and the post-validation cost projection.
    blockers: list[str] = []
    if not approved:
        blockers.append(approval_message)
    configured_modal_gpu = runtime.gpu_profiles[runtime.selected_gpu].modal_gpu
    if modal_gpu is not None and modal_gpu != configured_modal_gpu:
        blockers.append(
            f"{modal_gpu_env_var} must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
            "for full-run cost accounting"
        )
    if estimated_full_cost_usd > float(runtime.full_run_max_cost_usd):
        blockers.append("projected full-run cost exceeds runtime.full_run_max_cost_usd")
    return blockers


def build_modal_launch_command(
    *,
    config_path: Path,
    runtime: ModalRuntime,
    gpu_env_var: str,
    timeout_env_var: str,
    script_path: str,
    mode_flag: str,
) -> str:
    selected = runtime.selected_gpu
    modal_gpu = runtime.gpu_profiles[selected].modal_gpu
    timeout_hours = runtime.timeout_hours
    return (
        f"{gpu_env_var}='{modal_gpu}' {timeout_env_var}={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} "
        f"modal run --detach {script_path} "
        f"--config {config_path.as_posix()}{mode_flag} {LAUNCH_APPROVAL_FLAG} --json"
    )


def require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise LaunchError(f"{label} is missing required keys: {', '.join(missing)}")
    if extra:
        raise LaunchError(f"{label} has unsupported keys: {', '.join(extra)}")


def object_field(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LaunchError(f"{label} must be an object")
    return value


def str_field(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LaunchError(f"{label} must be a non-empty string")
    return value


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LaunchError(f"{label} must be a positive integer")
    return value


def validate_base_bundle_config(payload: BundleConfig, *, require_exists: bool) -> Path:
    if payload.format != "llm_pretrain_dense_v1":
        raise LaunchError("base_bundle.format must be llm_pretrain_dense_v1")
    if payload.model_family != "DenseBackbone":
        raise LaunchError("base_bundle.model_family must be DenseBackbone")
    if payload.read_only is not True:
        raise LaunchError("base_bundle.read_only must be true")
    path = Path(str_field(payload.path, "base_bundle.path")).expanduser().resolve()
    if require_exists and not path.is_dir():
        raise LaunchError(f"base_bundle.path does not exist: {path}")
    return path


def validate_modal_runtime(
    payload: ModalRuntime,
    *,
    full_run_spend_cap_usd: float,
    full_run_cap_label: str,
    modal_volume: str,
    require_smoke_profile_metrics: bool,
) -> ModalRuntime:
    if payload.provider != "modal":
        raise LaunchError("runtime.provider must be modal")
    if payload.precision not in {"fp32", "bf16"}:
        raise LaunchError("runtime.precision must be fp32 or bf16")
    if payload.allow_retries is not False:
        raise LaunchError("runtime.allow_retries must be false")
    if payload.smoke_max_cost_usd > SMOKE_SPEND_CAP_USD:
        raise LaunchError("runtime.smoke_max_cost_usd must be <= 2")
    if payload.runtime_spend_stop_usd > SMOKE_SPEND_CAP_USD:
        raise LaunchError("runtime.runtime_spend_stop_usd must be <= 2")
    if payload.full_run_max_cost_usd > full_run_spend_cap_usd:
        raise LaunchError(f"runtime.full_run_max_cost_usd must be <= {full_run_cap_label}")
    if payload.full_run_runtime_spend_stop_usd > full_run_spend_cap_usd:
        raise LaunchError(
            f"runtime.full_run_runtime_spend_stop_usd must be <= {full_run_cap_label}"
        )
    if payload.timeout_hours <= 0 or payload.timeout_hours > 24:
        raise LaunchError("runtime.timeout_hours must be between 1 and 24")
    if payload.modal_volume != modal_volume:
        raise LaunchError(f"runtime.modal_volume must be {modal_volume}")
    _validate_gpu_profiles(payload, require_smoke_profile_metrics=require_smoke_profile_metrics)
    return payload


def validate_output_artifacts(
    payload: OutputArtifacts, *, expected_files: tuple[str, ...], manifest_label: str
) -> Path:
    output_dir = Path(str_field(payload.output_dir, "artifacts.output_dir"))
    if output_dir.is_absolute() or ".." in output_dir.parts:
        raise LaunchError("artifacts.output_dir must stay inside the repository")
    if output_dir.parts[:1] != ("runs",):
        raise LaunchError("artifacts.output_dir must be under runs/")
    required_files = payload.required_files
    if tuple(required_files) != expected_files:
        raise LaunchError(f"artifacts.required_files must match the {manifest_label}")
    return output_dir


def _validate_gpu_profiles(payload: ModalRuntime, *, require_smoke_profile_metrics: bool) -> None:
    profiles = payload.gpu_profiles
    if len(profiles) < 2:
        raise LaunchError("runtime.gpu_profiles must include at least two GPU profiles")
    if payload.selected_gpu not in profiles:
        raise LaunchError("runtime.selected_gpu must name a gpu_profiles entry")
    for name, profile in profiles.items():
        if profile.modal_gpu != name:
            raise LaunchError(f"runtime.gpu_profiles.{name}.modal_gpu must be {name}")
        if profile.usd_per_hour <= 0:
            raise LaunchError(f"runtime.gpu_profiles.{name}.usd_per_hour must be positive")
        if profile.projected_tokens_per_second <= 0:
            raise LaunchError(
                f"runtime.gpu_profiles.{name}.projected_tokens_per_second must be positive"
            )
        if require_smoke_profile_metrics:
            if not isinstance(profile, SmokeGPUProfile):
                raise LaunchError(f"runtime.gpu_profiles.{name} requires smoke metrics")
            for key, value in (
                ("measured_tokens_per_second", profile.measured_tokens_per_second),
                ("measured_smoke_tokens_per_second", profile.measured_smoke_tokens_per_second),
            ):
                if value is not None and value <= 0:
                    raise LaunchError(f"runtime.gpu_profiles.{name}.{key} must be positive or null")
            if profile.expected_smoke_duration_minutes <= 0:
                raise LaunchError(
                    f"runtime.gpu_profiles.{name}.expected_smoke_duration_minutes must be positive"
                )
        str_field(profile.rationale, f"runtime.gpu_profiles.{name}.rationale")
