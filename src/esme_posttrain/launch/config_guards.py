from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from esme_posttrain.launch.config_models import ModalRuntime
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
    modal_gpu: str | None,
    modal_gpu_env_var: str,
) -> list[str]:
    # ModalRuntime already rejects caps and timeouts out of bounds at parse time.
    # Only the projection and the GPU env var, read at launch, can still block here.
    blockers = _modal_gpu_blockers(runtime, modal_gpu, modal_gpu_env_var, "smoke")
    if estimated_smoke_cost_usd > float(runtime.smoke_max_cost_usd):
        blockers.append("projected Modal smoke cost exceeds runtime.smoke_max_cost_usd")
    return blockers


def _modal_gpu_blockers(
    runtime: ModalRuntime, modal_gpu: str | None, modal_gpu_env_var: str, run: str
) -> list[str]:
    configured_modal_gpu = runtime.gpu_profiles[runtime.selected_gpu].modal_gpu
    if modal_gpu is None or modal_gpu == configured_modal_gpu:
        return []
    return [
        f"{modal_gpu_env_var} must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
        f"for {run} cost accounting"
    ]


def full_launch_blockers(
    *,
    runtime: ModalRuntime,
    estimated_full_cost_usd: float,
    approved: bool,
    modal_gpu: str | None,
    approval_message: str,
    modal_gpu_env_var: str,
) -> list[str]:
    # ModalRuntime already rejects caps out of bounds at parse time. Blockers cover
    # only launch-time state: approval, the GPU env var, and the cost projection.
    blockers: list[str] = []
    if not approved:
        blockers.append(approval_message)
    blockers.extend(_modal_gpu_blockers(runtime, modal_gpu, modal_gpu_env_var, "full-run"))
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
