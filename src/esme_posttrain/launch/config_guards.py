from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    runtime: dict[str, Any],
    estimated_smoke_cost_usd: float,
) -> list[str]:
    # Cap and timeout bounds are already enforced by validate_modal_runtime,
    # which raises before any blocker check can run. Only the projection —
    # computed after validation — can still block here.
    blockers: list[str] = []
    if estimated_smoke_cost_usd > float(runtime["smoke_max_cost_usd"]):
        blockers.append("projected Modal smoke cost exceeds runtime.smoke_max_cost_usd")
    return blockers


def full_launch_blockers(
    *,
    runtime: dict[str, Any],
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
    configured_modal_gpu = runtime["gpu_profiles"][runtime["selected_gpu"]]["modal_gpu"]
    if modal_gpu is not None and modal_gpu != configured_modal_gpu:
        blockers.append(
            f"{modal_gpu_env_var} must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
            "for full-run cost accounting"
        )
    if estimated_full_cost_usd > float(runtime["full_run_max_cost_usd"]):
        blockers.append("projected full-run cost exceeds runtime.full_run_max_cost_usd")
    return blockers


def build_modal_launch_command(
    *,
    config_path: Path,
    runtime: dict[str, Any],
    gpu_env_var: str,
    timeout_env_var: str,
    script_path: str,
    mode_flag: str,
) -> str:
    selected = runtime["selected_gpu"]
    modal_gpu = runtime["gpu_profiles"][selected]["modal_gpu"]
    timeout_hours = runtime["timeout_hours"]
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


def str_field(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LaunchError(f"{label} must be a non-empty string")
    return value


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LaunchError(f"{label} must be a positive integer")
    return value


def validate_base_bundle_config(payload: dict[str, Any], *, require_exists: bool) -> Path:
    require_keys(payload, {"path", "format", "model_family", "read_only"}, "base_bundle")
    if payload["format"] != "llm_pretrain_dense_v1":
        raise LaunchError("base_bundle.format must be llm_pretrain_dense_v1")
    if payload["model_family"] != "DenseBackbone":
        raise LaunchError("base_bundle.model_family must be DenseBackbone")
    if payload["read_only"] is not True:
        raise LaunchError("base_bundle.read_only must be true")
    path = Path(str_field(payload["path"], "base_bundle.path")).expanduser().resolve()
    if require_exists and not path.is_dir():
        raise LaunchError(f"base_bundle.path does not exist: {path}")
    return path


def validate_modal_runtime(
    payload: dict[str, Any],
    *,
    full_run_spend_cap_usd: float,
    full_run_cap_label: str,
    modal_volume: str,
    require_smoke_profile_metrics: bool,
) -> dict[str, Any]:
    require_keys(
        payload,
        {
            "provider",
            "selected_gpu",
            "gpu_profiles",
            "precision",
            "smoke_max_cost_usd",
            "full_run_max_cost_usd",
            "runtime_spend_stop_usd",
            "full_run_runtime_spend_stop_usd",
            "allow_retries",
            "modal_volume",
            "timeout_hours",
        },
        "runtime",
    )
    if payload["provider"] != "modal":
        raise LaunchError("runtime.provider must be modal")
    if payload["precision"] not in {"fp32", "bf16"}:
        raise LaunchError("runtime.precision must be fp32 or bf16")
    if payload["allow_retries"] is not False:
        raise LaunchError("runtime.allow_retries must be false")
    if float(payload["smoke_max_cost_usd"]) > SMOKE_SPEND_CAP_USD:
        raise LaunchError("runtime.smoke_max_cost_usd must be <= 2")
    if float(payload["runtime_spend_stop_usd"]) > SMOKE_SPEND_CAP_USD:
        raise LaunchError("runtime.runtime_spend_stop_usd must be <= 2")
    if float(payload["full_run_max_cost_usd"]) > full_run_spend_cap_usd:
        raise LaunchError(f"runtime.full_run_max_cost_usd must be <= {full_run_cap_label}")
    if float(payload["full_run_runtime_spend_stop_usd"]) > full_run_spend_cap_usd:
        raise LaunchError(
            f"runtime.full_run_runtime_spend_stop_usd must be <= {full_run_cap_label}"
        )
    if int(payload["timeout_hours"]) <= 0 or int(payload["timeout_hours"]) > 24:
        raise LaunchError("runtime.timeout_hours must be between 1 and 24")
    if payload["modal_volume"] != modal_volume:
        raise LaunchError(f"runtime.modal_volume must be {modal_volume}")
    _validate_gpu_profiles(payload, require_smoke_profile_metrics=require_smoke_profile_metrics)
    return payload


def validate_output_artifacts(
    payload: dict[str, Any], *, expected_files: tuple[str, ...], manifest_label: str
) -> Path:
    require_keys(payload, {"output_dir", "required_files"}, "artifacts")
    output_dir = Path(str_field(payload["output_dir"], "artifacts.output_dir"))
    if output_dir.is_absolute() or ".." in output_dir.parts:
        raise LaunchError("artifacts.output_dir must stay inside the repository")
    if output_dir.parts[:1] != ("runs",):
        raise LaunchError("artifacts.output_dir must be under runs/")
    required_files = payload["required_files"]
    if not isinstance(required_files, list) or tuple(required_files) != expected_files:
        raise LaunchError(f"artifacts.required_files must match the {manifest_label}")
    return output_dir


def validate_sft_loss(payload: dict[str, Any]) -> None:
    require_keys(
        payload,
        {"assistant_only_loss", "completion_only_loss", "ignore_index"},
        "loss",
    )
    if payload["assistant_only_loss"] is not True:
        raise LaunchError("loss.assistant_only_loss must be true")
    if payload["completion_only_loss"] is not True:
        raise LaunchError("loss.completion_only_loss must be true")
    if payload["ignore_index"] != -100:
        raise LaunchError("loss.ignore_index must be -100")


def validate_adamw_optimizer(
    payload: dict[str, Any], *, effective_batch_size: int | None = None
) -> dict[str, Any]:
    require_keys(
        payload,
        {
            "name",
            "learning_rate",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "max_steps",
            "smoke_max_steps",
            "seed",
            "grad_clip",
            "scheduler",
            "warmup_steps",
            "weight_decay",
        },
        "optimizer",
    )
    if payload["name"] != "AdamW":
        raise LaunchError("optimizer.name must be AdamW")
    if not isinstance(payload["learning_rate"], int | float) or payload["learning_rate"] <= 0:
        raise LaunchError("optimizer.learning_rate must be positive")
    for key in (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "max_steps",
        "smoke_max_steps",
        "seed",
    ):
        positive_int(payload[key], f"optimizer.{key}")
    if int(payload["micro_batch_size"]) * int(payload["gradient_accumulation_steps"]) != int(
        payload["effective_batch_size"]
    ):
        raise LaunchError(
            "optimizer.effective_batch_size must equal micro_batch_size * "
            "gradient_accumulation_steps"
        )
    if effective_batch_size is not None and int(payload["effective_batch_size"]) != int(
        effective_batch_size
    ):
        raise LaunchError(
            f"optimizer.effective_batch_size must be {effective_batch_size} for this recipe"
        )
    if payload["smoke_max_steps"] > payload["max_steps"]:
        raise LaunchError("optimizer.smoke_max_steps must be <= optimizer.max_steps")
    if not isinstance(payload["grad_clip"], int | float) or payload["grad_clip"] <= 0:
        raise LaunchError("optimizer.grad_clip must be positive")
    if payload["scheduler"] not in {"constant", "linear_warmup_decay", "cosine_decay"}:
        raise LaunchError(
            "optimizer.scheduler must be constant, linear_warmup_decay, or cosine_decay"
        )
    if (
        isinstance(payload["warmup_steps"], bool)
        or not isinstance(payload["warmup_steps"], int)
        or payload["warmup_steps"] < 0
    ):
        raise LaunchError("optimizer.warmup_steps must be a non-negative integer")
    if int(payload["warmup_steps"]) > int(payload["max_steps"]):
        raise LaunchError("optimizer.warmup_steps must be <= optimizer.max_steps")
    if not isinstance(payload["weight_decay"], int | float) or payload["weight_decay"] < 0:
        raise LaunchError("optimizer.weight_decay must be non-negative")
    return payload


def validate_full_tuning(payload: dict[str, Any]) -> None:
    require_keys(payload, {"mode"}, "tuning")
    if payload["mode"] != "full":
        raise LaunchError("only tuning.mode='full' is supported")


def validate_unpacked_sequence(payload: dict[str, Any]) -> None:
    require_keys(
        payload,
        {"sequence_packing", "pad_to_multiple_of", "no_packing_rationale"},
        "sequence",
    )
    if payload["sequence_packing"] is not False:
        raise LaunchError("sequence.sequence_packing must be false until packing is implemented")
    value = payload["pad_to_multiple_of"]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise LaunchError("sequence.pad_to_multiple_of must be a positive integer or null")
    str_field(payload["no_packing_rationale"], "sequence.no_packing_rationale")


def matched_interval_eval_sweep_blockers(evidence: Any) -> list[str]:
    label = "learning_gate.evidence.bounded_matched_interval_eval_sweep"
    if not isinstance(evidence, dict):
        return [f"{label} must be an object"]
    missing = sorted(
        {
            "kind",
            "eval_metric",
            "baseline_step",
            "step0_response_loss",
            "best_response_loss",
            "interval_eval_steps",
            "evidence_uri",
        }
        - set(evidence)
    )
    if missing:
        return [f"{label} is missing required keys: " + ", ".join(missing)]
    blockers: list[str] = []
    if evidence["kind"] != "bounded_matched_interval_eval_sweep":
        blockers.append(f"{label}.kind must be bounded_matched_interval_eval_sweep")
    if evidence["eval_metric"] != "eval/matched/response_loss":
        blockers.append(f"{label}.eval_metric must be eval/matched/response_loss")
    if evidence["baseline_step"] != 0:
        blockers.append(f"{label}.baseline_step must be 0")
    step0 = evidence["step0_response_loss"]
    best = evidence["best_response_loss"]
    if (
        isinstance(step0, bool)
        or isinstance(best, bool)
        or not isinstance(step0, int | float)
        or not isinstance(best, int | float)
        or step0 <= 0
        or best <= 0
    ):
        blockers.append(f"{label} response losses must be positive numbers")
    elif best >= step0:
        blockers.append(f"{label} must show best_response_loss < step0_response_loss")
    steps = evidence["interval_eval_steps"]
    if (
        not isinstance(steps, list)
        or not steps
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
    ):
        blockers.append(f"{label}.interval_eval_steps must list positive steps")
    if not isinstance(evidence["evidence_uri"], str) or not evidence["evidence_uri"]:
        blockers.append(f"{label}.evidence_uri must be a non-empty string")
    return blockers


def _validate_gpu_profiles(payload: dict[str, Any], *, require_smoke_profile_metrics: bool) -> None:
    profiles = object_field(payload["gpu_profiles"], "runtime.gpu_profiles")
    if len(profiles) < 2:
        raise LaunchError("runtime.gpu_profiles must include at least two GPU profiles")
    selected = str_field(payload["selected_gpu"], "runtime.selected_gpu")
    if selected not in profiles:
        raise LaunchError("runtime.selected_gpu must name a gpu_profiles entry")
    required_keys = {
        "modal_gpu",
        "usd_per_hour",
        "projected_tokens_per_second",
        "rationale",
        "projection_source",
        "measured",
    }
    if require_smoke_profile_metrics:
        required_keys |= {
            "measured_tokens_per_second",
            "measured_smoke_tokens_per_second",
            "expected_smoke_duration_minutes",
        }
    for name, profile in profiles.items():
        gpu_profile = object_field(profile, f"runtime.gpu_profiles.{name}")
        require_keys(gpu_profile, required_keys, f"runtime.gpu_profiles.{name}")
        if gpu_profile["modal_gpu"] != name:
            raise LaunchError(f"runtime.gpu_profiles.{name}.modal_gpu must be {name}")
        if float(gpu_profile["usd_per_hour"]) <= 0:
            raise LaunchError(f"runtime.gpu_profiles.{name}.usd_per_hour must be positive")
        if float(gpu_profile["projected_tokens_per_second"]) <= 0:
            raise LaunchError(
                f"runtime.gpu_profiles.{name}.projected_tokens_per_second must be positive"
            )
        if require_smoke_profile_metrics:
            for key in ("measured_tokens_per_second", "measured_smoke_tokens_per_second"):
                value = gpu_profile[key]
                if value is not None and float(value) <= 0:
                    raise LaunchError(f"runtime.gpu_profiles.{name}.{key} must be positive or null")
            if float(gpu_profile["expected_smoke_duration_minutes"]) <= 0:
                raise LaunchError(
                    f"runtime.gpu_profiles.{name}.expected_smoke_duration_minutes must be positive"
                )
        str_field(gpu_profile["rationale"], f"runtime.gpu_profiles.{name}.rationale")
