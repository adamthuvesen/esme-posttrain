from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    WrapSerializer,
    model_validator,
)

from esme_posttrain.launch.errors import LaunchError

SMOKE_SPEND_CAP_USD = 2.0


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def to_dict(self) -> dict[str, object]:
        # Config hashes depend on optional-key presence and the original numeric types.
        return self.model_dump(mode="json", exclude_unset=True)


def parse_config[ConfigT: FrozenConfig](
    model: type[ConfigT], payload: object, *, source: Path
) -> ConfigT:
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        details = []
        for item in error.errors(include_url=False):
            location = item["loc"]
            numeric_branch = item["type"] in {
                "int_type",
                "float_type",
                "finite_number",
            } and location[-1:] in (("int",), ("float",))
            if numeric_branch:
                location = location[:-1]
            path = ".".join(str(part) for part in location)
            if item["type"] == "value_error" and isinstance(
                cause := item.get("ctx", {}).get("error"), LaunchError
            ):
                # Recipe validators raise messages relative to their own model.
                details.append(f"{path}.{cause}" if path else str(cause))
            elif item["type"] == "missing":
                parent, _, key = path.rpartition(".")
                details.append(f"{parent or 'config'} is missing required keys: {key}")
            elif item["type"] == "extra_forbidden":
                parent, _, key = path.rpartition(".")
                details.append(f"{parent or 'config'} has unsupported keys: {key}")
            elif numeric_branch and item["type"] in {"int_type", "float_type"}:
                details.append(f"{path} must be a number")
            else:
                details.append(f"{path}: {item['msg']}")
        message = "; ".join(dict.fromkeys(details))
        raise LaunchError(f"{source}: {message}") from error


def str_field(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LaunchError(f"{label} must be a non-empty string")
    return value


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LaunchError(f"{label} must be a positive integer")
    return value


type FrozenJSON = (
    StrictStr
    | StrictInt
    | StrictFloat
    | StrictBool
    | tuple[FrozenJSON, ...]
    | Annotated[
        Mapping[str, FrozenJSON],
        AfterValidator(lambda value: MappingProxyType(dict(value))),
        WrapSerializer(lambda value, handler: handler(dict(value))),
    ]
    | None
)

JSONEvidence = Annotated[
    Mapping[str, FrozenJSON],
    AfterValidator(lambda value: MappingProxyType(dict(value))),
    WrapSerializer(lambda value, handler: handler(dict(value))),
]


class BundleConfig(FrozenConfig):
    path: StrictStr
    format: StrictStr
    model_family: StrictStr
    read_only: StrictBool

    @model_validator(mode="after")
    def _check_bundle(self) -> Self:
        if self.format != "llm_pretrain_dense_v1":
            raise LaunchError("format must be llm_pretrain_dense_v1")
        if self.model_family != "DenseBackbone":
            raise LaunchError("model_family must be DenseBackbone")
        if self.read_only is not True:
            raise LaunchError("read_only must be true")
        str_field(self.path, "path")
        return self


class GPUProfile(FrozenConfig):
    modal_gpu: StrictStr
    usd_per_hour: StrictFloat | StrictInt
    projected_tokens_per_second: StrictFloat | StrictInt
    rationale: StrictStr
    projection_source: StrictStr
    measured: StrictBool


class SmokeGPUProfile(GPUProfile):
    measured_tokens_per_second: StrictFloat | StrictInt | None
    measured_smoke_tokens_per_second: StrictFloat | StrictInt | None
    expected_smoke_duration_minutes: StrictFloat | StrictInt


GPUProfiles = Annotated[
    Mapping[str, GPUProfile],
    AfterValidator(lambda profiles: MappingProxyType(dict(profiles))),
    WrapSerializer(lambda profiles, handler: handler(dict(profiles))),
]
SmokeGPUProfiles = Annotated[
    Mapping[str, SmokeGPUProfile],
    AfterValidator(lambda profiles: MappingProxyType(dict(profiles))),
    WrapSerializer(lambda profiles, handler: handler(dict(profiles))),
]


class ModalRuntime(FrozenConfig):
    provider: StrictStr
    selected_gpu: StrictStr
    gpu_profiles: GPUProfiles
    precision: StrictStr
    smoke_max_cost_usd: StrictFloat | StrictInt
    full_run_max_cost_usd: StrictFloat | StrictInt
    runtime_spend_stop_usd: StrictFloat | StrictInt
    full_run_runtime_spend_stop_usd: StrictFloat | StrictInt
    allow_retries: StrictBool
    modal_volume: StrictStr
    timeout_hours: StrictInt

    @model_validator(mode="after")
    def _check_runtime(self) -> Self:
        if self.provider != "modal":
            raise LaunchError("provider must be modal")
        if self.precision not in {"fp32", "bf16"}:
            raise LaunchError("precision must be fp32 or bf16")
        if self.allow_retries is not False:
            raise LaunchError("allow_retries must be false")
        if self.smoke_max_cost_usd > SMOKE_SPEND_CAP_USD:
            raise LaunchError("smoke_max_cost_usd must be <= 2")
        if self.runtime_spend_stop_usd > SMOKE_SPEND_CAP_USD:
            raise LaunchError("runtime_spend_stop_usd must be <= 2")
        if self.timeout_hours <= 0 or self.timeout_hours > 24:
            raise LaunchError("timeout_hours must be between 1 and 24")
        if len(self.gpu_profiles) < 2:
            raise LaunchError("gpu_profiles must include at least two GPU profiles")
        if self.selected_gpu not in self.gpu_profiles:
            raise LaunchError("selected_gpu must name a gpu_profiles entry")
        for name, profile in self.gpu_profiles.items():
            _check_gpu_profile(name, profile)
        return self


class SmokeModalRuntime(ModalRuntime):
    gpu_profiles: SmokeGPUProfiles  # pyright: ignore[reportIncompatibleVariableOverride]  # the smoke profile narrows the field type


def check_stage_runtime(
    runtime: ModalRuntime, *, full_run_spend_cap_usd: float, modal_volume: str
) -> None:
    cap = f"{full_run_spend_cap_usd:g}"
    if runtime.full_run_max_cost_usd > full_run_spend_cap_usd:
        raise LaunchError(f"runtime.full_run_max_cost_usd must be <= {cap}")
    if runtime.full_run_runtime_spend_stop_usd > full_run_spend_cap_usd:
        raise LaunchError(f"runtime.full_run_runtime_spend_stop_usd must be <= {cap}")
    if runtime.modal_volume != modal_volume:
        raise LaunchError(f"runtime.modal_volume must be {modal_volume}")


def _check_gpu_profile(name: str, profile: GPUProfile) -> None:
    label = f"gpu_profiles.{name}"
    if profile.modal_gpu != name:
        raise LaunchError(f"{label}.modal_gpu must be {name}")
    if profile.usd_per_hour <= 0:
        raise LaunchError(f"{label}.usd_per_hour must be positive")
    if profile.projected_tokens_per_second <= 0:
        raise LaunchError(f"{label}.projected_tokens_per_second must be positive")
    if isinstance(profile, SmokeGPUProfile):
        for key, value in (
            ("measured_tokens_per_second", profile.measured_tokens_per_second),
            ("measured_smoke_tokens_per_second", profile.measured_smoke_tokens_per_second),
        ):
            if value is not None and value <= 0:
                raise LaunchError(f"{label}.{key} must be positive or null")
        if profile.expected_smoke_duration_minutes <= 0:
            raise LaunchError(f"{label}.expected_smoke_duration_minutes must be positive")
    str_field(profile.rationale, f"{label}.rationale")


class OutputArtifacts(FrozenConfig):
    output_dir: StrictStr
    required_files: tuple[StrictStr, ...]


def check_output_artifacts(
    artifacts: OutputArtifacts, *, expected_files: tuple[str, ...], manifest_label: str
) -> None:
    output_dir = Path(str_field(artifacts.output_dir, "artifacts.output_dir"))
    if output_dir.is_absolute() or ".." in output_dir.parts:
        raise LaunchError("artifacts.output_dir must stay inside the repository")
    if output_dir.parts[:1] != ("runs",):
        raise LaunchError("artifacts.output_dir must be under runs/")
    if tuple(artifacts.required_files) != expected_files:
        raise LaunchError(f"artifacts.required_files must match the {manifest_label}")


class PaddedSequence(FrozenConfig):
    pad_to_multiple_of: StrictInt | None
