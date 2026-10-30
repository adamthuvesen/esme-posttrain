from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

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
)

from esme_posttrain.launch.errors import LaunchError


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
            if item["type"] == "missing":
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


type FrozenJSON = (
    StrictStr
    | StrictInt
    | StrictFloat
    | StrictBool
    | None
    | tuple[FrozenJSON, ...]
    | Annotated[
        Mapping[str, FrozenJSON],
        AfterValidator(lambda value: MappingProxyType(dict(value))),
        WrapSerializer(lambda value, handler: handler(dict(value))),
    ]
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


class SmokeModalRuntime(ModalRuntime):
    gpu_profiles: SmokeGPUProfiles


class OutputArtifacts(FrozenConfig):
    output_dir: StrictStr
    required_files: tuple[StrictStr, ...]


class PaddedSequence(FrozenConfig):
    pad_to_multiple_of: StrictInt | None
