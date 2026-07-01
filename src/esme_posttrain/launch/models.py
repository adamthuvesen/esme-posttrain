from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuProfileBlock:
    name: str
    modal_gpu: str
    usd_per_hour: float
    projected_tokens_per_second: float


@dataclass(frozen=True)
class RuntimeBlock:
    selected_gpu: str
    selected_profile: GpuProfileBlock
    smoke_max_cost_usd: float
    full_run_max_cost_usd: float
    timeout_hours: int

    @classmethod
    def from_validated_payload(cls, payload: dict[str, Any]) -> RuntimeBlock:
        selected = str(payload["selected_gpu"])
        profile = payload["gpu_profiles"][selected]
        return cls(
            selected_gpu=selected,
            selected_profile=GpuProfileBlock(
                name=selected,
                modal_gpu=str(profile["modal_gpu"]),
                usd_per_hour=float(profile["usd_per_hour"]),
                projected_tokens_per_second=float(profile["projected_tokens_per_second"]),
            ),
            smoke_max_cost_usd=float(payload["smoke_max_cost_usd"]),
            full_run_max_cost_usd=float(payload["full_run_max_cost_usd"]),
            timeout_hours=int(payload["timeout_hours"]),
        )


@dataclass(frozen=True)
class ArtifactBlock:
    output_dir: Path
    required_files: tuple[str, ...]
