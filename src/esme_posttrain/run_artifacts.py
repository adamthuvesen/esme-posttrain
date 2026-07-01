"""Shared writers for run-evidence artifacts (env capture, manifests, JSON dumps).

These are pure leaf utilities with no family-specific behavior, shared across the
SFT / multi-turn / DPO full-run, sweep, probe, and smoke paths so a fix lands once.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.bundle import file_sha256


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_environment(path: Path, *, device: torch.device) -> None:
    lines = [
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"selected_device={device.type}",
    ]
    if torch.cuda.is_available():
        lines.extend(
            [
                f"cuda_device_count={torch.cuda.device_count()}",
                f"cuda_device_name={torch.cuda.get_device_name(0)}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selected_row_manifest(path: Path, examples: tuple[Any, ...]) -> None:
    rows = (json.dumps(example.manifest_entry(), sort_keys=True) + "\n" for example in examples)
    path.write_text("".join(rows), encoding="utf-8")


def refresh_manifest_files(output_dir: Path, expected_artifacts: tuple[str, ...]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.setdefault("files", {})
    for name in expected_artifacts:
        if name == "manifest.json":
            continue
        path = output_dir / name
        if path.is_file():
            files[name] = {"path": name, "sha256": file_sha256(path)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class RuntimeSpendTracker:
    started: float
    usd_per_hour: float
    stop_usd: float
    output_dir: Path
    paid_compute: bool = True

    def estimated_cost_usd(self) -> float:
        return (time.perf_counter() - self.started) * self.usd_per_hour / 3600.0

    def write_cost(self, *, step: int, status: str) -> dict[str, Any]:
        return write_runtime_cost(
            self.output_dir,
            started=self.started,
            usd_per_hour=self.usd_per_hour,
            stop_usd=self.stop_usd,
            step=step,
            status=status,
            paid_compute=self.paid_compute,
        )


def write_runtime_cost(
    output_dir: Path,
    *,
    started: float,
    usd_per_hour: float,
    stop_usd: float,
    step: int,
    status: str,
    paid_compute: bool = True,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    payload = {
        "paid_compute": paid_compute,
        "status": status,
        "elapsed_seconds": elapsed,
        "usd_per_hour": usd_per_hour,
        "estimated_cost_usd": elapsed * usd_per_hour / 3600.0,
        "runtime_spend_stop_usd": stop_usd,
        "step": step,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "cost.json", payload)
    return payload
