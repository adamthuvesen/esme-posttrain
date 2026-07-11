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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


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


def write_eval_suite_manifests(
    output_dir: Path, matched_eval_reports: dict[str, Any], no_robots_examples: tuple[Any, ...]
) -> None:
    for name, report in matched_eval_reports.items():
        write_selected_row_manifest(output_dir / f"eval-{name}-manifest.jsonl", report.examples)
    write_selected_row_manifest(output_dir / "eval-no_robots-manifest.jsonl", no_robots_examples)


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
    write_json(manifest_path, manifest)


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
        elapsed = time.perf_counter() - self.started
        payload = {
            "paid_compute": self.paid_compute,
            "status": status,
            "elapsed_seconds": elapsed,
            "usd_per_hour": self.usd_per_hour,
            "estimated_cost_usd": elapsed * self.usd_per_hour / 3600.0,
            "runtime_spend_stop_usd": self.stop_usd,
            "step": step,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "cost.json", payload)
        return payload

    def check_cap(
        self,
        step: int,
        *,
        label: str,
        error_type: type[Exception] = RuntimeError,
    ) -> None:
        cost = self.estimated_cost_usd()
        if cost > self.stop_usd:
            self.write_cost(step=step, status="runtime_cap_exceeded")
            raise error_type(
                f"{label} runtime spend estimate ${cost:.4f} exceeded ${self.stop_usd:.2f}"
            )
