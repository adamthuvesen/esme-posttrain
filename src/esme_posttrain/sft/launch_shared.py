from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from esme_posttrain.launch.config_models import FrozenJSON


def prepare_evidence_dir(default_output_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        evidence_dir = (Path.cwd() / default_output_dir.parent / "local-cpu-fixture").resolve()
        if evidence_dir.exists():
            shutil.rmtree(evidence_dir)
        evidence_dir.mkdir(parents=True)
        return evidence_dir

    evidence_dir = output_dir.expanduser().resolve()
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise ValueError(f"custom output_dir must be empty or absent: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def matched_interval_eval_sweep_blockers(evidence: Mapping[str, FrozenJSON]) -> list[str]:
    label = "learning_gate.evidence.bounded_matched_interval_eval_sweep"
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
        not isinstance(steps, tuple)
        or not steps
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
    ):
        blockers.append(f"{label}.interval_eval_steps must list positive steps")
    if not isinstance(evidence["evidence_uri"], str) or not evidence["evidence_uri"]:
        blockers.append(f"{label}.evidence_uri must be a non-empty string")
    return blockers
