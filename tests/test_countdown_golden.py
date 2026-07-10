"""Golden-fixture tests that pin the Countdown-Lite eval path output.

The golden files under ``fixtures/outputs/countdown_lite_golden/`` are the
"before" photo for the shared evaluation contract migration: existing
Countdown-Lite metrics must match them exactly. New output fields may be
added, but every recorded key must keep its recorded value.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from esme_posttrain.rl.countdown_lite import (
    load_countdown_lite_rows,
    verify_countdown_lite_expression,
)
from esme_posttrain.rl.countdown_lite_baseline import (
    CountdownBaselineProgressError,
    CountdownBaselineRequest,
    run_countdown_lite_baseline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "fixtures" / "outputs" / "countdown_lite_golden"
TINY_MANIFEST = REPO_ROOT / "fixtures" / "manifests" / "rl_tasks_tiny.json"
TINY_BUNDLE = REPO_ROOT / "fixtures" / "tiny_bundle"


def assert_matches_golden(golden: object, fresh: object, path: str = "$") -> None:
    """Every key and value recorded in the golden must survive unchanged.

    Additive keys in the fresh output are allowed; changed or missing ones
    are not. Lists must keep their exact length and order.
    """
    if isinstance(golden, dict):
        assert isinstance(fresh, dict), f"{path}: expected dict, got {type(fresh).__name__}"
        for key, value in golden.items():
            assert key in fresh, f"{path}.{key}: missing from fresh output"
            assert_matches_golden(value, fresh[key], f"{path}.{key}")
    elif isinstance(golden, list):
        assert isinstance(fresh, list), f"{path}: expected list, got {type(fresh).__name__}"
        assert len(fresh) == len(golden), f"{path}: length {len(fresh)} != golden {len(golden)}"
        for index, (golden_item, fresh_item) in enumerate(zip(golden, fresh, strict=True)):
            assert_matches_golden(golden_item, fresh_item, f"{path}[{index}]")
    else:
        assert fresh == golden, f"{path}: {fresh!r} != golden {golden!r}"


def golden_baseline_request(output_dir: Path) -> CountdownBaselineRequest:
    return CountdownBaselineRequest(
        manifest_path=TINY_MANIFEST,
        bundle_path=TINY_BUNDLE,
        output_dir=output_dir,
        split="eval",
        samples_per_task=8,
        max_new_tokens=8,
        seed=214,
        device="cpu",
        progress_label="golden_eval",
        eval_profile="golden_fixture_8x8",
        config_hash="golden-fixture-config",
        model_id="tiny-bundle-fixture",
    )


def normalize_report_paths(report: dict[str, object]) -> dict[str, object]:
    normalized = dict(report)
    for key in ("bundle_path", "manifest_path"):
        normalized[key] = str(normalized[key]).replace(str(REPO_ROOT), "<repo>")
    return normalized


def test_baseline_report_matches_golden(tmp_path: Path) -> None:
    run_countdown_lite_baseline(golden_baseline_request(tmp_path))

    fresh = normalize_report_paths(
        json.loads((tmp_path / "baseline-report.json").read_text(encoding="utf-8"))
    )
    golden = json.loads((GOLDEN_DIR / "baseline-report.json").read_text(encoding="utf-8"))

    assert_matches_golden(golden, fresh)


def test_baseline_partial_lines_match_golden(tmp_path: Path) -> None:
    run_countdown_lite_baseline(golden_baseline_request(tmp_path))

    fresh_lines = [
        json.loads(line)
        for line in (tmp_path / "baseline-partial.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    golden_lines = [
        json.loads(line)
        for line in (GOLDEN_DIR / "baseline-partial.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert_matches_golden(golden_lines, fresh_lines)


def test_resume_rejects_partial_with_incomplete_sample_budget(tmp_path: Path) -> None:
    first_request = CountdownBaselineRequest(
        manifest_path=TINY_MANIFEST,
        bundle_path=TINY_BUNDLE,
        output_dir=tmp_path,
        split="eval",
        samples_per_task=2,
        max_new_tokens=2,
        seed=214,
        device="cpu",
        progress_label="budget_eval",
        eval_profile="budget_fixture_2x2",
        config_hash="budget-fixture-config",
        model_id="tiny-bundle-fixture",
    )
    run_countdown_lite_baseline(first_request)

    partial_path = tmp_path / "baseline-partial.jsonl"
    line = json.loads(partial_path.read_text(encoding="utf-8").splitlines()[0])
    # Truncate the recorded samples but keep the counts consistent, so the
    # only defect left is the incomplete sample budget.
    task_result = line["task_result"]
    task_result["samples"] = task_result["samples"][:1]
    task_result["valid_samples"] = sum(
        sample["is_valid_expression"] for sample in task_result["samples"]
    )
    task_result["exact_samples"] = sum(
        sample["is_exact_solve"] for sample in task_result["samples"]
    )
    partial_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

    with pytest.raises(CountdownBaselineProgressError, match="recorded 1 samples, budget is 2"):
        run_countdown_lite_baseline(
            CountdownBaselineRequest(
                manifest_path=TINY_MANIFEST,
                bundle_path=TINY_BUNDLE,
                output_dir=tmp_path,
                split="eval",
                samples_per_task=2,
                max_new_tokens=2,
                seed=214,
                device="cpu",
                progress_label="budget_eval",
                eval_profile="budget_fixture_2x2",
                config_hash="budget-fixture-config",
                model_id="tiny-bundle-fixture",
                resume_from_partial=True,
            )
        )


def test_verifier_scores_match_golden() -> None:
    golden_records = json.loads((GOLDEN_DIR / "verifier-scores.json").read_text(encoding="utf-8"))
    rows_by_task_id = {str(row["task_id"]): row for row in load_countdown_lite_rows(TINY_MANIFEST)}
    assert len(golden_records) == 20

    for record in golden_records:
        row = rows_by_task_id[record["task_id"]]
        result = verify_countdown_lite_expression(
            record["candidate"],
            numbers=[int(number) for number in row["numbers"]],
            target=int(row["target"]),
        )
        assert_matches_golden(
            record["result"], asdict(result), f"$.{record['task_id']}.{record['candidate']!r}"
        )
