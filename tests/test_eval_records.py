"""Tests for the typed evaluation records and their loud-failure behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from esme_posttrain.evals.records import (
    CountdownEvalResumeLine,
    CountdownEvalSummary,
    CountdownSampleScore,
    CountdownTaskResult,
    CountdownTaskRow,
    dump_record,
)
from esme_posttrain.rl.countdown_lite_baseline import (
    CountdownBaselineRequest,
    _summarize,
    _validated_task_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "fixtures" / "outputs" / "countdown_lite_golden"

VALID_ROW = {
    "task_id": "countdown_lite_fixture_eval_easy_0000",
    "split": "eval",
    "difficulty": "easy",
    "prompt": "Numbers: 4, 5\nTarget: 20\nExpression:",
    "reward_name": "countdown_lite_exact_solve",
    "numbers": [4, 5],
    "target": 20,
    "solution": "4 * 5",
}


def test_task_row_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="unexpected_field"):
        CountdownTaskRow.model_validate({**VALID_ROW, "unexpected_field": 1})


def test_task_row_rejects_missing_fields() -> None:
    row = dict(VALID_ROW)
    del row["target"]
    with pytest.raises(ValidationError, match="target"):
        CountdownTaskRow.model_validate(row)


def test_task_row_rejects_mistyped_numbers() -> None:
    with pytest.raises(ValidationError, match="numbers"):
        CountdownTaskRow.model_validate({**VALID_ROW, "numbers": "4, 5"})


def test_validated_task_rows_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValueError, match="duplicate Countdown-Lite task_id"):
        _validated_task_rows((VALID_ROW, VALID_ROW))


def test_sample_score_requires_all_three_score_axes() -> None:
    with pytest.raises(ValidationError, match="is_well_formed"):
        CountdownSampleScore.model_validate(
            {
                "output": "4 * 5",
                "extracted_expression": "4 * 5",
                "is_valid_expression": True,
                "is_exact_solve": True,
                "value": 20,
                "reason": "exact_solve",
            }
        )


def test_sample_score_rejects_invalid_verifier_tiers() -> None:
    consistent = {
        "output": "4 * 5",
        "extracted_expression": "4 * 5",
        "is_well_formed": True,
        "is_valid_expression": True,
        "is_exact_solve": True,
        "value": 20,
        "reason": "exact_solve",
    }
    CountdownSampleScore.model_validate(consistent)

    with pytest.raises(ValidationError, match="exact solve without a valid expression"):
        CountdownSampleScore.model_validate({**consistent, "is_valid_expression": False})
    with pytest.raises(ValidationError, match="not well-formed"):
        CountdownSampleScore.model_validate(
            {**consistent, "is_exact_solve": False, "is_well_formed": False}
        )


def test_task_result_rejects_counts_that_contradict_samples() -> None:
    golden = json.loads((GOLDEN_DIR / "baseline-report.json").read_text(encoding="utf-8"))
    task = golden["tasks"][0]
    for sample in task["samples"]:
        sample.setdefault("is_well_formed", False)

    CountdownTaskResult.model_validate(task)

    with pytest.raises(ValidationError, match="do not match samples"):
        CountdownTaskResult.model_validate({**task, "exact_samples": task["exact_samples"] + 1})


def test_summarize_rejects_incomplete_sample_budget() -> None:
    golden = json.loads((GOLDEN_DIR / "baseline-report.json").read_text(encoding="utf-8"))
    task = golden["tasks"][0]
    for sample in task["samples"]:
        sample.setdefault("is_well_formed", False)
    task["samples"] = task["samples"][:3]
    task["valid_samples"] = sum(sample["is_valid_expression"] for sample in task["samples"])
    task["exact_samples"] = sum(sample["is_exact_solve"] for sample in task["samples"])

    request = CountdownBaselineRequest(
        manifest_path=Path("unused-manifest.json"),
        bundle_path=Path("unused-bundle"),
        output_dir=Path("unused-output"),
        samples_per_task=8,
    )
    with pytest.raises(ValueError, match="incomplete sample budget"):
        _summarize(request, rows=[], all_results=[task])


def test_golden_report_round_trips_through_summary_record() -> None:
    golden = json.loads((GOLDEN_DIR / "baseline-report.json").read_text(encoding="utf-8"))
    # The golden predates the record fields that are additive-only.
    for task in golden["tasks"]:
        for sample in task["samples"]:
            sample.setdefault("is_well_formed", False)

    record = CountdownEvalSummary.model_validate(golden)
    dumped = dump_record(record)

    assert dumped["pass@1"] == golden["pass@1"]
    assert dumped["valid_expression_rate"] == golden["valid_expression_rate"]
    assert dumped["decision"] == golden["decision"]
    assert "pass@32" not in dumped
    assert dumped["verifier_name"] == "countdown_lite_exact_solve"
    assert dumped["verifier_version"] == 1


def test_resume_line_round_trips_and_keeps_null_fields() -> None:
    golden_line = (
        (GOLDEN_DIR / "baseline-partial.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    payload = json.loads(golden_line)
    for sample in payload["task_result"]["samples"]:
        sample.setdefault("is_well_formed", False)

    record = CountdownEvalResumeLine.model_validate(payload)
    dumped = dump_record(record)

    assert dumped["task_id"] == payload["task_id"]
    assert dumped["task_result"]["samples"][0]["value"] is None
    assert "value" in dumped["task_result"]["samples"][0]
