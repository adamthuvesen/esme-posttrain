from __future__ import annotations

import json
from pathlib import Path

import pytest

from esme_posttrain.cli import main
from esme_posttrain.rl.countdown_lite import (
    build_countdown_lite_tasks,
    load_countdown_lite_rows,
    verify_countdown_lite_expression,
)
from esme_posttrain.rl.countdown_lite_baseline import _decision


def test_exact_verifier_accepts_valid_countdown_expression() -> None:
    result = verify_countdown_lite_expression("(2 + 3) * 4", numbers=(2, 3, 4), target=20)

    assert result.is_valid_expression is True
    assert result.is_exact_solve is True
    assert result.value == 20


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("2 + 3", "use each supplied number exactly once"),
        ("2 + 3 + 4", "evaluated to 9"),
        ("2 / 3 + 4", "unsupported characters"),
        ("2 + (3 * 4", "missing closing parenthesis"),
    ],
)
def test_exact_verifier_rejects_invalid_expressions(candidate: str, reason: str) -> None:
    result = verify_countdown_lite_expression(candidate, numbers=(2, 3, 4), target=20)

    assert result.is_exact_solve is False
    assert reason in result.reason


@pytest.mark.parametrize(
    "candidate",
    [
        "The numbers are 2 and 3 and 4\n(2 + 3) * 4",
        "either/or, hmm\n(2 + 3) * 4",
        "Expression: (2 + 3) * 4",
    ],
)
def test_exact_verifier_prefers_expression_line_over_preamble(candidate: str) -> None:
    result = verify_countdown_lite_expression(candidate, numbers=(2, 3, 4), target=20)

    assert result.is_exact_solve is True
    assert result.expression == "(2 + 3) * 4"


def test_countdown_lite_tasks_are_deterministic_and_bucketed() -> None:
    tasks = build_countdown_lite_tasks()

    assert len(tasks) == 360
    assert build_countdown_lite_tasks() == tasks
    assert {task.split for task in tasks} == {"train", "dev", "eval"}
    assert {task.difficulty for task in tasks} == {"easy", "medium", "hard"}

    for task in tasks[:20]:
        result = verify_countdown_lite_expression(
            task.solution,
            numbers=task.numbers,
            target=task.target,
        )
        assert result.is_exact_solve is True


def test_countdown_lite_build_data_cli_writes_manifest_and_jsonl(tmp_path: Path) -> None:
    assert main(["rlvr-countdown-lite-build-data", "--repo-root", str(tmp_path), "--json"]) == 0

    manifest_path = tmp_path / "data" / "manifests" / "esme-214m-rl.tasks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_type"] == "rl_tasks"
    assert manifest["sample_budget"] == 360
    assert [entry["records"] for entry in manifest["data_files"]] == [300, 30, 30]
    eval_rows = load_countdown_lite_rows(manifest_path, split="eval")
    assert len(eval_rows) == 30
    assert eval_rows[0]["reward_name"] == "countdown_lite_exact_solve"


def test_countdown_lite_decision_requires_exact_signal() -> None:
    no_signal = [{"difficulty": "easy", "pass@32": False}]
    valid_but_no_exact = [{"difficulty": "easy", "pass@32": False}]
    exact_easy_signal = [{"difficulty": "easy", "pass@32": True}]

    assert (
        _decision(no_signal, valid_count=0, exact_count=0, samples_per_task=32)
        == "blocked-with-evidence"
    )
    assert (
        _decision(valid_but_no_exact, valid_count=2, exact_count=0, samples_per_task=32)
        == "needs SFT/hint cold-start"
    )
    assert (
        _decision(exact_easy_signal, valid_count=2, exact_count=1, samples_per_task=32)
        == "GRPO-ready"
    )


def test_countdown_lite_decision_uses_largest_honest_pass_at_k() -> None:
    one_sample_signal = [{"difficulty": "easy", "pass@1": True}]

    assert (
        _decision(one_sample_signal, valid_count=1, exact_count=1, samples_per_task=1)
        == "GRPO-ready"
    )
