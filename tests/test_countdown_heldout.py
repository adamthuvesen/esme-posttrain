from __future__ import annotations

import json
from pathlib import Path

from esme_posttrain.cli import main
from esme_posttrain.rl.countdown_heldout import (
    SHIFT_TARGET_MAX,
    SHIFT_TARGET_MIN,
    build_heldout_fresh_tasks,
    build_heldout_shift_tasks,
    countdown_lite_task_keys,
)
from esme_posttrain.rl.countdown_lite import (
    build_countdown_lite_tasks,
    load_countdown_lite_rows,
    verify_countdown_lite_expression,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_heldout_fresh_matches_declared_mix_and_distribution() -> None:
    tasks = build_heldout_fresh_tasks()

    assert len(tasks) == 30
    assert build_heldout_fresh_tasks() == tasks
    difficulty_counts = {
        difficulty: sum(task.difficulty == difficulty for task in tasks)
        for difficulty in ("easy", "medium", "hard")
    }
    # Easy is capped at the 5 unseen easy tasks the generator has left;
    # medium:hard keeps the acceptance 3:2 ratio.
    assert difficulty_counts == {"easy": 5, "medium": 15, "hard": 10}
    for task in tasks:
        assert task.split == "heldout_fresh"
        assert 2 <= len(task.numbers) <= 3
        assert all(1 <= number <= 9 for number in task.numbers)
        assert 0 <= task.target <= 64


def test_heldout_shift_moves_only_the_target_range() -> None:
    tasks = build_heldout_shift_tasks()

    assert len(tasks) == 30
    assert build_heldout_shift_tasks() == tasks
    for task in tasks:
        assert task.split == "heldout_shift"
        assert 2 <= len(task.numbers) <= 3
        assert all(1 <= number <= 9 for number in task.numbers)
        assert SHIFT_TARGET_MIN <= task.target <= SHIFT_TARGET_MAX


def test_heldout_sets_are_disjoint_from_countdown_lite_and_each_other() -> None:
    existing_keys = countdown_lite_task_keys()
    fresh_keys = {(task.numbers, task.target) for task in build_heldout_fresh_tasks()}
    shift_keys = {(task.numbers, task.target) for task in build_heldout_shift_tasks()}

    assert len(existing_keys) == 360
    assert not fresh_keys & existing_keys
    assert not shift_keys & existing_keys
    assert not fresh_keys & shift_keys
    assert len(fresh_keys) == 30
    assert len(shift_keys) == 30


def test_heldout_solutions_are_exact_solves() -> None:
    for task in build_heldout_fresh_tasks() + build_heldout_shift_tasks():
        result = verify_countdown_lite_expression(
            task.solution, numbers=task.numbers, target=task.target
        )
        assert result.is_exact_solve is True


def test_heldout_build_data_cli_writes_manifest_and_jsonl(tmp_path: Path) -> None:
    assert main(["rlvr-countdown-heldout-build-data", "--repo-root", str(tmp_path), "--json"]) == 0

    manifest_path = tmp_path / "data" / "manifests" / "esme-214m-rl-heldout.tasks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["manifest_type"] == "rl_tasks"
    assert manifest["sample_budget"] == 60
    assert [entry["records"] for entry in manifest["data_files"]] == [30, 30]
    fresh_rows = load_countdown_lite_rows(manifest_path, split="heldout_fresh")
    shift_rows = load_countdown_lite_rows(manifest_path, split="heldout_shift")
    assert len(fresh_rows) == 30
    assert len(shift_rows) == 30
    assert fresh_rows[0]["reward_name"] == "countdown_lite_exact_solve"


def test_countdown_lite_committed_dataset_is_unchanged_by_heldout_helpers() -> None:
    # Regression guard: the seed-parameterized score refactor must not move
    # the committed Countdown-Lite selection.
    manifest_path = REPO_ROOT / "data" / "manifests" / "esme-214m-rl.tasks.json"
    committed_rows = load_countdown_lite_rows(manifest_path)
    generated_rows = [task.to_row() for task in build_countdown_lite_tasks()]

    assert [dict(row) for row in committed_rows] == generated_rows
