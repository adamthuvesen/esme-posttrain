"""Held-out Countdown-Lite task sets for the RL transfer eval.

Two eval-only sets, both dedup-verified disjoint from every task in the
committed Countdown-Lite train/dev/eval dataset (dedup key: canonical sorted
numbers tuple + target, the same key the Countdown-Lite generator dedups on):

- ``heldout_fresh``: same generator, same distribution (numbers 1-9, widths
  2-3, targets 0-64), re-ranked with a new selection seed and restricted to
  tasks the original dataset never selected. The easy bucket is capped by the
  remaining unseen population (see ``FRESH_COUNTS``).
- ``heldout_shift``: one modest distribution shift on the target axis the
  generator exposes: targets 65-128 instead of 0-64, with numbers, widths,
  and operators unchanged. Targets above 64 are unreachable in the original
  dataset, so the shift set is disjoint by construction (and still verified).
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from esme_posttrain.rl.countdown_lite import (
    COUNTDOWN_REWARD_NAME,
    CountdownLiteError,
    CountdownTask,
    _candidate_pools,
    _difficulty_for,
    _solutions_for_numbers,
    build_countdown_lite_tasks,
    stable_task_score,
)

HELDOUT_SEED = 4126
FRESH_SPLIT = "heldout_fresh"
SHIFT_SPLIT = "heldout_shift"
# The acceptance eval mix is 10 easy / 12 medium / 8 hard, but the committed
# dataset already consumed 110 of the 115 unique easy tasks the generator can
# produce, so the fresh set takes the entire remaining easy population (5) and
# keeps the acceptance medium:hard ratio (3:2) for the other 25 tasks.
FRESH_COUNTS = (("easy", 5), ("medium", 15), ("hard", 10))
SHIFT_TASK_COUNT = 30
SHIFT_TARGET_MIN = 65
SHIFT_TARGET_MAX = 128

TaskKey = tuple[tuple[int, ...], int]


def countdown_lite_task_keys() -> frozenset[TaskKey]:
    """Dedup keys (sorted numbers, target) of all committed Countdown-Lite tasks."""
    return frozenset((task.numbers, task.target) for task in build_countdown_lite_tasks())


def build_heldout_fresh_tasks() -> tuple[CountdownTask, ...]:
    used_keys = countdown_lite_task_keys()
    pools = _candidate_pools()
    tasks: list[CountdownTask] = []
    for difficulty, count in FRESH_COUNTS:
        unused = [item for item in pools[difficulty] if (item[0], item[1]) not in used_keys]
        unused.sort(
            key=lambda item: (
                stable_task_score(HELDOUT_SEED, difficulty, *item),
                item[0],
                item[1],
            )
        )
        selected = unused[:count]
        if len(selected) != count:
            raise CountdownLiteError(
                f"not enough unused {difficulty} tasks for {FRESH_SPLIT}: {len(selected)}"
            )
        for index, (numbers, target, solution) in enumerate(selected):
            tasks.append(
                CountdownTask(
                    task_id=f"countdown_heldout_fresh_{difficulty}_{index:04d}",
                    split=FRESH_SPLIT,
                    difficulty=difficulty,
                    numbers=numbers,
                    target=target,
                    solution=solution,
                )
            )
    _require_disjoint(tasks, used_keys, set_name=FRESH_SPLIT)
    return tuple(tasks)


def build_heldout_shift_tasks() -> tuple[CountdownTask, ...]:
    used_keys = countdown_lite_task_keys()
    seen: set[TaskKey] = set()
    candidates: list[tuple[tuple[int, ...], int, str]] = []
    for width in (2, 3):
        for numbers in product(range(1, 10), repeat=width):
            sorted_numbers = tuple(sorted(numbers))
            for target, solution in sorted(_solutions_for_numbers(sorted_numbers).items()):
                if not SHIFT_TARGET_MIN <= target <= SHIFT_TARGET_MAX:
                    continue
                key = (sorted_numbers, target)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((sorted_numbers, target, solution))
    candidates.sort(
        key=lambda item: (
            stable_task_score(HELDOUT_SEED, SHIFT_SPLIT, *item),
            item[0],
            item[1],
        )
    )
    selected = candidates[:SHIFT_TASK_COUNT]
    if len(selected) != SHIFT_TASK_COUNT:
        raise CountdownLiteError(
            f"not enough shifted-target tasks for {SHIFT_SPLIT}: {len(selected)}"
        )
    tasks = [
        CountdownTask(
            task_id=f"countdown_heldout_shift_{index:04d}",
            split=SHIFT_SPLIT,
            difficulty=_difficulty_for(numbers, target, solution),
            numbers=numbers,
            target=target,
            solution=solution,
        )
        for index, (numbers, target, solution) in enumerate(selected)
    ]
    _require_disjoint(tasks, used_keys, set_name=SHIFT_SPLIT)
    return tuple(tasks)


def write_countdown_heldout_dataset(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.expanduser().resolve()
    fresh_tasks = build_heldout_fresh_tasks()
    shift_tasks = build_heldout_shift_tasks()
    fresh_keys = {(task.numbers, task.target) for task in fresh_tasks}
    shift_keys = {(task.numbers, task.target) for task in shift_tasks}
    overlap = fresh_keys & shift_keys
    if overlap:
        raise CountdownLiteError(f"held-out sets overlap on {len(overlap)} task keys")

    data_dir = repo_root / "data" / "rl" / "countdown_heldout"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split, tasks in ((FRESH_SPLIT, fresh_tasks), (SHIFT_SPLIT, shift_tasks)):
        rows = [task.to_row() for task in tasks]
        split_counts[split] = len(rows)
        with (data_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "manifest_type": "rl_tasks",
        "name": "esme_214m_rl_countdown_heldout",
        "sample_budget": len(fresh_tasks) + len(shift_tasks),
        "reward_definitions": [
            {
                "name": COUNTDOWN_REWARD_NAME,
                "reward_type": "execution_check",
                "verifiable": True,
                "verifier": ("esme_posttrain.rl.countdown_lite.verify_countdown_lite_expression"),
                "pass_condition": (
                    "candidate is a valid arithmetic expression using each supplied "
                    "number exactly once and evaluating exactly to target"
                ),
            }
        ],
        "data_files": [
            {
                "path": f"../rl/countdown_heldout/{split}.jsonl",
                "format": "jsonl",
                "records": split_counts[split],
            }
            for split in (FRESH_SPLIT, SHIFT_SPLIT)
        ],
    }
    manifest_path = repo_root / "data" / "manifests" / "esme-214m-rl-heldout.tasks.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "data_dir": str(data_dir),
        "records": len(fresh_tasks) + len(shift_tasks),
        "split_counts": split_counts,
    }


def _require_disjoint(
    tasks: list[CountdownTask], used_keys: frozenset[TaskKey], *, set_name: str
) -> None:
    collisions = [task.task_id for task in tasks if (task.numbers, task.target) in used_keys]
    if collisions:
        raise CountdownLiteError(
            f"{set_name} collides with committed Countdown-Lite tasks: {collisions}"
        )
