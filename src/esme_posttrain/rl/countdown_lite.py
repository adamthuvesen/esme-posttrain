"""Countdown-Lite task generation and exact expression verification."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
from typing import Any

COUNTDOWN_REWARD_NAME = "countdown_lite_exact_solve"
DEFAULT_OPERATORS = ("+", "-", "*")
DEFAULT_SEED = 214


class CountdownLiteError(ValueError):
    pass


@dataclass(frozen=True)
class CountdownTask:
    task_id: str
    split: str
    difficulty: str
    numbers: tuple[int, ...]
    target: int
    solution: str

    @property
    def prompt(self) -> str:
        numbers_text = ", ".join(str(number) for number in self.numbers)
        return (
            "Solve this Countdown-Lite task. Use each supplied number exactly once. "
            "Allowed operators are +, -, and *. Parentheses are allowed. "
            "Return only the arithmetic expression, with no explanation.\n"
            f"Numbers: {numbers_text}\n"
            f"Target: {self.target}\n"
            "Expression:"
        )

    def to_row(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "reward_name": COUNTDOWN_REWARD_NAME,
            "numbers": list(self.numbers),
            "target": self.target,
            "solution": self.solution,
        }


@dataclass(frozen=True)
class VerificationResult:
    is_valid_expression: bool
    is_exact_solve: bool
    value: int | None
    reason: str
    expression: str | None = None
    # True when a candidate expression parsed as arithmetic, even if it broke a
    # task constraint (wrong numbers, non-integer value). Grades the reward tier
    # between "no expression at all" and "valid expression".
    is_well_formed: bool = False


_TOKEN_PATTERN = re.compile(r"\s*(\d+|[()+\-*])")


def render_chat_prompt(prompt: str) -> str:
    # Must match the repo newline chat template used by SFT and dense export.
    return f"user\n{prompt}\nassistant\n"


def build_countdown_lite_tasks() -> tuple[CountdownTask, ...]:
    specs = (
        ("train", "easy", 90),
        ("train", "medium", 120),
        ("train", "hard", 90),
        ("dev", "easy", 10),
        ("dev", "medium", 12),
        ("dev", "hard", 8),
        ("eval", "easy", 10),
        ("eval", "medium", 12),
        ("eval", "hard", 8),
    )
    pools = _candidate_pools()
    tasks: list[CountdownTask] = []
    for split, difficulty, count in specs:
        selected = pools[difficulty][:count]
        if len(selected) != count:
            raise CountdownLiteError(f"not enough {difficulty} tasks for {split}: {len(selected)}")
        del pools[difficulty][:count]
        for index, (numbers, target, solution) in enumerate(selected):
            task_id = f"countdown_lite_{split}_{difficulty}_{index:04d}"
            tasks.append(
                CountdownTask(
                    task_id=task_id,
                    split=split,
                    difficulty=difficulty,
                    numbers=numbers,
                    target=target,
                    solution=solution,
                )
            )
    return tuple(tasks)


def write_countdown_lite_dataset(repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.expanduser().resolve()
    tasks = build_countdown_lite_tasks()
    data_dir = repo_root / "data" / "rl" / "countdown_lite"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split in ("train", "dev", "eval"):
        rows = [task.to_row() for task in tasks if task.split == split]
        split_counts[split] = len(rows)
        _write_jsonl(data_dir / f"{split}.jsonl", rows)

    manifest = {
        "schema_version": 1,
        "manifest_type": "rl_tasks",
        "name": "esme_214m_rl_countdown_lite",
        "sample_budget": len(tasks),
        "token_budget": 512_000,
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
                "path": "../rl/countdown_lite/train.jsonl",
                "format": "jsonl",
                "records": split_counts["train"],
            },
            {
                "path": "../rl/countdown_lite/dev.jsonl",
                "format": "jsonl",
                "records": split_counts["dev"],
            },
            {
                "path": "../rl/countdown_lite/eval.jsonl",
                "format": "jsonl",
                "records": split_counts["eval"],
            },
        ],
    }
    manifest_path = repo_root / "data" / "manifests" / "esme-214m-rl.tasks.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "data_dir": str(data_dir),
        "records": len(tasks),
        "split_counts": split_counts,
    }


def verify_countdown_lite_expression(
    candidate: str,
    *,
    numbers: tuple[int, ...] | list[int],
    target: int,
) -> VerificationResult:
    expression = extract_candidate_expression(candidate)
    if expression is None:
        return VerificationResult(False, False, None, "no candidate expression found")
    try:
        parser = _ExpressionParser(expression)
        value, used_numbers = parser.parse()
    except CountdownLiteError as error:
        return VerificationResult(False, False, None, str(error), expression=expression)

    expected_numbers = Counter(int(number) for number in numbers)
    if Counter(used_numbers) != expected_numbers:
        return VerificationResult(
            False,
            False,
            int(value) if value.denominator == 1 else None,
            "expression must use each supplied number exactly once",
            expression=expression,
            is_well_formed=True,
        )
    if value.denominator != 1:
        return VerificationResult(
            False,
            False,
            None,
            "expression did not evaluate to an integer",
            expression,
            is_well_formed=True,
        )
    integer_value = int(value)
    if integer_value != int(target):
        return VerificationResult(
            True,
            False,
            integer_value,
            f"expression evaluated to {integer_value}, not target {target}",
            expression=expression,
            is_well_formed=True,
        )
    return VerificationResult(
        True, True, integer_value, "exact_solve", expression=expression, is_well_formed=True
    )


def extract_candidate_expression(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates = [line.removeprefix("Expression:").strip() for line in lines]
    candidates.append(text)
    cleaned_candidates = [candidate.strip().strip("`").strip() for candidate in candidates]
    for cleaned in cleaned_candidates:
        # A whole-line expression wins over fragments salvaged from chatty lines.
        # Division is included so a whole-line division attempt reaches the parser
        # and is graded invalid instead of being shadowed by a salvaged fragment.
        if re.fullmatch(r"[0-9\s()+\-*/]+", cleaned):
            return cleaned
    for cleaned in cleaned_candidates:
        match = re.search(r"[\d(][0-9\s()+\-*]*", cleaned)
        if match:
            return match.group(0).strip()
    return None


def load_countdown_lite_rows(
    manifest_path: Path, *, split: str | None = None
) -> tuple[dict[str, Any], ...]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = manifest_path.parent.parent
    rows: list[dict[str, Any]] = []
    for data_file in manifest["data_files"]:
        path = (manifest_path.parent / str(data_file["path"])).resolve()
        if not path.is_relative_to(data_root):
            raise CountdownLiteError(f"data file path escapes data root: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if split is None or row.get("split") == split:
                    rows.append(row)
    return tuple(rows)


def _candidate_pools() -> dict[str, list[tuple[tuple[int, ...], int, str]]]:
    seen: set[tuple[tuple[int, ...], int]] = set()
    pools: dict[str, list[tuple[tuple[int, ...], int, str]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }
    for width in (2, 3):
        for numbers in product(range(1, 10), repeat=width):
            sorted_numbers = tuple(sorted(numbers))
            solutions = _solutions_for_numbers(sorted_numbers)
            for target, solution in sorted(solutions.items()):
                if target < 0 or target > 64:
                    continue
                key = (sorted_numbers, target)
                if key in seen:
                    continue
                seen.add(key)
                difficulty = _difficulty_for(sorted_numbers, target, solution)
                pools[difficulty].append((sorted_numbers, target, solution))
    for difficulty, tasks in pools.items():
        tasks.sort(key=lambda item: (_stable_score(difficulty, *item), item[0], item[1]))
    return pools


def _difficulty_for(numbers: tuple[int, ...], target: int, solution: str) -> str:
    if len(numbers) == 2 and target <= 24:
        return "easy"
    if len(numbers) == 3 and 0 <= target <= 36 and solution.count("*") <= 1:
        return "medium"
    return "hard"


def _stable_score(difficulty: str, numbers: tuple[int, ...], target: int, solution: str) -> int:
    return stable_task_score(DEFAULT_SEED, difficulty, numbers, target, solution)


def stable_task_score(
    seed: int, difficulty: str, numbers: tuple[int, ...], target: int, solution: str
) -> int:
    text = f"{seed}:{difficulty}:{numbers}:{target}:{solution}"
    score = 0
    for char in text:
        score = (score * 131 + ord(char)) % 1_000_003
    return score


def _solutions_for_numbers(numbers: tuple[int, ...]) -> dict[int, str]:
    solutions: dict[int, str] = {}
    for ordered_numbers in set(permutations(numbers)):
        if len(ordered_numbers) == 2:
            a, b = ordered_numbers
            for op in DEFAULT_OPERATORS:
                value = _apply(Fraction(a), Fraction(b), op)
                if value.denominator == 1:
                    solutions.setdefault(int(value), f"{a} {op} {b}")
        else:
            a, b, c = ordered_numbers
            for left_op, right_op in product(DEFAULT_OPERATORS, repeat=2):
                left_value = _apply(
                    _apply(Fraction(a), Fraction(b), left_op),
                    Fraction(c),
                    right_op,
                )
                if left_value.denominator == 1:
                    solutions.setdefault(int(left_value), f"({a} {left_op} {b}) {right_op} {c}")
                right_value = _apply(
                    Fraction(a),
                    _apply(Fraction(b), Fraction(c), right_op),
                    left_op,
                )
                if right_value.denominator == 1:
                    solutions.setdefault(int(right_value), f"{a} {left_op} ({b} {right_op} {c})")
    return solutions


def _apply(left: Fraction, right: Fraction, operator: str) -> Fraction:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    raise CountdownLiteError(f"unsupported operator: {operator}")


class _ExpressionParser:
    def __init__(self, text: str) -> None:
        self._text = text
        self._tokens = self._tokenize(text)
        self._index = 0

    def parse(self) -> tuple[Fraction, tuple[int, ...]]:
        value, used_numbers = self._parse_expression()
        if self._peek() is not None:
            raise CountdownLiteError(f"unexpected token: {self._peek()}")
        return value, tuple(used_numbers)

    def _parse_expression(self) -> tuple[Fraction, list[int]]:
        value, used_numbers = self._parse_term()
        while self._peek() in {"+", "-"}:
            operator = self._consume()
            right_value, right_numbers = self._parse_term()
            value = _apply(value, right_value, operator)
            used_numbers.extend(right_numbers)
        return value, used_numbers

    def _parse_term(self) -> tuple[Fraction, list[int]]:
        value, used_numbers = self._parse_factor()
        while self._peek() == "*":
            self._consume()
            right_value, right_numbers = self._parse_factor()
            value *= right_value
            used_numbers.extend(right_numbers)
        return value, used_numbers

    def _parse_factor(self) -> tuple[Fraction, list[int]]:
        token = self._peek()
        if token is None:
            raise CountdownLiteError("incomplete expression")
        if token.isdigit():
            self._consume()
            return Fraction(int(token)), [int(token)]
        if token == "(":
            self._consume()
            value, used_numbers = self._parse_expression()
            if self._peek() != ")":
                raise CountdownLiteError("missing closing parenthesis")
            self._consume()
            return value, used_numbers
        raise CountdownLiteError(f"unexpected token: {token}")

    def _peek(self) -> str | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _consume(self) -> str:
        token = self._tokens[self._index]
        self._index += 1
        return token

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        tokens: list[str] = []
        index = 0
        while index < len(text):
            match = _TOKEN_PATTERN.match(text, index)
            if match is None:
                raise CountdownLiteError("expression contains unsupported characters")
            tokens.append(match.group(1))
            index = match.end()
        if not tokens:
            raise CountdownLiteError("empty expression")
        return tuple(tokens)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
