"""Typed records for the evaluation contract.

These records pin the shapes the Countdown-Lite eval path reads and writes:
dataset task rows, per-sample verifier scores, per-task results, resume lines,
and the aggregate report. Every record forbids unknown fields so malformed or
mistyped data fails loudly instead of being dropped.

The records are named for Countdown-Lite on purpose. The shared shells for a
multi-task contract get extracted when the second verifier task lands, not
before (roadmap stop condition: no registry, no plugin loader, no generic
framework beyond what the real tasks need).

Serialized shapes are pinned by the golden fixtures under
``fixtures/outputs/countdown_lite_golden/``: every key recorded there must
keep its recorded value. New fields may only be added, never renamed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

COUNTDOWN_VERIFIER_NAME = "countdown_lite_exact_solve"
COUNTDOWN_VERIFIER_VERSION = 1

_PASS_AT_FIELDS = ("pass_at_1", "pass_at_8", "pass_at_32")


class EvalRecordError(ValueError):
    """A record failed validation; the payload is malformed or mistyped."""


class _EvalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())


class CountdownTaskRow(_EvalRecord):
    """One dataset row: task identity, split, difficulty, and prompt inputs."""

    task_id: str
    split: str
    difficulty: str
    prompt: str
    reward_name: str
    numbers: tuple[int, ...]
    target: int
    solution: str


class CountdownSampleScore(_EvalRecord):
    """One verified completion, scored on three separate axes.

    ``is_well_formed`` grades format (the output parsed as arithmetic),
    ``is_valid_expression`` grades validity (task constraints hold), and
    ``is_exact_solve`` grades task success (the value hits the target).
    """

    output: str
    extracted_expression: str | None
    is_well_formed: bool
    is_valid_expression: bool
    is_exact_solve: bool
    value: int | None
    reason: str


class CountdownTaskResult(_EvalRecord):
    """All scored samples for one task, with pass@k over the sample prefix."""

    task_id: str
    split: str
    difficulty: str
    numbers: tuple[int, ...]
    target: int
    solution: str
    pass_at_1: bool | None = Field(default=None, alias="pass@1")
    pass_at_8: bool | None = Field(default=None, alias="pass@8")
    pass_at_32: bool | None = Field(default=None, alias="pass@32")
    valid_samples: int
    exact_samples: int
    samples: tuple[CountdownSampleScore, ...]


class CountdownEvalResumeLine(_EvalRecord):
    """One line of the partial-eval log: completed ranges and resume identity.

    Resume matches on the identity fields (phase, eval profile, config hash,
    model id, split, and decoding settings); the range fields record which
    tasks and samples a resumed run may skip.
    """

    schema_version: int
    event: str
    phase: str
    eval_profile: str | None
    config_hash: str | None
    model_id: str | None
    split: str
    task_index: int
    task_start: int
    task_end: int
    tasks_completed: int
    tasks_total: int
    sample_start: int
    sample_end: int
    samples_completed: int
    samples_total: int
    samples_per_task: int
    max_new_tokens: int
    task_id: str
    task_result: CountdownTaskResult


class CountdownDifficultyAggregate(_EvalRecord):
    """Aggregate rates for one difficulty bucket."""

    tasks: int
    pass_at_1: float | None = Field(default=None, alias="pass@1")
    pass_at_8: float | None = Field(default=None, alias="pass@8")
    pass_at_32: float | None = Field(default=None, alias="pass@32")
    valid_expression_rate: float
    exact_solve_rate: float


class CountdownEvalSummary(_EvalRecord):
    """The aggregate baseline report, including every per-task result."""

    split: str
    task_count: int
    samples_per_task: int
    max_new_tokens: int
    seed: int
    bundle_path: str
    manifest_path: str
    verifier_name: str = COUNTDOWN_VERIFIER_NAME
    verifier_version: int = COUNTDOWN_VERIFIER_VERSION
    pass_at_1: float | None = Field(default=None, alias="pass@1")
    pass_at_8: float | None = Field(default=None, alias="pass@8")
    pass_at_32: float | None = Field(default=None, alias="pass@32")
    valid_expression_rate: float
    exact_solve_rate: float
    difficulty_breakdown: dict[str, CountdownDifficultyAggregate]
    decision: str
    tasks: tuple[CountdownTaskResult, ...]


def dump_record(record: _EvalRecord) -> dict[str, Any]:
    """Serialize a record to the pinned JSON shape.

    pass@k keys appear only when the run produced them (k <= samples per
    task); every other field is always present, including explicit nulls.
    JSON mode keeps the payload identical to what lands on disk (tuples
    become lists), so in-memory consumers and file readers see one shape.
    """
    payload = record.model_dump(mode="json", by_alias=True)
    return _drop_missing_pass_at(payload)


def _drop_missing_pass_at(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _drop_missing_pass_at(value)
            for key, value in payload.items()
            if not (key in ("pass@1", "pass@8", "pass@32") and value is None)
        }
    if isinstance(payload, list):
        return [_drop_missing_pass_at(item) for item in payload]
    return payload
