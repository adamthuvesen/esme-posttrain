"""Baseline evaluation for Countdown-Lite using a local Esme chat bundle."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import ValidationError

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.evals.records import (
    CountdownEvalResumeLine,
    CountdownEvalSummary,
    CountdownSampleScore,
    CountdownTaskResult,
    CountdownTaskRow,
    EvalRecordError,
    dump_record,
)
from esme_posttrain.rl.countdown_lite import (
    load_countdown_lite_rows,
    render_chat_prompt,
    verify_countdown_lite_expression,
)

BaselineProgressCallback = Callable[[str, dict[str, Any]], None]

PASS_AT_KS = (1, 8, 32)


def emitted_pass_at_keys(samples_per_task: int) -> tuple[str, ...]:
    """pass@k labels this run can honestly report: only k <= samples_per_task."""
    return tuple(f"pass@{k}" for k in PASS_AT_KS if k <= samples_per_task)


@dataclass(frozen=True)
class CountdownBaselineRequest:
    manifest_path: Path
    bundle_path: Path
    output_dir: Path
    split: str = "eval"
    samples_per_task: int = 32
    max_tasks: int | None = None
    max_new_tokens: int = 4
    seed: int = 214
    device: str = "cpu"
    progress_label: str = "eval"
    progress_callback: BaselineProgressCallback | None = None
    progress_interval_tasks: int = 1
    progress_interval_samples: int = 32
    sample_batch_size: int | None = None
    wall_timeout_seconds: float | None = None
    no_progress_timeout_seconds: float | None = None
    resume_from_partial: bool = False
    eval_profile: str | None = None
    config_hash: str | None = None
    model_id: str | None = None
    time_source: Callable[[], float] = time.perf_counter


class CountdownBaselineProgressError(RuntimeError):
    pass


def run_countdown_lite_baseline(request: CountdownBaselineRequest) -> dict[str, object]:
    if request.samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    if request.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if request.progress_interval_tasks <= 0:
        raise ValueError("progress_interval_tasks must be positive")
    if request.progress_interval_samples <= 0:
        raise ValueError("progress_interval_samples must be positive")
    if request.sample_batch_size is not None and request.sample_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive when set")
    if request.wall_timeout_seconds is not None and request.wall_timeout_seconds <= 0:
        raise ValueError("wall_timeout_seconds must be positive when set")
    if request.no_progress_timeout_seconds is not None and request.no_progress_timeout_seconds <= 0:
        raise ValueError("no_progress_timeout_seconds must be positive when set")

    rows = _validated_task_rows(
        load_countdown_lite_rows(request.manifest_path, split=request.split)
    )
    if request.max_tasks is not None:
        rows = rows[: request.max_tasks]
    if not rows:
        raise ValueError(f"no Countdown-Lite rows found for split: {request.split}")
    sample_batch_size = request.sample_batch_size or request.samples_per_task
    total_samples = len(rows) * request.samples_per_task
    progress = _BaselineProgress(
        request,
        total_tasks=len(rows),
        total_samples=total_samples,
        sample_batch_size=sample_batch_size,
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = request.output_dir / "baseline-partial.jsonl"
    all_results = (
        _load_partial_results(partial_path, request) if request.resume_from_partial else []
    )
    completed_task_ids = {str(item["task_id"]) for item in all_results}
    samples_completed = len(all_results) * request.samples_per_task
    progress.emit(
        "generation_start",
        tasks_completed=len(all_results),
        samples_completed=samples_completed,
        resumed_tasks=len(all_results),
    )

    target_device = torch.device(request.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Countdown-Lite baseline requested CUDA, but CUDA is not available")
    loaded = load_dense_backbone_bundle(request.bundle_path, map_location=target_device)
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = next(model.parameters()).device
    eos_token_id = tokenizer.token_to_id("<eos>")

    torch.manual_seed(request.seed)
    for task_index, row in enumerate(rows):
        task_id = row.task_id
        if task_id in completed_task_ids:
            progress.maybe_emit(
                tasks_completed=task_index + 1,
                samples_completed=samples_completed,
                task_id=task_id,
            )
            continue
        progress.check_timeout(
            "task_start",
            tasks_completed=task_index,
            samples_completed=samples_completed,
            task_id=task_id,
        )
        prompt = render_chat_prompt(row.prompt)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        if not prompt_ids:
            raise ValueError(f"task {row.task_id} produced an empty prompt")
        samples = []
        greedy_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        torch.manual_seed(request.seed + task_index * 10_000)
        samples.extend(
            _decode_batch(
                model=model,
                tokenizer=tokenizer,
                input_tensor=greedy_input,
                prompt_tokens=len(prompt_ids),
                max_new_tokens=request.max_new_tokens,
                temperature=0.0,
                eos_token_id=eos_token_id,
                row=row,
            )
        )
        samples_completed += 1
        progress.maybe_emit(
            tasks_completed=task_index,
            samples_completed=samples_completed,
            task_id=task_id,
        )
        stochastic_count = request.samples_per_task - 1
        stochastic_done = 0
        torch.manual_seed(request.seed + task_index * 10_000 + 1)
        while stochastic_done < stochastic_count:
            progress.check_timeout(
                "sample_batch_start",
                tasks_completed=task_index,
                samples_completed=samples_completed,
                task_id=task_id,
            )
            batch_count = min(sample_batch_size, stochastic_count - stochastic_done)
            stochastic_input = torch.tensor(
                [prompt_ids] * batch_count,
                dtype=torch.long,
                device=device,
            )
            samples.extend(
                _decode_batch(
                    model=model,
                    tokenizer=tokenizer,
                    input_tensor=stochastic_input,
                    prompt_tokens=len(prompt_ids),
                    max_new_tokens=request.max_new_tokens,
                    temperature=0.8,
                    eos_token_id=eos_token_id,
                    row=row,
                )
            )
            stochastic_done += batch_count
            samples_completed += batch_count
            progress.maybe_emit(
                tasks_completed=task_index,
                samples_completed=samples_completed,
                task_id=task_id,
            )
        task_result = _task_result(row, samples)
        all_results.append(dump_record(task_result))
        _append_partial_result(
            partial_path,
            task_result,
            request,
            task_index=task_index,
            task_count=len(rows),
            samples_completed=samples_completed,
            total_samples=total_samples,
        )
        progress.maybe_emit(
            tasks_completed=task_index + 1,
            samples_completed=samples_completed,
            task_id=task_id,
        )

    progress.check_timeout(
        "generation_finish",
        tasks_completed=len(rows),
        samples_completed=samples_completed,
        task_id=rows[-1].task_id,
    )
    report = _summarize(request, rows, all_results)
    json_path = request.output_dir / "baseline-report.json"
    markdown_path = request.output_dir / "baseline-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    progress.emit(
        "generation_complete",
        tasks_completed=len(rows),
        samples_completed=samples_completed,
    )
    return {
        **report,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "partial_path": str(partial_path),
    }


def _validated_task_rows(rows: tuple[dict[str, Any], ...]) -> list[CountdownTaskRow]:
    validated: list[CountdownTaskRow] = []
    seen_task_ids: set[str] = set()
    for index, row in enumerate(rows):
        try:
            task_row = CountdownTaskRow.model_validate(row)
        except ValidationError as error:
            raise EvalRecordError(f"malformed Countdown-Lite task row {index}: {error}") from error
        if task_row.task_id in seen_task_ids:
            raise EvalRecordError(f"duplicate Countdown-Lite task_id: {task_row.task_id}")
        seen_task_ids.add(task_row.task_id)
        validated.append(task_row)
    return validated


def _load_partial_results(path: Path, request: CountdownBaselineRequest) -> list[dict[str, object]]:
    if not path.exists():
        return []
    results: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            resume_line = CountdownEvalResumeLine.model_validate_json(line)
        except ValidationError as error:
            raise CountdownBaselineProgressError(
                f"cannot resume from malformed partial eval line {line_number}: {error}"
            ) from error
        if not _partial_matches_request(resume_line, request):
            raise CountdownBaselineProgressError(
                "cannot resume partial eval: profile/config/model/sample metadata mismatch"
            )
        recorded_samples = len(resume_line.task_result.samples)
        if recorded_samples != request.samples_per_task:
            raise CountdownBaselineProgressError(
                f"cannot resume partial eval: task {resume_line.task_id} recorded "
                f"{recorded_samples} samples, budget is {request.samples_per_task}"
            )
        results.append(dump_record(resume_line.task_result))
    return results


def _partial_matches_request(
    resume_line: CountdownEvalResumeLine, request: CountdownBaselineRequest
) -> bool:
    return (
        resume_line.phase == request.progress_label
        and resume_line.eval_profile == request.eval_profile
        and resume_line.config_hash == request.config_hash
        and resume_line.model_id == request.model_id
        and resume_line.samples_per_task == request.samples_per_task
        and resume_line.max_new_tokens == request.max_new_tokens
        and resume_line.split == request.split
    )


def _append_partial_result(
    path: Path,
    task_result: CountdownTaskResult,
    request: CountdownBaselineRequest,
    *,
    task_index: int,
    task_count: int,
    samples_completed: int,
    total_samples: int,
) -> None:
    resume_line = CountdownEvalResumeLine(
        schema_version=1,
        event="countdown_lite_eval_task_complete",
        phase=request.progress_label,
        eval_profile=request.eval_profile,
        config_hash=request.config_hash,
        model_id=request.model_id,
        split=request.split,
        task_index=task_index,
        task_start=task_index,
        task_end=task_index + 1,
        tasks_completed=task_index + 1,
        tasks_total=task_count,
        sample_start=samples_completed - request.samples_per_task,
        sample_end=samples_completed,
        samples_completed=samples_completed,
        samples_total=total_samples,
        samples_per_task=request.samples_per_task,
        max_new_tokens=request.max_new_tokens,
        task_id=task_result.task_id,
        task_result=task_result,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dump_record(resume_line), sort_keys=True) + "\n")


class _BaselineProgress:
    def __init__(
        self,
        request: CountdownBaselineRequest,
        *,
        total_tasks: int,
        total_samples: int,
        sample_batch_size: int,
    ) -> None:
        self._request = request
        self._total_tasks = total_tasks
        self._total_samples = total_samples
        self._sample_batch_size = sample_batch_size
        self._started = request.time_source()
        self._last_progress = self._started
        self._last_task_milestone = 0
        self._last_sample_milestone = 0
        self._last_seen_samples = 0

    def maybe_emit(
        self,
        *,
        tasks_completed: int,
        samples_completed: int,
        task_id: str,
    ) -> None:
        if samples_completed > self._last_seen_samples:
            self._last_seen_samples = samples_completed
            self._last_progress = self._request.time_source()
        task_delta = tasks_completed - self._last_task_milestone
        sample_delta = samples_completed - self._last_sample_milestone
        if (
            tasks_completed >= self._total_tasks
            or samples_completed >= self._total_samples
            or task_delta >= self._request.progress_interval_tasks
            or sample_delta >= self._request.progress_interval_samples
        ):
            self.emit(
                "generation_progress",
                tasks_completed=tasks_completed,
                samples_completed=samples_completed,
                task_id=task_id,
            )
            self._last_task_milestone = tasks_completed
            self._last_sample_milestone = samples_completed

    def check_timeout(
        self,
        stage: str,
        *,
        tasks_completed: int,
        samples_completed: int,
        task_id: str,
    ) -> None:
        elapsed = self._elapsed()
        if (
            self._request.wall_timeout_seconds is not None
            and elapsed > self._request.wall_timeout_seconds
        ):
            self.emit(
                "generation_timeout",
                tasks_completed=tasks_completed,
                samples_completed=samples_completed,
                task_id=task_id,
                timeout_kind="wall",
                timeout_stage=stage,
                timeout_seconds=self._request.wall_timeout_seconds,
            )
            raise CountdownBaselineProgressError(
                f"{self._request.progress_label} wall timeout after {elapsed:.2f}s "
                f"at {tasks_completed}/{self._total_tasks} tasks and "
                f"{samples_completed}/{self._total_samples} samples"
            )
        idle = self._request.time_source() - self._last_progress
        if (
            self._request.no_progress_timeout_seconds is not None
            and idle > self._request.no_progress_timeout_seconds
        ):
            self.emit(
                "generation_timeout",
                tasks_completed=tasks_completed,
                samples_completed=samples_completed,
                task_id=task_id,
                timeout_kind="no_progress",
                timeout_stage=stage,
                timeout_seconds=self._request.no_progress_timeout_seconds,
            )
            raise CountdownBaselineProgressError(
                f"{self._request.progress_label} made no progress for {idle:.2f}s "
                f"at {tasks_completed}/{self._total_tasks} tasks and "
                f"{samples_completed}/{self._total_samples} samples"
            )

    def emit(self, stage: str, **fields: Any) -> None:
        callback = self._request.progress_callback
        if callback is None:
            return
        payload = {
            "phase": self._request.progress_label,
            "tasks_total": self._total_tasks,
            "samples_per_task": self._request.samples_per_task,
            "samples_total": self._total_samples,
            "sample_batch_size": self._sample_batch_size,
            "elapsed_seconds": round(self._elapsed(), 3),
            **fields,
        }
        callback(f"{self._request.progress_label}_{stage}", payload)

    def _elapsed(self) -> float:
        return self._request.time_source() - self._started


def _decode_batch(
    *,
    model: Any,
    tokenizer: Any,
    input_tensor: torch.Tensor,
    prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int | None,
    row: CountdownTaskRow,
) -> list[CountdownSampleScore]:
    generated = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    samples: list[CountdownSampleScore] = []
    for generated_row in generated.detach().cpu().tolist():
        generated_ids = _truncate_at_eos(generated_row[prompt_tokens:], eos_token_id)
        output = tokenizer.decode(generated_ids, skip_special_tokens=False)
        verification = verify_countdown_lite_expression(
            output,
            numbers=row.numbers,
            target=row.target,
        )
        samples.append(
            CountdownSampleScore(
                output=output,
                extracted_expression=verification.expression,
                is_well_formed=verification.is_well_formed,
                is_valid_expression=verification.is_valid_expression,
                is_exact_solve=verification.is_exact_solve,
                value=verification.value,
                reason=verification.reason,
            )
        )
    return samples


def _truncate_at_eos(token_ids: list[int], eos_token_id: int | None) -> list[int]:
    if eos_token_id is None:
        return token_ids
    try:
        eos_index = token_ids.index(eos_token_id)
    except ValueError:
        return token_ids
    return token_ids[:eos_index]


def _task_result(row: CountdownTaskRow, samples: list[CountdownSampleScore]) -> CountdownTaskResult:
    pass_at = {}
    for k in PASS_AT_KS:
        if k > len(samples):
            continue
        prefix = samples[:k]
        pass_at[f"pass_at_{k}"] = any(sample.is_exact_solve for sample in prefix)
    return CountdownTaskResult(
        task_id=row.task_id,
        split=row.split,
        difficulty=row.difficulty,
        numbers=row.numbers,
        target=row.target,
        solution=row.solution,
        **pass_at,
        valid_samples=sum(sample.is_valid_expression for sample in samples),
        exact_samples=sum(sample.is_exact_solve for sample in samples),
        samples=tuple(samples),
    )


def _summarize(
    request: CountdownBaselineRequest,
    rows: list[CountdownTaskRow],
    all_results: list[dict[str, object]],
) -> dict[str, object]:
    for result in all_results:
        recorded_samples = len(cast(list[object], result["samples"]))
        if recorded_samples != request.samples_per_task:
            raise EvalRecordError(
                f"incomplete sample budget: task {result['task_id']} recorded "
                f"{recorded_samples} samples, budget is {request.samples_per_task}"
            )
    sample_count = len(all_results) * request.samples_per_task
    valid_count = sum(cast(int, result["valid_samples"]) for result in all_results)
    exact_count = sum(cast(int, result["exact_samples"]) for result in all_results)
    pass_at_keys = emitted_pass_at_keys(request.samples_per_task)
    summary = {
        "split": request.split,
        "task_count": len(rows),
        "samples_per_task": request.samples_per_task,
        "max_new_tokens": request.max_new_tokens,
        "seed": request.seed,
        "bundle_path": str(request.bundle_path.expanduser().resolve()),
        "manifest_path": str(request.manifest_path.expanduser().resolve()),
        **{key: _pass_rate(all_results, key) for key in pass_at_keys},
        "valid_expression_rate": valid_count / sample_count,
        "exact_solve_rate": exact_count / sample_count,
        "difficulty_breakdown": _difficulty_breakdown(
            all_results, request.samples_per_task, pass_at_keys
        ),
        "decision": _decision(
            all_results, valid_count, exact_count, samples_per_task=request.samples_per_task
        ),
        "tasks": all_results,
    }
    try:
        record = CountdownEvalSummary.model_validate(summary)
    except ValidationError as error:
        raise EvalRecordError(f"baseline summary failed record validation: {error}") from error
    return dump_record(record)


def _pass_rate(results: list[dict[str, object]], key: str) -> float:
    return sum(bool(result[key]) for result in results) / len(results)


def _difficulty_breakdown(
    results: list[dict[str, object]],
    samples_per_task: int,
    pass_at_keys: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["difficulty"])].append(result)

    breakdown: dict[str, dict[str, object]] = {}
    for difficulty, bucket in sorted(grouped.items()):
        sample_count = len(bucket) * samples_per_task
        breakdown[difficulty] = {
            "tasks": len(bucket),
            **{key: _pass_rate(bucket, key) for key in pass_at_keys},
            "valid_expression_rate": (
                sum(cast(int, result["valid_samples"]) for result in bucket) / sample_count
            ),
            "exact_solve_rate": sum(cast(int, result["exact_samples"]) for result in bucket)
            / sample_count,
        }
    return breakdown


def _decision(
    results: list[dict[str, object]],
    valid_count: int,
    exact_count: int,
    *,
    samples_per_task: int,
) -> str:
    easy_results = [result for result in results if result["difficulty"] == "easy"]
    top_pass_key = emitted_pass_at_keys(samples_per_task)[-1]
    easy_pass_top = _pass_rate(easy_results, top_pass_key) if easy_results else 0.0
    if exact_count > 0 and easy_pass_top > 0:
        return "GRPO-ready"
    if valid_count > 0:
        return "needs SFT/hint cold-start"
    return "blocked-with-evidence"


def _markdown_report(report: dict[str, object]) -> str:
    pass_at_keys = emitted_pass_at_keys(cast(int, report["samples_per_task"]))
    lines = [
        "# RLVR Countdown-Lite Baseline",
        "",
        f"- Bundle: `{report['bundle_path']}`",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Split: `{report['split']}`",
        f"- Tasks: `{report['task_count']}`",
        f"- Samples per task: `{report['samples_per_task']}`",
        *(f"- {key}: `{_percent(float(cast(int | float, report[key])))}`" for key in pass_at_keys),
        f"- valid-expression rate: "
        f"`{_percent(float(cast(int | float, report['valid_expression_rate'])))}`",
        f"- exact-solve rate: `{_percent(float(cast(int | float, report['exact_solve_rate'])))}`",
        f"- Decision: `{report['decision']}`",
        "",
        "## Difficulty Breakdown",
        "",
        "| Difficulty | Tasks | "
        + " | ".join(pass_at_keys)
        + " | Valid expressions | Exact solves |",
        "| --- | ---: | " + " | ".join("---:" for _key in pass_at_keys) + " | ---: | ---: |",
    ]
    breakdown = report["difficulty_breakdown"]
    if not isinstance(breakdown, dict):
        raise ValueError("difficulty_breakdown must be a dict")
    for difficulty, raw_bucket in sorted(breakdown.items()):
        bucket = dict(raw_bucket)
        pass_cells = " | ".join(_percent(float(bucket[key])) for key in pass_at_keys)
        lines.append(
            "| "
            f"{difficulty} | {bucket['tasks']} | {pass_cells} | "
            f"{_percent(float(bucket['valid_expression_rate']))} | "
            f"{_percent(float(bucket['exact_solve_rate']))} |"
        )
    lines.extend(
        [
            "",
            "## Spend",
            "",
            "No Modal, GPU, paid API, remote dataset download, or full training was started.",
            "This baseline used the local `Esme-214M-Chat` bundle on CPU.",
            "",
        ]
    )
    return "\n".join(lines)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"
