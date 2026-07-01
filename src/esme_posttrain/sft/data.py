from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.sft.adapters import AdapterError, iter_adapter_examples
from esme_posttrain.sft.data_types import (
    IGNORE_INDEX as IGNORE_INDEX,
)
from esme_posttrain.sft.data_types import (
    ChatTurn as ChatTurn,
)
from esme_posttrain.sft.data_types import (
    DataError,
    DatasetBuildResult,
    DatasetSource,
    FormattedExample,
    LossSemantics,
    MultiTurnExample,
    SingleTurnExample,
    SourceSurvivorCounts,
    TokenizedExample,
)


def sequence_efficiency_report(
    examples: tuple[TokenizedExample, ...],
    *,
    max_sequence_tokens: int,
    micro_batch_size: int,
    sequence_packing: bool,
    pad_to_multiple_of: int | None,
    no_packing_rationale: str,
) -> dict[str, Any]:
    if not examples:
        raise DataError("sequence efficiency requires at least one example")
    if max_sequence_tokens <= 0:
        raise DataError("max_sequence_tokens must be positive")
    if micro_batch_size <= 0:
        raise DataError("micro_batch_size must be positive")
    if sequence_packing:
        raise DataError("sequence_packing=true is not implemented for this SFT recipe")
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise DataError("pad_to_multiple_of must be positive when set")
    unpadded_tokens = sum(len(example.input_ids) for example in examples)
    batch_padded_tokens = 0
    for start in range(0, len(examples), micro_batch_size):
        batch = examples[start : start + micro_batch_size]
        padded_length = max(len(example.input_ids) for example in batch)
        if pad_to_multiple_of is not None:
            padded_length = int(math.ceil(padded_length / pad_to_multiple_of) * pad_to_multiple_of)
        batch_padded_tokens += padded_length * len(batch)
    max_sequence_slots = len(examples) * max_sequence_tokens
    return {
        "sequence_packing": False,
        "no_packing_rationale": no_packing_rationale,
        "pad_to_multiple_of": pad_to_multiple_of,
        "examples": len(examples),
        "unpadded_tokens": unpadded_tokens,
        "batch_padded_tokens": batch_padded_tokens,
        "padding_tokens": batch_padded_tokens - unpadded_tokens,
        "padding_efficiency": unpadded_tokens / max(1, batch_padded_tokens),
        "max_sequence_slot_efficiency": unpadded_tokens / max(1, max_sequence_slots),
    }


def format_single_turn(example: SingleTurnExample) -> FormattedExample:
    instruction = _clean_text(example.instruction)
    if example.constraints:
        constraints = "\n".join(f"- {_clean_text(item)}" for item in example.constraints)
        instruction = f"{instruction}\n\nConstraints:\n{constraints}"
    response = _clean_text(example.response)
    if not instruction or not response:
        raise DataError("single-turn examples require non-empty instruction and response")
    return FormattedExample(
        prompt=f"user\n{instruction}\nassistant\n",
        response=response,
        source=example.source,
        row_id=example.row_id,
        source_revision=example.source_revision,
        source_dataset=example.source_dataset or example.source,
        prompt_chars=len(instruction),
        response_chars=len(response),
    )


def tokenize_single_turn(
    tokenizer: Tokenizer,
    example: SingleTurnExample,
    *,
    max_sequence_tokens: int,
    eos_token: str = "<eos>",
    loss_semantics: LossSemantics | None = None,
) -> TokenizedExample:
    loss_semantics = loss_semantics or LossSemantics()
    formatted = format_single_turn(example)
    prompt_ids = tuple(tokenizer.encode(formatted.prompt, add_special_tokens=False).ids)
    response_ids = list(tokenizer.encode(formatted.response, add_special_tokens=False).ids)
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids.append(int(eos_id))
    if not prompt_ids:
        raise DataError(f"{example.source}:{example.row_id} produced zero prompt tokens")
    if not response_ids:
        raise DataError(f"{example.source}:{example.row_id} produced zero response tokens")
    input_ids = prompt_ids + tuple(response_ids)
    if len(input_ids) > max_sequence_tokens:
        raise DataError(
            f"{example.source}:{example.row_id} has {len(input_ids)} tokens, "
            f"above max_sequence_tokens={max_sequence_tokens}"
        )
    labels = (loss_semantics.ignore_index,) * len(prompt_ids) + tuple(response_ids)
    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
        source=example.source,
        row_id=example.row_id,
        source_revision=example.source_revision,
        source_dataset=example.source_dataset or example.source,
        prompt_chars=formatted.prompt_chars,
        response_chars=formatted.response_chars,
    )


def measure_single_turn_lengths(
    tokenizer: Tokenizer,
    example: SingleTurnExample,
    *,
    max_sequence_tokens: int,
    eos_token: str = "<eos>",
) -> dict[str, Any]:
    formatted = format_single_turn(example)
    prompt_ids = tuple(tokenizer.encode(formatted.prompt, add_special_tokens=False).ids)
    response_ids = list(tokenizer.encode(formatted.response, add_special_tokens=False).ids)
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids.append(int(eos_id))
    total_tokens = len(prompt_ids) + len(response_ids)
    prompt_truncated = len(prompt_ids) >= max_sequence_tokens
    assistant_target_truncated = not prompt_truncated and total_tokens > max_sequence_tokens
    return {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "input_tokens": total_tokens,
        "prompt_truncated": prompt_truncated,
        "assistant_target_truncated": assistant_target_truncated,
    }


def _render_chat_segments(example: MultiTurnExample) -> list[tuple[str, bool]]:
    """Render a conversation into (text, supervised) segments.

    The template mirrors the single-turn `user\\n...\\nassistant\\n` markers and
    extends them to N turns. Every assistant turn's content plus its trailing
    `<eos>` is a supervised segment; system/user turns and the role markers
    themselves are masked.
    """
    segments: list[tuple[str, bool]] = []
    for turn in example.turns:
        content = _clean_text(turn.content)
        if turn.role == "assistant":
            if not content:
                raise DataError(f"{example.source}:{example.row_id} has an empty assistant turn")
            segments.append(("assistant\n", False))
            segments.append((content, True))
        elif turn.role == "system":
            segments.append((f"system\n{content}\n", False))
        else:
            segments.append((f"user\n{content}\n", False))
    return segments


def tokenize_multi_turn(
    tokenizer: Tokenizer,
    example: MultiTurnExample,
    *,
    max_sequence_tokens: int,
    eos_token: str = "<eos>",
    loss_semantics: LossSemantics | None = None,
) -> TokenizedExample:
    loss_semantics = loss_semantics or LossSemantics()
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise DataError(f"tokenizer is missing the {eos_token} token required for chat turns")
    segments = _render_chat_segments(example)

    input_ids: list[int] = []
    labels: list[int] = []
    prompt_tokens = 0
    response_tokens = 0
    response_chars = 0
    saw_assistant = False
    for text, supervised in segments:
        token_ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
        if supervised:
            token_ids.append(int(eos_id))
            input_ids.extend(token_ids)
            labels.extend(token_ids)
            response_tokens += len(token_ids)
            response_chars += len(text)
            saw_assistant = True
        else:
            input_ids.extend(token_ids)
            labels.extend(loss_semantics.ignore_index for _ in token_ids)
            if not saw_assistant:
                prompt_tokens += len(token_ids)

    if not saw_assistant:
        raise DataError(
            f"{example.source}:{example.row_id} produced no supervised assistant tokens"
        )
    if response_tokens == 0:
        raise DataError(f"{example.source}:{example.row_id} produced zero response tokens")
    if len(input_ids) > max_sequence_tokens:
        raise DataError(
            f"{example.source}:{example.row_id} has {len(input_ids)} tokens, "
            f"above max_sequence_tokens={max_sequence_tokens}"
        )

    prompt_chars = sum(
        len(_clean_text(turn.content)) for turn in example.turns if turn.role != "assistant"
    )
    return TokenizedExample(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        source=example.source,
        row_id=example.row_id,
        source_revision=example.source_revision,
        source_dataset=example.source_dataset or example.source,
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        turns=len(example.turns),
        assistant_turns=example.assistant_turns,
    )


def measure_multi_turn_lengths(
    tokenizer: Tokenizer,
    example: MultiTurnExample,
    *,
    max_sequence_tokens: int,
    eos_token: str = "<eos>",
) -> dict[str, Any]:
    eos_id = tokenizer.token_to_id(eos_token)
    segments = _render_chat_segments(example)
    total_tokens = 0
    first_assistant_offset = 0
    saw_assistant = False
    for text, supervised in segments:
        count = len(tokenizer.encode(text, add_special_tokens=False).ids)
        if supervised and eos_id is not None:
            count += 1
        if not supervised and not saw_assistant:
            first_assistant_offset += count
        if supervised:
            saw_assistant = True
        total_tokens += count
    prompt_truncated = first_assistant_offset >= max_sequence_tokens
    assistant_target_truncated = not prompt_truncated and total_tokens > max_sequence_tokens
    return {
        "input_tokens": total_tokens,
        "prompt_truncated": prompt_truncated,
        "assistant_target_truncated": assistant_target_truncated,
    }


def build_training_mix(
    sources: tuple[DatasetSource, ...],
    tokenizer: Tokenizer,
    *,
    max_samples: int,
    max_tokens: int,
    max_sequence_tokens: int,
    allow_remote_download: bool = False,
) -> DatasetBuildResult:
    if max_samples <= 0:
        raise DataError("max_samples must be positive")
    if max_tokens <= 0:
        raise DataError("max_tokens must be positive")
    train_sources = tuple(source for source in sources if source.role == "train")
    if len(train_sources) != 2:
        raise DataError("SFT training mix must contain exactly two train sources")
    ratio_sum = sum(source.mix_ratio for source in train_sources)
    if abs(ratio_sum - 1.0) > 1e-9:
        raise DataError("train source mix ratios must sum to 1.0")

    target_counts = _target_counts(train_sources, max_samples)
    counts = {source.name: SourceSurvivorCounts() for source in train_sources}
    selected_by_source: dict[str, list[TokenizedExample]] = {
        source.name: [] for source in train_sources
    }
    selected_tokens = 0
    budget_stop: str | None = None

    for source in train_sources:
        target = target_counts[source.name]
        source_counts = counts[source.name]
        for tokenized in _iter_tokenized_single_turns(
            source,
            tokenizer,
            source_counts,
            max_sequence_tokens=max_sequence_tokens,
            allow_remote_download=allow_remote_download,
        ):
            if selected_tokens + len(tokenized.input_ids) > max_tokens:
                budget_stop = "token_cap"
                break
            selected_by_source[source.name].append(tokenized)
            source_counts.selected += 1
            selected_tokens += len(tokenized.input_ids)
            if source_counts.selected >= target:
                break
        if budget_stop:
            break

    mixed = _interleave_sources(train_sources, selected_by_source)
    if len(mixed) >= max_samples:
        budget_stop = budget_stop or "sample_cap"
        mixed = mixed[:max_samples]

    shortfalls = tuple(
        f"{source.name}: selected {counts[source.name].selected}/{target_counts[source.name]}"
        for source in train_sources
        if counts[source.name].selected < target_counts[source.name]
    )
    return DatasetBuildResult(
        examples=tuple(mixed),
        counts_by_source=counts,
        selected_tokens=sum(len(example.input_ids) for example in mixed),
        selected_supervised_tokens=sum(example.supervised_tokens for example in mixed),
        selected_prompt_tokens=sum(example.prompt_tokens for example in mixed),
        sample_cap=max_samples,
        token_cap=max_tokens,
        budget_stop=budget_stop,
        shortfalls=shortfalls,
    )


def build_matched_eval_sets(
    sources: tuple[DatasetSource, ...],
    tokenizer: Tokenizer,
    *,
    skip_selected_by_source: dict[str, int],
    max_samples_per_source: int,
    max_tokens_per_source: int,
    max_sequence_tokens: int,
    allow_remote_download: bool = False,
) -> dict[str, DatasetBuildResult]:
    if max_samples_per_source <= 0:
        raise DataError("max_samples_per_source must be positive")
    if max_tokens_per_source <= 0:
        raise DataError("max_tokens_per_source must be positive")
    train_sources = tuple(source for source in sources if source.role == "train")
    if len(train_sources) != 2:
        raise DataError("matched eval requires exactly two train sources")
    return {
        source.name: _build_source_holdout(
            source,
            tokenizer,
            skip_selected=int(skip_selected_by_source.get(source.name, 0)),
            max_samples=max_samples_per_source,
            max_tokens=max_tokens_per_source,
            max_sequence_tokens=max_sequence_tokens,
            allow_remote_download=allow_remote_download,
        )
        for source in train_sources
    }


def build_eval_set(
    source: DatasetSource,
    tokenizer: Tokenizer,
    *,
    max_samples: int,
    max_tokens: int,
    max_sequence_tokens: int,
    allow_remote_download: bool = False,
) -> DatasetBuildResult:
    if source.role != "eval":
        raise DataError("eval source must have role='eval'")
    if source.train_allowed:
        raise DataError(
            f"{source.name} is marked train_allowed=true but eval holdout must be false"
        )
    return _build_single_source_set(
        source,
        tokenizer,
        skip_selected=0,
        max_samples=max_samples,
        max_tokens=max_tokens,
        max_sequence_tokens=max_sequence_tokens,
        allow_remote_download=allow_remote_download,
    )


def _build_source_holdout(
    source: DatasetSource,
    tokenizer: Tokenizer,
    *,
    skip_selected: int,
    max_samples: int,
    max_tokens: int,
    max_sequence_tokens: int,
    allow_remote_download: bool,
) -> DatasetBuildResult:
    return _build_single_source_set(
        source,
        tokenizer,
        skip_selected=skip_selected,
        max_samples=max_samples,
        max_tokens=max_tokens,
        max_sequence_tokens=max_sequence_tokens,
        allow_remote_download=allow_remote_download,
    )


def _build_single_source_set(
    source: DatasetSource,
    tokenizer: Tokenizer,
    *,
    skip_selected: int,
    max_samples: int,
    max_tokens: int,
    max_sequence_tokens: int,
    allow_remote_download: bool,
) -> DatasetBuildResult:
    counts = {source.name: SourceSurvivorCounts()}
    selected: list[TokenizedExample] = []
    selected_tokens = 0
    budget_stop: str | None = None
    skipped_tokenized = 0
    source_counts = counts[source.name]
    for tokenized in _iter_tokenized_single_turns(
        source,
        tokenizer,
        source_counts,
        max_sequence_tokens=max_sequence_tokens,
        allow_remote_download=allow_remote_download,
    ):
        if skipped_tokenized < skip_selected:
            skipped_tokenized += 1
            continue
        if selected_tokens + len(tokenized.input_ids) > max_tokens:
            budget_stop = "token_cap"
            break
        selected.append(tokenized)
        selected_tokens += len(tokenized.input_ids)
        source_counts.selected += 1
        if len(selected) >= max_samples:
            budget_stop = "sample_cap"
            break

    return DatasetBuildResult(
        examples=tuple(selected),
        counts_by_source=counts,
        selected_tokens=selected_tokens,
        selected_supervised_tokens=sum(example.supervised_tokens for example in selected),
        selected_prompt_tokens=sum(example.prompt_tokens for example in selected),
        sample_cap=max_samples,
        token_cap=max_tokens,
        budget_stop=budget_stop,
        shortfalls=() if selected else (f"{source.name}: selected 0/{max_samples}",),
    )


def _iter_tokenized_single_turns(
    source: DatasetSource,
    tokenizer: Tokenizer,
    counts: SourceSurvivorCounts,
    *,
    max_sequence_tokens: int,
    allow_remote_download: bool,
) -> Iterator[TokenizedExample]:
    rows = _iter_rows(source, allow_remote_download=allow_remote_download)
    try:
        adapter_rows = iter_adapter_examples(source, rows)
    except AdapterError as error:
        raise DataError(str(error)) from error
    try:
        for row_id, adapter_example in adapter_rows:
            counts.rows_seen += 1
            if adapter_example is None:
                counts.rejected_non_single_turn += 1
                continue
            example = SingleTurnExample(
                instruction=adapter_example.instruction,
                response=adapter_example.response,
                source=source.name,
                row_id=row_id,
                source_revision=source.revision,
                source_dataset=source.source,
                constraints=adapter_example.constraints,
            )
            if (
                len(example.instruction) > source.max_prompt_chars
                or len(example.response) > source.max_response_chars
            ):
                counts.rejected_too_long_chars += 1
                continue
            counts.survivors += 1
            try:
                yield tokenize_single_turn(
                    tokenizer,
                    example,
                    max_sequence_tokens=max_sequence_tokens,
                )
            except DataError:
                _record_token_truncation(
                    counts,
                    tokenizer,
                    example,
                    max_sequence_tokens=max_sequence_tokens,
                )
                counts.rejected_too_long_tokens += 1
                continue
    except AdapterError as error:
        raise DataError(str(error)) from error


def _record_token_truncation(
    counts: SourceSurvivorCounts,
    tokenizer: Tokenizer,
    example: SingleTurnExample,
    *,
    max_sequence_tokens: int,
) -> None:
    try:
        lengths = measure_single_turn_lengths(
            tokenizer,
            example,
            max_sequence_tokens=max_sequence_tokens,
        )
    except DataError:
        return
    if lengths["prompt_truncated"]:
        counts.prompt_truncation_count += 1
    if lengths["assistant_target_truncated"]:
        counts.assistant_target_truncation_count += 1


def _target_counts(sources: tuple[DatasetSource, ...], max_samples: int) -> dict[str, int]:
    first = int(max_samples * sources[0].mix_ratio)
    return {sources[0].name: first, sources[1].name: max_samples - first}


def _interleave_sources(
    sources: tuple[DatasetSource, ...], selected_by_source: dict[str, list[TokenizedExample]]
) -> list[TokenizedExample]:
    if len(sources) == 2 and sources[0].mix_ratio == 0.8 and sources[1].mix_ratio == 0.2:
        return _interleave_80_20(sources, selected_by_source)
    mixed: list[TokenizedExample] = []
    positions = {source.name: 0 for source in sources}
    while True:
        emitted = False
        for source in sources:
            position = positions[source.name]
            rows = selected_by_source[source.name]
            if position < len(rows):
                mixed.append(rows[position])
                positions[source.name] += 1
                emitted = True
        if not emitted:
            return mixed


def _interleave_80_20(
    sources: tuple[DatasetSource, ...], selected_by_source: dict[str, list[TokenizedExample]]
) -> list[TokenizedExample]:
    primary, secondary = sources
    positions = {primary.name: 0, secondary.name: 0}
    mixed: list[TokenizedExample] = []
    while True:
        emitted = False
        for source, quota in ((primary, 4), (secondary, 1)):
            rows = selected_by_source[source.name]
            for _ in range(quota):
                position = positions[source.name]
                if position >= len(rows):
                    break
                mixed.append(rows[position])
                positions[source.name] += 1
                emitted = True
        if not emitted:
            return mixed


def _iter_rows(source: DatasetSource, *, allow_remote_download: bool) -> Iterator[tuple[str, Any]]:
    if source.path is not None:
        yield from _iter_jsonl(source.path)
        return
    if not allow_remote_download:
        raise DataError(f"remote dataset access is disabled for {source.source}")
    datasets = importlib.import_module("datasets")
    kwargs: dict[str, Any] = {
        "path": source.source,
        "split": source.split,
        "revision": source.revision,
        "streaming": True,
    }
    if source.subset is not None:
        kwargs["name"] = source.subset
    stream = datasets.load_dataset(**kwargs)
    for index, row in enumerate(stream):
        yield str(index), row


def _iter_jsonl(path: Path) -> Iterator[tuple[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise DataError(f"{path}:{line_number}: blank JSONL lines are not allowed")
            try:
                yield str(line_number), json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(f"{path}:{line_number}: malformed JSON: {error.msg}") from error


def _clean_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines()).strip()
