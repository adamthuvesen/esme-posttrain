"""Multi-turn SFT data path: full smol-smoltalk conversations with all-assistant masking.

This sits on top of the shared primitives in ``sft_data`` (DatasetSource,
SourceSurvivorCounts, DatasetBuildResult, TokenizedExample). It differs from the
single-turn Instruct path in three ways:

- conversations keep every turn and supervise every assistant turn,
- capacity-filtered subsets (function calling, hardest reasoning) are dropped,
- the train mix accepts one or two sources (smol-smoltalk alone, or with a small
  tulu-personas instruction slice), not exactly two.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer

from esme_posttrain.sft.adapters import (
    AdapterError,
    MultiTurnAdapterExample,
    iter_multi_turn_examples,
)
from esme_posttrain.sft.data import (
    ChatTurn,
    DataError,
    DatasetBuildResult,
    DatasetSource,
    MultiTurnExample,
    SourceSurvivorCounts,
    TokenizedExample,
    iter_rows,
    measure_multi_turn_lengths,
    tokenize_multi_turn,
)


def _to_multi_turn_example(
    source: DatasetSource, row_id: str, parsed: MultiTurnAdapterExample
) -> MultiTurnExample:
    turns = tuple(ChatTurn(role=turn.role, content=turn.content) for turn in parsed.turns)
    return MultiTurnExample(
        turns=turns,
        source=source.name,
        row_id=row_id,
        source_revision=source.revision,
        source_dataset=source.source,
    )


def _example_chars(parsed: MultiTurnAdapterExample) -> tuple[int, int]:
    prompt_chars = sum(len(turn.content) for turn in parsed.turns if turn.role != "assistant")
    response_chars = sum(len(turn.content) for turn in parsed.turns if turn.role == "assistant")
    return prompt_chars, response_chars


def _record_token_truncation(
    counts: SourceSurvivorCounts,
    tokenizer: Tokenizer,
    example: MultiTurnExample,
    *,
    max_sequence_tokens: int,
) -> None:
    try:
        lengths = measure_multi_turn_lengths(
            tokenizer, example, max_sequence_tokens=max_sequence_tokens
        )
    except DataError:
        return
    if lengths["prompt_truncated"]:
        counts.prompt_truncation_count += 1
    if lengths["assistant_target_truncated"]:
        counts.assistant_target_truncation_count += 1


def _select_multi_turn_rows(
    source: DatasetSource,
    tokenizer: Tokenizer,
    *,
    counts: SourceSurvivorCounts,
    target: int,
    max_tokens: int,
    max_sequence_tokens: int,
    skip_selected: int,
    allow_remote_download: bool,
) -> tuple[list[TokenizedExample], int, str | None]:
    selected: list[TokenizedExample] = []
    selected_tokens = 0
    skipped = 0
    budget_stop: str | None = None
    rows = iter_rows(source, allow_remote_download=allow_remote_download)
    try:
        for row_id, parsed in iter_multi_turn_examples(source, rows):
            counts.rows_seen += 1
            if parsed.rejection == "capacity_filtered":
                counts.rejected_capacity_filtered += 1
                continue
            if parsed.rejection == "unparsable" or not parsed.turns:
                counts.rejected_unparsable += 1
                continue
            prompt_chars, response_chars = _example_chars(parsed)
            if prompt_chars > source.max_prompt_chars or response_chars > source.max_response_chars:
                counts.rejected_too_long_chars += 1
                continue
            example = _to_multi_turn_example(source, row_id, parsed)
            try:
                tokenized = tokenize_multi_turn(
                    tokenizer, example, max_sequence_tokens=max_sequence_tokens
                )
            except DataError:
                _record_token_truncation(
                    counts, tokenizer, example, max_sequence_tokens=max_sequence_tokens
                )
                counts.rejected_too_long_tokens += 1
                continue
            counts.survivors += 1
            if skipped < skip_selected:
                skipped += 1
                continue
            if selected_tokens + len(tokenized.input_ids) > max_tokens:
                budget_stop = "token_cap"
                break
            selected.append(tokenized)
            selected_tokens += len(tokenized.input_ids)
            counts.selected += 1
            counts.record_selected_turns(
                turns=tokenized.turns, assistant_turns=tokenized.assistant_turns
            )
            if counts.selected >= target:
                break
    except AdapterError as error:
        raise DataError(str(error)) from error
    return selected, selected_tokens, budget_stop


def _target_counts(sources: tuple[DatasetSource, ...], max_samples: int) -> dict[str, int]:
    if len(sources) == 1:
        return {sources[0].name: max_samples}
    first = int(max_samples * sources[0].mix_ratio)
    return {sources[0].name: first, sources[1].name: max_samples - first}


def _interleave(
    sources: tuple[DatasetSource, ...], selected_by_source: dict[str, list[TokenizedExample]]
) -> list[TokenizedExample]:
    if len(sources) == 1:
        return list(selected_by_source[sources[0].name])
    primary, secondary = sources
    primary_quota = max(1, round(primary.mix_ratio / max(1e-9, secondary.mix_ratio)))
    positions = {primary.name: 0, secondary.name: 0}
    mixed: list[TokenizedExample] = []
    while True:
        emitted = False
        for source, quota in ((primary, primary_quota), (secondary, 1)):
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


@dataclass(frozen=True)
class TurnDistribution:
    single_turn: int
    multi_turn: int
    assistant_turns: int
    turn_histogram: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        total = self.single_turn + self.multi_turn
        return {
            "single_turn_examples": self.single_turn,
            "multi_turn_examples": self.multi_turn,
            "multi_turn_fraction": self.multi_turn / max(1, total),
            "total_assistant_turns": self.assistant_turns,
            "mean_assistant_turns": self.assistant_turns / max(1, total),
            "turn_count_histogram": self.turn_histogram,
        }


def turn_distribution(examples: tuple[TokenizedExample, ...]) -> TurnDistribution:
    single = sum(1 for example in examples if not example.is_multi_turn)
    multi = sum(1 for example in examples if example.is_multi_turn)
    assistant_turns = sum(example.assistant_turns for example in examples)
    histogram: dict[str, int] = {}
    for example in examples:
        key = str(example.turns)
        histogram[key] = histogram.get(key, 0) + 1
    return TurnDistribution(
        single_turn=single,
        multi_turn=multi,
        assistant_turns=assistant_turns,
        turn_histogram=dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    )


def build_multi_turn_mix(
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
    if not 1 <= len(train_sources) <= 2:
        raise DataError("multi-turn SFT mix must contain one or two train sources")
    if len(train_sources) == 2:
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
        rows, source_tokens, source_stop = _select_multi_turn_rows(
            source,
            tokenizer,
            counts=counts[source.name],
            target=target_counts[source.name],
            max_tokens=max(1, max_tokens - selected_tokens),
            max_sequence_tokens=max_sequence_tokens,
            skip_selected=0,
            allow_remote_download=allow_remote_download,
        )
        selected_by_source[source.name] = rows
        selected_tokens += source_tokens
        if source_stop:
            budget_stop = source_stop
            break

    mixed = _interleave(train_sources, selected_by_source)
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


def build_multi_turn_matched_eval_sets(
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
    if not 1 <= len(train_sources) <= 2:
        raise DataError("matched eval requires one or two train sources")
    reports: dict[str, DatasetBuildResult] = {}
    for source in train_sources:
        counts = SourceSurvivorCounts()
        rows, selected_tokens, budget_stop = _select_multi_turn_rows(
            source,
            tokenizer,
            counts=counts,
            target=max_samples_per_source,
            max_tokens=max_tokens_per_source,
            max_sequence_tokens=max_sequence_tokens,
            skip_selected=int(skip_selected_by_source.get(source.name, 0)),
            allow_remote_download=allow_remote_download,
        )
        if len(rows) >= max_samples_per_source:
            budget_stop = budget_stop or "sample_cap"
        reports[source.name] = DatasetBuildResult(
            examples=tuple(rows),
            counts_by_source={source.name: counts},
            selected_tokens=selected_tokens,
            selected_supervised_tokens=sum(example.supervised_tokens for example in rows),
            selected_prompt_tokens=sum(example.prompt_tokens for example in rows),
            sample_cap=max_samples_per_source,
            token_cap=max_tokens_per_source,
            budget_stop=budget_stop,
            shortfalls=() if rows else (f"{source.name}: selected 0/{max_samples_per_source}",),
        )
    return reports


def build_multi_turn_eval_set(
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
    counts = SourceSurvivorCounts()
    rows, selected_tokens, budget_stop = _select_multi_turn_rows(
        source,
        tokenizer,
        counts=counts,
        target=max_samples,
        max_tokens=max_tokens,
        max_sequence_tokens=max_sequence_tokens,
        skip_selected=0,
        allow_remote_download=allow_remote_download,
    )
    if len(rows) >= max_samples:
        budget_stop = budget_stop or "sample_cap"
    return DatasetBuildResult(
        examples=tuple(rows),
        counts_by_source={source.name: counts},
        selected_tokens=selected_tokens,
        selected_supervised_tokens=sum(example.supervised_tokens for example in rows),
        selected_prompt_tokens=sum(example.prompt_tokens for example in rows),
        sample_cap=max_samples,
        token_cap=max_tokens,
        budget_stop=budget_stop,
        shortfalls=() if rows else (f"{source.name}: selected 0/{max_samples}",),
    )
