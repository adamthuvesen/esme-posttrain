from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.sft.data_types import (
    IGNORE_INDEX as IGNORE_INDEX,
)
from esme_posttrain.sft.data_types import (
    ChatTurn as ChatTurn,
)
from esme_posttrain.sft.data_types import (
    DataError,
    DatasetSource,
    MultiTurnExample,
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


def _render_chat_segments(example: MultiTurnExample) -> list[tuple[str, bool]]:
    """Render a conversation into (text, supervised) segments.

    Every assistant turn's content plus its trailing `<eos>` is supervised;
    system/user turns and role markers are masked.
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
) -> TokenizedExample:
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
            labels.extend(IGNORE_INDEX for _ in token_ids)
            if not saw_assistant:
                prompt_tokens += len(token_ids)

    if not saw_assistant:
        raise DataError(
            f"{example.source}:{example.row_id} produced no supervised assistant tokens"
        )
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


def iter_rows(source: DatasetSource, *, allow_remote_download: bool) -> Iterator[tuple[str, Any]]:
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
