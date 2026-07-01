"""DPO preference-pair data path for the Esme-214M-Chat polish stage.

This reuses the repo's existing chat template and EOS conventions (the same
``user\\n...\\nassistant\\n`` markers and supervised ``<eos>`` that
``sft_data._render_chat_segments`` emits) so the DPO policy sees exactly the
format the SFT foundation was trained on. A preference pair shares one prompt and
has a chosen and a rejected completion; the prompt tokens are masked in both
completions so only the response tokens score in the DPO loss.

UltraFeedback-binarized (``HuggingFaceH4/ultrafeedback_binarized``) is the
off-policy/static source for v1. Rows arrive as ``chosen`` / ``rejected`` message
lists that share a prompt prefix; this module parses, light-filters, chat-templates,
and tokenizes them into :class:`TokenizedPreferencePair` objects.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.sft.data import (
    IGNORE_INDEX,
    ChatTurn,
    DataError,
    DatasetSource,
    _clean_text,
    _iter_rows,
)

# UltraFeedback-binarized has a handful of rows with an empty/whitespace chosen or
# rejected response; the strict per-pair validator rejects them, so the loader
# drops and counts them rather than aborting the whole run. But a *high* drop rate
# means a templating/parsing bug, not a few bad source rows -- so if the fraction
# of survivors lost to per-row validation exceeds this floor, raise loudly.
MAX_VALIDATION_DROP_FRACTION = 0.02


@dataclass(frozen=True)
class PreferencePair:
    """A parsed, untokenized preference pair sharing one prompt context."""

    prompt_turns: tuple[ChatTurn, ...]
    chosen: str
    rejected: str
    source: str
    row_id: str
    source_revision: str = "local"
    source_dataset: str = "local"

    def __post_init__(self) -> None:
        if not self.prompt_turns:
            raise DataError("preference pair requires at least one prompt turn")
        if self.prompt_turns[-1].role == "assistant":
            raise DataError("preference prompt must end on a user/system turn, not assistant")
        if not self.chosen.strip() or not self.rejected.strip():
            raise DataError("preference pair requires non-empty chosen and rejected responses")


@dataclass(frozen=True)
class TokenizedCompletion:
    """One side (chosen or rejected) of a tokenized preference pair.

    ``input_ids`` is prompt + response; ``labels`` masks every prompt token with
    ``IGNORE_INDEX`` so only response tokens (and the trailing ``<eos>``) score.
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    response_tokens: int

    @property
    def response_supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


@dataclass(frozen=True)
class TokenizedPreferencePair:
    prompt_ids: tuple[int, ...]
    chosen: TokenizedCompletion
    rejected: TokenizedCompletion
    source: str
    row_id: str
    source_revision: str = "local"
    source_dataset: str = "local"

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_dataset": self.source_dataset,
            "revision": self.source_revision,
            "row_id": self.row_id,
            "prompt_tokens": len(self.prompt_ids),
            "chosen_tokens": len(self.chosen.input_ids),
            "rejected_tokens": len(self.rejected.input_ids),
            "chosen_response_tokens": self.chosen.response_tokens,
            "rejected_response_tokens": self.rejected.response_tokens,
        }


@dataclass
class PreferenceSurvivorCounts:
    rows_seen: int = 0
    survivors: int = 0
    selected: int = 0
    # Identical/mismatched/malformed rows are all folded into rejected_unparsable;
    # the UltraFeedback parser returns None for every clean-pair violation.
    rejected_unparsable: int = 0
    rejected_too_long_chars: int = 0
    rejected_too_long_tokens: int = 0
    # Rows that hit a per-row DataError (empty/whitespace chosen or rejected, zero
    # prompt/response tokens, etc.) -- dropped and counted by reason, never fatal.
    dropped_validation: int = 0
    prompt_truncation_count: int = 0
    response_truncation_count: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def record_drop(self, reason: str) -> None:
        self.dropped_validation += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_seen": self.rows_seen,
            "survivors": self.survivors,
            "selected": self.selected,
            "rejected_unparsable": self.rejected_unparsable,
            "rejected_too_long_chars": self.rejected_too_long_chars,
            "rejected_too_long_tokens": self.rejected_too_long_tokens,
            "dropped_validation": self.dropped_validation,
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
            "prompt_truncation_count": self.prompt_truncation_count,
            "response_truncation_count": self.response_truncation_count,
        }


@dataclass(frozen=True)
class PreferenceBuildResult:
    pairs: tuple[TokenizedPreferencePair, ...]
    counts: PreferenceSurvivorCounts
    selected_tokens: int
    selected_chosen_response_tokens: int
    selected_rejected_response_tokens: int
    sample_cap: int
    token_cap: int
    budget_stop: str | None = None
    shortfalls: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_pairs": len(self.pairs),
            "selected_tokens": self.selected_tokens,
            "selected_chosen_response_tokens": self.selected_chosen_response_tokens,
            "selected_rejected_response_tokens": self.selected_rejected_response_tokens,
            "sample_cap": self.sample_cap,
            "token_cap": self.token_cap,
            "budget_stop": self.budget_stop,
            "shortfalls": list(self.shortfalls),
            "counts": self.counts.to_dict(),
            "selected_row_manifest": [pair.manifest_entry() for pair in self.pairs],
        }


def _render_prompt(prompt_turns: tuple[ChatTurn, ...]) -> str:
    """Render the prompt context, mirroring ``sft_data._render_chat_segments``.

    The final ``assistant\\n`` marker that opens the response is included here so
    both completions tokenize against an identical, supervised-aligned prompt.
    """
    parts: list[str] = []
    for turn in prompt_turns:
        content = _clean_text(turn.content)
        if turn.role == "system":
            parts.append(f"system\n{content}\n")
        elif turn.role == "user":
            parts.append(f"user\n{content}\n")
        else:  # pragma: no cover - guarded by PreferencePair.__post_init__
            raise DataError("prompt turns must be system or user")
    parts.append("assistant\n")
    return "".join(parts)


def tokenize_preference_pair(
    tokenizer: Tokenizer,
    pair: PreferencePair,
    *,
    max_length: int,
    max_prompt_length: int,
    eos_token: str = "<eos>",
) -> TokenizedPreferencePair:
    """Tokenize a preference pair with prompt masking on both completions.

    Mirrors ``tokenize_multi_turn``: the assistant response content plus a single
    trailing ``<eos>`` are the only supervised tokens; the shared prompt span is
    masked with ``IGNORE_INDEX`` in both the chosen and rejected ``labels``.
    """
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise DataError(f"tokenizer is missing the {eos_token} token required for chat turns")
    prompt_text = _render_prompt(pair.prompt_turns)
    prompt_ids = tuple(tokenizer.encode(prompt_text, add_special_tokens=False).ids)
    if not prompt_ids:
        raise DataError(f"{pair.source}:{pair.row_id} produced zero prompt tokens")
    if len(prompt_ids) > max_prompt_length:
        raise DataError(
            f"{pair.source}:{pair.row_id} prompt has {len(prompt_ids)} tokens, "
            f"above max_prompt_length={max_prompt_length}"
        )
    chosen = _tokenize_completion(
        tokenizer, prompt_ids, pair.chosen, eos_id=eos_id, max_length=max_length, pair=pair
    )
    rejected = _tokenize_completion(
        tokenizer, prompt_ids, pair.rejected, eos_id=eos_id, max_length=max_length, pair=pair
    )
    return TokenizedPreferencePair(
        prompt_ids=prompt_ids,
        chosen=chosen,
        rejected=rejected,
        source=pair.source,
        row_id=pair.row_id,
        source_revision=pair.source_revision,
        source_dataset=pair.source_dataset,
    )


def _tokenize_completion(
    tokenizer: Tokenizer,
    prompt_ids: tuple[int, ...],
    response: str,
    *,
    eos_id: int,
    max_length: int,
    pair: PreferencePair,
) -> TokenizedCompletion:
    response_ids = list(tokenizer.encode(_clean_text(response), add_special_tokens=False).ids)
    if not response_ids:
        raise DataError(f"{pair.source}:{pair.row_id} produced zero response tokens")
    if response_ids[-1] != eos_id:
        response_ids.append(int(eos_id))
    input_ids = prompt_ids + tuple(response_ids)
    if len(input_ids) > max_length:
        raise DataError(
            f"{pair.source}:{pair.row_id} completion has {len(input_ids)} tokens, "
            f"above max_length={max_length}"
        )
    labels = (IGNORE_INDEX,) * len(prompt_ids) + tuple(response_ids)
    return TokenizedCompletion(
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
    )


def measure_preference_lengths(
    tokenizer: Tokenizer,
    pair: PreferencePair,
    *,
    max_length: int,
    max_prompt_length: int,
    eos_token: str = "<eos>",
) -> dict[str, Any]:
    eos_id = tokenizer.token_to_id(eos_token)
    prompt_ids = tokenizer.encode(_render_prompt(pair.prompt_turns), add_special_tokens=False).ids
    prompt_len = len(prompt_ids)

    def _response_len(text: str) -> int:
        ids = list(tokenizer.encode(_clean_text(text), add_special_tokens=False).ids)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            ids.append(int(eos_id))
        return len(ids)

    chosen_total = prompt_len + _response_len(pair.chosen)
    rejected_total = prompt_len + _response_len(pair.rejected)
    prompt_truncated = prompt_len > max_prompt_length
    response_truncated = not prompt_truncated and max(chosen_total, rejected_total) > max_length
    return {
        "prompt_tokens": prompt_len,
        "chosen_tokens": chosen_total,
        "rejected_tokens": rejected_total,
        "prompt_truncated": prompt_truncated,
        "response_truncated": response_truncated,
    }


# --- UltraFeedback parsing ----------------------------------------------------


def parse_ultrafeedback_row(source: DatasetSource, row_id: str, row: Any) -> PreferencePair | None:
    """Parse one UltraFeedback-binarized row into a :class:`PreferencePair`.

    The dataset stores ``chosen`` and ``rejected`` as message lists that repeat a
    shared prompt and append one assistant turn. A ``prompt`` string field is also
    present. Rows whose chosen/rejected are not a clean shared-prompt single-turn
    pair, or whose responses are identical, return ``None`` (a parse rejection).
    """
    if not isinstance(row, dict):
        return None
    chosen_messages = row.get("chosen")
    rejected_messages = row.get("rejected")
    if not isinstance(chosen_messages, list) or not isinstance(rejected_messages, list):
        return None
    chosen_turns = _coerce_turns(chosen_messages)
    rejected_turns = _coerce_turns(rejected_messages)
    if chosen_turns is None or rejected_turns is None:
        return None
    if not chosen_turns or not rejected_turns:
        return None
    if chosen_turns[-1].role != "assistant" or rejected_turns[-1].role != "assistant":
        return None
    prompt_turns = tuple(chosen_turns[:-1])
    # The two responses must share the same prompt prefix; otherwise the pair is
    # not a clean shared-prompt contrast and we drop it loudly via None.
    if tuple(rejected_turns[:-1]) != prompt_turns:
        return None
    if not prompt_turns or prompt_turns[-1].role == "assistant":
        return None
    chosen_response = chosen_turns[-1].content
    rejected_response = rejected_turns[-1].content
    if _clean_text(chosen_response) == _clean_text(rejected_response):
        return None
    return PreferencePair(
        prompt_turns=prompt_turns,
        chosen=chosen_response,
        rejected=rejected_response,
        source=source.name,
        row_id=row_id,
        source_revision=source.revision,
        source_dataset=source.source,
    )


def _coerce_turns(messages: list[Any]) -> tuple[ChatTurn, ...] | None:
    turns: list[ChatTurn] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        turns.append(ChatTurn(role=role, content=content))
    return tuple(turns)


def build_preference_set(
    source: DatasetSource,
    tokenizer: Tokenizer,
    *,
    max_pairs: int,
    max_tokens: int,
    max_length: int,
    max_prompt_length: int,
    min_pairs: int | None = None,
    skip_selected: int = 0,
    allow_remote_download: bool = False,
) -> PreferenceBuildResult:
    """Build a bounded preference set from one source (UltraFeedback for v1).

    ``max_pairs`` is a CAP ("use at most N"); ``min_pairs`` is the sufficiency
    FLOOR. UltraFeedback responses are long, so at ``max_length=1024`` roughly half
    are length-filtered and selecting fewer than the cap is normal -- a shortfall is
    reported only when fewer than ``min_pairs`` clean pairs survive. ``min_pairs``
    defaults to ``max_pairs`` (cap is also the floor) for callers that don't set it.

    ``skip_selected`` lets a held-out eval set start past the rows a train build
    already consumed, keeping train/eval disjoint exactly like the SFT path.
    """
    if max_pairs <= 0:
        raise DataError("max_pairs must be positive")
    if max_tokens <= 0:
        raise DataError("max_tokens must be positive")
    floor = max_pairs if min_pairs is None else min_pairs
    if floor <= 0:
        raise DataError("min_pairs must be positive")
    if floor > max_pairs:
        raise DataError("min_pairs (floor) must be <= max_pairs (cap)")
    counts = PreferenceSurvivorCounts()
    selected: list[TokenizedPreferencePair] = []
    selected_tokens = 0
    skipped = 0
    budget_stop: str | None = None
    tokenized_rows = _iter_tokenized_preference_pairs(
        source,
        tokenizer,
        counts,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        allow_remote_download=allow_remote_download,
    )
    for tokenized in tokenized_rows:
        if skipped < skip_selected:
            skipped += 1
            continue
        pair_tokens = len(tokenized.chosen.input_ids) + len(tokenized.rejected.input_ids)
        if selected_tokens + pair_tokens > max_tokens:
            budget_stop = "token_cap"
            break
        selected.append(tokenized)
        selected_tokens += pair_tokens
        counts.selected += 1
        if counts.selected >= max_pairs:
            budget_stop = budget_stop or "sample_cap"
            break

    _assert_validation_drop_floor(source, counts)

    # Selecting fewer than the cap because of length/filtering is normal; a
    # shortfall fires only below the sufficiency floor.
    shortfalls = (
        (f"{source.name}: selected {counts.selected}/{floor} (floor); cap was {max_pairs}",)
        if counts.selected < floor
        else ()
    )
    return PreferenceBuildResult(
        pairs=tuple(selected),
        counts=counts,
        selected_tokens=selected_tokens,
        selected_chosen_response_tokens=sum(p.chosen.response_tokens for p in selected),
        selected_rejected_response_tokens=sum(p.rejected.response_tokens for p in selected),
        sample_cap=max_pairs,
        token_cap=max_tokens,
        budget_stop=budget_stop,
        shortfalls=shortfalls,
    )


def _iter_tokenized_preference_pairs(
    source: DatasetSource,
    tokenizer: Tokenizer,
    counts: PreferenceSurvivorCounts,
    *,
    max_length: int,
    max_prompt_length: int,
    allow_remote_download: bool,
) -> Iterator[TokenizedPreferencePair]:
    for row_id, row in _iter_rows(source, allow_remote_download=allow_remote_download):
        counts.rows_seen += 1
        # Parse-time per-row DataError (empty/whitespace chosen or rejected, prompt
        # ending on assistant) must drop the row, not abort the whole build.
        try:
            parsed = parse_ultrafeedback_row(source, row_id, row)
        except DataError as error:
            counts.record_drop(_drop_reason(error))
            continue
        if parsed is None:
            counts.rejected_unparsable += 1
            continue
        prompt_chars = sum(len(_clean_text(turn.content)) for turn in parsed.prompt_turns)
        if (
            prompt_chars > source.max_prompt_chars
            or len(parsed.chosen) > source.max_response_chars
            or len(parsed.rejected) > source.max_response_chars
        ):
            counts.rejected_too_long_chars += 1
            continue
        counts.survivors += 1
        try:
            yield tokenize_preference_pair(
                tokenizer,
                parsed,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
            )
        except DataError as error:
            # Over-length is a truncation reject; any other tokenize-time DataError
            # (e.g. zero prompt/response tokens) is a counted validation drop.
            if _is_length_error(error):
                _record_truncation(
                    counts,
                    tokenizer,
                    parsed,
                    max_length=max_length,
                    max_prompt_length=max_prompt_length,
                )
                counts.rejected_too_long_tokens += 1
            else:
                counts.record_drop(_drop_reason(error))
            continue


def _is_length_error(error: DataError) -> bool:
    message = str(error)
    return "max_length" in message or "max_prompt_length" in message


def _drop_reason(error: DataError) -> str:
    message = str(error)
    if "non-empty chosen and rejected" in message:
        return "empty_response"
    if "user/system turn" in message:
        return "prompt_ends_on_assistant"
    if "zero prompt tokens" in message:
        return "zero_prompt_tokens"
    if "zero response tokens" in message:
        return "zero_response_tokens"
    if "at least one prompt turn" in message:
        return "no_prompt_turn"
    return "other_validation_error"


def _assert_validation_drop_floor(source: DatasetSource, counts: PreferenceSurvivorCounts) -> None:
    """Drops are normal for a few bad source rows; a high rate means a bug.

    The denominator is rows that parsed into a candidate pair (survivors plus the
    parse-time validation drops) -- structural `None` rejections (mismatched or
    malformed rows) are a separate, expected class and excluded.
    """
    considered = counts.survivors + counts.dropped_validation
    if considered <= 0 or counts.dropped_validation == 0:
        return
    drop_fraction = counts.dropped_validation / considered
    if drop_fraction > MAX_VALIDATION_DROP_FRACTION:
        raise DataError(
            f"{source.name}: per-row validation dropped {counts.dropped_validation}/"
            f"{considered} candidate pairs ({drop_fraction:.1%}), above the "
            f"{MAX_VALIDATION_DROP_FRACTION:.0%} floor -- likely a templating/parsing bug, "
            f"not bad source rows; drop reasons: {dict(sorted(counts.drop_reasons.items()))}"
        )


def _record_truncation(
    counts: PreferenceSurvivorCounts,
    tokenizer: Tokenizer,
    pair: PreferencePair,
    *,
    max_length: int,
    max_prompt_length: int,
) -> None:
    try:
        lengths = measure_preference_lengths(
            tokenizer, pair, max_length=max_length, max_prompt_length=max_prompt_length
        )
    except DataError:
        return
    if lengths["prompt_truncated"]:
        counts.prompt_truncation_count += 1
    if lengths["response_truncated"]:
        counts.response_truncation_count += 1
