from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

IGNORE_INDEX = -100


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class LossSemantics:
    assistant_only_loss: bool = True
    completion_only_loss: bool = True
    ignore_index: int = IGNORE_INDEX

    def __post_init__(self) -> None:
        if not self.assistant_only_loss:
            raise DataError("only assistant_only_loss=true is supported for Instruct SFT")
        if not self.completion_only_loss:
            raise DataError("only completion_only_loss=true is supported for Instruct SFT")
        if self.ignore_index != IGNORE_INDEX:
            raise DataError(f"loss.ignore_index must be {IGNORE_INDEX}")


@dataclass(frozen=True)
class DatasetSource:
    name: str
    source: str
    revision: str
    license: str
    split: str
    role: Literal["train", "eval"]
    mix_ratio: float = 0.0
    subset: str | None = None
    path: Path | None = None
    train_allowed: bool = True
    max_prompt_chars: int = 1600
    max_response_chars: int = 1600


@dataclass(frozen=True)
class SingleTurnExample:
    instruction: str
    response: str
    source: str
    row_id: str
    source_revision: str = "local"
    source_dataset: str | None = None
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class MultiTurnExample:
    turns: tuple[ChatTurn, ...]
    source: str
    row_id: str
    source_revision: str = "local"
    source_dataset: str | None = None

    def __post_init__(self) -> None:
        if not self.turns:
            raise DataError("multi-turn examples require at least one turn")
        if not any(turn.role == "assistant" for turn in self.turns):
            raise DataError("multi-turn examples require at least one assistant turn")

    @property
    def assistant_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.role == "assistant")


@dataclass(frozen=True)
class FormattedExample:
    prompt: str
    response: str
    source: str
    row_id: str
    source_revision: str
    source_dataset: str
    prompt_chars: int
    response_chars: int


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    response_tokens: int
    source: str
    row_id: str
    source_revision: str = "local"
    source_dataset: str = "local"
    prompt_chars: int = 0
    response_chars: int = 0
    turns: int = 2
    assistant_turns: int = 1

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)

    @property
    def is_multi_turn(self) -> bool:
        return self.assistant_turns > 1

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_dataset": self.source_dataset,
            "revision": self.source_revision,
            "row_id": self.row_id,
            "input_tokens": len(self.input_ids),
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "supervised_tokens": self.supervised_tokens,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "turns": self.turns,
            "assistant_turns": self.assistant_turns,
        }


@dataclass
class SourceSurvivorCounts:
    rows_seen: int = 0
    survivors: int = 0
    selected: int = 0
    rejected_non_single_turn: int = 0
    rejected_unparsable: int = 0
    rejected_capacity_filtered: int = 0
    rejected_too_long_chars: int = 0
    rejected_too_long_tokens: int = 0
    prompt_truncation_count: int = 0
    assistant_target_truncation_count: int = 0
    selected_single_turn: int = 0
    selected_multi_turn: int = 0
    selected_assistant_turns: int = 0
    selected_turn_histogram: dict[int, int] = field(default_factory=dict)

    def record_selected_turns(self, *, turns: int, assistant_turns: int) -> None:
        if assistant_turns <= 1:
            self.selected_single_turn += 1
        else:
            self.selected_multi_turn += 1
        self.selected_assistant_turns += assistant_turns
        self.selected_turn_histogram[turns] = self.selected_turn_histogram.get(turns, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_seen": self.rows_seen,
            "survivors": self.survivors,
            "selected": self.selected,
            "rejected_non_single_turn": self.rejected_non_single_turn,
            "rejected_unparsable": self.rejected_unparsable,
            "rejected_capacity_filtered": self.rejected_capacity_filtered,
            "rejected_too_long_chars": self.rejected_too_long_chars,
            "rejected_too_long_tokens": self.rejected_too_long_tokens,
            "prompt_truncation_count": self.prompt_truncation_count,
            "assistant_target_truncation_count": self.assistant_target_truncation_count,
            "selected_single_turn": self.selected_single_turn,
            "selected_multi_turn": self.selected_multi_turn,
            "selected_assistant_turns": self.selected_assistant_turns,
            "selected_turn_histogram": {
                str(turns): count for turns, count in sorted(self.selected_turn_histogram.items())
            },
        }


@dataclass(frozen=True)
class DatasetBuildResult:
    examples: tuple[TokenizedExample, ...]
    counts_by_source: dict[str, SourceSurvivorCounts]
    selected_tokens: int
    selected_supervised_tokens: int
    selected_prompt_tokens: int
    sample_cap: int
    token_cap: int
    budget_stop: str | None = None
    shortfalls: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_samples": len(self.examples),
            "selected_tokens": self.selected_tokens,
            "selected_supervised_tokens": self.selected_supervised_tokens,
            "selected_prompt_tokens": self.selected_prompt_tokens,
            "unused_examples": sum(
                max(0, counts.survivors - counts.selected)
                for counts in self.counts_by_source.values()
            ),
            "selected_row_manifest": [example.manifest_entry() for example in self.examples],
            "sample_cap": self.sample_cap,
            "token_cap": self.token_cap,
            "budget_stop": self.budget_stop,
            "shortfalls": list(self.shortfalls),
            "counts_by_source": {
                source: counts.to_dict() for source, counts in self.counts_by_source.items()
            },
        }
