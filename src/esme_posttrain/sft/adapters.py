from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol


class AdapterError(ValueError):
    pass


# Tasks beyond a 214M model's capacity. SmolLM2 dropped function calling and the
# hardest reasoning from the small-model SFT mix; smol-smoltalk tags each row with
# its `source` sub-dataset. Both smol-smoltalk adapters (SmolTalkAdapter and
# SmolTalkMultiTurnAdapter) drop rows from these subsets.
CAPACITY_FILTERED_SUBSETS: frozenset[str] = frozenset(
    {
        "apigen-80k",
        "xlam-function-calling-60k",
        "self-oss-instruct-sc2-exec-filter-50k",
    }
)


class DatasetSourceLike(Protocol):
    name: str
    role: Literal["train", "eval"]
    train_allowed: bool


@dataclass(frozen=True)
class AdapterExample:
    instruction: str
    response: str
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapacityFiltered:
    """Rejection marker: the row's subset is beyond a 214M model's capacity."""


@dataclass(frozen=True)
class ChatTurnLike:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class MultiTurnAdapterExample:
    """A parsed conversation, or a capacity-filtered/unparsable rejection marker.

    Exactly one of `turns` (accepted) or `rejection` (dropped) is set.
    """

    turns: tuple[ChatTurnLike, ...] = ()
    rejection: Literal["", "unparsable", "capacity_filtered"] = ""


class DatasetAdapter(Protocol):
    def parse(
        self, source: DatasetSourceLike, row_id: str, row: Any
    ) -> AdapterExample | CapacityFiltered | None: ...


class SmolTalkAdapter:
    def parse(
        self, source: DatasetSourceLike, row_id: str, row: Any
    ) -> AdapterExample | CapacityFiltered | None:
        del source, row_id
        if isinstance(row, dict):
            subset = row.get("source")
            if isinstance(subset, str) and subset in CAPACITY_FILTERED_SUBSETS:
                return CapacityFiltered()
        return _single_turn_from_row(row)


class TuluPersonasAdapter:
    def parse(self, source: DatasetSourceLike, row_id: str, row: Any) -> AdapterExample | None:
        del source, row_id
        example = _single_turn_from_row(row)
        if example is None or not isinstance(row, dict):
            return example
        return AdapterExample(
            instruction=example.instruction,
            response=example.response,
            constraints=_constraints_from_row(row),
        )


class NoRobotsEvalAdapter:
    def parse(self, source: DatasetSourceLike, row_id: str, row: Any) -> AdapterExample | None:
        del row_id
        if source.role != "eval" or source.train_allowed:
            raise AdapterError("HuggingFaceH4/no_robots is non-commercial and eval-only here")
        return _single_turn_from_row(row)


def adapter_for(source: DatasetSourceLike) -> DatasetAdapter:
    if source.name == "smol-smoltalk":
        return SmolTalkAdapter()
    if source.name == "tulu-3-personas":
        return TuluPersonasAdapter()
    if source.name == "no_robots":
        return NoRobotsEvalAdapter()
    raise AdapterError(f"unsupported SFT dataset adapter: {source.name}")


def _single_turn_from_row(row: Any) -> AdapterExample | None:
    if not isinstance(row, dict):
        return None
    messages = row.get("messages")
    if isinstance(messages, list):
        return _single_turn_from_messages(messages)
    pair = _instruction_response_pair(row)
    if pair is not None:
        instruction, response = pair
        return AdapterExample(instruction=instruction, response=response)
    return None


def _single_turn_from_messages(messages: list[Any]) -> AdapterExample | None:
    turns = _turns_from_messages(messages)
    if turns is None:
        return None
    roles = tuple(turn.role for turn in turns)
    contents = tuple(turn.content for turn in turns)
    if roles == ("user", "assistant"):
        return AdapterExample(instruction=contents[0], response=contents[1])
    if roles == ("system", "user", "assistant"):
        return AdapterExample(
            instruction=f"{contents[0].strip()}\n\n{contents[1].strip()}",
            response=contents[2],
        )
    return None


def _instruction_response_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    instruction = row.get("instruction") or row.get("prompt")
    response = row.get("response") or row.get("completion")
    if isinstance(instruction, str) and isinstance(response, str):
        return instruction, response
    return None


def _turns_from_messages(messages: list[Any]) -> tuple[ChatTurnLike, ...] | None:
    turns: list[ChatTurnLike] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        turns.append(ChatTurnLike(role=role, content=content))
    return tuple(turns)


def _constraints_from_row(row: dict[str, Any]) -> tuple[str, ...]:
    constraints = row.get("constraints")
    if not isinstance(constraints, list):
        return ()
    return tuple(str(item) for item in constraints if str(item).strip())


def iter_adapter_examples(
    source: DatasetSourceLike, rows: Iterator[tuple[str, Any]]
) -> Iterator[tuple[str, AdapterExample | CapacityFiltered | None]]:
    adapter = adapter_for(source)
    for row_id, row in rows:
        yield row_id, adapter.parse(source, row_id, row)


class MultiTurnAdapter(Protocol):
    def parse(
        self, source: DatasetSourceLike, row_id: str, row: Any
    ) -> MultiTurnAdapterExample: ...


class SmolTalkMultiTurnAdapter:
    """Parses full smol-smoltalk conversations and drops capacity-filtered subsets."""

    def parse(self, source: DatasetSourceLike, row_id: str, row: Any) -> MultiTurnAdapterExample:
        del source, row_id
        if not isinstance(row, dict):
            return MultiTurnAdapterExample(rejection="unparsable")
        subset = row.get("source")
        if isinstance(subset, str) and subset in CAPACITY_FILTERED_SUBSETS:
            return MultiTurnAdapterExample(rejection="capacity_filtered")
        return _multi_turn_from_row(row)


class TuluPersonasMultiTurnAdapter:
    """Single-turn instruction-following rows folded into the chat template.

    The personas set carries constraints; they are appended to the user turn so a
    multi-turn loss still supervises the assistant answer.
    """

    def parse(self, source: DatasetSourceLike, row_id: str, row: Any) -> MultiTurnAdapterExample:
        del source, row_id
        parsed = _multi_turn_from_row(row)
        if parsed.rejection or not isinstance(row, dict):
            return parsed
        constraints = _constraints_from_row(row)
        if not constraints:
            return parsed
        rendered = "\n".join(f"- {item}" for item in constraints)
        turns = list(parsed.turns)
        for index, turn in enumerate(turns):
            if turn.role == "user":
                turns[index] = ChatTurnLike(
                    role="user", content=f"{turn.content}\n\nConstraints:\n{rendered}"
                )
                break
        return MultiTurnAdapterExample(turns=tuple(turns))


class NoRobotsMultiTurnAdapter:
    def parse(self, source: DatasetSourceLike, row_id: str, row: Any) -> MultiTurnAdapterExample:
        del row_id
        if source.role != "eval" or source.train_allowed:
            raise AdapterError("HuggingFaceH4/no_robots is non-commercial and eval-only here")
        return _multi_turn_from_row(row)


def multi_turn_adapter_for(source: DatasetSourceLike) -> MultiTurnAdapter:
    if source.name == "smol-smoltalk":
        return SmolTalkMultiTurnAdapter()
    if source.name == "tulu-3-personas":
        return TuluPersonasMultiTurnAdapter()
    if source.name == "no_robots":
        return NoRobotsMultiTurnAdapter()
    raise AdapterError(f"unsupported multi-turn SFT dataset adapter: {source.name}")


def _multi_turn_from_row(row: Any) -> MultiTurnAdapterExample:
    if not isinstance(row, dict):
        return MultiTurnAdapterExample(rejection="unparsable")
    messages = row.get("messages")
    if isinstance(messages, list):
        return _multi_turn_from_messages(messages)
    pair = _instruction_response_pair(row)
    if pair is not None:
        instruction, response = pair
        return MultiTurnAdapterExample(
            turns=(
                ChatTurnLike(role="user", content=instruction),
                ChatTurnLike(role="assistant", content=response),
            )
        )
    return MultiTurnAdapterExample(rejection="unparsable")


def _multi_turn_from_messages(messages: list[Any]) -> MultiTurnAdapterExample:
    turns = _turns_from_messages(messages)
    if not turns or turns[-1].role != "assistant":
        return MultiTurnAdapterExample(rejection="unparsable")
    return MultiTurnAdapterExample(turns=turns)


def iter_multi_turn_examples(
    source: DatasetSourceLike, rows: Iterator[tuple[str, Any]]
) -> Iterator[tuple[str, MultiTurnAdapterExample]]:
    adapter = multi_turn_adapter_for(source)
    for row_id, row in rows:
        yield row_id, adapter.parse(source, row_id, row)
