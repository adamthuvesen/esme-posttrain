from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

import torch

from esme_posttrain.sft.data import IGNORE_INDEX
from esme_posttrain.training.errors import TrainerError
from esme_posttrain.training.runtime import resolve_torch_device


class CollatableSequence(Protocol):
    input_ids: Sequence[int]
    labels: Sequence[int]
    source: str
    row_id: str


CollatableT = TypeVar("CollatableT", bound=CollatableSequence)


def token_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    mask = targets != IGNORE_INDEX
    if int(mask.sum().item()) == 0:
        return 0
    predictions = logits.argmax(dim=-1)
    return int(((predictions == targets) & mask).sum().item())


def collate_batch(
    batch: tuple[CollatableSequence, ...],
    *,
    device: torch.device | str | None = None,
    pad_to_multiple_of: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise TrainerError("batch must not be empty")
    max_len = max(len(example.input_ids) for example in batch)
    if pad_to_multiple_of is not None:
        if pad_to_multiple_of <= 0:
            raise TrainerError("pad_to_multiple_of must be positive")
        remainder = max_len % pad_to_multiple_of
        if remainder:
            max_len += pad_to_multiple_of - remainder
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for example in batch:
        if len(example.input_ids) != len(example.labels):
            raise TrainerError(f"{example.source}:{example.row_id} input/label length mismatch")
        pad = max_len - len(example.input_ids)
        input_rows.append([*example.input_ids, *([0] * pad)])
        label_rows.append([*example.labels, *([IGNORE_INDEX] * pad)])
    target_device = resolve_torch_device(device or "cpu")
    return (
        torch.tensor(input_rows, dtype=torch.long, device=target_device),
        torch.tensor(label_rows, dtype=torch.long, device=target_device),
    )


def cyclic_batch(
    examples: tuple[CollatableT, ...], *, batch_index: int, batch_size: int
) -> tuple[CollatableT, ...]:
    start = batch_index * batch_size
    return tuple(examples[(start + offset) % len(examples)] for offset in range(batch_size))
