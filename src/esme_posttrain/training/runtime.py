from __future__ import annotations

import random
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch

from esme_posttrain.training.errors import TrainerError


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise TrainerError(
            "CUDA was requested for SFT training but torch.cuda.is_available() is false"
        )
    return resolved


def constant_lr(step: int) -> float:
    """Flat multiplier: the optimizer's base learning rate at every step."""
    del step
    return 1.0


def cosine_decay_lr(step: int, *, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup, then cosine decay of the LR multiplier toward ~0 by ``max_steps``."""
    if warmup_steps and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / decay_steps))
    return max(1e-8, 0.5 * (1.0 + np.cos(np.pi * progress)))


def linear_warmup_decay_lr(step: int, *, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup, then linear decay of the LR multiplier toward ~0 by ``max_steps``."""
    if warmup_steps and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    decay_steps = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / decay_steps))
    return max(1e-8, 1.0 - progress)


def lr_lambda_factory(
    *, scheduler: str, warmup_steps: int, max_steps: int
) -> Callable[[int], float]:
    """Map a config's scheduler name onto the matching multiplier for ``LambdaLR``."""
    if scheduler == "constant":
        return constant_lr
    if scheduler == "cosine_decay":
        return partial(cosine_decay_lr, warmup_steps=warmup_steps, max_steps=max_steps)
    if scheduler == "linear_warmup_decay":
        return partial(linear_warmup_decay_lr, warmup_steps=warmup_steps, max_steps=max_steps)
    raise TrainerError(f"unknown scheduler: {scheduler}")


def validate_precision(precision: str, device: torch.device) -> None:
    if precision == "fp32":
        return
    if device.type != "cuda":
        raise TrainerError("bf16 precision requires CUDA for this trainer")
    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise TrainerError("bf16 precision was requested but this CUDA device does not support it")


def precision_context(precision: str, device: torch.device) -> Any:
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)
