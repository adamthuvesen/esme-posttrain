from __future__ import annotations

import random
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


def lr_lambda(config: Any) -> Any:
    if config.scheduler == "constant":
        return lambda _step: 1.0

    if config.scheduler == "cosine_decay":

        def cosine_warmup_decay(step: int) -> float:
            if config.warmup_steps and step < config.warmup_steps:
                return max(1e-8, float(step + 1) / float(config.warmup_steps))
            decay_steps = max(1, config.max_steps - config.warmup_steps)
            progress = min(1.0, max(0.0, float(step - config.warmup_steps) / decay_steps))
            return max(1e-8, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return cosine_warmup_decay

    def linear_warmup_decay(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return max(1e-8, float(step + 1) / float(config.warmup_steps))
        decay_steps = max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, float(step - config.warmup_steps) / decay_steps))
        return max(1e-8, 1.0 - progress)

    return linear_warmup_decay


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
