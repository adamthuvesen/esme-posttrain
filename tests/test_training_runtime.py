from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from esme_posttrain.training.errors import TrainerError
from esme_posttrain.training.runtime import (
    constant_lr,
    cosine_decay_lr,
    linear_warmup_decay_lr,
    lr_lambda_factory,
)


@dataclass(frozen=True)
class _LegacyScheduleConfig:
    scheduler: str
    warmup_steps: int
    max_steps: int


def _legacy_lr_lambda(config: _LegacyScheduleConfig):
    """The pre-refactor config-driven implementation, kept verbatim as the oracle."""
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


@pytest.mark.parametrize("scheduler", ["constant", "cosine_decay", "linear_warmup_decay"])
@pytest.mark.parametrize(("warmup_steps", "max_steps"), [(0, 40), (5, 40), (40, 40), (3, 4)])
def test_lr_factory_matches_legacy_schedule_exactly(
    scheduler: str, warmup_steps: int, max_steps: int
) -> None:
    legacy = _legacy_lr_lambda(
        _LegacyScheduleConfig(scheduler=scheduler, warmup_steps=warmup_steps, max_steps=max_steps)
    )
    current = lr_lambda_factory(scheduler=scheduler, warmup_steps=warmup_steps, max_steps=max_steps)
    for step in range(max_steps + 10):
        assert current(step) == legacy(step), f"{scheduler} diverged at step {step}"


def test_named_schedule_functions_cover_warmup_and_decay() -> None:
    assert constant_lr(0) == 1.0
    assert constant_lr(999) == 1.0
    assert cosine_decay_lr(0, warmup_steps=4, max_steps=10) == pytest.approx(0.25)
    assert cosine_decay_lr(10, warmup_steps=4, max_steps=10) == pytest.approx(1e-8)
    assert linear_warmup_decay_lr(0, warmup_steps=4, max_steps=10) == pytest.approx(0.25)
    assert linear_warmup_decay_lr(7, warmup_steps=4, max_steps=10) == pytest.approx(0.5)


def test_lr_factory_rejects_unknown_scheduler() -> None:
    with pytest.raises(TrainerError, match="unknown scheduler"):
        lr_lambda_factory(scheduler="wsd", warmup_steps=0, max_steps=10)
