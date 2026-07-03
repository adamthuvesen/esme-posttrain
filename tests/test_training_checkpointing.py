from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.smoke_instruct import tiny_backbone_config
from esme_posttrain.training.checkpointing import (
    CheckpointError,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)


def _tiny_model() -> DenseBackbone:
    torch.manual_seed(7)
    return DenseBackbone(tiny_backbone_config())


def test_v3_checkpoint_round_trips_rng_state_and_data_position(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        step=5,
        metrics={"event": "periodic"},
        rng_state=capture_rng_state(),
        data_position=20,
    )
    expected = [random.random(), float(np.random.random()), float(torch.rand(1))]

    loaded = load_training_checkpoint(path)
    assert loaded.step == 5
    assert loaded.data_position == 20
    assert loaded.rng_state is not None
    restore_rng_state(loaded.rng_state)
    resumed = [random.random(), float(np.random.random()), float(torch.rand(1))]
    assert resumed == expected


def test_old_shape_payload_without_rng_keys_still_loads(tmp_path: Path) -> None:
    model = _tiny_model()
    for version in (1, 2):
        path = tmp_path / f"checkpoint-v{version}.pt"
        torch.save(
            {
                "format_version": version,
                "config": model.config.to_dict(),
                "model_state": model.state_dict(),
                "optimizer_state": None,
                "scheduler_state": None,
                "step": 3,
                "metrics": {"event": "final"},
            },
            path,
        )
        loaded = load_training_checkpoint(path)
        assert loaded.step == 3
        assert loaded.rng_state is None
        assert loaded.data_position is None


def test_checkpoint_without_rng_arguments_loads_with_none_fields(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, model=model, step=1)
    loaded = load_training_checkpoint(path)
    assert loaded.rng_state is None
    assert loaded.data_position is None


def test_malformed_data_position_is_rejected_loudly(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 3,
            "config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "scheduler_state": None,
            "step": 1,
            "metrics": {},
            "rng_state": None,
            "data_position": "twenty",
        },
        path,
    )
    with pytest.raises(CheckpointError, match="data_position"):
        load_training_checkpoint(path)


def test_restore_rng_state_is_a_noop_for_absent_state() -> None:
    random.seed(11)
    expected = random.random()
    random.seed(11)
    restore_rng_state(None)
    restore_rng_state({})
    assert random.random() == expected
