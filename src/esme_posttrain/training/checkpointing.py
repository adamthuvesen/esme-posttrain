from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.modeling import BackboneConfig, DenseBackbone

TRAINING_CHECKPOINT_FORMAT = 2
_CHECKPOINT_RE = re.compile(r"step-(\d{6,})$")


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedTrainingCheckpoint:
    model: DenseBackbone
    config: BackboneConfig
    step: int
    metrics: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step-{step:06d}"


def save_training_checkpoint(
    path: Path,
    *,
    model: DenseBackbone,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": TRAINING_CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "metrics": dict(metrics or {}),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_training_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> LoadedTrainingCheckpoint:
    if not path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be an object")
    version = payload.get("format_version")
    if version not in {1, TRAINING_CHECKPOINT_FORMAT}:
        raise CheckpointError("unsupported training checkpoint format")
    config = BackboneConfig.from_dict(
        _require_checkpoint_object(payload.get("config"), "checkpoint.config")
    )
    model = DenseBackbone(config)
    model.load_state_dict(
        _require_checkpoint_object(payload.get("model_state"), "checkpoint.model_state")
    )
    model.eval()
    return LoadedTrainingCheckpoint(
        model=model,
        config=config,
        step=int(payload["step"]),
        metrics=_require_checkpoint_object(payload.get("metrics"), "checkpoint.metrics"),
        optimizer_state=_optional_checkpoint_object(
            payload.get("optimizer_state"), "checkpoint.optimizer"
        ),
        scheduler_state=_optional_checkpoint_object(
            payload.get("scheduler_state"), "checkpoint.scheduler"
        ),
    )


def latest_checkpoint_path(output_dir: Path) -> Path | None:
    candidates = []
    checkpoints_root = output_dir / "checkpoints"
    if checkpoints_root.is_dir():
        for path in checkpoints_root.iterdir():
            match = _CHECKPOINT_RE.match(path.name)
            checkpoint = path / "checkpoint.pt"
            if match and checkpoint.is_file():
                candidates.append((int(match.group(1)), checkpoint))
    final = output_dir / "checkpoint.pt"
    if final.is_file():
        try:
            loaded = load_training_checkpoint(final)
        except CheckpointError:
            loaded = None
        if loaded is not None:
            candidates.append((loaded.step, final))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def retain_last_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints_root = output_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        return
    ordered = sorted(_checkpoint_step_dirs(checkpoints_root), key=lambda item: item[0])
    for _step, path in ordered[:-keep]:
        shutil.rmtree(path)


def _checkpoint_step_dirs(path: Path) -> Iterable[tuple[int, Path]]:
    for child in path.iterdir():
        match = _CHECKPOINT_RE.match(child.name)
        if match and child.is_dir():
            yield int(match.group(1)), child


def _require_checkpoint_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{label} must be an object")
    return value


def _optional_checkpoint_object(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _require_checkpoint_object(value, label)
