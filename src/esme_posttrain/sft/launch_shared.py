from __future__ import annotations

from typing import Any

from esme_posttrain.launch.common import LaunchError, require_keys
from esme_posttrain.sft.data import DatasetSource


def validate_eval_source(
    payload: dict[str, Any], expected_dataset: dict[str, Any]
) -> DatasetSource:
    require_keys(
        payload,
        {"name", "source", "revision", "license", "split", "role", "train_allowed", "usage"},
        "datasets.eval_holdout",
    )
    for key, value in expected_dataset.items():
        if payload[key] != value:
            raise LaunchError(f"datasets.eval_holdout.{key} must be {value}")
    if payload["role"] != "eval":
        raise LaunchError("datasets.eval_holdout.role must be eval")
    if payload["usage"] != "eval_only":
        raise LaunchError("datasets.eval_holdout.usage must be eval_only")
    return DatasetSource(
        name=expected_dataset["name"],
        source=expected_dataset["source"],
        revision=expected_dataset["revision"],
        license=expected_dataset["license"],
        split=expected_dataset["split"],
        role="eval",
        train_allowed=False,
    )
