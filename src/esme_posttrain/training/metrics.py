from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TRAIN_METRIC_NAMES = {
    "train/loss",
    "train/learning_rate",
    "train/grad_norm",
    "train/tokens",
    "train/supervised_tokens",
    "train/token_accuracy",
}
EVAL_METRIC_NAMES = {
    "eval/response_loss",
    "eval/perplexity",
    "eval/supervised_tokens",
    "eval/examples",
    "eval/matched/response_loss",
    "eval/smol-smoltalk/response_loss",
    "eval/tulu-3-personas/response_loss",
    "eval/no_robots/response_loss",
}


def train_metric_payload(
    *,
    step: int,
    loss: float,
    learning_rate: float,
    grad_norm: float,
    tokens: int,
    supervised_tokens: int,
    token_accuracy: float,
    total_tokens: int,
    total_supervised_tokens: int,
    components: dict[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "train",
        "step": step,
        "train/loss": loss,
        "train/learning_rate": learning_rate,
        "train/grad_norm": grad_norm,
        "train/tokens": tokens,
        "train/supervised_tokens": supervised_tokens,
        "train/token_accuracy": token_accuracy,
        "train/total_tokens": total_tokens,
        "train/total_supervised_tokens": total_supervised_tokens,
    }
    payload.update({f"train/{key}": value for key, value in components.items()})
    return payload


def eval_suite_metric_payload(
    *,
    step: int,
    selector_metric: str,
    selector_response_loss: float,
    split_metrics: dict[str, dict[str, Any]],
    selector_weights: dict[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "eval",
        "step": step,
        "split": "eval_suite",
        "eval/response_loss": selector_response_loss,
        "eval/perplexity": math.exp(selector_response_loss)
        if selector_response_loss < 50
        else float("inf"),
        "eval/supervised_tokens": sum(
            int(metrics["supervised_tokens"]) for metrics in split_metrics.values()
        ),
        "eval/examples": sum(int(metrics["examples"]) for metrics in split_metrics.values()),
        "eval/selector_metric": selector_metric,
        "eval/selector_weights": selector_weights,
    }
    matched_tokens = 0
    matched_examples = 0
    for name, metrics in split_metrics.items():
        prefix = f"eval/{name}"
        payload[f"{prefix}/response_loss"] = metrics["response_loss"]
        payload[f"{prefix}/perplexity"] = metrics["perplexity"]
        payload[f"{prefix}/supervised_tokens"] = metrics["supervised_tokens"]
        payload[f"{prefix}/examples"] = metrics["examples"]
        if selector_weights.get(name, 0.0) > 0:
            matched_tokens += int(metrics["supervised_tokens"])
            matched_examples += int(metrics["examples"])
    payload["eval/matched/response_loss"] = selector_response_loss
    payload["eval/matched/perplexity"] = payload["eval/perplexity"]
    payload["eval/matched/supervised_tokens"] = matched_tokens
    payload["eval/matched/examples"] = matched_examples
    return payload


def append_metric(path: Path, payload: dict[str, Any], wandb_run: Any | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    if wandb_run is not None:
        wandb_run.log(payload, step=payload.get("step"))
