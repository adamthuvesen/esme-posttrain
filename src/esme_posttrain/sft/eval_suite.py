from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from esme_posttrain.modeling import DenseBackbone, language_model_loss, perplexity
from esme_posttrain.sft.data import IGNORE_INDEX, TokenizedExample
from esme_posttrain.training.collate import collate_batch
from esme_posttrain.training.errors import TrainerError
from esme_posttrain.training.metrics import eval_suite_metric_payload
from esme_posttrain.training.runtime import precision_context


@dataclass(frozen=True)
class EvalMetrics:
    response_loss: float
    perplexity: float
    supervised_tokens: int
    examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_loss": self.response_loss,
            "perplexity": self.perplexity,
            "supervised_tokens": self.supervised_tokens,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalMetrics:
        return cls(
            response_loss=float(payload["response_loss"]),
            perplexity=float(payload["perplexity"]),
            supervised_tokens=int(payload["supervised_tokens"]),
            examples=int(payload["examples"]),
        )


@dataclass(frozen=True)
class EvalSplit:
    name: str
    examples: tuple[TokenizedExample, ...]
    selector_weight: float = 0.0
    guardrail: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise TrainerError("eval split name must be non-empty")
        if "/" in self.name:
            raise TrainerError("eval split name must not contain '/'")
        if not self.examples:
            raise TrainerError(f"eval split {self.name} must not be empty")
        if self.selector_weight < 0:
            raise TrainerError("eval selector weights must be non-negative")


@dataclass(frozen=True)
class EvalSuiteResult:
    selector_metric_name: str
    selector_response_loss: float
    split_metrics: dict[str, EvalMetrics]
    selector_weights: dict[str, float]

    @property
    def selector_eval(self) -> EvalMetrics:
        matched_splits = [
            (name, metrics)
            for name, metrics in self.split_metrics.items()
            if self.selector_weights.get(name, 0.0) > 0
        ]
        return EvalMetrics(
            response_loss=self.selector_response_loss,
            perplexity=perplexity(self.selector_response_loss),
            supervised_tokens=sum(metrics.supervised_tokens for _name, metrics in matched_splits),
            examples=sum(metrics.examples for _name, metrics in matched_splits),
        )

    def to_metric_payload(self, *, step: int) -> dict[str, Any]:
        return eval_suite_metric_payload(
            step=step,
            selector_metric=self.selector_metric_name,
            selector_response_loss=self.selector_response_loss,
            split_metrics={name: metrics.to_dict() for name, metrics in self.split_metrics.items()},
            selector_weights=self.selector_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_metric": self.selector_metric_name,
            "selector_response_loss": self.selector_response_loss,
            "selector_weights": self.selector_weights,
            "splits": {name: metrics.to_dict() for name, metrics in self.split_metrics.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalSuiteResult:
        raw_splits = payload["splits"]
        raw_weights = payload["selector_weights"]
        if not isinstance(raw_splits, dict):
            raise TrainerError("selected_eval_suite.splits must be an object")
        if not isinstance(raw_weights, dict):
            raise TrainerError("selected_eval_suite.selector_weights must be an object")
        return cls(
            selector_metric_name=str(payload["selector_metric"]),
            selector_response_loss=float(payload["selector_response_loss"]),
            split_metrics={
                str(name): EvalMetrics.from_dict(_dict_field(metrics, f"splits.{name}"))
                for name, metrics in raw_splits.items()
            },
            selector_weights={str(name): float(weight) for name, weight in raw_weights.items()},
        )


def _dict_field(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainerError(f"selected_eval_suite.{label} must be an object")
    return value


def evaluate_response_loss(
    model: DenseBackbone,
    examples: tuple[TokenizedExample, ...],
    *,
    batch_size: int,
) -> EvalMetrics:
    if not examples:
        raise TrainerError("eval examples must not be empty")
    if batch_size <= 0:
        raise TrainerError("batch_size must be positive")
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids, labels = collate_batch(batch, device=device)
            with precision_context("fp32", device):
                logits = model(input_ids[:, :-1])
            targets = labels[:, 1:]
            loss, _components = language_model_loss(
                logits,
                targets,
                z_loss_weight=0.0,
                logit_soft_cap=model.config.logit_soft_cap,
                ignore_index=IGNORE_INDEX,
            )
            supervised = int((targets != IGNORE_INDEX).sum().item())
            total_loss += float(loss.detach()) * supervised
            total_tokens += supervised
    if was_training:
        model.train()
    if total_tokens <= 0:
        raise TrainerError("eval examples contain no supervised response tokens")
    response_loss = total_loss / total_tokens
    return EvalMetrics(
        response_loss=response_loss,
        perplexity=perplexity(response_loss),
        supervised_tokens=total_tokens,
        examples=len(examples),
    )


def evaluate_eval_suite(
    model: DenseBackbone,
    eval_splits: tuple[EvalSplit, ...],
    *,
    batch_size: int,
) -> EvalSuiteResult:
    if not eval_splits:
        raise TrainerError("eval_splits must not be empty")
    split_metrics = {
        split.name: evaluate_response_loss(model, split.examples, batch_size=batch_size)
        for split in eval_splits
    }
    selector_weights = {
        split.name: split.selector_weight for split in eval_splits if split.selector_weight > 0
    }
    weight_sum = sum(selector_weights.values())
    if weight_sum <= 0:
        raise TrainerError("at least one eval split must have a selector weight")
    selector_loss = (
        sum(split_metrics[name].response_loss * weight for name, weight in selector_weights.items())
        / weight_sum
    )
    return EvalSuiteResult(
        selector_metric_name=configured_selector_metric(eval_splits),
        selector_response_loss=selector_loss,
        split_metrics=split_metrics,
        selector_weights=selector_weights,
    )


def configured_selector_metric(eval_splits: tuple[EvalSplit, ...]) -> str:
    selector_names = tuple(split.name for split in eval_splits if split.selector_weight > 0)
    if selector_names == ("smol-smoltalk", "tulu-3-personas"):
        return "eval/matched/response_loss"
    if len(selector_names) == 1:
        return f"eval/{selector_names[0]}/response_loss"
    return "eval/matched/response_loss"


def normalize_eval_splits(
    eval_examples: tuple[TokenizedExample, ...], eval_splits: tuple[EvalSplit, ...] | None
) -> tuple[EvalSplit, ...]:
    if eval_splits is None:
        return (EvalSplit("heldout", eval_examples, selector_weight=1.0),)
    names = [split.name for split in eval_splits]
    if len(set(names)) != len(names):
        raise TrainerError("eval split names must be unique")
    if sum(split.selector_weight for split in eval_splits) <= 0:
        raise TrainerError("at least one eval split must have a selector weight")
    return eval_splits


def split_response_loss(eval_suite: EvalSuiteResult, name: str) -> float | None:
    metrics = eval_suite.split_metrics.get(name)
    return metrics.response_loss if metrics is not None else None


def no_robots_catastrophic_regression(
    eval_suite: EvalSuiteResult, *, baseline: float | None, multiplier: float
) -> bool:
    current = split_response_loss(eval_suite, "no_robots")
    if baseline is None or current is None:
        return False
    return current >= baseline * multiplier
