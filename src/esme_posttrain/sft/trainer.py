from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.bundle import file_sha256
from esme_posttrain.modeling import BackboneConfig, DenseBackbone, language_model_loss
from esme_posttrain.sft.data import IGNORE_INDEX, TokenizedExample
from esme_posttrain.sft.eval_suite import (
    EvalMetrics,
    EvalSplit,
    EvalSuiteResult,
    configured_selector_metric,
    evaluate_eval_suite,
)
from esme_posttrain.sft.eval_suite import (
    no_robots_catastrophic_regression as _no_robots_catastrophic_regression,
)
from esme_posttrain.sft.eval_suite import (
    normalize_eval_splits as _normalize_eval_splits,
)
from esme_posttrain.sft.eval_suite import (
    split_response_loss as _split_response_loss,
)
from esme_posttrain.sft.sampling import (
    generate_samples as _generate_samples,
)
from esme_posttrain.sft.sampling import (
    markdown_fenced_text as _markdown_fenced_text,
)
from esme_posttrain.sft.sampling import (
    sample_prompt_examples as _sample_prompt_examples,
)
from esme_posttrain.training.checkpointing import (
    LoadedTrainingCheckpoint,
    checkpoint_dir,
    latest_checkpoint_path,
    load_training_checkpoint,
    retain_last_checkpoints,
    save_training_checkpoint,
)
from esme_posttrain.training.collate import (
    collate_batch as _collate_batch,
)
from esme_posttrain.training.collate import (
    cyclic_batch as _cyclic_batch,
)
from esme_posttrain.training.collate import (
    token_correct as _token_correct,
)
from esme_posttrain.training.errors import TrainerError
from esme_posttrain.training.metrics import append_metric, train_metric_payload
from esme_posttrain.training.runtime import (
    lr_lambda as _lr_lambda,
)
from esme_posttrain.training.runtime import (
    precision_context as _precision_context,
)
from esme_posttrain.training.runtime import (
    resolve_torch_device as _resolve_torch_device,
)
from esme_posttrain.training.runtime import (
    set_reproducible_seed as _set_reproducible_seed,
)
from esme_posttrain.training.runtime import (
    validate_precision as _validate_precision,
)
from esme_posttrain.training.wandb_init import WandbConfig
from esme_posttrain.training.wandb_init import start_wandb as _start_wandb

StepCallback = Callable[[int], None]


@dataclass(frozen=True)
class SFTTrainerConfig:
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    seed: int
    output_dir: Path
    scheduler: str = "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    precision: str = "fp32"
    pad_to_multiple_of: int | None = None
    assistant_only_loss: bool = True
    completion_only_loss: bool = True
    tuning_mode: str = "full"
    artifact_name: str = "Esme-214M-Instruct"
    base_artifact_name: str = "Esme-214M-Base"
    grad_clip: float = 1.0
    log_interval: int = 1
    eval_interval: int = 0
    checkpoint_interval: int = 0
    retain_last_checkpoints: int = 2
    early_stopping_patience: int = 0
    no_robots_catastrophic_regression_multiplier: float = 1.5
    resume_from_latest: bool = False
    sample_new_tokens: int = 16
    device: str = "cpu"
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "grad_clip",
            "log_interval",
        ):
            if getattr(self, field_name) <= 0:
                raise TrainerError(f"{field_name} must be positive")
        if not self.assistant_only_loss or not self.completion_only_loss:
            raise TrainerError("only assistant/completion-only supervised loss is supported")
        if self.tuning_mode != "full":
            raise TrainerError(f"unsupported tuning_mode for SFT trainer: {self.tuning_mode}")
        if self.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
            raise TrainerError("scheduler must be constant, linear_warmup_decay, or cosine_decay")
        for field_name in (
            "warmup_steps",
            "weight_decay",
            "eval_interval",
            "checkpoint_interval",
            "retain_last_checkpoints",
            "early_stopping_patience",
        ):
            if getattr(self, field_name) < 0:
                raise TrainerError(f"{field_name} must be non-negative")
        if self.warmup_steps > self.max_steps:
            raise TrainerError("warmup_steps must be <= max_steps")
        if self.precision not in {"fp32", "bf16"}:
            raise TrainerError("precision must be fp32 or bf16")
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise TrainerError("pad_to_multiple_of must be positive when set")
        if self.no_robots_catastrophic_regression_multiplier <= 1.0:
            raise TrainerError("no_robots_catastrophic_regression_multiplier must be > 1")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class SFTTrainResult:
    output_dir: Path
    checkpoint_path: Path
    manifest_path: Path
    metrics_path: Path
    samples_path: Path
    best_checkpoint_path: Path
    best_checkpoint_metadata_path: Path
    base_eval: EvalMetrics
    instruct_eval: EvalMetrics
    base_eval_suite: EvalSuiteResult
    selected_eval_suite: EvalSuiteResult
    steps_completed: int
    selected_step: int
    selected_metric_name: str
    selected_metric_value: float
    trained_tokens: int
    supervised_tokens: int
    selected_examples: int
    eval_examples: int
    effective_epochs: float
    response_loss_decreased: bool
    instruct_beats_base: bool
    training_mode: str
    start_step: int
    early_stopped: bool = False
    early_stop_reason: str | None = None
    resumed_from_checkpoint: str | None = None
    wandb_run_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "manifest_path": str(self.manifest_path),
            "metrics_path": str(self.metrics_path),
            "samples_path": str(self.samples_path),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "best_checkpoint_metadata_path": str(self.best_checkpoint_metadata_path),
            "base_eval": self.base_eval.to_dict(),
            "instruct_eval": self.instruct_eval.to_dict(),
            "base_eval_suite": self.base_eval_suite.to_dict(),
            "selected_eval_suite": self.selected_eval_suite.to_dict(),
            "steps_completed": self.steps_completed,
            "selected_step": self.selected_step,
            "selected_metric_name": self.selected_metric_name,
            "selected_metric_value": self.selected_metric_value,
            "trained_tokens": self.trained_tokens,
            "supervised_tokens": self.supervised_tokens,
            "selected_examples": self.selected_examples,
            "eval_examples": self.eval_examples,
            "effective_epochs": self.effective_epochs,
            "response_loss_decreased": self.response_loss_decreased,
            "instruct_beats_base": self.instruct_beats_base,
            "training_mode": self.training_mode,
            "start_step": self.start_step,
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "wandb_run_url": self.wandb_run_url,
        }


@dataclass(frozen=True)
class LoadedSFTCheckpoint:
    model: DenseBackbone
    config: BackboneConfig
    step: int
    metrics: dict[str, Any]


def build_full_finetune_optimizer(
    model: DenseBackbone, config: SFTTrainerConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda(config))
    return optimizer, scheduler


def run_sft_training(
    model: DenseBackbone,
    tokenizer: Tokenizer,
    train_examples: tuple[TokenizedExample, ...],
    eval_examples: tuple[TokenizedExample, ...],
    config: SFTTrainerConfig,
    *,
    eval_splits: tuple[EvalSplit, ...] | None = None,
    base_bundle_manifest: dict[str, Any] | None = None,
    step_callback: StepCallback | None = None,
) -> SFTTrainResult:
    if not train_examples:
        raise TrainerError("train_examples must not be empty")
    if not eval_examples:
        raise TrainerError("eval_examples must not be empty")
    eval_suite = _normalize_eval_splits(eval_examples, eval_splits)
    _set_reproducible_seed(config.seed)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    samples_path = output_dir / "samples.md"
    checkpoint_path = output_dir / "checkpoint.pt"
    best_checkpoint_path = output_dir / "best-checkpoint.pt"
    best_checkpoint_metadata_path = output_dir / "best-checkpoint.json"
    manifest_path = output_dir / "manifest.json"

    device = _resolve_torch_device(config.device)
    _validate_precision(config.precision, device)
    model.to(device)

    base_eval_suite = evaluate_eval_suite(model, eval_suite, batch_size=config.micro_batch_size)
    base_eval = base_eval_suite.selector_eval
    sample_examples = _sample_prompt_examples(eval_suite)
    base_generations = _generate_samples(
        model,
        tokenizer,
        sample_examples,
        config.sample_new_tokens,
    )
    optimizer, scheduler = build_full_finetune_optimizer(model, config)
    start_step = 0
    trained_tokens = 0
    supervised_tokens = 0
    resumed_from_checkpoint: Path | None = None
    if config.resume_from_latest:
        resumed = _resume_latest_checkpoint(output_dir, model, optimizer, scheduler, device)
        if resumed is not None:
            start_step = resumed.step
            resumed_from_checkpoint = latest_checkpoint_path(output_dir)
            totals = resumed.metrics.get("totals", {})
            trained_tokens = int(totals.get("trained_tokens", 0))
            supervised_tokens = int(totals.get("supervised_tokens", 0))
            restored_best = _load_best_checkpoint_state(
                best_checkpoint_path, best_checkpoint_metadata_path
            )

    wandb_run = _start_wandb(config, base_bundle_manifest)
    if resumed_from_checkpoint is None:
        # The base eval predates the checkpoint load; logging it on resume would
        # append a base-model row mislabeled with the resumed step.
        append_metric(
            metrics_path,
            base_eval_suite.to_metric_payload(step=start_step),
            wandb_run,
        )

    model.train()
    last_components: dict[str, float] = {}
    current_step = start_step
    last_completed_step = start_step
    if resumed_from_checkpoint is not None:
        best_selector_value = restored_best.selected_metric_value
        selected_step = restored_best.selected_step
        selected_eval_suite = restored_best.selected_eval_suite
        evals_without_improvement = restored_best.evals_without_improvement
    else:
        best_selector_value = float("inf")
        selected_step = start_step
        selected_eval_suite = base_eval_suite
        evals_without_improvement = 0
    last_eval_suite = base_eval_suite
    last_eval_step = start_step if start_step == 0 else -1
    early_stopped = False
    early_stop_reason: str | None = None
    no_robots_baseline = _split_response_loss(base_eval_suite, "no_robots")
    failure_checkpoint: Path | None = None
    failure_checkpoint_error: str | None = None
    try:
        for step in range(start_step + 1, config.max_steps + 1):
            current_step = step
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_tokens = 0
            step_supervised_tokens = 0
            step_correct = 0
            for accumulation_index in range(config.gradient_accumulation_steps):
                batch_index = (step - 1) * config.gradient_accumulation_steps + accumulation_index
                batch = _cyclic_batch(
                    train_examples,
                    batch_index=batch_index,
                    batch_size=config.micro_batch_size,
                )
                input_ids, labels = _collate_batch(
                    batch, device=device, pad_to_multiple_of=config.pad_to_multiple_of
                )
                with _precision_context(config.precision, device):
                    logits = model(input_ids[:, :-1])
                targets = labels[:, 1:]
                loss, components = language_model_loss(
                    logits,
                    targets,
                    z_loss_weight=model.config.z_loss_weight,
                    logit_soft_cap=model.config.logit_soft_cap,
                    ignore_index=IGNORE_INDEX,
                )
                if not torch.isfinite(loss):
                    raise TrainerError("SFT loss became NaN or inf")
                (loss / config.gradient_accumulation_steps).backward()
                micro_supervised = int((targets != IGNORE_INDEX).sum().item())
                step_loss += float(loss.detach()) * max(1, micro_supervised)
                step_tokens += sum(len(example.input_ids) for example in batch)
                step_supervised_tokens += micro_supervised
                step_correct += _token_correct(logits, targets)
                last_components = components
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            trained_tokens += step_tokens
            supervised_tokens += step_supervised_tokens
            last_completed_step = step
            if step_callback is not None:
                step_callback(step)
            if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
                loss_value = step_loss / max(1, step_supervised_tokens)
                append_metric(
                    metrics_path,
                    train_metric_payload(
                        step=step,
                        loss=loss_value,
                        learning_rate=float(scheduler.get_last_lr()[0]),
                        grad_norm=float(grad_norm_tensor.detach()),
                        tokens=step_tokens,
                        supervised_tokens=step_supervised_tokens,
                        token_accuracy=step_correct / max(1, step_supervised_tokens),
                        total_tokens=trained_tokens,
                        total_supervised_tokens=supervised_tokens,
                        components=last_components,
                    ),
                    wandb_run,
                )
            if config.eval_interval and (
                step % config.eval_interval == 0 or step == config.max_steps
            ):
                interval_eval = evaluate_eval_suite(
                    model, eval_suite, batch_size=config.micro_batch_size
                )
                append_metric(
                    metrics_path,
                    interval_eval.to_metric_payload(step=step),
                    wandb_run,
                )
                last_eval_suite = interval_eval
                last_eval_step = step
                if _no_robots_catastrophic_regression(
                    interval_eval,
                    baseline=no_robots_baseline,
                    multiplier=config.no_robots_catastrophic_regression_multiplier,
                ):
                    raise TrainerError("no_robots catastrophic regression threshold fired")
                if interval_eval.selector_response_loss < best_selector_value:
                    best_selector_value = interval_eval.selector_response_loss
                    selected_step = step
                    selected_eval_suite = interval_eval
                    evals_without_improvement = 0
                    _save_best_checkpoint(
                        best_checkpoint_path,
                        best_checkpoint_metadata_path,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        selected_metric=configured_selector_metric(eval_suite),
                        eval_suite=interval_eval,
                        trained_tokens=trained_tokens,
                        supervised_tokens=supervised_tokens,
                        evals_without_improvement=evals_without_improvement,
                        early_stopping_patience=config.early_stopping_patience,
                    )
                else:
                    evals_without_improvement += 1
                    _write_best_checkpoint_metadata(
                        best_checkpoint_metadata_path,
                        step=selected_step,
                        selected_metric=configured_selector_metric(eval_suite),
                        eval_suite=selected_eval_suite,
                        trained_tokens=trained_tokens,
                        supervised_tokens=supervised_tokens,
                        evals_without_improvement=evals_without_improvement,
                        early_stopping_patience=config.early_stopping_patience,
                    )
                if (
                    config.early_stopping_patience
                    and evals_without_improvement >= config.early_stopping_patience
                ):
                    early_stopped = True
                    early_stop_reason = (
                        f"{configured_selector_metric(eval_suite)} did not improve for "
                        f"{evals_without_improvement} evals"
                    )
                    break
                model.train()
            if config.checkpoint_interval and (
                step % config.checkpoint_interval == 0 or step == config.max_steps
            ):
                _save_periodic_checkpoint(
                    output_dir,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trained_tokens=trained_tokens,
                    supervised_tokens=supervised_tokens,
                    retain_last=config.retain_last_checkpoints,
                )
        model.eval()
        if last_eval_step != last_completed_step:
            last_eval_suite = evaluate_eval_suite(
                model, eval_suite, batch_size=config.micro_batch_size
            )
            append_metric(
                metrics_path,
                last_eval_suite.to_metric_payload(step=last_completed_step),
                wandb_run,
            )
            if _no_robots_catastrophic_regression(
                last_eval_suite,
                baseline=no_robots_baseline,
                multiplier=config.no_robots_catastrophic_regression_multiplier,
            ):
                raise TrainerError("no_robots catastrophic regression threshold fired")
    except KeyboardInterrupt:
        failure_checkpoint, failure_checkpoint_error = _save_failure_checkpoint(
            output_dir,
            step=last_completed_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trained_tokens=trained_tokens,
            supervised_tokens=supervised_tokens,
        )
        _write_failure_report(
            output_dir,
            step=last_completed_step,
            status="interrupted",
            message="training interrupted before completion",
            checkpoint_path=failure_checkpoint,
            checkpoint_error=failure_checkpoint_error,
        )
        raise
    except Exception as error:
        failure_checkpoint, failure_checkpoint_error = _save_failure_checkpoint(
            output_dir,
            step=last_completed_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trained_tokens=trained_tokens,
            supervised_tokens=supervised_tokens,
        )
        status = "non_finite_loss" if "NaN or inf" in str(error) else "failed"
        _write_failure_report(
            output_dir,
            step=current_step,
            status=status,
            message=str(error),
            checkpoint_path=failure_checkpoint,
            checkpoint_error=failure_checkpoint_error,
        )
        raise

    if last_eval_suite.selector_response_loss < best_selector_value:
        best_selector_value = last_eval_suite.selector_response_loss
        selected_step = last_completed_step
        selected_eval_suite = last_eval_suite
        evals_without_improvement = 0
        _save_best_checkpoint(
            best_checkpoint_path,
            best_checkpoint_metadata_path,
            step=selected_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            selected_metric=configured_selector_metric(eval_suite),
            eval_suite=selected_eval_suite,
            trained_tokens=trained_tokens,
            supervised_tokens=supervised_tokens,
            evals_without_improvement=evals_without_improvement,
            early_stopping_patience=config.early_stopping_patience,
        )
    if not best_checkpoint_path.is_file():
        _save_best_checkpoint(
            best_checkpoint_path,
            best_checkpoint_metadata_path,
            step=last_completed_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            selected_metric=configured_selector_metric(eval_suite),
            eval_suite=last_eval_suite,
            trained_tokens=trained_tokens,
            supervised_tokens=supervised_tokens,
            evals_without_improvement=evals_without_improvement,
            early_stopping_patience=config.early_stopping_patience,
        )
        selected_step = last_completed_step
        selected_eval_suite = last_eval_suite
        best_selector_value = last_eval_suite.selector_response_loss

    save_sft_checkpoint(
        checkpoint_path,
        model=model,
        step=last_completed_step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={
            "event": "final",
            "totals": {
                "trained_tokens": trained_tokens,
                "supervised_tokens": supervised_tokens,
            },
        },
    )
    loaded_best = load_training_checkpoint(best_checkpoint_path, map_location=device)
    model.load_state_dict(loaded_best.model.state_dict())
    instruct_eval = selected_eval_suite.selector_eval
    _write_samples(
        samples_path,
        model,
        tokenizer,
        sample_examples,
        config.sample_new_tokens,
        base_generations=base_generations,
        selected_step=selected_step,
    )
    _write_tokenizer(tokenizer, output_dir / "tokenizer.json")
    manifest = _write_manifest(
        manifest_path,
        config,
        model.config,
        base_eval,
        instruct_eval,
        base_eval_suite=base_eval_suite,
        selected_eval_suite=selected_eval_suite,
        trained_tokens=trained_tokens,
        supervised_tokens=supervised_tokens,
        selected_examples=len(train_examples),
        selected_train_tokens=sum(len(example.input_ids) for example in train_examples),
        eval_examples=sum(len(split.examples) for split in eval_suite),
        selected_step=selected_step,
        selected_metric_name=configured_selector_metric(eval_suite),
        selected_metric_value=best_selector_value,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        base_bundle_manifest=base_bundle_manifest,
        training_mode="resumed" if resumed_from_checkpoint is not None else "fresh",
        start_step=start_step,
        resumed_from_checkpoint=resumed_from_checkpoint,
    )
    wandb_run_url = getattr(wandb_run, "url", None) if wandb_run is not None else None
    if wandb_run is not None:
        wandb_run.log({"manifest": manifest})
        wandb_run.finish()
    instruct_beats_base = instruct_eval.response_loss < base_eval.response_loss
    return SFTTrainResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        samples_path=samples_path,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_metadata_path=best_checkpoint_metadata_path,
        base_eval=base_eval,
        instruct_eval=instruct_eval,
        base_eval_suite=base_eval_suite,
        selected_eval_suite=selected_eval_suite,
        steps_completed=last_completed_step,
        selected_step=selected_step,
        selected_metric_name=configured_selector_metric(eval_suite),
        selected_metric_value=best_selector_value,
        trained_tokens=trained_tokens,
        supervised_tokens=supervised_tokens,
        selected_examples=len(train_examples),
        eval_examples=sum(len(split.examples) for split in eval_suite),
        effective_epochs=trained_tokens
        / max(1, sum(len(example.input_ids) for example in train_examples)),
        response_loss_decreased=instruct_beats_base,
        instruct_beats_base=instruct_beats_base,
        training_mode="resumed" if resumed_from_checkpoint is not None else "fresh",
        start_step=start_step,
        early_stopped=early_stopped,
        early_stop_reason=early_stop_reason,
        resumed_from_checkpoint=str(resumed_from_checkpoint)
        if resumed_from_checkpoint is not None
        else None,
        wandb_run_url=wandb_run_url,
    )


def save_sft_checkpoint(
    path: Path,
    *,
    model: DenseBackbone,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    save_training_checkpoint(
        path,
        model=model,
        step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
    )


def _save_best_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    step: int,
    model: DenseBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    selected_metric: str,
    eval_suite: EvalSuiteResult,
    trained_tokens: int,
    supervised_tokens: int,
    evals_without_improvement: int,
    early_stopping_patience: int,
) -> None:
    metadata = _best_checkpoint_metadata(
        step=step,
        selected_metric=selected_metric,
        eval_suite=eval_suite,
        trained_tokens=trained_tokens,
        supervised_tokens=supervised_tokens,
        evals_without_improvement=evals_without_improvement,
        early_stopping_patience=early_stopping_patience,
    )
    save_sft_checkpoint(
        checkpoint_path,
        model=model,
        step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metadata,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _write_best_checkpoint_metadata(
    metadata_path: Path,
    *,
    step: int,
    selected_metric: str,
    eval_suite: EvalSuiteResult,
    trained_tokens: int,
    supervised_tokens: int,
    evals_without_improvement: int,
    early_stopping_patience: int,
) -> None:
    metadata = _best_checkpoint_metadata(
        step=step,
        selected_metric=selected_metric,
        eval_suite=eval_suite,
        trained_tokens=trained_tokens,
        supervised_tokens=supervised_tokens,
        evals_without_improvement=evals_without_improvement,
        early_stopping_patience=early_stopping_patience,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _best_checkpoint_metadata(
    *,
    step: int,
    selected_metric: str,
    eval_suite: EvalSuiteResult,
    trained_tokens: int,
    supervised_tokens: int,
    evals_without_improvement: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    return {
        "selected_metric": selected_metric,
        "selected_step": step,
        "selected_metric_value": eval_suite.selector_response_loss,
        "selected_eval_suite": eval_suite.to_dict(),
        "component_eval_losses": {
            name: metrics.response_loss for name, metrics in eval_suite.split_metrics.items()
        },
        "selector_weights": eval_suite.selector_weights,
        "patience_state": {
            "evals_without_improvement": evals_without_improvement,
            "early_stopping_patience": early_stopping_patience,
            "ordinary_early_stop_metric": selected_metric,
        },
        "totals": {
            "trained_tokens": trained_tokens,
            "supervised_tokens": supervised_tokens,
        },
    }


def load_sft_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> LoadedSFTCheckpoint:
    try:
        loaded = load_training_checkpoint(path, map_location=map_location)
    except ValueError as error:
        raise TrainerError(str(error)) from error
    return LoadedSFTCheckpoint(
        model=loaded.model,
        config=loaded.config,
        step=loaded.step,
        metrics=loaded.metrics,
    )


@dataclass(frozen=True)
class _BestCheckpointState:
    selected_metric_value: float
    selected_step: int
    selected_eval_suite: EvalSuiteResult
    evals_without_improvement: int


def _load_best_checkpoint_state(checkpoint_path: Path, metadata_path: Path) -> _BestCheckpointState:
    if not checkpoint_path.is_file():
        raise TrainerError(f"resume requires existing best checkpoint: {checkpoint_path}")
    if not metadata_path.is_file():
        raise TrainerError(f"resume requires existing best checkpoint metadata: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("metadata payload must be an object")
        patience_state = payload["patience_state"]
        if not isinstance(patience_state, dict):
            raise TypeError("patience_state must be an object")
        selected_eval_suite = payload["selected_eval_suite"]
        if not isinstance(selected_eval_suite, dict):
            raise TypeError("selected_eval_suite must be an object")
        return _BestCheckpointState(
            selected_metric_value=float(payload["selected_metric_value"]),
            selected_step=int(payload["selected_step"]),
            selected_eval_suite=EvalSuiteResult.from_dict(selected_eval_suite),
            evals_without_improvement=int(patience_state["evals_without_improvement"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TrainerError(
            f"best checkpoint metadata is malformed and cannot seed resume state: {metadata_path}"
        ) from error


def _resume_latest_checkpoint(
    output_dir: Path,
    model: DenseBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> LoadedTrainingCheckpoint | None:
    checkpoint = latest_checkpoint_path(output_dir)
    if checkpoint is None:
        return None
    loaded = load_training_checkpoint(checkpoint, map_location=device)
    if loaded.config != model.config:
        raise TrainerError("latest checkpoint model config does not match current model")
    model.load_state_dict(loaded.model.state_dict())
    if loaded.optimizer_state is not None:
        optimizer.load_state_dict(loaded.optimizer_state)
    if loaded.scheduler_state is not None:
        scheduler.load_state_dict(loaded.scheduler_state)
    return loaded


def _save_failure_checkpoint(
    output_dir: Path,
    *,
    step: int,
    model: DenseBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trained_tokens: int,
    supervised_tokens: int,
) -> tuple[Path | None, str | None]:
    path = checkpoint_dir(output_dir, step) / "checkpoint.pt"
    try:
        save_sft_checkpoint(
            path,
            model=model,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={
                "status": "failure_checkpoint",
                "totals": {
                    "trained_tokens": trained_tokens,
                    "supervised_tokens": supervised_tokens,
                },
            },
        )
    except Exception as error:  # pragma: no cover - only hit on storage failure.
        return None, str(error)
    return path, None


def _save_periodic_checkpoint(
    output_dir: Path,
    *,
    step: int,
    model: DenseBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trained_tokens: int,
    supervised_tokens: int,
    retain_last: int,
) -> None:
    path = checkpoint_dir(output_dir, step) / "checkpoint.pt"
    save_sft_checkpoint(
        path,
        model=model,
        step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={
            "totals": {
                "trained_tokens": trained_tokens,
                "supervised_tokens": supervised_tokens,
            }
        },
    )
    retain_last_checkpoints(output_dir, retain_last)


def _write_failure_report(
    output_dir: Path,
    *,
    step: int,
    status: str,
    message: str,
    checkpoint_path: Path | None,
    checkpoint_error: str | None,
) -> None:
    payload = {
        "status": status,
        "step": step,
        "message": message,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_error": checkpoint_error,
    }
    (output_dir / "failure-report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_samples(
    path: Path,
    model: DenseBackbone,
    tokenizer: Tokenizer,
    eval_examples: tuple[TokenizedExample, ...],
    sample_new_tokens: int,
    *,
    base_generations: tuple[str, ...],
    selected_step: int,
) -> None:
    instruct_generations = _generate_samples(model, tokenizer, eval_examples, sample_new_tokens)
    lines = ["# SFT Samples", "", f"Selected checkpoint step: {selected_step}", ""]
    for index, example in enumerate(eval_examples[:3], start=1):
        prompt_ids = example.input_ids[: example.prompt_tokens]
        prompt = tokenizer.decode(list(prompt_ids), skip_special_tokens=False)
        lines.extend(
            [
                f"## Sample {index}",
                "",
                "Prompt:",
                "",
                *_markdown_fenced_text(prompt),
                "",
                "Base generation:",
                "",
                *_markdown_fenced_text(base_generations[index - 1]),
                "",
                "Selected Instruct generation:",
                "",
                *_markdown_fenced_text(instruct_generations[index - 1]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tokenizer.save(str(tmp_path))
    tmp_path.replace(path)


def _write_manifest(
    path: Path,
    config: SFTTrainerConfig,
    model_config: BackboneConfig,
    base_eval: EvalMetrics,
    instruct_eval: EvalMetrics,
    *,
    base_eval_suite: EvalSuiteResult,
    selected_eval_suite: EvalSuiteResult,
    trained_tokens: int,
    supervised_tokens: int,
    selected_examples: int,
    selected_train_tokens: int,
    eval_examples: int,
    selected_step: int,
    selected_metric_name: str,
    selected_metric_value: float,
    early_stopped: bool,
    early_stop_reason: str | None,
    base_bundle_manifest: dict[str, Any] | None,
    training_mode: str,
    start_step: int,
    resumed_from_checkpoint: Path | None,
) -> dict[str, Any]:
    output_dir = path.parent
    files = {}
    for name in (
        "checkpoint.pt",
        "best-checkpoint.pt",
        "best-checkpoint.json",
        "metrics.jsonl",
        "samples.md",
        "tokenizer.json",
    ):
        file_path = output_dir / name
        files[name] = {"path": name, "sha256": file_sha256(file_path)}
    manifest = {
        "schema_version": 1,
        "format": "llm_posttrain_instruct_sft_v1",
        "artifact_name": config.artifact_name,
        "base_artifact_name": config.base_artifact_name,
        "model_family": "DenseBackbone",
        "model_config": model_config.to_dict(),
        "trainer": {
            "max_steps": config.max_steps,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
            "scheduler": config.scheduler,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "precision": config.precision,
            "pad_to_multiple_of": config.pad_to_multiple_of,
            "tuning_mode": config.tuning_mode,
            "assistant_only_loss": config.assistant_only_loss,
            "completion_only_loss": config.completion_only_loss,
            "seed": config.seed,
            "device": config.device,
            "checkpoint_interval": config.checkpoint_interval,
            "retain_last_checkpoints": config.retain_last_checkpoints,
            "early_stopping_patience": config.early_stopping_patience,
            "no_robots_catastrophic_regression_multiplier": (
                config.no_robots_catastrophic_regression_multiplier
            ),
            "training_mode": training_mode,
            "start_step": start_step,
            "resumed_from_checkpoint": str(resumed_from_checkpoint)
            if resumed_from_checkpoint is not None
            else None,
        },
        "token_accounting": {
            "trained_tokens": trained_tokens,
            "supervised_tokens": supervised_tokens,
            "selected_examples": selected_examples,
            "eval_examples": eval_examples,
            "effective_epochs": trained_tokens / max(1, selected_train_tokens),
        },
        "base_bundle": base_bundle_manifest,
        "eval": {
            "base": base_eval.to_dict(),
            "instruct": instruct_eval.to_dict(),
            "base_suite": base_eval_suite.to_dict(),
            "selected_suite": selected_eval_suite.to_dict(),
            "selected_metric": selected_metric_name,
            "selected_metric_value": selected_metric_value,
            "selected_step": selected_step,
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "instruct_beats_base": instruct_eval.response_loss < base_eval.response_loss,
            "acceptance_rule": "weighted matched response loss must be lower than Base",
            "no_robots_role": "OOD guardrail/reporting only; not the ordinary selector",
        },
        "files": files,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
