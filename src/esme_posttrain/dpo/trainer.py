"""Vanilla offline DPO trainer for the Esme-214M-Chat polish stage.

This implements the same vanilla DPO objective TRL's ``DPOTrainer`` ships with
``loss_type="sigmoid"`` (Rafailov et al. 2023), plus the length-normalized
variant as a config flag — but on this repo's native ``DenseBackbone`` model and
``tokenizers.Tokenizer``, reusing the SFT collate/checkpoint/scheduler/metric
infra. TRL is not used because it requires a HuggingFace ``PreTrainedModel`` and
``AutoTokenizer``; the from-scratch Esme stack is neither, and the project keeps
sibling repos artifact-only with minimal runtime dependencies.

Loss (per pair):

    pi_logratio  = logp_pi(chosen)  - logp_pi(rejected)
    ref_logratio = logp_ref(chosen) - logp_ref(rejected)
    logits       = pi_logratio - ref_logratio
    loss         = -log_sigmoid(beta * logits)

``logp`` is the summed token log-probability of the response span (prompt masked).
With ``length_normalized=True`` each response logp is divided by its supervised
token count before the difference, the dominant mitigation for verbosity
reward-hacking. The reference model is a frozen copy of the SFT foundation.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.bundle import file_sha256
from esme_posttrain.modeling import DenseBackbone, soft_cap_logits
from esme_posttrain.training.checkpointing import (
    capture_rng_state,
    latest_checkpoint_path,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from esme_posttrain.training.collate import IGNORE_INDEX, collate_batch, cyclic_batch
from esme_posttrain.training.errors import TrainerError
from esme_posttrain.training.metrics import append_metric
from esme_posttrain.training.runtime import (
    lr_lambda_factory,
    precision_context,
    resolve_torch_device,
    set_reproducible_seed,
    validate_precision,
)
from esme_posttrain.training.wandb_init import (
    WandbConfig,
    start_wandb,
)

# Likelihood displacement (Razin et al., arXiv:2410.08847) is a *substantial* fall
# in the chosen response's log-prob during DPO, not the sub-noise jitter that any
# eval set shows step to step. A zero-tolerance `selected < base` guard flags eval
# noise as failure. Eval jitter around 0.03% is below the collapse threshold.
#
# `chosen_logp` is a *summed* response-token log-prob, so its magnitude scales with
# response length; a fixed-nats threshold would not generalize across response
# lengths. A relative band does: collapse fires only when the drop exceeds this
# fraction of |base_chosen_logp|. 1% sits well above eval jitter (~0.03%) and well
# below a genuine displacement event.
CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE = 0.01


def is_chosen_logp_collapsed(
    base_chosen_logp: float,
    selected_chosen_logp: float,
    *,
    rel_tolerance: float = CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE,
) -> bool:
    """True only when the chosen log-prob falls by more than ``rel_tolerance`` of
    its baseline magnitude — meaningful likelihood displacement, not eval noise.

    The drop is measured relative to ``|base_chosen_logp|`` so the threshold tracks
    response length (the log-prob is a sum over response tokens). A rise or a
    sub-tolerance dip is never a collapse.
    """
    drop = base_chosen_logp - selected_chosen_logp
    if drop <= 0:
        return False
    return drop > rel_tolerance * abs(base_chosen_logp)


class DPOTrainerError(TrainerError):
    pass


StepCallback = Callable[[int], None]


@dataclass(frozen=True)
class DPOTrainerConfig:
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    beta: float
    seed: int
    output_dir: Path
    length_normalized: bool = False
    scheduler: str = "cosine_decay"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    precision: str = "fp32"
    pad_to_multiple_of: int | None = None
    grad_clip: float = 1.0
    artifact_name: str = "Esme-214M-Chat"
    reference_artifact_name: str = "Esme-214M-Instruct"
    log_interval: int = 1
    eval_interval: int = 0
    checkpoint_interval: int = 0
    resume_from_latest: bool = False
    device: str = "cpu"
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "beta",
            "grad_clip",
            "log_interval",
        ):
            if getattr(self, field_name) <= 0:
                raise DPOTrainerError(f"{field_name} must be positive")
        if self.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
            raise DPOTrainerError(
                "scheduler must be constant, linear_warmup_decay, or cosine_decay"
            )
        for field_name in ("warmup_steps",):
            if getattr(self, field_name) < 0:
                raise DPOTrainerError(f"{field_name} must be non-negative")
        if self.warmup_steps > self.max_steps:
            raise DPOTrainerError("warmup_steps must be <= max_steps")
        if self.precision not in {"fp32", "bf16"}:
            raise DPOTrainerError("precision must be fp32 or bf16")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class DPOEvalMetrics:
    """DPO held-out preference metrics; the primary cheap proxy for selection."""

    preference_accuracy: float
    margin: float
    chosen_logp: float
    rejected_logp: float
    loss: float
    pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference_accuracy": self.preference_accuracy,
            "margin": self.margin,
            "chosen_logp": self.chosen_logp,
            "rejected_logp": self.rejected_logp,
            "loss": self.loss,
            "pairs": self.pairs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DPOEvalMetrics:
        return cls(
            preference_accuracy=float(payload["preference_accuracy"]),
            margin=float(payload["margin"]),
            chosen_logp=float(payload["chosen_logp"]),
            rejected_logp=float(payload["rejected_logp"]),
            loss=float(payload["loss"]),
            pairs=int(payload["pairs"]),
        )


@dataclass(frozen=True)
class DPOTrainResult:
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    best_checkpoint_metadata_path: Path
    metrics_path: Path
    manifest_path: Path
    base_eval: DPOEvalMetrics
    selected_eval: DPOEvalMetrics
    steps_completed: int
    selected_step: int
    selected_metric_name: str
    selected_metric_value: float
    margin_increased: bool
    chosen_logp_collapsed: bool
    base_chosen_logp: float
    selected_chosen_logp: float
    start_step: int = 0
    resumed_from_checkpoint: str | None = None
    wandb_run_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "best_checkpoint_metadata_path": str(self.best_checkpoint_metadata_path),
            "metrics_path": str(self.metrics_path),
            "manifest_path": str(self.manifest_path),
            "base_eval": self.base_eval.to_dict(),
            "selected_eval": self.selected_eval.to_dict(),
            "steps_completed": self.steps_completed,
            "selected_step": self.selected_step,
            "selected_metric_name": self.selected_metric_name,
            "selected_metric_value": self.selected_metric_value,
            "margin_increased": self.margin_increased,
            "chosen_logp_collapsed": self.chosen_logp_collapsed,
            "base_chosen_logp": self.base_chosen_logp,
            "selected_chosen_logp": self.selected_chosen_logp,
            "start_step": self.start_step,
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "wandb_run_url": self.wandb_run_url,
        }


def sequence_logprob(
    model: DenseBackbone,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    length_normalized: bool,
) -> torch.Tensor:
    """Summed (or mean, if length-normalized) response-token log-prob per row.

    Prompt tokens are masked with ``IGNORE_INDEX`` in ``labels`` and excluded.
    Matches the next-token shift the SFT trainer uses (predict ``labels[:, 1:]``
    from ``input_ids[:, :-1]``). Returns shape ``[batch]``.
    """
    logits = model(input_ids[:, :-1])
    targets = labels[:, 1:]
    capped = soft_cap_logits(logits.float(), model.config.logit_soft_cap)
    log_probs = torch.log_softmax(capped, dim=-1)
    mask = targets != IGNORE_INDEX
    gather_targets = targets.clamp_min(0).unsqueeze(-1)
    token_logp = log_probs.gather(-1, gather_targets).squeeze(-1)
    token_logp = token_logp * mask
    summed = token_logp.sum(dim=-1)
    if length_normalized:
        token_counts = mask.sum(dim=-1).clamp_min(1)
        return summed / token_counts
    return summed


def dpo_pair_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    reference_chosen_logp: torch.Tensor,
    reference_rejected_logp: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla DPO sigmoid loss + the implicit reward margin, per pair.

    Returns ``(loss_per_pair, margin_per_pair)`` where ``margin`` is the
    difference of implicit rewards (``beta * (pi_logratio - ref_logratio)``); a
    positive margin means the policy prefers chosen over rejected more than the
    reference does.
    """
    pi_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = reference_chosen_logp - reference_rejected_logp
    logits = pi_logratio - ref_logratio
    loss = -torch.nn.functional.logsigmoid(beta * logits)
    margin = beta * logits
    return loss, margin


def run_dpo_training(
    policy: DenseBackbone,
    reference: DenseBackbone,
    tokenizer: Tokenizer,
    train_pairs: tuple[Any, ...],
    eval_pairs: tuple[Any, ...],
    config: DPOTrainerConfig,
    *,
    reference_bundle_manifest: dict[str, Any] | None = None,
    step_callback: StepCallback | None = None,
) -> DPOTrainResult:
    if not train_pairs:
        raise DPOTrainerError("train_pairs must not be empty")
    if not eval_pairs:
        raise DPOTrainerError("eval_pairs must not be empty")
    if policy.config != reference.config:
        raise DPOTrainerError("policy and reference must share the same model config")
    set_reproducible_seed(config.seed)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    best_checkpoint_path = output_dir / "best-checkpoint.pt"
    best_checkpoint_metadata_path = output_dir / "best-checkpoint.json"
    manifest_path = output_dir / "manifest.json"

    device = resolve_torch_device(config.device)
    validate_precision(config.precision, device)
    policy.to(device)
    reference.to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    base_eval = evaluate_preferences(
        policy,
        reference,
        eval_pairs,
        beta=config.beta,
        length_normalized=config.length_normalized,
        batch_size=config.micro_batch_size,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_factory(
            scheduler=config.scheduler,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        ),
    )

    start_step = 0
    resumed_from_checkpoint: Path | None = None
    if config.resume_from_latest:
        resumed = _resume_latest_checkpoint(output_dir, policy, optimizer, scheduler, device)
        if resumed is not None:
            start_step = resumed.step
            resumed_from_checkpoint = latest_checkpoint_path(output_dir)

    wandb_run = start_wandb(
        config.wandb,
        run_config={
            "artifact_name": config.artifact_name,
            "reference_artifact_name": config.reference_artifact_name,
            "beta": config.beta,
            "length_normalized": config.length_normalized,
            "max_steps": config.max_steps,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
            "scheduler": config.scheduler,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "precision": config.precision,
            "seed": config.seed,
        },
        base_bundle_manifest=reference_bundle_manifest,
    )
    if resumed_from_checkpoint is None:
        append_metric(metrics_path, _dpo_eval_payload(step=0, metrics=base_eval), wandb_run)

    if resumed_from_checkpoint is not None and best_checkpoint_metadata_path.is_file():
        restored_best = _load_best_checkpoint_state(
            best_checkpoint_path, best_checkpoint_metadata_path
        )
        best_selector_value = restored_best.selected_metric_value
        selected_step = restored_best.selected_step
        selected_eval = restored_best.selected_eval
    else:
        best_selector_value = -float("inf")
        selected_step = start_step
        selected_eval = base_eval
    last_completed_step = start_step

    policy.train()
    for step in range(start_step + 1, config.max_steps + 1):
        last_completed_step = step
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_margin = 0.0
        step_correct = 0
        step_pairs = 0
        for accumulation_index in range(config.gradient_accumulation_steps):
            batch_index = (step - 1) * config.gradient_accumulation_steps + accumulation_index
            batch = cyclic_batch(
                train_pairs, batch_index=batch_index, batch_size=config.micro_batch_size
            )
            loss, margin, correct = _forward_batch(
                policy,
                reference,
                batch,
                config=config,
                device=device,
            )
            (loss / config.gradient_accumulation_steps).backward()
            step_loss += float(loss.detach()) * len(batch)
            step_margin += float(margin.detach())
            step_correct += correct
            step_pairs += len(batch)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        if step_callback is not None:
            step_callback(step)
        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            append_metric(
                metrics_path,
                {
                    "event": "train",
                    "step": step,
                    "train/loss": step_loss / max(1, step_pairs),
                    "train/margin": step_margin / max(1, step_pairs),
                    "train/preference_accuracy": step_correct / max(1, step_pairs),
                    "train/learning_rate": float(scheduler.get_last_lr()[0]),
                    "train/grad_norm": float(grad_norm.detach()),
                    "train/pairs": step_pairs,
                },
                wandb_run,
            )
        if config.eval_interval and (step % config.eval_interval == 0 or step == config.max_steps):
            interval_eval = evaluate_preferences(
                policy,
                reference,
                eval_pairs,
                beta=config.beta,
                length_normalized=config.length_normalized,
                batch_size=config.micro_batch_size,
                pad_to_multiple_of=config.pad_to_multiple_of,
            )
            append_metric(
                metrics_path, _dpo_eval_payload(step=step, metrics=interval_eval), wandb_run
            )
            if interval_eval.preference_accuracy > best_selector_value:
                best_selector_value = interval_eval.preference_accuracy
                selected_step = step
                selected_eval = interval_eval
                _save_best_checkpoint(
                    best_checkpoint_path,
                    best_checkpoint_metadata_path,
                    step=step,
                    model=policy,
                    metrics=interval_eval,
                    selected_metric="eval/preference_accuracy",
                )
            policy.train()
        if config.checkpoint_interval and (
            step % config.checkpoint_interval == 0 or step == config.max_steps
        ):
            save_training_checkpoint(
                output_dir / "checkpoints" / f"step-{step:06d}" / "checkpoint.pt",
                model=policy,
                step=step,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"event": "periodic"},
                rng_state=capture_rng_state(),
                data_position=step * config.gradient_accumulation_steps,
            )

    policy.eval()
    if not best_checkpoint_path.is_file():
        # No interval eval fired (e.g. eval_interval=0); select the final state.
        final_eval = evaluate_preferences(
            policy,
            reference,
            eval_pairs,
            beta=config.beta,
            length_normalized=config.length_normalized,
            batch_size=config.micro_batch_size,
            pad_to_multiple_of=config.pad_to_multiple_of,
        )
        append_metric(
            metrics_path, _dpo_eval_payload(step=last_completed_step, metrics=final_eval), wandb_run
        )
        selected_step = last_completed_step
        selected_eval = final_eval
        best_selector_value = final_eval.preference_accuracy
        _save_best_checkpoint(
            best_checkpoint_path,
            best_checkpoint_metadata_path,
            step=last_completed_step,
            model=policy,
            metrics=final_eval,
            selected_metric="eval/preference_accuracy",
        )

    save_training_checkpoint(
        checkpoint_path,
        model=policy,
        step=last_completed_step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics={"event": "final"},
        rng_state=capture_rng_state(),
        data_position=last_completed_step * config.gradient_accumulation_steps,
    )
    loaded_best = load_training_checkpoint(best_checkpoint_path, map_location=device)
    policy.load_state_dict(loaded_best.model.state_dict())
    _write_tokenizer(tokenizer, output_dir / "tokenizer.json")

    chosen_logp_collapsed = is_chosen_logp_collapsed(
        base_eval.chosen_logp, selected_eval.chosen_logp
    )
    manifest = _write_manifest(
        manifest_path,
        config,
        base_eval=base_eval,
        selected_eval=selected_eval,
        selected_step=selected_step,
        selected_metric_value=best_selector_value,
        reference_bundle_manifest=reference_bundle_manifest,
        chosen_logp_collapsed=chosen_logp_collapsed,
    )
    wandb_run_url = getattr(wandb_run, "url", None) if wandb_run is not None else None
    if wandb_run is not None:
        wandb_run.log({"manifest": manifest})
        wandb_run.finish()
    return DPOTrainResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_metadata_path=best_checkpoint_metadata_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        base_eval=base_eval,
        selected_eval=selected_eval,
        steps_completed=last_completed_step,
        selected_step=selected_step,
        selected_metric_name="eval/preference_accuracy",
        selected_metric_value=best_selector_value,
        margin_increased=selected_eval.margin > base_eval.margin,
        chosen_logp_collapsed=chosen_logp_collapsed,
        base_chosen_logp=base_eval.chosen_logp,
        selected_chosen_logp=selected_eval.chosen_logp,
        start_step=start_step,
        resumed_from_checkpoint=str(resumed_from_checkpoint)
        if resumed_from_checkpoint is not None
        else None,
        wandb_run_url=wandb_run_url,
    )


def _forward_batch(
    policy: DenseBackbone,
    reference: DenseBackbone,
    batch: tuple[Any, ...],
    *,
    config: DPOTrainerConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    chosen_ids, chosen_labels = collate_batch(
        tuple(_completion_as_collate_row(pair.chosen) for pair in batch),
        device=device,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )
    rejected_ids, rejected_labels = collate_batch(
        tuple(_completion_as_collate_row(pair.rejected) for pair in batch),
        device=device,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )
    with precision_context(config.precision, device):
        policy_chosen = sequence_logprob(
            policy, chosen_ids, chosen_labels, length_normalized=config.length_normalized
        )
        policy_rejected = sequence_logprob(
            policy, rejected_ids, rejected_labels, length_normalized=config.length_normalized
        )
    with torch.no_grad():
        reference_chosen = sequence_logprob(
            reference, chosen_ids, chosen_labels, length_normalized=config.length_normalized
        )
        reference_rejected = sequence_logprob(
            reference, rejected_ids, rejected_labels, length_normalized=config.length_normalized
        )
    loss_per_pair, margin_per_pair = dpo_pair_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=config.beta,
    )
    if not torch.isfinite(loss_per_pair).all():
        raise DPOTrainerError("DPO loss became NaN or inf")
    loss = loss_per_pair.mean()
    correct = int((margin_per_pair > 0).sum().item())
    return loss, margin_per_pair.sum(), correct


def evaluate_preferences(
    policy: DenseBackbone,
    reference: DenseBackbone,
    eval_pairs: tuple[Any, ...],
    *,
    beta: float,
    length_normalized: bool,
    batch_size: int,
    pad_to_multiple_of: int | None = None,
) -> DPOEvalMetrics:
    if not eval_pairs:
        raise DPOTrainerError("eval_pairs must not be empty")
    if batch_size <= 0:
        raise DPOTrainerError("batch_size must be positive")
    was_training = policy.training
    policy.eval()
    device = next(policy.parameters()).device
    total_loss = 0.0
    total_margin = 0.0
    total_chosen = 0.0
    total_rejected = 0.0
    correct = 0
    pairs = 0
    with torch.no_grad():
        for start in range(0, len(eval_pairs), batch_size):
            batch = eval_pairs[start : start + batch_size]
            chosen_ids, chosen_labels = collate_batch(
                tuple(_completion_as_collate_row(pair.chosen) for pair in batch),
                device=device,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            rejected_ids, rejected_labels = collate_batch(
                tuple(_completion_as_collate_row(pair.rejected) for pair in batch),
                device=device,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            policy_chosen = sequence_logprob(
                policy, chosen_ids, chosen_labels, length_normalized=length_normalized
            )
            policy_rejected = sequence_logprob(
                policy, rejected_ids, rejected_labels, length_normalized=length_normalized
            )
            reference_chosen = sequence_logprob(
                reference, chosen_ids, chosen_labels, length_normalized=length_normalized
            )
            reference_rejected = sequence_logprob(
                reference, rejected_ids, rejected_labels, length_normalized=length_normalized
            )
            loss_per_pair, margin_per_pair = dpo_pair_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=beta,
            )
            total_loss += float(loss_per_pair.sum().detach())
            total_margin += float(margin_per_pair.sum().detach())
            total_chosen += float(policy_chosen.sum().detach())
            total_rejected += float(policy_rejected.sum().detach())
            correct += int((margin_per_pair > 0).sum().item())
            pairs += len(batch)
    if was_training:
        policy.train()
    return DPOEvalMetrics(
        preference_accuracy=correct / max(1, pairs),
        margin=total_margin / max(1, pairs),
        chosen_logp=total_chosen / max(1, pairs),
        rejected_logp=total_rejected / max(1, pairs),
        loss=total_loss / max(1, pairs),
        pairs=pairs,
    )


def _completion_as_collate_row(completion: Any) -> Any:
    """Adapt a TokenizedCompletion to the (input_ids, labels) shape collate_batch wants.

    ``collate_batch`` reads ``.input_ids``, ``.labels``, ``.source``, ``.row_id``;
    TokenizedCompletion has the first two, so wrap it with the diagnostic fields.
    """
    return _DPOCompletionCollateRow(
        input_ids=completion.input_ids,
        labels=completion.labels,
    )


@dataclass(frozen=True)
class _DPOCompletionCollateRow:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    source: str = "dpo"
    row_id: str = "pair"


def _dpo_eval_payload(*, step: int, metrics: DPOEvalMetrics) -> dict[str, Any]:
    return {
        "event": "eval",
        "step": step,
        "split": "preference_heldout",
        "eval/preference_accuracy": metrics.preference_accuracy,
        "eval/margin": metrics.margin,
        "eval/chosen_logp": metrics.chosen_logp,
        "eval/rejected_logp": metrics.rejected_logp,
        "eval/loss": metrics.loss,
        "eval/pairs": metrics.pairs,
    }


def _save_best_checkpoint(
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    step: int,
    model: DenseBackbone,
    metrics: DPOEvalMetrics,
    selected_metric: str,
) -> None:
    metadata = {
        "selected_metric": selected_metric,
        "selected_step": step,
        "selected_metric_value": metrics.preference_accuracy,
        "eval": metrics.to_dict(),
    }
    save_training_checkpoint(checkpoint_path, model=model, step=step, metrics=metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _resume_latest_checkpoint(
    output_dir: Path,
    policy: DenseBackbone,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> Any | None:
    checkpoint = latest_checkpoint_path(output_dir)
    if checkpoint is None:
        return None
    loaded = load_training_checkpoint(checkpoint, map_location=device)
    if loaded.config != policy.config:
        raise DPOTrainerError("latest checkpoint model config does not match current policy")
    if loaded.optimizer_state is None or loaded.scheduler_state is None:
        raise DPOTrainerError(
            f"checkpoint lacks optimizer/scheduler state and cannot seed a faithful "
            f"resume: {checkpoint}"
        )
    policy.load_state_dict(loaded.model.state_dict())
    optimizer.load_state_dict(loaded.optimizer_state)
    scheduler.load_state_dict(loaded.scheduler_state)
    # Pre-v3 checkpoints carry no RNG state; restore is a no-op for them.
    restore_rng_state(loaded.rng_state)
    return loaded


@dataclass(frozen=True)
class _BestCheckpointState:
    selected_metric_value: float
    selected_step: int
    selected_eval: DPOEvalMetrics


def _load_best_checkpoint_state(checkpoint_path: Path, metadata_path: Path) -> _BestCheckpointState:
    if not checkpoint_path.is_file():
        raise DPOTrainerError(f"resume requires existing best checkpoint: {checkpoint_path}")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("metadata payload must be an object")
        eval_payload = payload["eval"]
        if not isinstance(eval_payload, dict):
            raise TypeError("eval must be an object")
        return _BestCheckpointState(
            selected_metric_value=float(payload["selected_metric_value"]),
            selected_step=int(payload["selected_step"]),
            selected_eval=DPOEvalMetrics.from_dict(eval_payload),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DPOTrainerError(
            f"best checkpoint metadata is malformed and cannot seed resume state: {metadata_path}"
        ) from error


def _write_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tokenizer.save(str(tmp_path))
    tmp_path.replace(path)


def _write_manifest(
    path: Path,
    config: DPOTrainerConfig,
    *,
    base_eval: DPOEvalMetrics,
    selected_eval: DPOEvalMetrics,
    selected_step: int,
    selected_metric_value: float,
    reference_bundle_manifest: dict[str, Any] | None,
    chosen_logp_collapsed: bool,
) -> dict[str, Any]:
    output_dir = path.parent
    files = {}
    for name in (
        "checkpoint.pt",
        "best-checkpoint.pt",
        "best-checkpoint.json",
        "metrics.jsonl",
        "tokenizer.json",
    ):
        file_path = output_dir / name
        if file_path.is_file():
            files[name] = {"path": name, "sha256": file_sha256(file_path)}
    manifest = {
        "schema_version": 1,
        "format": "llm_posttrain_dpo_v1",
        "artifact_name": config.artifact_name,
        "reference_artifact_name": config.reference_artifact_name,
        "model_family": "DenseBackbone",
        "trainer": {
            "method": "vanilla_dpo_sigmoid",
            "beta": config.beta,
            "length_normalized": config.length_normalized,
            "max_steps": config.max_steps,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
            "scheduler": config.scheduler,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "precision": config.precision,
            "seed": config.seed,
            "device": config.device,
            "frozen_reference": True,
        },
        "reference_bundle": reference_bundle_manifest,
        "eval": {
            "base": base_eval.to_dict(),
            "selected": selected_eval.to_dict(),
            "selected_metric": "eval/preference_accuracy",
            "selected_metric_value": selected_metric_value,
            "selected_step": selected_step,
            "margin_increased": selected_eval.margin > base_eval.margin,
            "chosen_logp_collapsed": chosen_logp_collapsed,
            "chosen_logp_collapse_rel_tolerance": CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE,
            "likelihood_displacement_note": (
                "chosen_logp_collapsed=true means the chosen response's log-prob fell by "
                f"more than {CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE:.0%} of its baseline "
                "magnitude (meaningful likelihood displacement); a rise or a sub-tolerance "
                "dip is eval noise and does not trip it"
            ),
            "acceptance_rule": (
                "held-out preference accuracy and margin increase versus step 0 without "
                f"chosen-logp collapse (a >{CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE:.0%} relative "
                "drop in chosen log-prob)"
            ),
        },
        "environment": _environment_lines(),
        "files": files,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _environment_lines() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
