"""Minimal Countdown-Lite GRPO trainer for the Esme native DenseBackbone stack."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.bundle import BUNDLE_FORMAT, file_sha256
from esme_posttrain.modeling import DenseBackbone, soft_cap_logits
from esme_posttrain.rl.countdown_lite import (
    render_chat_prompt,
    verify_countdown_lite_expression,
)
from esme_posttrain.run_artifacts import write_json
from esme_posttrain.training.checkpointing import save_training_checkpoint
from esme_posttrain.training.collate import IGNORE_INDEX
from esme_posttrain.training.metrics import append_metric
from esme_posttrain.training.runtime import (
    lr_lambda_factory,
    precision_context,
    resolve_torch_device,
    set_reproducible_seed,
    validate_precision,
)

StepCallback = Callable[[int], None]

# Exact solves must clear the closeness ceiling (valid + closeness_weight) by at
# least this margin so near-miss farming can never rival solving the task.
EXACT_SOLVE_REWARD_MARGIN = 0.4


class CountdownGRPOTrainerError(ValueError):
    pass


@dataclass(frozen=True)
class CountdownGRPOTrainerConfig:
    max_steps: int
    prompts_per_step: int
    group_size: int
    max_new_tokens: int
    temperature: float
    kl_beta: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    scheduler: str
    grad_clip: float
    seed: int
    output_dir: Path
    max_rollout_tokens: int
    exact_solve_reward: float = 1.0
    valid_expression_reward: float = 0.1
    invalid_reward: float = 0.0
    format_expression_reward: float = 0.0
    closeness_weight: float = 0.0
    zero_variance_max_resamples: int = 0
    replay_buffer_max_age_steps: int = 0
    stratified_difficulty_sampling: bool = False
    write_final_bundle: bool = False
    precision: str = "fp32"
    device: str = "cpu"
    log_interval: int = 1
    checkpoint_interval: int = 0
    artifact_name: str = "Esme-214M-RL"
    reference_artifact_name: str = "Esme-214M-Chat"
    bundle_model_id: str = "esme-214m-rl"
    pad_token_id: int = 0
    source_manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "prompts_per_step",
            "group_size",
            "max_new_tokens",
            "learning_rate",
            "grad_clip",
            "seed",
            "log_interval",
            "max_rollout_tokens",
        ):
            if getattr(self, field_name) <= 0:
                raise CountdownGRPOTrainerError(f"{field_name} must be positive")
        if self.temperature <= 0:
            raise CountdownGRPOTrainerError("temperature must be positive")
        if self.kl_beta < 0:
            raise CountdownGRPOTrainerError("kl_beta must be non-negative")
        if self.weight_decay < 0:
            raise CountdownGRPOTrainerError("weight_decay must be non-negative")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise CountdownGRPOTrainerError("warmup_steps must be between 0 and max_steps")
        if self.scheduler not in {"constant", "linear_warmup_decay", "cosine_decay"}:
            raise CountdownGRPOTrainerError(
                "scheduler must be constant, linear_warmup_decay, or cosine_decay"
            )
        if self.precision not in {"fp32", "bf16"}:
            raise CountdownGRPOTrainerError("precision must be fp32 or bf16")
        if not (
            self.exact_solve_reward
            > self.valid_expression_reward
            >= self.format_expression_reward
            >= self.invalid_reward
        ):
            raise CountdownGRPOTrainerError(
                "reward order must be exact_solve > valid_expression >= "
                "format_expression >= invalid"
            )
        if self.closeness_weight < 0:
            raise CountdownGRPOTrainerError("closeness_weight must be non-negative")
        closeness_ceiling = self.valid_expression_reward + self.closeness_weight
        if self.exact_solve_reward < closeness_ceiling + EXACT_SOLVE_REWARD_MARGIN:
            raise CountdownGRPOTrainerError(
                "exact_solve_reward must exceed the closeness ceiling "
                f"(valid_expression_reward + closeness_weight) by at least "
                f"{EXACT_SOLVE_REWARD_MARGIN}"
            )
        if self.zero_variance_max_resamples < 0:
            raise CountdownGRPOTrainerError("zero_variance_max_resamples must be non-negative")
        if self.replay_buffer_max_age_steps < 0:
            raise CountdownGRPOTrainerError("replay_buffer_max_age_steps must be non-negative")


@dataclass(frozen=True)
class CountdownGRPOResult:
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    best_checkpoint_metadata_path: Path
    metrics_path: Path
    rollout_path: Path
    manifest_path: Path
    bundle_dir: Path
    bundle_final_dir: Path | None
    steps_completed: int
    selected_step: int
    selected_metric_name: str
    selected_metric_value: float
    rollout_tokens: int
    exact_rollouts: int
    valid_rollouts: int
    total_rollouts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "best_checkpoint_metadata_path": str(self.best_checkpoint_metadata_path),
            "metrics_path": str(self.metrics_path),
            "rollout_path": str(self.rollout_path),
            "manifest_path": str(self.manifest_path),
            "bundle_dir": str(self.bundle_dir),
            "bundle_final_dir": str(self.bundle_final_dir)
            if self.bundle_final_dir is not None
            else None,
            "steps_completed": self.steps_completed,
            "selected_step": self.selected_step,
            "selected_metric_name": self.selected_metric_name,
            "selected_metric_value": self.selected_metric_value,
            "rollout_tokens": self.rollout_tokens,
            "exact_rollouts": self.exact_rollouts,
            "valid_rollouts": self.valid_rollouts,
            "total_rollouts": self.total_rollouts,
        }


@dataclass(frozen=True)
class _Rollout:
    task_id: str
    difficulty: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    output: str
    extracted_expression: str | None
    reward: float
    is_valid_expression: bool
    is_exact_solve: bool
    is_well_formed: bool
    reason: str
    value: int | None
    group_index: int
    replayed: bool = False

    @property
    def token_count(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)

    def to_json(self, *, step: int) -> dict[str, Any]:
        return {
            "step": step,
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "output": self.output,
            "extracted_expression": self.extracted_expression,
            "reward": self.reward,
            "is_valid_expression": self.is_valid_expression,
            "is_exact_solve": self.is_exact_solve,
            "is_well_formed": self.is_well_formed,
            "reason": self.reason,
            "value": self.value,
            "completion_tokens": len(self.completion_ids),
            "group_index": self.group_index,
            "replayed": self.replayed,
        }


def run_countdown_lite_grpo(
    policy: DenseBackbone,
    reference: DenseBackbone,
    tokenizer: Tokenizer,
    train_rows: tuple[dict[str, Any], ...],
    config: CountdownGRPOTrainerConfig,
    *,
    step_callback: StepCallback | None = None,
    wandb_run: Any | None = None,
) -> CountdownGRPOResult:
    if not train_rows:
        raise CountdownGRPOTrainerError("train_rows must not be empty")
    if policy.config != reference.config:
        raise CountdownGRPOTrainerError("policy and reference must share the same model config")

    set_reproducible_seed(config.seed)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.jsonl"
    rollout_path = output_dir / "rollouts.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    best_checkpoint_path = output_dir / "best-checkpoint.pt"
    best_checkpoint_metadata_path = output_dir / "best-checkpoint.json"
    manifest_path = output_dir / "manifest.json"
    bundle_dir = output_dir / "bundle"
    bundle_final_dir = output_dir / "bundle-final"
    stale_paths = (
        metrics_path,
        rollout_path,
        checkpoint_path,
        best_checkpoint_path,
        best_checkpoint_metadata_path,
        manifest_path,
        bundle_dir,
        bundle_final_dir,
    )
    existing = [path.name for path in stale_paths if path.exists()]
    if existing:
        raise CountdownGRPOTrainerError(
            "output_dir contains stale GRPO artifacts: " + ", ".join(existing)
        )

    device = resolve_torch_device(config.device)
    validate_precision(config.precision, device)
    policy.to(device)
    reference.to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    eos_id = tokenizer.token_to_id("<eos>")
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

    write_json(output_dir / "config.json", _config_payload(config))
    write_json(output_dir / "data-report.json", _data_report(train_rows, config))
    tokenizer.save(str(output_dir / "tokenizer.json"))

    cycled_rows = (
        _stratified_rows(list(train_rows))
        if config.stratified_difficulty_sampling
        else list(train_rows)
    )
    replay_buffer = _SuccessReplayBuffer(
        max_age_steps=config.replay_buffer_max_age_steps,
        min_reward=config.valid_expression_reward,
    )

    rollout_tokens = 0
    total_rollouts = 0
    valid_rollouts = 0
    exact_rollouts = 0
    best_metric = -float("inf")
    best_step = 0
    best_state = _clone_state_dict(policy)
    last_step = 0

    for step in range(1, config.max_steps + 1):
        last_step = step
        step_rows = _cyclic_rows(
            cycled_rows,
            start=(step - 1) * config.prompts_per_step,
            count=config.prompts_per_step,
        )
        groups = [
            _sample_group(
                policy=policy,
                tokenizer=tokenizer,
                row=row,
                config=config,
                device=device,
                eos_id=eos_id,
                group_index=group_index,
            )
            for group_index, row in enumerate(step_rows)
        ]
        rollout_tokens += sum(rollout.token_count for group in groups for rollout in group)
        groups, intervention = _apply_group_interventions(
            groups,
            resample=_group_resampler(
                policy=policy,
                tokenizer=tokenizer,
                rows=step_rows,
                config=config,
                device=device,
                eos_id=eos_id,
            ),
            replay_buffer=replay_buffer,
            step=step,
            config=config,
        )
        rollout_tokens += intervention.resampled_tokens
        rollouts = tuple(rollout for group in groups for rollout in group)
        if rollout_tokens > config.max_rollout_tokens:
            raise CountdownGRPOTrainerError(
                "GRPO rollout token budget exceeded: "
                f"{rollout_tokens} > {config.max_rollout_tokens}"
            )
        total_rollouts += len(rollouts)
        valid_rollouts += sum(rollout.is_valid_expression for rollout in rollouts)
        exact_rollouts += sum(rollout.is_exact_solve for rollout in rollouts)

        input_ids, labels = _collate_rollouts(
            rollouts, device=device, pad_token_id=config.pad_token_id
        )
        with torch.no_grad():
            reference_logp = _sequence_logprob(reference, input_ids, labels)
        advantages = _group_advantages(rollouts).to(device)

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        with precision_context(config.precision, device):
            # One gradient step per rollout batch, so this is plain
            # REINFORCE-with-baseline plus a KL penalty against the reference;
            # a PPO-style ratio would be identically 1 here.
            policy_logp, token_entropy = _sequence_logprob_and_entropy(policy, input_ids, labels)
            objective = advantages * policy_logp
            log_ratio = reference_logp - policy_logp
            kl_penalty = torch.exp(log_ratio) - log_ratio - 1.0
            loss = -objective.mean() + config.kl_beta * kl_penalty.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        replay_buffer.record_step(step, groups)
        step_payload = _step_metrics(
            step=step,
            rollouts=rollouts,
            config=config,
            loss=float(loss.detach()),
            grad_norm=float(grad_norm.detach()),
            learning_rate=float(scheduler.get_last_lr()[0]),
            rollout_tokens=rollout_tokens,
            advantages=advantages.detach().cpu(),
            policy_logp=policy_logp.detach().cpu(),
            reference_logp=reference_logp.detach().cpu(),
            token_entropy=float(token_entropy.detach()),
            intervention=intervention,
            replay_buffer_size=len(replay_buffer),
        )
        _append_rollouts(rollout_path, step=step, rollouts=rollouts)
        if step == 1 or step % config.log_interval == 0 or step == config.max_steps:
            append_metric(metrics_path, step_payload, wandb_run)

        selector = float(step_payload["train/reward_mean"])
        if selector > best_metric:
            best_metric = selector
            best_step = step
            best_state = _clone_state_dict(policy)
            write_json(
                best_checkpoint_metadata_path,
                {
                    "selected_step": best_step,
                    "selected_metric_name": "train/reward_mean",
                    "selected_metric_value": best_metric,
                },
            )
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_training_checkpoint(
                output_dir / "checkpoints" / f"step-{step:06d}" / "checkpoint.pt",
                model=policy,
                step=step,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=step_payload,
            )
        if step_callback is not None:
            step_callback(step)

    final_metrics = {
        "selected_step": best_step,
        "selected_metric_name": "train/reward_mean",
        "selected_metric_value": best_metric,
        "rollout_tokens": rollout_tokens,
    }
    save_training_checkpoint(
        checkpoint_path,
        model=policy,
        step=last_step,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=final_metrics,
    )
    if config.write_final_bundle:
        _write_bundle(
            bundle_final_dir,
            model=policy,
            tokenizer=tokenizer,
            config=config,
            checkpoint_step=last_step,
        )
    policy.load_state_dict(best_state)
    save_training_checkpoint(
        best_checkpoint_path,
        model=policy,
        step=best_step,
        optimizer=None,
        scheduler=None,
        metrics=final_metrics,
    )
    if not best_checkpoint_metadata_path.is_file():
        write_json(
            best_checkpoint_metadata_path,
            {
                "selected_step": best_step,
                "selected_metric_name": "train/reward_mean",
                "selected_metric_value": best_metric,
            },
        )
    _write_bundle(
        bundle_dir,
        model=policy,
        tokenizer=tokenizer,
        config=config,
        checkpoint_step=best_step,
    )
    _write_manifest(
        manifest_path,
        output_dir=output_dir,
        config=config,
        selected_step=best_step,
        selected_metric_value=best_metric,
    )
    return CountdownGRPOResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_metadata_path=best_checkpoint_metadata_path,
        metrics_path=metrics_path,
        rollout_path=rollout_path,
        manifest_path=manifest_path,
        bundle_dir=bundle_dir,
        bundle_final_dir=bundle_final_dir if config.write_final_bundle else None,
        steps_completed=last_step,
        selected_step=best_step,
        selected_metric_name="train/reward_mean",
        selected_metric_value=best_metric,
        rollout_tokens=rollout_tokens,
        exact_rollouts=exact_rollouts,
        valid_rollouts=valid_rollouts,
        total_rollouts=total_rollouts,
    )


def _sample_group(
    *,
    policy: DenseBackbone,
    tokenizer: Tokenizer,
    row: dict[str, Any],
    config: CountdownGRPOTrainerConfig,
    device: torch.device,
    eos_id: int | None,
    group_index: int,
) -> tuple[_Rollout, ...]:
    policy.eval()
    rollouts: list[_Rollout] = []
    with torch.no_grad():
        prompt_ids = tuple(
            tokenizer.encode(render_chat_prompt(str(row["prompt"])), add_special_tokens=False).ids
        )
        if not prompt_ids:
            raise CountdownGRPOTrainerError(f"{row['task_id']} produced an empty prompt")
        prompt_tensor = torch.tensor(
            [list(prompt_ids)] * config.group_size,
            dtype=torch.long,
            device=device,
        )
        generated = policy.generate(
            prompt_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            eos_token_id=eos_id,
        )
        target = int(row["target"])
        for generated_row in generated.detach().cpu().tolist():
            completion_ids = _truncate_at_eos_inclusive(generated_row[len(prompt_ids) :], eos_id)
            output_ids = _without_terminal_eos(completion_ids, eos_id)
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            verification = verify_countdown_lite_expression(
                output,
                numbers=_as_int_tuple(row["numbers"]),
                target=target,
            )
            reward = _reward_for(verification, config, target=target)
            rollouts.append(
                _Rollout(
                    task_id=str(row["task_id"]),
                    difficulty=str(row["difficulty"]),
                    prompt_ids=prompt_ids,
                    completion_ids=tuple(int(token_id) for token_id in completion_ids),
                    output=output,
                    extracted_expression=verification.expression,
                    reward=reward,
                    is_valid_expression=verification.is_valid_expression,
                    is_exact_solve=verification.is_exact_solve,
                    is_well_formed=verification.is_well_formed,
                    reason=verification.reason,
                    value=verification.value,
                    group_index=group_index,
                )
            )
    return tuple(rollouts)


def _group_resampler(
    *,
    policy: DenseBackbone,
    tokenizer: Tokenizer,
    rows: tuple[dict[str, Any], ...],
    config: CountdownGRPOTrainerConfig,
    device: torch.device,
    eos_id: int | None,
) -> Callable[[int], tuple[_Rollout, ...]]:
    def resample(group_index: int) -> tuple[_Rollout, ...]:
        return _sample_group(
            policy=policy,
            tokenizer=tokenizer,
            row=rows[group_index],
            config=config,
            device=device,
            eos_id=eos_id,
            group_index=group_index,
        )

    return resample


@dataclass(frozen=True)
class _GroupInterventionStats:
    zero_variance_groups_sampled: int
    zero_variance_groups_final: int
    zero_variance_resamples: int
    zero_variance_cap_hits: int
    replay_injections: int
    resampled_tokens: int
    group_count: int


class _SuccessReplayBuffer:
    """ARPO-style per-task cache of recent successful rollouts.

    Keeps the highest-reward rollout per task that at least produced a valid
    expression, so an all-failed group can be re-seeded with one known success
    before advantage computation. Entries expire after ``max_age_steps``.
    """

    def __init__(self, *, max_age_steps: int, min_reward: float) -> None:
        self._max_age_steps = max_age_steps
        self._min_reward = min_reward
        self._entries: dict[str, tuple[int, _Rollout]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return self._max_age_steps > 0

    def record_step(self, step: int, groups: list[tuple[_Rollout, ...]]) -> None:
        if not self.enabled:
            return
        for group in groups:
            for rollout in group:
                if rollout.replayed or rollout.reward < self._min_reward:
                    continue
                existing = self._entries.get(rollout.task_id)
                if existing is None or rollout.reward >= existing[1].reward:
                    self._entries[rollout.task_id] = (step, rollout)
        self._evict(step)

    def lookup(self, task_id: str, *, step: int) -> _Rollout | None:
        if not self.enabled:
            return None
        entry = self._entries.get(task_id)
        if entry is None or step - entry[0] > self._max_age_steps:
            return None
        return entry[1]

    def _evict(self, step: int) -> None:
        stale = [
            task_id
            for task_id, (recorded_step, _rollout) in self._entries.items()
            if step - recorded_step > self._max_age_steps
        ]
        for task_id in stale:
            del self._entries[task_id]


def _apply_group_interventions(
    groups: list[tuple[_Rollout, ...]],
    *,
    resample: Callable[[int], tuple[_Rollout, ...]],
    replay_buffer: _SuccessReplayBuffer,
    step: int,
    config: CountdownGRPOTrainerConfig,
) -> tuple[list[tuple[_Rollout, ...]], _GroupInterventionStats]:
    zero_variance_sampled = sum(1 for group in groups if _is_zero_variance(group))
    resamples = 0
    cap_hits = 0
    resampled_tokens = 0
    refilled: list[tuple[_Rollout, ...]] = []
    for group in groups:
        # DAPO-style dynamic sampling: a group with identical rewards has zero
        # advantage everywhere, so resample it up to the configured cap.
        attempts = 0
        while _is_zero_variance(group) and attempts < config.zero_variance_max_resamples:
            attempts += 1
            resamples += 1
            candidate = resample(group[0].group_index)
            resampled_tokens += sum(rollout.token_count for rollout in candidate)
            group = candidate
        if attempts and _is_zero_variance(group):
            cap_hits += 1
            print(
                "countdown_grpo zero-variance resample cap hit: "
                f"step={step} group_index={group[0].group_index} "
                f"task_id={group[0].task_id} attempts={attempts}",
                flush=True,
            )
        refilled.append(group)

    injections = 0
    final_groups: list[tuple[_Rollout, ...]] = []
    for group in refilled:
        # ARPO-style success replay: re-seed an all-failed group with the most
        # recent cached success for the same task.
        cached = (
            replay_buffer.lookup(group[0].task_id, step=step)
            if _is_all_failed(group, min_reward=config.valid_expression_reward)
            else None
        )
        if cached is not None:
            injected = replace(cached, group_index=group[0].group_index, replayed=True)
            weakest = min(range(len(group)), key=lambda index: group[index].reward)
            group = tuple(
                injected if index == weakest else rollout for index, rollout in enumerate(group)
            )
            injections += 1
        final_groups.append(group)

    stats = _GroupInterventionStats(
        zero_variance_groups_sampled=zero_variance_sampled,
        zero_variance_groups_final=sum(1 for group in final_groups if _is_zero_variance(group)),
        zero_variance_resamples=resamples,
        zero_variance_cap_hits=cap_hits,
        replay_injections=injections,
        resampled_tokens=resampled_tokens,
        group_count=len(final_groups),
    )
    return final_groups, stats


def _is_zero_variance(group: tuple[_Rollout, ...]) -> bool:
    return len({rollout.reward for rollout in group}) <= 1


def _is_all_failed(group: tuple[_Rollout, ...], *, min_reward: float) -> bool:
    return all(rollout.reward < min_reward for rollout in group)


def _stratified_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically interleave task difficulties so every batch mixes strata.

    The raw train split is grouped by difficulty, which walked cyclically gives
    long single-difficulty blocks (a single-difficulty schedule collapses
    exactly at the easy-to-medium boundary). Largest-remainder interleaving
    keeps each difficulty's share while spreading it across every window of rows.
    """
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[str(row["difficulty"])].append(row)
    total = len(rows)
    taken = dict.fromkeys(by_difficulty, 0)
    ordered: list[dict[str, Any]] = []
    for position in range(1, total + 1):
        # Pick the difficulty that is furthest behind its proportional share.
        _deficit, choice = min(
            (taken[difficulty] - position * len(pool) / total, difficulty)
            for difficulty, pool in by_difficulty.items()
            if taken[difficulty] < len(pool)
        )
        ordered.append(by_difficulty[choice][taken[choice]])
        taken[choice] += 1
    return ordered


def _sequence_logprob(
    model: DenseBackbone,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    sequence_logp, _token_entropy = _sequence_logprob_and_entropy(model, input_ids, labels)
    return sequence_logp


def _sequence_logprob_and_entropy(
    model: DenseBackbone,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(input_ids[:, :-1])
    targets = labels[:, 1:]
    capped = soft_cap_logits(logits.float(), model.config.logit_soft_cap)
    log_probs = torch.log_softmax(capped, dim=-1)
    mask = targets != IGNORE_INDEX
    gather_targets = targets.clamp_min(0).unsqueeze(-1)
    token_logp = log_probs.gather(-1, gather_targets).squeeze(-1)
    token_logp = token_logp * mask
    token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    mask_total = mask.sum().clamp_min(1)
    mean_token_entropy = (token_entropy * mask).sum() / mask_total
    return token_logp.sum(dim=-1), mean_token_entropy


def _collate_rollouts(
    rollouts: tuple[_Rollout, ...], *, device: torch.device, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(rollout.prompt_ids) + len(rollout.completion_ids) for rollout in rollouts)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for rollout in rollouts:
        input_ids = [*rollout.prompt_ids, *rollout.completion_ids]
        labels = [IGNORE_INDEX] * len(rollout.prompt_ids) + list(rollout.completion_ids)
        pad = max_len - len(input_ids)
        input_rows.append([*input_ids, *([pad_token_id] * pad)])
        label_rows.append([*labels, *([IGNORE_INDEX] * pad)])
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(label_rows, dtype=torch.long, device=device),
    )


def _group_advantages(rollouts: tuple[_Rollout, ...]) -> torch.Tensor:
    # Dr. GRPO mean-only baseline: advantage = reward - group mean, with no
    # std division. Normalizing by std blows the advantage up exactly when a
    # group is nearly uniform, which is the least informative case.
    grouped: dict[int, list[float]] = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout.group_index].append(float(rollout.reward))
    means = {group_index: sum(rewards) / len(rewards) for group_index, rewards in grouped.items()}
    return torch.tensor(
        [float(rollout.reward) - means[rollout.group_index] for rollout in rollouts],
        dtype=torch.float32,
    )


def _reward_for(verification: Any, config: CountdownGRPOTrainerConfig, *, target: int) -> float:
    if verification.is_exact_solve:
        return float(config.exact_solve_reward)
    if verification.is_valid_expression:
        return float(config.valid_expression_reward) + _closeness_bonus(
            verification.value, target=target, weight=config.closeness_weight
        )
    if verification.is_well_formed:
        return float(config.format_expression_reward)
    return float(config.invalid_reward)


def _closeness_bonus(value: int | None, *, target: int, weight: float) -> float:
    if weight <= 0.0 or value is None:
        return 0.0
    return weight * math.exp(-abs(value - target) / max(target, 1))


def _step_metrics(
    *,
    step: int,
    rollouts: tuple[_Rollout, ...],
    config: CountdownGRPOTrainerConfig,
    loss: float,
    grad_norm: float,
    learning_rate: float,
    rollout_tokens: int,
    advantages: torch.Tensor,
    policy_logp: torch.Tensor,
    reference_logp: torch.Tensor,
    token_entropy: float,
    intervention: _GroupInterventionStats,
    replay_buffer_size: int,
) -> dict[str, Any]:
    rewards = torch.tensor([rollout.reward for rollout in rollouts], dtype=torch.float32)
    exact = sum(rollout.is_exact_solve for rollout in rollouts)
    valid = sum(rollout.is_valid_expression for rollout in rollouts)
    format_only = sum(
        rollout.is_well_formed and not rollout.is_valid_expression for rollout in rollouts
    )
    invalid = sum(not rollout.is_well_formed for rollout in rollouts)
    closeness_bonuses = [
        rollout.reward - config.valid_expression_reward
        for rollout in rollouts
        if rollout.is_valid_expression and not rollout.is_exact_solve
    ]
    group_count = max(intervention.group_count, 1)
    return {
        "event": "train",
        "step": step,
        "train/loss": loss,
        "train/reward_mean": float(rewards.mean()),
        "train/reward_max": float(rewards.max()),
        "train/reward_std": float(rewards.std(unbiased=False)),
        "train/valid_expression_rate": valid / len(rollouts),
        "train/exact_solve_rate": exact / len(rollouts),
        "train/format_only_rate": format_only / len(rollouts),
        "train/invalid_rate": invalid / len(rollouts),
        "train/reward_closeness_mean": (
            sum(closeness_bonuses) / len(closeness_bonuses) if closeness_bonuses else 0.0
        ),
        "train/token_entropy": token_entropy,
        "train/completion_tokens_mean": (
            sum(len(rollout.completion_ids) for rollout in rollouts) / len(rollouts)
        ),
        "train/frac_zero_variance_groups": intervention.zero_variance_groups_final / group_count,
        "train/frac_zero_variance_groups_sampled": (
            intervention.zero_variance_groups_sampled / group_count
        ),
        "train/zero_variance_resamples": intervention.zero_variance_resamples,
        "train/zero_variance_cap_hits": intervention.zero_variance_cap_hits,
        "train/replay_injections": intervention.replay_injections,
        "train/replay_buffer_size": replay_buffer_size,
        "train/advantage_mean": float(advantages.mean()),
        "train/advantage_abs_mean": float(advantages.abs().mean()),
        "train/policy_logp_mean": float(policy_logp.mean()),
        "train/reference_logp_mean": float(reference_logp.mean()),
        "train/learning_rate": learning_rate,
        "train/grad_norm": grad_norm,
        "train/rollouts": len(rollouts),
        "train/rollout_tokens": rollout_tokens,
    }


def _append_rollouts(path: Path, *, step: int, rollouts: tuple[_Rollout, ...]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout.to_json(step=step), sort_keys=True) + "\n")


def _cyclic_rows(
    rows: list[dict[str, Any]], *, start: int, count: int
) -> tuple[dict[str, Any], ...]:
    return tuple(rows[(start + offset) % len(rows)] for offset in range(count))


def _truncate_at_eos_inclusive(token_ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is None:
        return [int(token_id) for token_id in token_ids]
    try:
        index = token_ids.index(eos_id)
    except ValueError:
        return [int(token_id) for token_id in token_ids]
    return [int(token_id) for token_id in token_ids[: index + 1]]


def _without_terminal_eos(token_ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is not None and token_ids and token_ids[-1] == eos_id:
        return token_ids[:-1]
    return token_ids


def _as_int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CountdownGRPOTrainerError("row.numbers must be a list")
    return tuple(int(number) for number in value)


def _clone_state_dict(model: DenseBackbone) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _config_payload(config: CountdownGRPOTrainerConfig) -> dict[str, Any]:
    return {
        "mode": "countdown_lite_grpo",
        "artifact_name": config.artifact_name,
        "reference_artifact_name": config.reference_artifact_name,
        "max_steps": config.max_steps,
        "prompts_per_step": config.prompts_per_step,
        "group_size": config.group_size,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "kl_beta": config.kl_beta,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "scheduler": config.scheduler,
        "grad_clip": config.grad_clip,
        "seed": config.seed,
        "max_rollout_tokens": config.max_rollout_tokens,
        "exact_solve_reward": config.exact_solve_reward,
        "valid_expression_reward": config.valid_expression_reward,
        "invalid_reward": config.invalid_reward,
        "format_expression_reward": config.format_expression_reward,
        "closeness_weight": config.closeness_weight,
        "zero_variance_max_resamples": config.zero_variance_max_resamples,
        "replay_buffer_max_age_steps": config.replay_buffer_max_age_steps,
        "stratified_difficulty_sampling": config.stratified_difficulty_sampling,
        "precision": config.precision,
        "device": config.device,
    }


def _data_report(
    train_rows: tuple[dict[str, Any], ...], config: CountdownGRPOTrainerConfig
) -> dict[str, Any]:
    difficulty_counts: dict[str, int] = {}
    for row in train_rows:
        difficulty = str(row["difficulty"])
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    return {
        "mode": "countdown_lite_grpo",
        "train_tasks": len(train_rows),
        "difficulty_counts": difficulty_counts,
        "remote_dataset_download": False,
        "paid_api": False,
        "max_rollout_tokens": config.max_rollout_tokens,
        "reward_policy": {
            "exact_solve_reward": config.exact_solve_reward,
            "valid_expression_reward": config.valid_expression_reward,
            "invalid_reward": config.invalid_reward,
            "format_expression_reward": config.format_expression_reward,
            "closeness_weight": config.closeness_weight,
            "verifiable_only": True,
        },
        "selected_task_manifest": [
            {
                "task_id": row["task_id"],
                "split": row["split"],
                "difficulty": row["difficulty"],
                "numbers": row["numbers"],
                "target": row["target"],
                "reward_name": row["reward_name"],
            }
            for row in train_rows
        ],
    }


def _write_bundle(
    bundle_dir: Path,
    *,
    model: DenseBackbone,
    tokenizer: Tokenizer,
    config: CountdownGRPOTrainerConfig,
    checkpoint_step: int,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "config.json").write_text(
        json.dumps(model.config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tokenizer.save(str(bundle_dir / "tokenizer.json"))
    weights_payload = {
        "format_version": 1,
        "format": BUNDLE_FORMAT,
        "key_format": BUNDLE_FORMAT,
        "state_dict_key": "model_state",
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(weights_payload, bundle_dir / "weights.pt")
    manifest = {
        "schema_version": 1,
        "format": BUNDLE_FORMAT,
        "weights_format": BUNDLE_FORMAT,
        "model_family": "DenseBackbone",
        "model": {"id": config.bundle_model_id, "name": config.artifact_name, "stage": "rlvr"},
        "model_config": model.config.to_dict(),
        "files": {
            "config": {
                "path": "config.json",
                "sha256": file_sha256(bundle_dir / "config.json"),
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": file_sha256(bundle_dir / "tokenizer.json"),
            },
            "weights": {"path": "weights.pt", "sha256": file_sha256(bundle_dir / "weights.pt")},
        },
        "tokenizer": {
            "path": "tokenizer.json",
            "format": "tokenizers-json",
            "add_special_tokens": False,
        },
        "eos_token_ids": _eos_token_ids(tokenizer),
        "decoding": {
            "eos_token_ids": _eos_token_ids(tokenizer),
            "default_add_special_tokens": False,
        },
        "provenance": {
            "source": "countdown_lite_grpo",
            "starts_from": config.reference_artifact_name,
            "checkpoint_step": checkpoint_step,
            "source_manifest": config.source_manifest,
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    config: CountdownGRPOTrainerConfig,
    selected_step: int,
    selected_metric_value: float,
) -> None:
    files: dict[str, dict[str, str]] = {}
    for name in (
        "config.json",
        "data-report.json",
        "rollouts.jsonl",
        "metrics.jsonl",
        "checkpoint.pt",
        "best-checkpoint.pt",
        "best-checkpoint.json",
        "tokenizer.json",
    ):
        path = output_dir / name
        if path.is_file():
            files[name] = {"path": name, "sha256": file_sha256(path)}
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_name": config.artifact_name,
            "stage": "rlvr",
            "method": "grpo",
            "selected_step": selected_step,
            "selected_metric_name": "train/reward_mean",
            "selected_metric_value": selected_metric_value,
            "bundle_dir": str(output_dir / "bundle"),
            "files": files,
        },
    )


def _eos_token_ids(tokenizer: Tokenizer) -> list[int]:
    eos_id = tokenizer.token_to_id("<eos>")
    return [int(eos_id)] if eos_id is not None else []
