"""Unit tests for the GRPO stability bundle (Dr. GRPO / DAPO / ARPO)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from esme_posttrain.launch.config_guards import LaunchError
from esme_posttrain.rl.countdown_lite import verify_countdown_lite_expression
from esme_posttrain.rl.grpo import (
    CountdownGRPOTrainerConfig,
    CountdownGRPOTrainerError,
    _apply_group_interventions,
    _group_advantages,
    _is_zero_variance,
    _reward_for,
    _Rollout,
    _stratified_rows,
    _SuccessReplayBuffer,
)
from esme_posttrain.rl.launch import load_rlvr_config

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_CONFIG = REPO_ROOT / "configs" / "esme-214m-rl.json"


def _trainer_config(**overrides: object) -> CountdownGRPOTrainerConfig:
    payload: dict[str, object] = {
        "max_steps": 4,
        "prompts_per_step": 2,
        "group_size": 4,
        "max_new_tokens": 4,
        "temperature": 1.0,
        "kl_beta": 0.001,
        "learning_rate": 5e-7,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "scheduler": "constant",
        "grad_clip": 1.0,
        "seed": 214,
        "output_dir": Path("/tmp/unused"),
        "max_rollout_tokens": 10_000,
        "exact_solve_reward": 1.0,
        "valid_expression_reward": 0.3,
        "invalid_reward": 0.0,
        "format_expression_reward": 0.05,
        "closeness_weight": 0.3,
        "zero_variance_max_resamples": 1,
        "replay_buffer_max_age_steps": 20,
        "stratified_difficulty_sampling": True,
    }
    payload.update(overrides)
    return CountdownGRPOTrainerConfig(**payload)  # type: ignore[arg-type]


def _rollout(
    *,
    task_id: str = "task-a",
    reward: float,
    group_index: int = 0,
    is_valid: bool = False,
    is_exact: bool = False,
    is_well_formed: bool = False,
    replayed: bool = False,
) -> _Rollout:
    return _Rollout(
        task_id=task_id,
        difficulty="easy",
        prompt_ids=(1, 2, 3),
        completion_ids=(4, 5),
        output="7 + 7",
        extracted_expression="7 + 7",
        reward=reward,
        is_valid_expression=is_valid,
        is_exact_solve=is_exact,
        is_well_formed=is_well_formed,
        reason="fixture",
        value=None,
        group_index=group_index,
        replayed=replayed,
    )


def test_group_advantages_are_mean_only_without_std_division() -> None:
    rollouts = (
        _rollout(reward=1.0, group_index=0),
        _rollout(reward=0.0, group_index=0),
        _rollout(reward=0.9, group_index=1),
        _rollout(reward=1.0, group_index=1),
    )

    advantages = _group_advantages(rollouts)

    # Group 0: mean 0.5 -> +/-0.5. With v1 std division these were +/-1.0.
    assert torch.allclose(advantages[:2], torch.tensor([0.5, -0.5]))
    # Group 1: low variance must NOT blow the advantage up.
    assert torch.allclose(advantages[2:], torch.tensor([-0.05, 0.05]), atol=1e-6)


def test_group_advantages_zero_for_zero_variance_group() -> None:
    rollouts = tuple(_rollout(reward=0.3, group_index=0) for _ in range(4))
    assert torch.equal(_group_advantages(rollouts), torch.zeros(4))


def test_graded_reward_is_monotonic_across_tiers() -> None:
    config = _trainer_config()
    numbers = [3, 5]
    target = 15

    invalid = _reward_for(
        verify_countdown_lite_expression("no idea", numbers=numbers, target=target),
        config,
        target=target,
    )
    format_only = _reward_for(
        verify_countdown_lite_expression("7 + 7", numbers=numbers, target=target),
        config,
        target=target,
    )
    valid_far = _reward_for(
        verify_countdown_lite_expression("5 - 3", numbers=numbers, target=target),
        config,
        target=target,
    )
    valid_near = _reward_for(
        verify_countdown_lite_expression("3 * 5 - 1", numbers=[3, 5, 1], target=target),
        config,
        target=target,
    )
    exact = _reward_for(
        verify_countdown_lite_expression("3 * 5", numbers=numbers, target=target),
        config,
        target=target,
    )

    assert invalid == 0.0
    assert invalid < format_only < valid_far < valid_near < exact
    assert format_only == 0.05
    assert valid_far == pytest.approx(0.3 + 0.3 * math.exp(-13 / 15))
    assert valid_near == pytest.approx(0.3 + 0.3 * math.exp(-1 / 15))
    assert exact == 1.0
    # Exact stays at least the anti-proximity-hacking margin above the ceiling.
    assert exact - (0.3 + 0.3) >= 0.4


def test_trainer_config_rejects_exact_reward_inside_closeness_margin() -> None:
    with pytest.raises(CountdownGRPOTrainerError, match="closeness ceiling"):
        _trainer_config(valid_expression_reward=0.4, closeness_weight=0.3)


def test_trainer_config_rejects_format_reward_above_valid() -> None:
    with pytest.raises(CountdownGRPOTrainerError, match="reward order"):
        _trainer_config(format_expression_reward=0.4)


def test_zero_variance_group_is_refilled_by_resample() -> None:
    config = _trainer_config()
    degenerate = tuple(_rollout(reward=0.0) for _ in range(4))
    varied = (
        _rollout(reward=0.0),
        _rollout(reward=0.05, is_well_formed=True),
        _rollout(reward=0.0),
        _rollout(reward=0.0),
    )
    buffer = _SuccessReplayBuffer(max_age_steps=0, min_reward=0.3)

    groups, stats = _apply_group_interventions(
        [degenerate],
        resample=lambda group_index: varied,
        replay_buffer=buffer,
        step=1,
        config=config,
    )

    assert groups == [varied]
    assert stats.zero_variance_groups_sampled == 1
    assert stats.zero_variance_groups_final == 0
    assert stats.zero_variance_resamples == 1
    assert stats.zero_variance_cap_hits == 0
    assert stats.resampled_tokens == sum(rollout.token_count for rollout in varied)


def test_zero_variance_resample_cap_hit_is_counted(capsys: pytest.CaptureFixture[str]) -> None:
    config = _trainer_config()
    degenerate = tuple(_rollout(reward=0.0) for _ in range(4))
    buffer = _SuccessReplayBuffer(max_age_steps=0, min_reward=0.3)

    groups, stats = _apply_group_interventions(
        [degenerate],
        resample=lambda group_index: degenerate,
        replay_buffer=buffer,
        step=3,
        config=config,
    )

    assert stats.zero_variance_resamples == 1
    assert stats.zero_variance_cap_hits == 1
    assert stats.zero_variance_groups_final == 1
    assert _is_zero_variance(groups[0])
    assert "zero-variance resample cap hit" in capsys.readouterr().out


def test_replay_buffer_injects_cached_success_into_all_failed_group() -> None:
    config = _trainer_config(zero_variance_max_resamples=0)
    buffer = _SuccessReplayBuffer(max_age_steps=20, min_reward=0.3)
    success = _rollout(reward=1.0, is_valid=True, is_exact=True, is_well_formed=True)
    buffer.record_step(5, [(success,)])
    failed = tuple(_rollout(reward=0.0) for _ in range(4))

    groups, stats = _apply_group_interventions(
        [failed],
        resample=lambda group_index: failed,
        replay_buffer=buffer,
        step=10,
        config=config,
    )

    injected = [rollout for rollout in groups[0] if rollout.replayed]
    assert stats.replay_injections == 1
    assert len(injected) == 1
    assert injected[0].reward == 1.0
    assert injected[0].group_index == failed[0].group_index


def test_replay_buffer_skips_stale_entries_and_evicts() -> None:
    config = _trainer_config(zero_variance_max_resamples=0)
    buffer = _SuccessReplayBuffer(max_age_steps=20, min_reward=0.3)
    success = _rollout(reward=1.0, is_valid=True, is_exact=True, is_well_formed=True)
    buffer.record_step(1, [(success,)])
    failed = tuple(_rollout(reward=0.0) for _ in range(4))

    _groups, stats = _apply_group_interventions(
        [failed],
        resample=lambda group_index: failed,
        replay_buffer=buffer,
        step=30,
        config=config,
    )

    assert stats.replay_injections == 0
    buffer.record_step(30, [()])
    assert len(buffer) == 0


def test_replay_buffer_ignores_failures_and_replayed_rollouts() -> None:
    buffer = _SuccessReplayBuffer(max_age_steps=20, min_reward=0.3)
    replayed_success = _rollout(reward=1.0, is_valid=True, is_exact=True, replayed=True)
    failure = _rollout(reward=0.05, is_well_formed=True)

    buffer.record_step(1, [(replayed_success, failure)])

    assert len(buffer) == 0
    assert buffer.lookup("task-a", step=1) is None


def test_stratified_rows_mix_difficulties_in_every_batch_window() -> None:
    rows = (
        [{"difficulty": "easy", "task_id": f"e{i}"} for i in range(90)]
        + [{"difficulty": "medium", "task_id": f"m{i}"} for i in range(120)]
        + [{"difficulty": "hard", "task_id": f"h{i}"} for i in range(90)]
    )

    ordered = _stratified_rows(list(rows))

    assert len(ordered) == len(rows)
    assert sorted(row["task_id"] for row in ordered) == sorted(row["task_id"] for row in rows)
    # Every 8-row batch window (prompts_per_step in the config) must mix
    # difficulties; a single-difficulty schedule walks ~22-step blocks and
    # collapses at the easy-to-medium boundary.
    for start in range(0, len(ordered) - 7, 8):
        window = {row["difficulty"] for row in ordered[start : start + 8]}
        assert len(window) >= 2, f"single-difficulty window at rows {start}..{start + 8}"
    # Deterministic given the same input.
    assert [row["task_id"] for row in _stratified_rows(list(rows))] == [
        row["task_id"] for row in ordered
    ]


def test_config_loads_and_projects_worst_case_resample_cost() -> None:
    config = load_rlvr_config(RL_CONFIG)

    grpo = config.grpo
    sequences = int(grpo["prompts_per_step"]) * int(grpo["group_size"]) * int(grpo["max_steps"])
    resample_factor = 1 + int(grpo["zero_variance_max_resamples"])
    assert config.estimated_train_tokens % (sequences * resample_factor) == 0
    per_sequence = config.estimated_train_tokens // (sequences * resample_factor)
    assert per_sequence > int(grpo["max_new_tokens"])
    assert config.estimated_full_cost_usd <= 15.0
    assert float(config.runtime["full_run_max_cost_usd"]) == 20.0
    assert float(config.runtime["full_run_runtime_spend_stop_usd"]) == 18.0
    assert tuple(config.payload["artifacts"]["required_files"])[-1] == "eval-after-final.json"


def test_reward_policy_margin_is_enforced_at_launch_validation(tmp_path: Path) -> None:
    import json

    payload = json.loads(RL_CONFIG.read_text(encoding="utf-8"))
    payload["reward_policy"]["closeness_weight"] = 0.5
    (tmp_path / "configs").mkdir()
    broken = tmp_path / "configs" / "esme-214m-rl-broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "exports").symlink_to(REPO_ROOT / "exports")
    (tmp_path / "data").symlink_to(REPO_ROOT / "data")

    with pytest.raises(LaunchError, match="closeness ceiling"):
        load_rlvr_config(broken)


def test_verifier_flags_well_formed_expressions() -> None:
    parsed_but_wrong_numbers = verify_countdown_lite_expression("7 + 7", numbers=[3, 5], target=15)
    assert parsed_but_wrong_numbers.is_well_formed
    assert not parsed_but_wrong_numbers.is_valid_expression

    unparseable = verify_countdown_lite_expression("3 + (", numbers=[3, 5], target=15)
    assert not unparseable.is_well_formed

    no_expression = verify_countdown_lite_expression("hello there", numbers=[3, 5], target=15)
    assert not no_expression.is_well_formed
