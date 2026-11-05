from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from esme_posttrain.cli import main
from esme_posttrain.dpo.launch import DPOLaunchConfig, load_dpo_config, validate_dpo_payload
from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    build_modal_launch_command,
    estimate_cost_usd,
    full_launch_blockers,
    load_json_object,
    smoke_launch_blockers,
)
from esme_posttrain.launch.config_models import SmokeModalRuntime
from esme_posttrain.launch.errors import LaunchError
from esme_posttrain.launch.modal_cli import format_payload, validate_output_stem
from esme_posttrain.rl.launch import RLVRLaunchConfig, load_rlvr_config, validate_rlvr_payload
from esme_posttrain.sft.launch_multiturn import (
    MultiTurnLaunchConfig,
    load_multi_turn_config,
    validate_multi_turn_payload,
)

type LaunchConfig = MultiTurnLaunchConfig | DPOLaunchConfig | RLVRLaunchConfig


@pytest.fixture
def runtime() -> SmokeModalRuntime:
    return load_multi_turn_config(Path("configs/esme-214m-sft-multiturn.json")).runtime


def test_load_json_object_rejects_non_object(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(LaunchError, match="config must be a JSON object"):
        load_json_object(config_path)


def test_estimate_cost_requires_positive_inputs() -> None:
    with pytest.raises(LaunchError, match="tokens must be positive"):
        estimate_cost_usd(tokens=0, projected_tokens_per_second=10.0, usd_per_hour=1.0)


def test_shared_smoke_launch_blockers_flag_projection_over_budget(
    runtime: SmokeModalRuntime,
) -> None:

    def blockers(cost: float, gpu: str | None = None) -> list[str]:
        return smoke_launch_blockers(
            runtime=runtime,
            estimated_smoke_cost_usd=cost,
            modal_gpu=gpu,
            modal_gpu_env_var="SFT_MODAL_GPU",
            timeout_hours=None,
            timeout_env_var="SFT_TIMEOUT_HOURS",
        )

    assert blockers(4.0) == ["projected Modal smoke cost exceeds runtime.smoke_max_cost_usd"]
    assert blockers(1.0) == []
    assert blockers(1.0, runtime.gpu_profiles[runtime.selected_gpu].modal_gpu) == []
    assert blockers(1.0, "H100:8") == [
        "SFT_MODAL_GPU must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
        "for smoke cost accounting"
    ]


def test_shared_full_launch_blockers_keep_pipeline_wording(runtime: SmokeModalRuntime) -> None:

    assert full_launch_blockers(
        runtime=runtime,
        estimated_full_cost_usd=42.0,
        approved=False,
        modal_gpu="H100",
        approval_message="full multi-turn SFT launch requires --approved",
        modal_gpu_env_var="SFT_MODAL_GPU",
    ) == [
        "full multi-turn SFT launch requires --approved",
        "SFT_MODAL_GPU must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
        "for full-run cost accounting",
        "projected full-run cost exceeds runtime.full_run_max_cost_usd",
    ]
    assert (
        full_launch_blockers(
            runtime=runtime,
            estimated_full_cost_usd=39.0,
            approved=True,
            modal_gpu="A100",
            approval_message="full multi-turn SFT launch requires --approved",
            modal_gpu_env_var="SFT_MODAL_GPU",
        )
        == []
    )


def test_build_modal_launch_command_preserves_approval_flag(
    tmp_path: Path, runtime: SmokeModalRuntime
) -> None:

    command = build_modal_launch_command(
        config_path=tmp_path / "config.json",
        runtime=runtime,
        gpu_env_var="SFT_MODAL_GPU",
        timeout_env_var="SFT_TIMEOUT_HOURS",
        script_path="scripts/modal_chat_sft.py",
        mode_flag=" --full-run",
    )

    expected_suffix = (
        f"--config {(tmp_path / 'config.json').as_posix()} --full-run {LAUNCH_APPROVAL_FLAG} --json"
    )
    assert command.endswith(expected_suffix)
    assert command.startswith("SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=12 ")


def test_format_payload_keeps_pipeline_specific_keys() -> None:
    payload = {
        "status": "ready",
        "chat_eval_command": "run chat eval",
        "launch_blockers": [],
        "unused": "hidden",
    }

    assert format_payload(payload, json_output=False, keys=("chat_eval_command",)) == (
        "status: ready\nchat_eval_command: run chat eval\nlaunch_blockers: none"
    )


def test_validate_output_stem_rejects_forbidden_substrings() -> None:
    with pytest.raises(ValueError, match="must not contain -excellence"):
        validate_output_stem(
            "model-excellence",
            env_var="SFT_MODAL_FULL_OUTPUT_STEM",
            forbidden_substrings=("-excellence",),
        )


def test_load_json_object_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(LaunchError, match="config path does not exist"):
        load_json_object(missing)


@pytest.mark.parametrize(
    ("config_path", "loader", "validator"),
    [
        (
            Path("configs/esme-214m-sft-multiturn.json"),
            load_multi_turn_config,
            validate_multi_turn_payload,
        ),
        (Path("configs/esme-214m-chat-dpo.json"), load_dpo_config, validate_dpo_payload),
        (
            Path("fixtures/configs/esme-214m-rl.fixture.json"),
            load_rlvr_config,
            validate_rlvr_payload,
        ),
    ],
)
def test_launch_settings_preserve_serialized_config_and_isolate_mutation(
    config_path: Path,
    loader: Callable[[Path], LaunchConfig],
    validator: Callable[[object, Path], LaunchConfig],
) -> None:
    original = json.loads(config_path.read_text(encoding="utf-8"))
    source = json.loads(json.dumps(original))
    config = validator(source, config_path.resolve())
    assert json.dumps(config.settings.to_dict(), sort_keys=True) == json.dumps(
        original, sort_keys=True
    )
    assert loader(config_path).settings == config.settings

    source["runtime"]["gpu_profiles"]["A100"]["usd_per_hour"] = 999
    source["budgets"].clear()
    detached: dict[str, Any] = config.settings.to_dict()
    detached["runtime"]["gpu_profiles"]["A100"]["usd_per_hour"] = 999
    assert config.settings.to_dict() == original

    with pytest.raises(ValidationError, match="frozen"):
        config.settings.runtime.timeout_hours = 99
    with pytest.raises(ValidationError, match="frozen"):
        config.selected_gpu_profile.usd_per_hour = 999
    with pytest.raises(TypeError):
        config.settings.runtime.gpu_profiles["A100"] = config.selected_gpu_profile  # pyright: ignore[reportIndexIssue]
    assert isinstance(config.settings.artifacts.required_files, tuple)


def test_learning_gate_evidence_is_deeply_immutable() -> None:
    config_path = Path("configs/esme-214m-sft-multiturn.json")
    source = json.loads(config_path.read_text(encoding="utf-8"))
    config = validate_multi_turn_payload(source, config_path.resolve())
    evidence: Any = config.settings.learning_gate.evidence
    sweep = evidence["bounded_matched_interval_eval_sweep"]
    original_steps = tuple(sweep["interval_eval_steps"])
    source["learning_gate"]["evidence"]["bounded_matched_interval_eval_sweep"][
        "interval_eval_steps"
    ].clear()
    assert sweep["interval_eval_steps"] == original_steps
    with pytest.raises(TypeError):
        sweep["step0_response_loss"] = 0


def test_legacy_rl_settings_keep_optional_defaults_out_of_persisted_config() -> None:
    config_path = Path("fixtures/configs/esme-214m-rl.fixture.json")
    source = json.loads(config_path.read_text(encoding="utf-8"))
    for name in (
        "zero_variance_max_resamples",
        "replay_buffer_max_age_steps",
        "stratified_difficulty_sampling",
        "reward_mode",
        "random_reward_seed",
        "advantage_normalization",
    ):
        source["grpo"].pop(name, None)
    source["reward_policy"].pop("format_expression_reward")
    source["reward_policy"].pop("closeness_weight")
    config = validate_rlvr_payload(source, config_path.resolve())

    assert config.grpo.zero_variance_max_resamples == 0
    assert config.grpo.reward_mode == "verifier"
    assert config.grpo.advantage_normalization == "mean"
    assert config.settings.reward_policy.closeness_weight == 0.0
    assert config.settings.to_dict() == source


@pytest.mark.parametrize(
    ("config_path", "section", "validator"),
    [
        (
            Path("configs/esme-214m-sft-multiturn.json"),
            "optimizer",
            validate_multi_turn_payload,
        ),
        (Path("configs/esme-214m-chat-dpo.json"), "optimizer", validate_dpo_payload),
        (
            Path("fixtures/configs/esme-214m-rl.fixture.json"),
            "grpo",
            validate_rlvr_payload,
        ),
    ],
)
def test_launch_settings_reject_boolean_learning_rates(
    config_path: Path,
    section: str,
    validator: Callable[[object, Path], LaunchConfig],
) -> None:
    source = json.loads(config_path.read_text(encoding="utf-8"))
    source[section]["learning_rate"] = True
    with pytest.raises(LaunchError, match=rf"{section}.learning_rate"):
        validator(source, config_path.resolve())


def test_cli_config_error_identifies_source_and_nested_field_without_partial_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = json.loads(Path("configs/esme-214m-sft-multiturn.json").read_text(encoding="utf-8"))
    source["runtime"]["gpu_profiles"]["A100"]["projected_tokens_per_second"] = True
    config_path = tmp_path / "invalid-sft.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(["sft-multiturn-dry-run", "--config", str(config_path), "--json"])

    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert str(config_path) in output.err
    assert "runtime.gpu_profiles.A100.projected_tokens_per_second must be a number" in output.err
    assert "Traceback" not in output.err
    assert ".projected_tokens_per_second.float" not in output.err


@pytest.mark.parametrize("unknown_field", ["int", "float"])
def test_config_error_preserves_field_names_that_match_numeric_types(unknown_field: str) -> None:
    config_path = Path("configs/esme-214m-chat-dpo.json")
    source = json.loads(config_path.read_text(encoding="utf-8"))
    source[unknown_field] = 3

    with pytest.raises(LaunchError, match=f"unsupported keys: {unknown_field}"):
        validate_dpo_payload(source, config_path)
