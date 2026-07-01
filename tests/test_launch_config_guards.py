from __future__ import annotations

import json
from pathlib import Path

import pytest

from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    LaunchError,
    build_modal_launch_command,
    estimate_cost_usd,
    full_launch_blockers,
    load_json_object,
    smoke_launch_blockers,
)
from esme_posttrain.launch.modal_cli import format_payload, validate_output_stem
from esme_posttrain.launch.validate import (
    iter_jsonl,
    load_json_file,
    require_path,
)


def test_load_json_object_rejects_non_object(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(LaunchError, match="config must be a JSON object"):
        load_json_object(config_path)


def test_estimate_cost_requires_positive_inputs() -> None:
    with pytest.raises(LaunchError, match="tokens must be positive"):
        estimate_cost_usd(tokens=0, projected_tokens_per_second=10.0, usd_per_hour=1.0)


def test_shared_smoke_launch_blockers_preserve_cap_messages() -> None:
    runtime = {
        "smoke_max_cost_usd": 3.0,
        "runtime_spend_stop_usd": 3.0,
        "timeout_hours": 25,
    }

    assert smoke_launch_blockers(runtime=runtime, estimated_smoke_cost_usd=4.0) == [
        "runtime.smoke_max_cost_usd exceeds the approved $2 smoke cap",
        "runtime.runtime_spend_stop_usd exceeds the approved $2 smoke cap",
        "projected Modal smoke cost exceeds runtime.smoke_max_cost_usd",
        "runtime.timeout_hours exceeds Modal's 24h function maximum",
    ]


def test_shared_full_launch_blockers_keep_pipeline_wording() -> None:
    runtime = {
        "selected_gpu": "A100",
        "gpu_profiles": {"A100": {"modal_gpu": "A100"}},
        "full_run_max_cost_usd": 41.0,
        "full_run_runtime_spend_stop_usd": 41.0,
    }

    assert full_launch_blockers(
        runtime=runtime,
        estimated_full_cost_usd=42.0,
        approved=False,
        modal_gpu="H100",
        approval_message="full multi-turn SFT launch requires --approved",
        modal_gpu_env_var="SFT_MODAL_GPU",
        full_run_cap_usd=40.0,
        cap_label="$40 runaway cap",
    ) == [
        "full multi-turn SFT launch requires --approved",
        "SFT_MODAL_GPU must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
        "for full-run cost accounting",
        "runtime.full_run_max_cost_usd exceeds the $40 runaway cap",
        "runtime.full_run_runtime_spend_stop_usd exceeds the $40 runaway cap",
        "projected full-run cost exceeds runtime.full_run_max_cost_usd",
    ]


def test_build_modal_launch_command_preserves_approval_flag(tmp_path: Path) -> None:
    runtime = {
        "selected_gpu": "A100",
        "timeout_hours": 12,
        "gpu_profiles": {"A100": {"modal_gpu": "A100"}},
    }

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


def test_load_json_file_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(LaunchError, match="missing config"):
        load_json_file(missing, "config")


def test_require_path_resolves_relative_to_base_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "configs"
    base_dir.mkdir()
    payload = {"path": "../data/manifest.json"}
    resolved = require_path(payload, "path", base_dir, "config")
    assert resolved == (tmp_path / "data" / "manifest.json").resolve()


def test_iter_jsonl_rejects_blank_lines(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('{"prompt": "hi"}\n\n', encoding="utf-8")
    with pytest.raises(LaunchError, match="blank JSONL lines are not allowed"):
        list(iter_jsonl(jsonl_path))
