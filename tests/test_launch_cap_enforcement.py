"""Cap enforcement through real config validation, for every stage.

These tests pin the invariant the launch blockers rely on: a config whose
runtime exceeds a hardcoded spend cap or the Modal timeout bound can never
survive validation, so blocker functions only need to cover launch-time state
(approval, GPU env var, cost projection). Each case starts from a tracked
config file and mutates one runtime field, so the failure path runs through
the same validator the CLI uses.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from esme_posttrain.dpo.launch import validate_dpo_payload
from esme_posttrain.launch.errors import LaunchError
from esme_posttrain.rl.launch import validate_rlvr_payload
from esme_posttrain.sft.launch_instruct import validate_sft_payload
from esme_posttrain.sft.launch_multiturn import validate_multi_turn_payload

REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE_CASES = [
    pytest.param(
        REPO_ROOT / "configs" / "esme-214m-instruct.json",
        lambda payload, path: validate_sft_payload(payload, path, require_base_bundle_exists=False),
        25.0,
        id="instruct-sft",
    ),
    pytest.param(
        REPO_ROOT / "configs" / "esme-214m-sft-multiturn.json",
        lambda payload, path: validate_multi_turn_payload(
            payload, path, require_base_bundle_exists=False
        ),
        40.0,
        id="sft-multiturn",
    ),
    pytest.param(
        REPO_ROOT / "configs" / "esme-214m-chat-dpo.json",
        validate_dpo_payload,
        15.0,
        id="chat-dpo",
    ),
    pytest.param(
        REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json",
        validate_rlvr_payload,
        25.0,
        id="rlvr-grpo",
    ),
]


def _load_payload(config_path: Path) -> dict[str, Any]:
    return copy.deepcopy(json.loads(config_path.read_text(encoding="utf-8")))


@pytest.mark.parametrize(("config_path", "validate", "full_run_cap_usd"), STAGE_CASES)
def test_smoke_cost_over_hardcoded_cap_fails_validation(
    config_path: Path, validate: Any, full_run_cap_usd: float
) -> None:
    payload = _load_payload(config_path)
    payload["runtime"]["smoke_max_cost_usd"] = 2.5

    with pytest.raises(LaunchError, match="smoke_max_cost_usd"):
        validate(payload, config_path)


@pytest.mark.parametrize(("config_path", "validate", "full_run_cap_usd"), STAGE_CASES)
def test_full_run_cost_over_stage_cap_fails_validation(
    config_path: Path, validate: Any, full_run_cap_usd: float
) -> None:
    payload = _load_payload(config_path)
    payload["runtime"]["full_run_max_cost_usd"] = full_run_cap_usd + 1.0

    with pytest.raises(LaunchError, match="full_run_max_cost_usd"):
        validate(payload, config_path)


@pytest.mark.parametrize(("config_path", "validate", "full_run_cap_usd"), STAGE_CASES)
def test_runtime_spend_stop_over_smoke_cap_fails_validation(
    config_path: Path, validate: Any, full_run_cap_usd: float
) -> None:
    payload = _load_payload(config_path)
    payload["runtime"]["runtime_spend_stop_usd"] = 2.5

    with pytest.raises(LaunchError, match="runtime_spend_stop_usd"):
        validate(payload, config_path)


@pytest.mark.parametrize(("config_path", "validate", "full_run_cap_usd"), STAGE_CASES)
def test_timeout_over_modal_maximum_fails_validation(
    config_path: Path, validate: Any, full_run_cap_usd: float
) -> None:
    payload = _load_payload(config_path)
    payload["runtime"]["timeout_hours"] = 25

    with pytest.raises(LaunchError, match="timeout_hours"):
        validate(payload, config_path)
