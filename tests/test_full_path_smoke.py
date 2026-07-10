from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from esme_posttrain.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    CANONICAL_CONFIG_KEYS,
)
from esme_posttrain.cli import main
from esme_posttrain.dpo.smoke import run_dpo_cpu_fixture
from esme_posttrain.launch.errors import LaunchError
from esme_posttrain.launch.full_path_smoke import (
    load_full_path_smoke_config,
)
from esme_posttrain.sft.smoke_multiturn import run_multi_turn_cpu_fixture
from esme_posttrain.training.checkpointing import load_training_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_PATH_CONFIG = REPO_ROOT / "fixtures" / "configs" / "full-path-cpu-smoke.json"


def test_full_path_fixture_config_loads_validated_stage_configs() -> None:
    config = load_full_path_smoke_config(FULL_PATH_CONFIG)

    assert config.repo_root == REPO_ROOT
    assert config.base_bundle_path == REPO_ROOT / "fixtures" / "tiny_bundle"
    assert config.sft_steps == config.dpo_steps == config.rlvr_steps == 2
    assert config.rlvr_max_rollout_tokens == 512
    assert config.dpo_interrupt_after_step == 1
    assert config.eval_task_budget == config.samples_per_task == 1
    assert config.task_manifest_path == config.rlvr_config.dataset_manifest_path
    assert config.sft_config.run_id == "esme_214m_sft_multiturn"
    assert config.dpo_config.run_id == "esme_214m_chat_dpo"
    assert config.rlvr_config.run_id == "esme_214m_rlvr_countdown_lite_grpo"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update({"unknown": True}), "unsupported keys"),
        (lambda payload: payload["sft"].update({"max_steps": 3}), "sft.max_steps must be 2"),
        (
            lambda payload: payload["dpo"].update({"interrupt_after_step": 2}),
            "interrupt_after_step must be less than",
        ),
        (
            lambda payload: payload.update({"base_bundle_path": "../../../outside"}),
            "escapes the repository",
        ),
        (
            lambda payload: payload.update(
                {"base_bundle_path": str(REPO_ROOT / "fixtures" / "missing-bundle")}
            ),
            "does not exist",
        ),
    ),
)
def test_full_path_fixture_config_rejects_bad_inputs_before_training(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    payload = _portable_config_payload()
    mutation(payload)
    config_path = tmp_path / "full-path.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LaunchError, match=message):
        load_full_path_smoke_config(config_path, repo_root=REPO_ROOT)

    assert list(tmp_path.iterdir()) == [config_path]


def test_sft_to_dpo_fixture_resumes_exactly_like_uninterrupted(tmp_path: Path) -> None:
    config = load_full_path_smoke_config(FULL_PATH_CONFIG)
    sft_dir = tmp_path / "sft"
    sft = run_multi_turn_cpu_fixture(
        config.sft_config,
        output_dir=sft_dir,
        input_bundle_path=config.base_bundle_path,
        max_steps=config.sft_steps,
    )

    assert sft["result"]["steps_completed"] == 2
    sft_manifest = json.loads((sft_dir / "manifest.json").read_text(encoding="utf-8"))
    assert sft_manifest["base_bundle"]["model"]["id"] == "fixture-chat"
    assert sft_manifest["trainer"]["max_steps"] == 2

    reference_checkpoint = sft_dir / "best-checkpoint.pt"
    resumed_dir = tmp_path / "dpo-resumed"
    resumed = run_dpo_cpu_fixture(
        config.dpo_config,
        output_dir=resumed_dir,
        reference_checkpoint_path=reference_checkpoint,
        max_steps=config.dpo_steps,
        interrupt_after_step=config.dpo_interrupt_after_step,
    )
    control_dir = tmp_path / "dpo-uninterrupted"
    control = run_dpo_cpu_fixture(
        config.dpo_config,
        output_dir=control_dir,
        reference_checkpoint_path=reference_checkpoint,
        max_steps=config.dpo_steps,
    )

    assert resumed["interrupted_and_resumed"] is True
    assert resumed["interrupted_after_step"] == 1
    assert resumed["result"]["start_step"] == 1
    assert resumed["result"]["steps_completed"] == 2
    assert resumed["result"]["resumed_from_checkpoint"] == resumed["resume_checkpoint"]
    interruption_checkpoint = load_training_checkpoint(Path(resumed["resume_checkpoint"]))
    assert interruption_checkpoint.step == 1
    assert interruption_checkpoint.data_position == 1
    assert interruption_checkpoint.rng_state is not None
    assert interruption_checkpoint.optimizer_state is not None
    assert interruption_checkpoint.scheduler_state is not None

    resumed_checkpoint = load_training_checkpoint(resumed_dir / "checkpoint.pt")
    control_checkpoint = load_training_checkpoint(control_dir / "checkpoint.pt")
    assert resumed_checkpoint.step == control_checkpoint.step == 2
    assert resumed_checkpoint.data_position == control_checkpoint.data_position == 2
    for name, resumed_tensor in resumed_checkpoint.model.state_dict().items():
        assert torch.equal(resumed_tensor, control_checkpoint.model.state_dict()[name]), name
    _assert_nested_equal(resumed_checkpoint.optimizer_state, control_checkpoint.optimizer_state)
    _assert_nested_equal(resumed_checkpoint.scheduler_state, control_checkpoint.scheduler_state)

    resumed_step_2 = _metric_at_step(resumed_dir / "metrics.jsonl", event="train", step=2)
    control_step_2 = _metric_at_step(control_dir / "metrics.jsonl", event="train", step=2)
    assert resumed_step_2["train/loss"] == control_step_2["train/loss"]
    assert resumed_step_2["train/learning_rate"] == control_step_2["train/learning_rate"]
    assert resumed["result"]["selected_step"] == control["result"]["selected_step"]
    assert resumed["result"]["selected_eval"] == control["result"]["selected_eval"]

    dpo_manifest = json.loads((resumed_dir / "manifest.json").read_text(encoding="utf-8"))
    assert dpo_manifest["reference_bundle"] == sft_manifest
    assert dpo_manifest["trainer"]["max_steps"] == 2


def test_full_path_cpu_smoke_trains_exports_loads_and_scores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "full-path"
    assert (
        main(
            [
                "full-path-cpu-smoke",
                "--config",
                str(FULL_PATH_CONFIG),
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)

    assert report["status"] == "full_path_cpu_smoke_passed"
    assert report["steps"] == {"sft": 2, "dpo": 2, "rlvr": 2}
    assert report["dpo_resume"]["equivalent_to_uninterrupted"] is True
    assert report["paid_compute"] is False
    assert report["remote_dataset_download"] is False
    assert report["wandb_enabled"] is False
    assert report["bundles"]["base"]["schema_version"] == 1
    assert report["bundles"]["dpo"]["schema_version"] == 1
    assert report["bundles"]["rlvr_final"]["schema_version"] == 1
    assert report["bundles"]["dpo"]["file_set"] == [
        "README.md",
        "config.json",
        "manifest.json",
        "tokenizer.json",
        "weights.pt",
    ]
    assert report["bundles"]["rlvr_final"]["file_set"] == report["bundles"]["dpo"]["file_set"]
    _assert_canonical_v1_bundle(Path(report["bundles"]["dpo"]["bundle_dir"]))
    _assert_canonical_v1_bundle(Path(report["bundles"]["rlvr_final"]["bundle_dir"]))
    score = report["downstream_score"]
    assert score["task_id"]
    assert len(score["samples"]) == 1
    assert set(score["samples"][0]) >= {
        "output",
        "is_well_formed",
        "is_valid_expression",
        "is_exact_solve",
    }
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["status"] == report["status"]


def _portable_config_payload() -> dict[str, Any]:
    payload = json.loads(FULL_PATH_CONFIG.read_text(encoding="utf-8"))
    payload = deepcopy(payload)
    payload["base_bundle_path"] = str(REPO_ROOT / "fixtures" / "tiny_bundle")
    payload["sft"]["config_path"] = str(REPO_ROOT / "configs" / "esme-214m-sft-multiturn.json")
    payload["dpo"]["config_path"] = str(REPO_ROOT / "configs" / "esme-214m-chat-dpo.json")
    payload["rlvr"]["config_path"] = str(
        REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"
    )
    payload["evaluation"]["manifest_path"] = str(
        REPO_ROOT / "fixtures" / "manifests" / "rl_tasks_tiny.json"
    )
    return payload


def _assert_canonical_v1_bundle(bundle_dir: Path) -> None:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    weights = torch.load(bundle_dir / "weights.pt", weights_only=False)
    assert manifest["schema_version"] == weights["format_version"] == BUNDLE_SCHEMA_VERSION
    assert manifest["format"] == weights["format"] == BUNDLE_FORMAT
    assert manifest["tokenizer"] == {
        "path": "tokenizer.json",
        "format": "tokenizers-json",
    }
    assert manifest["checkpoint_step"] == weights["checkpoint_step"]
    assert manifest["source_checkpoint"] == weights["source_checkpoint"]
    assert manifest["source_checkpoint_sha256"] == weights["source_checkpoint_sha256"]
    assert len(manifest["source_checkpoint_sha256"]) == 64
    assert set(manifest["files"]) == {"config", "tokenizer", "weights", "readme"}
    assert set(config) == set(CANONICAL_CONFIG_KEYS)
    assert weights["metadata"]["key_format"] == BUNDLE_FORMAT


def _metric_at_step(path: Path, *, event: str, step: int) -> dict[str, Any]:
    matches = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("event") == event
        and json.loads(line).get("step") == step
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right
