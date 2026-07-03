from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from scripts import modal_rlvr_grpo

from esme_posttrain.cli import main
from esme_posttrain.launch.config_guards import LaunchError
from esme_posttrain.rl.launch import (
    build_rlvr_dry_run,
    full_launch_blockers,
    load_rlvr_config,
    pipeline_smoke_grpo_settings,
    validate_rl_task_manifest,
    validate_rlvr_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_CONFIG = REPO_ROOT / "configs" / "esme-214m-rl.json"
RL_FIXTURE_CONFIG = REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"
BASE_BUNDLE_MANIFEST = REPO_ROOT / "exports" / "esme-214m-chat" / "manifest.json"
requires_base_bundle = pytest.mark.skipif(
    not BASE_BUNDLE_MANIFEST.is_file(),
    reason="requires local exports/esme-214m-chat bundle (gitignored, absent in CI)",
)


def test_rlvr_fixture_dry_run_via_api() -> None:
    payload = build_rlvr_dry_run(RL_FIXTURE_CONFIG)

    assert payload["command"] == "rlvr-dry-run"
    assert payload["artifact_name"] == "Esme-214M-RL"
    assert payload["method"] == "grpo"
    assert payload["dataset_type"] == "rl_tasks"
    assert payload["sample_budget"] == 2
    assert payload["token_budget"] == 256
    assert payload["eval_rollouts"] == 1
    assert payload["debug_eval_task_budget"] == 1
    assert payload["debug_samples_per_eval_task"] == 1
    assert payload["wandb_project"] == "esme-posttrain"
    assert payload["wandb_required_for_modal"] is True
    assert "stage=rlvr" in payload["wandb_tags"]
    assert payload["eval_profile"] == "full_eval_1x1"
    assert payload["training_started"] is False
    assert payload["modal_gpu_or_paid_work_started"] is False
    assert payload["will_start_modal_job"] is False
    assert payload["projected_cost_usd"] < 1.0
    assert payload["timeout_cost_ceiling_usd"] == 2.0988
    assert payload["eval_wall_timeout_seconds"] == 60.0
    assert payload["eval_no_progress_timeout_seconds"] == 30.0
    assert payload["acceptance_preflight"]["decision"] == "ready_for_visible_modal_decision"
    assert payload["acceptance_preflight"]["full_acceptance_preserved"] is False
    assert payload["acceptance_preflight"]["total_samples"] == 1
    assert payload["resume_command"] == payload["full_launch_command"]
    assert payload["pipeline_smoke_profile"] == "pipeline_smoke"
    assert "rlvr-pipeline-smoke" in payload["pipeline_smoke_command"]
    assert "--modal-pipeline-smoke" in payload["modal_smoke_command"]
    assert payload["full_launch_blockers"] == ["full Esme-214M-RL GRPO launch requires --approved"]
    assert "uv run --with modal==1.5.1 modal run --detach" in payload["full_launch_command"]
    assert "python -m modal" not in payload["full_launch_command"]


def test_rlvr_fixture_dry_run_via_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rlvr-dry-run", "--config", str(RL_FIXTURE_CONFIG)]) == 0

    output = capsys.readouterr().out
    assert "dry_run: rlvr-dry-run" in output
    assert "method: grpo" in output
    assert "dataset_records_declared: 2" in output
    assert "sample_budget: 2" in output
    assert "token_budget: 256" in output
    assert "timeout_cost_ceiling_usd: 2.0988" in output
    assert "eval_profile: full_eval_1x1" in output
    assert "debug_eval_task_budget: 1" in output
    assert "pipeline_smoke_profile: pipeline_smoke" in output
    assert "wandb_required_for_modal: True" in output
    assert "approval_required: yes" in output
    assert "training_started: no" in output
    assert "modal_gpu_or_paid_work_started: no" in output


def test_full_launch_blockers_clear_with_approval() -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    assert full_launch_blockers(config, approved=True, modal_gpu="A100") == []


def test_full_launch_blocks_modal_gpu_mismatch() -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    assert full_launch_blockers(config, approved=True, modal_gpu="A10G") == [
        "RLVR_MODAL_GPU must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
        "for full-run cost accounting"
    ]


def test_timeout_cost_ceiling_must_stay_under_runtime_hard_stop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["runtime"])["full_run_runtime_spend_stop_usd"] = 1.0
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert (
        "runtime timeout cost ceiling exceeds runtime.full_run_runtime_spend_stop_usd"
        in capsys.readouterr().err
    )


@requires_base_bundle
def test_modal_launcher_dry_run_rejects_timeout_env_config_mismatch() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "RLVR_TIMEOUT_HOURS": "4",
    }

    result = subprocess.run(
        [
            sys.executable,
            "scripts/modal_rlvr_grpo.py",
            "--config",
            "configs/esme-214m-rl.json",
            "--dry-run",
            "--json",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "RLVR_TIMEOUT_HOURS must match runtime.timeout_hours" in result.stderr
    assert "env=4, config=8" in result.stderr
    assert "effective timeout cost ceiling $8.3952" in result.stderr


def test_modal_launcher_full_run_returns_receipt_without_get(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(
        load_rlvr_config(RL_FIXTURE_CONFIG),
        report_path=tmp_path / "grpo-report.json",
        doc_path=tmp_path / "grpo-report.md",
    )
    captured: dict[str, object] = {}

    class FakeCall:
        object_id = "fc-rlvr-test"
        app_id = "ap-rlvr-test"

        def get(self) -> dict[str, object]:
            raise AssertionError("launcher must not wait for the remote GRPO result")

    class FakeRunner:
        def spawn(self, *args: object) -> FakeCall:
            captured["args"] = args
            return FakeCall()

    monkeypatch.setattr(modal_rlvr_grpo, "RLVR_TIMEOUT_HOURS", 1)
    monkeypatch.setattr(modal_rlvr_grpo, "RLVR_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_rlvr_grpo, "modal", object())
    monkeypatch.setattr(modal_rlvr_grpo, "run_modal_grpo", FakeRunner())
    monkeypatch.setattr(modal_rlvr_grpo, "load_rlvr_config", lambda _path: config)

    assert (
        modal_rlvr_grpo.launch(["--config", "unused.json", "--full-run", "--approved", "--json"])
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    assert captured["args"][0] == config.payload
    assert captured["args"][3] == "esme-214m-rlvr-countdown-grpo"
    assert captured["args"][4] is False
    assert captured["args"][5] is False
    assert receipt["status"] == "modal_grpo_launch_in_flight"
    assert receipt["will_start_modal_job"] is True
    assert receipt["debug_before_eval"] is False
    assert receipt["pipeline_smoke"] is False
    assert receipt["eval_profile"] == "full_eval_1x1"
    assert receipt["wandb_project"] == "esme-posttrain"
    assert receipt["wandb_mode"] == "online"
    assert receipt["modal_result_awaited"] is False
    assert receipt["modal_app_id"] == "ap-rlvr-test"
    assert receipt["modal_call_id"] == "fc-rlvr-test"
    assert (
        receipt["modal_logs_command"]
        == "modal app logs ap-rlvr-test --timestamps --show-function-call-id "
        "--show-container-id"
    )
    assert receipt["modal_call_logs_command"].endswith("--function-call fc-rlvr-test")
    assert receipt["modal_stop_command"] == "modal app stop ap-rlvr-test --yes"
    assert receipt["modal_status_command"] == "modal app list --json"
    assert receipt["remote_status_path"].endswith(
        "/_rlvr-launch-status/esme-214m-rlvr-countdown-grpo.jsonl"
    )
    assert receipt["resume_command"] == receipt["full_launch_command"]
    assert receipt["report_path"] == str(tmp_path / "grpo-report.json")

    report = json.loads((tmp_path / "grpo-report.json").read_text(encoding="utf-8"))
    assert report["status"] == "modal_grpo_launch_in_flight"
    assert report["modal"]["app_id"] == "ap-rlvr-test"
    assert report["modal"]["call_id"] == "fc-rlvr-test"
    assert report["modal"]["call_logs_command"].endswith("--function-call fc-rlvr-test")
    assert report["modal"]["stop_command"] == "modal app stop ap-rlvr-test --yes"
    assert report["resume_command"].endswith("--full-run --approved --json")
    assert report["ready_for_hq_inspection"] is False
    assert report["wandb"]["required_for_modal"] is True
    assert report["eval_profile"] == "full_eval_1x1"


def test_modal_launcher_debug_before_eval_dry_run_uses_reduced_probe_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(modal_rlvr_grpo, "RLVR_TIMEOUT_HOURS", 1)
    monkeypatch.setattr(modal_rlvr_grpo, "RLVR_MODAL_GPU", "A100")

    assert (
        modal_rlvr_grpo.launch(
            [
                "--config",
                str(RL_FIXTURE_CONFIG),
                "--debug-before-eval",
                "--dry-run",
                "--approved",
                "--json",
            ]
        )
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "ready_for_before_eval_debug_probe"
    assert receipt["debug_before_eval"] is True
    assert receipt["debug_eval_task_budget"] == 1
    assert receipt["debug_samples_per_eval_task"] == 1
    assert "--debug-before-eval" in receipt["full_launch_command"]
    assert "--full-run" not in receipt["full_launch_command"]
    assert (
        "RLVR_MODAL_OUTPUT_STEM='esme-214m-rlvr-before-eval-debug'"
        in receipt["full_launch_command"]
    )


def test_modal_launcher_pipeline_smoke_dry_run_uses_dedicated_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(modal_rlvr_grpo, "RLVR_TIMEOUT_HOURS", 1)
    monkeypatch.setattr(modal_rlvr_grpo, "RLVR_MODAL_GPU", "A100")

    assert (
        modal_rlvr_grpo.launch(
            [
                "--config",
                str(RL_FIXTURE_CONFIG),
                "--modal-pipeline-smoke",
                "--dry-run",
                "--approved",
                "--json",
            ]
        )
        == 0
    )

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "ready_for_modal_pipeline_smoke"
    assert receipt["pipeline_smoke"] is True
    assert receipt["eval_profile"] == "pipeline_smoke"
    assert "--modal-pipeline-smoke" in receipt["full_launch_command"]
    assert "--full-run" not in receipt["full_launch_command"]
    assert (
        "RLVR_MODAL_OUTPUT_STEM='esme-214m-rlvr-pipeline-smoke'" in receipt["full_launch_command"]
    )


@requires_base_bundle
def test_full_acceptance_eval_stays_30x32_with_scaled_timeout() -> None:
    payload = build_rlvr_dry_run(RL_CONFIG)

    assert payload["eval_profile"] == "full_acceptance_30x32"
    assert payload["eval_task_budget"] == 30
    assert payload["samples_per_eval_task"] == 32
    assert payload["eval_rollouts"] == 960
    assert payload["eval_wall_timeout_seconds"] == 5760.0
    assert payload["eval_no_progress_timeout_seconds"] == 1440.0
    assert "960 samples" in payload["eval_timeout_basis"]
    assert payload["acceptance_preflight"]["full_acceptance_preserved"] is True
    assert payload["acceptance_preflight"]["total_samples"] == 960
    assert payload["acceptance_preflight"]["timeout_margin_seconds"] == 0.0
    config = load_rlvr_config(RL_CONFIG)
    assert pipeline_smoke_grpo_settings(config)["warmup_steps"] == 0


def test_modal_launcher_mounts_wandb_secret() -> None:
    source = (REPO_ROOT / "scripts" / "modal_rlvr_grpo.py").read_text(encoding="utf-8")

    assert 'modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])' in source


def test_modal_launcher_unhydrated_spawn_fails_loudly_without_app_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    class FakeRunner:
        def spawn(self, *args: object) -> object:
            del args
            raise RuntimeError("Function has not been hydrated")

    class FakeApp:
        def run(self) -> object:
            raise AssertionError("app.run fallback must not be used")

    monkeypatch.setattr(modal_rlvr_grpo, "run_modal_grpo", FakeRunner())
    monkeypatch.setattr(modal_rlvr_grpo, "app", FakeApp())

    with pytest.raises(RuntimeError, match="Refusing the old app.run\\(\\) fallback"):
        modal_rlvr_grpo._spawn_modal_grpo(config, "esme-214m-rlvr-countdown-grpo")


def test_missing_input_bundle_fails_loudly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["input_bundle"])["path"] = str(tmp_path / "missing-chat")
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "missing base bundle manifest" in capsys.readouterr().err


def test_missing_dataset_manifest_fails_loudly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["dataset"])["manifest_path"] = str(tmp_path / "missing-manifest.json")
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "config path does not exist" in capsys.readouterr().err


def test_rl_rewards_must_be_verifiable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _load_json(REPO_ROOT / "fixtures" / "manifests" / "rl_tasks_tiny.json")
    reward = _as_object(_as_list(manifest["reward_definitions"])[0])
    reward["name"] = "countdown_lite_exact_solve"
    reward["verifiable"] = False
    data_file = _as_object(_as_list(manifest["data_files"])[0])
    data_file["path"] = str(REPO_ROOT / "fixtures" / "datasets" / "rl_tasks_tiny.jsonl")
    manifest_path = _write_json(tmp_path / "rl_bad.json", manifest)

    config = _load_absolute_fixture_config()
    _as_object(config["dataset"])["manifest_path"] = str(manifest_path)
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "must be verifiable" in capsys.readouterr().err


def test_rl_style_terms_are_not_reward_terms(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _load_json(REPO_ROOT / "fixtures" / "manifests" / "rl_tasks_tiny.json")
    reward = _as_object(_as_list(manifest["reward_definitions"])[0])
    reward["name"] = "friendliness_score"
    data_file = _as_object(_as_list(manifest["data_files"])[0])
    data_file["path"] = str(REPO_ROOT / "fixtures" / "datasets" / "rl_tasks_tiny.jsonl")
    manifest_path = _write_json(tmp_path / "rl_style_bad.json", manifest)

    config = _load_absolute_fixture_config()
    _as_object(config["dataset"])["manifest_path"] = str(manifest_path)
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "uses eval-observation terms" in capsys.readouterr().err


def test_rl_manifest_validates_every_data_file(tmp_path: Path) -> None:
    manifest = _load_json(REPO_ROOT / "fixtures" / "manifests" / "rl_tasks_tiny.json")
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    shutil.copy(
        REPO_ROOT / "fixtures" / "datasets" / "rl_tasks_tiny.jsonl",
        datasets_dir / "rl_tasks_tiny.jsonl",
    )
    _as_list(manifest["data_files"]).append(
        {"path": "../datasets/missing.jsonl", "format": "jsonl", "records": 1}
    )
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    manifest_path = _write_json(manifests_dir / "rl_tasks.json", manifest)

    with pytest.raises(LaunchError, match=r"missing dataset manifest\.data_files\[1\]\.path"):
        validate_rl_task_manifest(manifest_path)


def test_output_dir_must_stay_under_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["artifacts"])["output_dir"] = "artifacts/posttrain/esme-214m-rl"
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "artifacts.output_dir must be a relative path under runs/" in capsys.readouterr().err


def test_gsm8k_lite_must_remain_eval_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["dataset"])["train_on_gsm8k_lite"] = True
    config_path = _write_json(tmp_path / "config.json", config)

    with pytest.raises(SystemExit) as exc:
        main(["rlvr-dry-run", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "dataset.train_on_gsm8k_lite must be false" in capsys.readouterr().err


def test_config_requires_modal_wandb() -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["monitoring"])["wandb_required_for_modal"] = False

    with pytest.raises(LaunchError, match="wandb_required_for_modal must be true"):
        validate_rlvr_payload(config, RL_FIXTURE_CONFIG)


def test_config_requires_stage_rlvr_wandb_tag() -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["monitoring"])["wandb_tags"] = ["countdown-lite"]

    with pytest.raises(LaunchError, match="wandb_tags must include stage=rlvr"):
        validate_rlvr_payload(config, RL_FIXTURE_CONFIG)


def test_config_rejects_removed_clip_epsilon_key() -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["grpo"])["clip_epsilon"] = 0.2

    with pytest.raises(LaunchError, match="grpo has unsupported keys: clip_epsilon"):
        validate_rlvr_payload(config, RL_FIXTURE_CONFIG)


def test_eval_rollouts_must_fit_configured_budget() -> None:
    config = _load_absolute_fixture_config()
    _as_object(config["monitoring"])["samples_per_eval_task"] = 2

    with pytest.raises(LaunchError, match="exceeds budgets.max_eval_rollouts"):
        validate_rlvr_payload(config, RL_FIXTURE_CONFIG)


def test_build_rlvr_dry_run_raises_launch_error_for_bad_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(LaunchError):
        build_rlvr_dry_run(config_path)


def _load_absolute_fixture_config() -> dict[str, object]:
    config = _load_json(RL_FIXTURE_CONFIG)
    config_dir = RL_FIXTURE_CONFIG.parent
    input_bundle = _as_object(config["input_bundle"])
    input_bundle["path"] = str((config_dir / str(input_bundle["path"])).resolve())
    dataset = _as_object(config["dataset"])
    dataset["manifest_path"] = str((config_dir / str(dataset["manifest_path"])).resolve())
    return config


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _as_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value
