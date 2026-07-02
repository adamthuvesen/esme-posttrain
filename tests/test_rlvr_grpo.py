from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from esme_posttrain.rl.countdown_lite_baseline import (
    CountdownBaselineProgressError,
    CountdownBaselineRequest,
    run_countdown_lite_baseline,
)
from esme_posttrain.rl.full import run_countdown_lite_grpo_job
from esme_posttrain.rl.launch import EXPECTED_ARTIFACTS_WITH_FINAL_EVAL, load_rlvr_config
from esme_posttrain.rl.pipeline_smoke import run_rlvr_pipeline_smoke
from esme_posttrain.rl.report import build_blocked_grpo_report, build_grpo_report

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_FIXTURE_CONFIG = REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"


def test_countdown_lite_grpo_fixture_job_writes_required_artifacts(tmp_path: Path) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)
    milestones: list[tuple[str, dict[str, Any]]] = []

    payload = run_countdown_lite_grpo_job(
        config,
        output_dir=tmp_path / "grpo-fixture",
        require_cuda=False,
        commit="fixture",
        dirty=False,
        milestone_callback=lambda stage, fields: milestones.append((stage, fields)),
    )

    assert payload["status"] == "modal_full_countdown_lite_grpo_complete"
    # The fixture eval profile is full_eval_1x1, so the run cannot claim an
    # acceptance verdict either way.
    assert payload["grpo_result"] == "not-acceptance-evidence"
    assert payload["gsm8k_lite"]["status"] == "not_run"
    assert payload["wandb_run"] is None
    assert set(payload["required_artifacts_present"]) == set(EXPECTED_ARTIFACTS_WITH_FINAL_EVAL)
    assert all(payload["required_artifacts_present"].values())
    assert (tmp_path / "grpo-fixture" / "bundle" / "manifest.json").is_file()
    assert (tmp_path / "grpo-fixture" / "bundle-final" / "manifest.json").is_file()
    assert (tmp_path / "grpo-fixture" / "eval-after-final.json").is_file()
    assert payload["after_final"] is not None
    assert payload["after_final"]["task_count"] == 1
    assert (tmp_path / "grpo-fixture" / "eval-progress.jsonl").is_file()
    stages = [stage for stage, _fields in milestones]
    assert "trainer_start" in stages
    assert "before_eval_generation_start" in stages
    assert "before_eval_generation_progress" in stages
    assert "before_eval_generation_complete" in stages
    assert "after_eval_generation_start" in stages
    assert "after_eval_generation_progress" in stages
    assert "after_eval_generation_complete" in stages
    before_progress = dict(milestones[stages.index("before_eval_generation_progress")][1])
    assert before_progress["tasks_total"] == 1
    assert before_progress["samples_total"] == 1
    assert before_progress["samples_completed"] == 1


def test_pipeline_smoke_reaches_lifecycle_and_writes_report(tmp_path: Path) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    payload = run_rlvr_pipeline_smoke(
        config,
        output_dir=tmp_path / "pipeline-smoke",
        report_path=tmp_path / "pipeline-smoke-report.json",
        doc_path=tmp_path / "pipeline-smoke-report.md",
        repo_root=REPO_ROOT,
    )

    assert payload["status"] == "pipeline_smoke_complete"
    assert payload["grpo_result"] == "pipeline_smoke_passed"
    assert payload["eval_profile"] == "pipeline_smoke"
    assert payload["paid_compute"] is False
    assert payload["cost"]["estimated_cost_usd"] == 0.0
    assert payload["will_start_modal_job"] is False
    assert payload["modal_gpu_or_paid_work_started"] is False
    assert payload["online_wandb"] is False
    assert (tmp_path / "pipeline-smoke-report.json").is_file()
    assert (tmp_path / "pipeline-smoke-report.md").is_file()
    stages = payload["lifecycle_milestones"]
    assert "before_eval_start" in stages
    assert "trainer_start" in stages
    assert "after_eval_start" in stages
    assert "after_eval_complete" in stages
    report = (tmp_path / "pipeline-smoke-report.json").read_text(encoding="utf-8")
    assert '"pipeline_smoke": true' in report
    assert '"eval_profile": "pipeline_smoke"' in report
    assert '"lifecycle_milestones"' in report
    assert '"modal_gpu_or_paid_work_started": false' in report
    assert '"online_wandb": false' in report
    assert '"recommendation": "review smoke evidence, then decide full acceptance"' in report


def test_countdown_lite_baseline_progress_timeout_fails_loudly(tmp_path: Path) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)
    milestones: list[tuple[str, dict[str, Any]]] = []
    ticks = iter([0.0, 2.0, 2.0])

    def fake_time() -> float:
        return next(ticks, 2.0)

    with pytest.raises(CountdownBaselineProgressError, match="wall timeout"):
        run_countdown_lite_baseline(
            CountdownBaselineRequest(
                manifest_path=config.dataset_manifest_path,
                bundle_path=config.input_bundle_path,
                output_dir=tmp_path / "timeout-baseline",
                samples_per_task=1,
                max_tasks=1,
                max_new_tokens=2,
                progress_label="before_eval",
                progress_callback=lambda stage, fields: milestones.append((stage, fields)),
                wall_timeout_seconds=1,
                no_progress_timeout_seconds=30,
                time_source=fake_time,
            )
        )

    assert milestones[-1][0] == "before_eval_generation_timeout"
    assert milestones[-1][1]["timeout_kind"] == "wall"


def test_countdown_lite_baseline_resumes_from_partial_without_duplicates(
    tmp_path: Path,
) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)
    output_dir = tmp_path / "resume-baseline"
    request = CountdownBaselineRequest(
        manifest_path=config.dataset_manifest_path,
        bundle_path=config.input_bundle_path,
        output_dir=output_dir,
        samples_per_task=1,
        max_tasks=1,
        max_new_tokens=2,
        progress_label="before_eval",
        eval_profile="full_eval_1x1",
        config_hash="fixture-config",
        model_id="fixture-model",
    )

    first = run_countdown_lite_baseline(request)
    resumed = run_countdown_lite_baseline(
        CountdownBaselineRequest(
            **{
                **request.__dict__,
                "resume_from_partial": True,
            }
        )
    )

    partial_path = output_dir / "baseline-partial.jsonl"
    partial_lines = partial_path.read_text(encoding="utf-8").splitlines()
    partial_payload = json.loads(partial_lines[0])
    assert len(partial_lines) == 1
    assert partial_payload["phase"] == "before_eval"
    assert partial_payload["eval_profile"] == "full_eval_1x1"
    assert partial_payload["config_hash"] == "fixture-config"
    assert partial_payload["model_id"] == "fixture-model"
    assert partial_payload["task_start"] == 0
    assert partial_payload["task_end"] == 1
    assert partial_payload["sample_start"] == 0
    assert partial_payload["sample_end"] == 1
    assert first["pass@1"] == resumed["pass@1"]
    assert resumed["partial_path"] == str(partial_path)


def test_one_sample_baseline_reports_only_pass_at_1(tmp_path: Path) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    report = run_countdown_lite_baseline(
        CountdownBaselineRequest(
            manifest_path=config.dataset_manifest_path,
            bundle_path=config.input_bundle_path,
            output_dir=tmp_path / "one-sample-baseline",
            samples_per_task=1,
            max_tasks=1,
            max_new_tokens=2,
        )
    )

    assert "pass@1" in report
    assert "pass@8" not in report
    assert "pass@32" not in report
    for task_result in report["tasks"]:
        assert "pass@1" in task_result
        assert "pass@8" not in task_result
        assert "pass@32" not in task_result
    for bucket in report["difficulty_breakdown"].values():
        assert "pass@1" in bucket
        assert "pass@8" not in bucket
        assert "pass@32" not in bucket
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")
    assert "pass@1" in markdown
    assert "pass@8" not in markdown
    assert "pass@32" not in markdown


def test_countdown_lite_grpo_fixture_logs_wandb_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)
    fake_run = _FakeWandbRun()
    init_payload: dict[str, Any] = {}

    class FakeWandb:
        @staticmethod
        def init(**kwargs: Any) -> _FakeWandbRun:
            init_payload.update(kwargs)
            return fake_run

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    payload = run_countdown_lite_grpo_job(
        config,
        output_dir=tmp_path / "grpo-fixture-wandb",
        require_cuda=False,
        commit="fixture",
        dirty=False,
        wandb_enabled=True,
        wandb_mode="offline",
    )

    assert init_payload["project"] == "esme-posttrain"
    assert init_payload["mode"] == "offline"
    assert "stage=rlvr" in init_payload["tags"]
    assert init_payload["config"]["eval_profile"] == "full_eval_1x1"
    assert payload["wandb_run"] == "https://wandb.local/rlvr-fixture"
    assert fake_run.finished is True
    logged_events = [payload.get("event") for payload, _step in fake_run.logs]
    assert "lifecycle" in logged_events
    assert "eval" in logged_events
    assert "train" in logged_events
    assert all(step is None for _payload, step in fake_run.logs)
    train_payload = next(
        payload for payload, _step in fake_run.logs if payload.get("event") == "train"
    )
    eval_payload = next(
        payload for payload, _step in fake_run.logs if payload.get("event") == "eval"
    )
    lifecycle_payload = next(
        payload for payload, _step in fake_run.logs if payload.get("event") == "lifecycle"
    )
    assert train_payload["train/step"] == 1
    assert eval_payload["eval/step"] == 0
    assert lifecycle_payload["lifecycle/index"] == 1


def test_blocked_grpo_report_requires_explicit_spend_evidence() -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    with pytest.raises(ValueError, match="blocked GRPO reports require spend evidence"):
        build_blocked_grpo_report(
            config,
            reason="fixture blocker",
            launch_command="fixture launch",
            spend_evidence={},
            modal_evidence=_modal_evidence(),
        )


def test_blocked_grpo_report_requires_explicit_modal_evidence() -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    with pytest.raises(ValueError, match="blocked GRPO reports require Modal/status evidence"):
        build_blocked_grpo_report(
            config,
            reason="fixture blocker",
            launch_command="fixture launch",
            spend_evidence=_spend_evidence(),
            modal_evidence={},
        )


def test_generic_grpo_report_rejects_blocked_result_without_cost_evidence() -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    with pytest.raises(ValueError, match="blocked GRPO reports require cost evidence"):
        build_grpo_report(config, {"grpo_result": "blocked-with-evidence"})


def test_blocked_grpo_report_records_spend_and_modal_evidence() -> None:
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    report = build_blocked_grpo_report(
        config,
        reason="fixture blocker",
        launch_command="fixture launch",
        spend_evidence=_spend_evidence(),
        modal_evidence=_modal_evidence(),
    )

    assert report["spend"]["paid_compute"] is True
    assert report["spend"]["actual_or_estimated_cost_usd"] == 0.5
    assert report["spend"]["cost_basis"] == "fixture estimate"
    assert report["spend"]["timeout_cost_ceiling_usd"] == 2.0988
    assert report["modal"]["app_id"] == "ap-fixture"
    assert report["modal"]["call_id"] == "fc-fixture"
    assert report["modal"]["post_stop_status"]["tasks"] == "0"
    assert report["ready_for_hq_inspection"] is True


def _spend_evidence() -> dict[str, object]:
    return {
        "paid_compute": True,
        "actual_or_estimated_cost_usd": 0.5,
        "cost_basis": "fixture estimate",
        "timeout_cost_ceiling_usd": 2.0988,
    }


def _modal_evidence() -> dict[str, object]:
    return {
        "app": "esme-posttrain-rlvr-countdown-grpo",
        "app_id": "ap-fixture",
        "call_id": "fc-fixture",
        "stop_command": "modal app stop ap-fixture --yes",
        "post_stop_status": {
            "command": "modal app list --json",
            "app_id": "ap-fixture",
            "state": "stopped",
            "tasks": "0",
            "summary": "fixture stopped",
        },
        "status_basis": "fixture status evidence",
    }


class _FakeWandbRun:
    url = "https://wandb.local/rlvr-fixture"
    _allowed_attrs = {"logs", "finished"}

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self._allowed_attrs:
            raise Exception(f"Attribute {name} is not supported on Run object.")
        super().__setattr__(name, value)

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, Any], int | None]] = []
        self.finished = False

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.logs.append((payload, step))

    def finish(self) -> None:
        self.finished = True
