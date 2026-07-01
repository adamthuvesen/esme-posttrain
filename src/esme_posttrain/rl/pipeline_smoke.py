"""No-spend RLVR pipeline smoke runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from esme_posttrain.launch.modal_cli import local_git_commit, local_git_dirty
from esme_posttrain.rl.full import CountdownGRPOFullRunError, run_countdown_lite_grpo_job
from esme_posttrain.rl.launch import PIPELINE_SMOKE_PROFILE, RLVRLaunchConfig
from esme_posttrain.rl.report import write_grpo_report_artifacts

REQUIRED_PIPELINE_SMOKE_STAGES = (
    "before_eval_start",
    "trainer_start",
    "after_eval_start",
    "after_eval_complete",
    "return_serialization",
)


def run_rlvr_pipeline_smoke(
    config: RLVRLaunchConfig,
    *,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    doc_path: Path | None = None,
    repo_root: Path | None = None,
    launch_command: str | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or config.config_path.parent.parent).resolve()
    smoke_config = replace(
        config,
        output_dir=(output_dir or config.pipeline_smoke_output_dir).expanduser().resolve(),
        report_path=(report_path or config.pipeline_smoke_report_path).expanduser().resolve(),
        doc_path=(doc_path or config.pipeline_smoke_doc_path).expanduser().resolve(),
    )
    milestones: list[tuple[str, dict[str, Any]]] = []
    payload = run_countdown_lite_grpo_job(
        smoke_config,
        output_dir=smoke_config.output_dir,
        require_cuda=False,
        commit=local_git_commit(repo_root),
        dirty=local_git_dirty(repo_root),
        milestone_callback=lambda stage, fields: milestones.append((stage, fields)),
        wandb_enabled=False,
        wandb_mode="disabled",
        pipeline_smoke=True,
        paid_compute=False,
    )
    stages = tuple(stage for stage, _fields in milestones)
    missing = [stage for stage in REQUIRED_PIPELINE_SMOKE_STAGES if stage not in stages]
    if missing:
        raise CountdownGRPOFullRunError(
            "pipeline_smoke missed lifecycle milestones: " + ", ".join(missing)
        )
    payload = {
        **payload,
        "pipeline_smoke": True,
        "eval_profile": PIPELINE_SMOKE_PROFILE,
        "lifecycle_milestones": list(stages),
        "report_generated": True,
        "will_start_modal_job": False,
        "modal_gpu_or_paid_work_started": False,
        "online_wandb": False,
        "wandb_mode": "disabled",
        "paid_api": False,
        "remote_dataset_download": False,
    }
    write_grpo_report_artifacts(smoke_config, payload, launch_command=launch_command)
    return {
        **payload,
        "report_path": str(smoke_config.report_path),
        "doc_path": str(smoke_config.doc_path),
    }
