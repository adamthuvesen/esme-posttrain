#!/usr/bin/env python3
# ruff: noqa: E402
"""Approval-gated Modal launcher for Countdown-Lite GRPO."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from esme_posttrain.launch.config_guards import (
    IMAGE_PACKAGE_PINS,
    LAUNCH_APPROVAL_FLAG,
    LaunchError,
)
from esme_posttrain.launch.modal_cli import (
    command_with_output_stem,
    format_payload,
    fresh_output_dir,
    local_git_commit,
    local_git_dirty,
    modal_call_id,
    validate_output_stem,
)
from esme_posttrain.rl.full import run_countdown_lite_grpo_job
from esme_posttrain.rl.launch import (
    BEFORE_EVAL_DEBUG_PROFILE,
    MODAL_VOLUME,
    PIPELINE_SMOKE_PROFILE,
    build_eval_profile,
    build_grpo_dry_run,
    full_launch_blockers,
    load_rlvr_config,
    validate_rlvr_payload,
)
from esme_posttrain.rl.pipeline_smoke import run_rlvr_pipeline_smoke
from esme_posttrain.rl.report import (
    build_blocked_grpo_report,
    write_grpo_report,
)

RLVR_TIMEOUT_ENV_VAR = "RLVR_TIMEOUT_HOURS"
RLVR_MODAL_GPU = os.environ.get("RLVR_MODAL_GPU", "A100")
RLVR_TIMEOUT_HOURS = int(float(os.environ.get(RLVR_TIMEOUT_ENV_VAR, "3")))
VOLUME_MOUNT = Path("/posttrain")
MODAL_APP_NAME = "esme-posttrain-rlvr-countdown-grpo"
DEFAULT_MODAL_OUTPUT_STEM = "esme-214m-rlvr-countdown-grpo"
DEFAULT_MODAL_DEBUG_OUTPUT_STEM = "esme-214m-rlvr-before-eval-debug"
DEFAULT_MODAL_PIPELINE_SMOKE_OUTPUT_STEM = "esme-214m-rlvr-pipeline-smoke"
MODAL_OUTPUT_STEM = os.environ.get("RLVR_MODAL_OUTPUT_STEM", DEFAULT_MODAL_OUTPUT_STEM)

try:
    import modal
except ImportError:  # pragma: no cover - Modal is a runtime dependency for launch.
    modal = None

run_modal_grpo = None
app = None

if modal is not None:  # pragma: no cover - exercised by Modal, not local unit tests.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PACKAGE_PINS.items()))
        .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
        .add_local_dir(str(REPO_ROOT / "data"), remote_path="/root/data")
        .add_local_dir(
            str(REPO_ROOT / "exports" / "esme-214m-chat"),
            remote_path="/root/exports/esme-214m-chat",
        )
    )
    posttrain_volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
    app = modal.App(MODAL_APP_NAME)

    @app.function(
        image=image,
        gpu=RLVR_MODAL_GPU,
        timeout=RLVR_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): posttrain_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_grpo(
        config_payload: dict[str, Any],
        commit: str,
        dirty: bool,
        output_stem: str,
        before_eval_only: bool,
        pipeline_smoke: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        status_path: Path | None = None
        try:
            _emit_remote_milestone(
                "remote_entry",
                output_stem=output_stem,
                commit=commit,
                dirty=dirty,
                pipeline_smoke=pipeline_smoke,
            )
            config = validate_rlvr_payload(config_payload, Path("/root/configs/esme-214m-rl.json"))
            _emit_remote_milestone("config_validated", run_id=config.run_id)
            blockers = full_launch_blockers(config, approved=True, modal_gpu=RLVR_MODAL_GPU)
            if blockers:
                _emit_remote_milestone("launch_blocked_inside_modal", blockers=blockers)
                raise RuntimeError("GRPO refused inside Modal: " + "; ".join(blockers))
            output_dir = fresh_output_dir(VOLUME_MOUNT, output_stem)
            status_path = _remote_status_path(output_dir)
            _record_remote_milestone(
                status_path,
                "output_dir_selected",
                output_dir=str(output_dir),
            )
            posttrain_volume.commit()

            def milestone_callback(stage: str, fields: dict[str, Any]) -> None:
                if status_path is None:
                    return
                _write_remote_milestone(status_path, stage, **fields)
                posttrain_volume.commit()

            return run_countdown_lite_grpo_job(
                config,
                output_dir=output_dir,
                require_cuda=True,
                started=started,
                commit=commit,
                dirty=dirty,
                milestone_callback=milestone_callback,
                wandb_enabled=True,
                before_eval_only=before_eval_only,
                pipeline_smoke=pipeline_smoke,
                paid_compute=True,
                skip_acceptance_eval=config.skip_acceptance_eval,
            )
        finally:
            _emit_remote_milestone(
                "remote_exit",
                status_path=str(status_path) if status_path is not None else None,
            )
            posttrain_volume.commit()

    @app.local_entrypoint()
    def main(
        config: str,
        approved: bool = False,
        json: bool = False,
        full_run: bool = False,
        debug_before_eval: bool = False,
        pipeline_smoke: bool = False,
        modal_pipeline_smoke: bool = False,
        dry_run: bool = False,
    ) -> None:
        argv = ["--config", config]
        if dry_run:
            argv.append("--dry-run")
        if full_run:
            argv.append("--full-run")
        if debug_before_eval:
            argv.append("--debug-before-eval")
        if pipeline_smoke:
            argv.append("--pipeline-smoke")
        if modal_pipeline_smoke:
            argv.append("--modal-pipeline-smoke")
        if approved:
            argv.append(LAUNCH_APPROVAL_FLAG)
        if json:
            argv.append("--json")
        raise SystemExit(launch(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_rlvr_grpo.py",
        description="Validate and launch the bounded Esme-214M-RL Countdown-Lite GRPO run.",
    )
    parser.add_argument("--config", required=True, type=Path, help="RLVR GRPO config JSON path.")
    parser.add_argument(LAUNCH_APPROVAL_FLAG, action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never starts Modal.")
    parser.add_argument(
        "--full-run", action="store_true", help="Run the approved bounded GRPO job."
    )
    parser.add_argument(
        "--debug-before-eval",
        action="store_true",
        help="Run only the reduced before-eval debug probe after approval.",
    )
    parser.add_argument(
        "--pipeline-smoke",
        action="store_true",
        help="Run the no-spend local pipeline_smoke lifecycle gate.",
    )
    parser.add_argument(
        "--modal-pipeline-smoke",
        action="store_true",
        help="Launch the approved Modal pipeline_smoke lifecycle gate.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def launch(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode_count = sum(
        bool(value)
        for value in (
            args.full_run,
            args.debug_before_eval,
            args.pipeline_smoke,
            args.modal_pipeline_smoke,
        )
    )
    if mode_count > 1:
        print("RLVR GRPO launch failed: choose exactly one launch mode", file=sys.stderr)
        return 2
    try:
        config = load_rlvr_config(args.config)
        output_stem = _validated_output_stem(
            _selected_output_stem(
                debug_before_eval=args.debug_before_eval,
                modal_pipeline_smoke=args.modal_pipeline_smoke,
            )
        )
        if not args.pipeline_smoke:
            _validate_timeout_env_matches_config(config)
    except (LaunchError, ValueError) as error:
        print(f"RLVR GRPO launch failed: {error}", file=sys.stderr)
        return 2

    if args.pipeline_smoke:
        try:
            payload = run_rlvr_pipeline_smoke(
                config,
                repo_root=REPO_ROOT,
                launch_command=(
                    f"uv run python scripts/modal_rlvr_grpo.py --config {args.config.as_posix()} "
                    "--pipeline-smoke --json"
                ),
            )
        except (LaunchError, ValueError, RuntimeError) as error:
            print(f"RLVR pipeline_smoke failed: {error}", file=sys.stderr)
            return 2
        print(_format_payload(payload, json_output=args.json))
        return 0

    launch_requested = args.full_run or args.debug_before_eval or args.modal_pipeline_smoke
    if args.dry_run or not launch_requested:
        payload = build_grpo_dry_run(
            config,
            full_run_approved=args.approved,
            full_run_modal_gpu=RLVR_MODAL_GPU if launch_requested else None,
        )
        if launch_requested:
            payload = _with_output_stem(
                payload,
                output_stem,
                before_eval_only=args.debug_before_eval,
                modal_pipeline_smoke=args.modal_pipeline_smoke,
            )
            if args.debug_before_eval:
                payload["status"] = (
                    "ready_for_before_eval_debug_probe"
                    if args.approved and not payload["full_launch_blockers"]
                    else "dry_run"
                )
                payload["debug_before_eval"] = True
                payload["debug_eval_task_budget"] = config.payload["monitoring"][
                    "debug_eval_task_budget"
                ]
                payload["debug_samples_per_eval_task"] = config.payload["monitoring"][
                    "debug_samples_per_eval_task"
                ]
            if args.modal_pipeline_smoke:
                payload["status"] = (
                    "ready_for_modal_pipeline_smoke"
                    if args.approved and not payload["full_launch_blockers"]
                    else "dry_run"
                )
                payload["pipeline_smoke"] = True
                payload["eval_profile"] = PIPELINE_SMOKE_PROFILE
        print(_format_payload(payload, json_output=args.json))
        return 0

    blockers = full_launch_blockers(config, approved=args.approved, modal_gpu=RLVR_MODAL_GPU)
    full_launch_command = _launch_command(
        config,
        output_stem,
        before_eval_only=args.debug_before_eval,
        modal_pipeline_smoke=args.modal_pipeline_smoke,
    )
    if blockers:
        reason = "; ".join(blockers)
        report = build_blocked_grpo_report(
            config,
            reason=reason,
            launch_command=full_launch_command,
            spend_evidence=_zero_spend_evidence("refused before Modal spawn"),
            modal_evidence=_modal_evidence(
                status_basis="refused before Modal spawn; no Modal call was created"
            ),
        )
        write_grpo_report(config, report)
        payload = {
            "status": _refused_status(
                before_eval_only=args.debug_before_eval,
                modal_pipeline_smoke=args.modal_pipeline_smoke,
            ),
            "will_start_modal_job": False,
            "full_launch_blockers": blockers,
            "full_launch_command": full_launch_command,
            "report_path": str(config.report_path),
            "doc_path": str(config.doc_path),
        }
        print(_format_payload(payload, json_output=args.json))
        return 2
    if modal is None or run_modal_grpo is None:
        reason = "modal is not installed"
        report = build_blocked_grpo_report(
            config,
            reason=reason,
            launch_command=full_launch_command,
            spend_evidence=_zero_spend_evidence("refused before Modal spawn"),
            modal_evidence=_modal_evidence(
                status_basis="modal import failed before Modal spawn; no Modal call was created"
            ),
        )
        write_grpo_report(config, report)
        print(f"RLVR GRPO launch failed: {reason}", file=sys.stderr)
        return 2

    try:
        function_call = _spawn_modal_grpo(
            config,
            output_stem,
            before_eval_only=args.debug_before_eval,
            pipeline_smoke=args.modal_pipeline_smoke,
        )
    except Exception as error:
        error_text = str(error) or repr(error)
        reason = f"Modal GRPO failed before FunctionCall receipt: {error_text}"
        report = build_blocked_grpo_report(
            config,
            reason=reason,
            launch_command=full_launch_command,
            spend_evidence=_zero_spend_evidence("spawn failed before FunctionCall receipt"),
            modal_evidence=_modal_evidence(
                app_id=_modal_app_id(),
                status_basis="Modal spawn failed before a FunctionCall id was returned",
            ),
        )
        write_grpo_report(config, report)
        print(f"RLVR GRPO launch failed: {reason}", file=sys.stderr)
        return 2

    call_id = modal_call_id(function_call)
    app_id = _modal_app_id(function_call)
    modal_evidence = _modal_evidence(
        app_id=app_id,
        call_id=call_id,
        status_basis="Modal call spawned; launch receipt returned without waiting for result",
    )
    spend_evidence = _launched_spend_evidence(config)
    in_flight_status = _in_flight_status(
        before_eval_only=args.debug_before_eval,
        modal_pipeline_smoke=args.modal_pipeline_smoke,
    )
    in_flight_reason = _in_flight_reason(
        before_eval_only=args.debug_before_eval,
        modal_pipeline_smoke=args.modal_pipeline_smoke,
    )
    in_flight_report = build_blocked_grpo_report(
        config,
        reason=in_flight_reason,
        launch_command=full_launch_command,
        spend_evidence=spend_evidence,
        modal_evidence=modal_evidence,
        status=in_flight_status,
        ready_for_hq_inspection=False,
    )
    write_grpo_report(config, in_flight_report)
    receipt = _launch_receipt(
        config,
        output_stem=output_stem,
        full_launch_command=full_launch_command,
        app_id=app_id,
        call_id=call_id,
        before_eval_only=args.debug_before_eval,
        pipeline_smoke=args.modal_pipeline_smoke,
    )
    print(_format_payload(receipt, json_output=args.json))
    return 0


def _spawn_modal_grpo(
    config: Any,
    output_stem: str,
    *,
    before_eval_only: bool = False,
    pipeline_smoke: bool = False,
) -> Any:
    if run_modal_grpo is None:
        raise RuntimeError("Modal function is not initialized")
    try:
        return run_modal_grpo.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            output_stem,
            before_eval_only,
            pipeline_smoke,
        )
    except Exception as error:
        if "has not been hydrated" not in str(error):
            raise
        raise RuntimeError(
            "Modal function is not hydrated; use the checked-in "
            "`modal run --detach scripts/modal_rlvr_grpo.py ...` launch command. "
            "Refusing the old app.run() fallback."
        ) from error


def _validate_timeout_env_matches_config(config: Any) -> None:
    config_timeout_hours = int(config.runtime["timeout_hours"])
    if config_timeout_hours == RLVR_TIMEOUT_HOURS:
        return
    usd_per_hour = float(config.selected_gpu_profile["usd_per_hour"])
    effective_ceiling = RLVR_TIMEOUT_HOURS * usd_per_hour
    configured_ceiling = config_timeout_hours * usd_per_hour
    hard_stop = float(config.runtime["full_run_runtime_spend_stop_usd"])
    raise LaunchError(
        f"{RLVR_TIMEOUT_ENV_VAR} must match runtime.timeout_hours for bounded GRPO launch: "
        f"env={RLVR_TIMEOUT_HOURS}, config={config_timeout_hours}; "
        f"effective timeout cost ceiling ${effective_ceiling:.4f}, "
        f"configured ceiling ${configured_ceiling:.4f}, "
        f"runtime hard stop ${hard_stop:.2f}"
    )


def _zero_spend_evidence(cost_basis: str) -> dict[str, Any]:
    return {
        "paid_compute": False,
        "actual_or_estimated_cost_usd": 0.0,
        "cost_basis": cost_basis,
        "timeout_cost_ceiling_usd": 0.0,
    }


def _launched_spend_evidence(config: Any) -> dict[str, Any]:
    timeout_cost_ceiling = _timeout_cost_ceiling_usd(config)
    return {
        "paid_compute": True,
        "actual_or_estimated_cost_usd": timeout_cost_ceiling,
        "cost_basis": (
            "timeout cost ceiling from configured Modal GPU hourly price and timeout; "
            "remote cost.json replaces this on durable success"
        ),
        "timeout_cost_ceiling_usd": timeout_cost_ceiling,
    }


def _timeout_cost_ceiling_usd(config: Any) -> float:
    return float(config.runtime["timeout_hours"]) * float(
        config.selected_gpu_profile["usd_per_hour"]
    )


def _modal_evidence(
    *,
    app_id: str | None = None,
    call_id: str | None = None,
    status_basis: str,
) -> dict[str, Any]:
    logs_command, call_logs_command = _modal_log_commands(app_id=app_id, call_id=call_id)
    return {
        "app": MODAL_APP_NAME,
        "app_id": app_id,
        "call_id": call_id,
        "logs_command": logs_command,
        "call_logs_command": call_logs_command,
        "stop_command": _modal_stop_command(app_id),
        "status_command": _modal_status_command(),
        "post_stop_status": None,
        "status_basis": status_basis,
    }


def _modal_app_id(function_call: Any | None = None) -> str | None:
    for owner in (function_call, app):
        if owner is None:
            continue
        for name in ("app_id", "_app_id"):
            value = getattr(owner, name, None)
            if value:
                return str(value)
    return None


def _launch_receipt(
    config: Any,
    *,
    output_stem: str,
    full_launch_command: str,
    app_id: str | None,
    call_id: str | None,
    before_eval_only: bool,
    pipeline_smoke: bool,
) -> dict[str, Any]:
    logs_command, call_logs_command = _modal_log_commands(app_id=app_id, call_id=call_id)
    eval_profile = _receipt_eval_profile(
        config, before_eval_only=before_eval_only, modal_pipeline_smoke=pipeline_smoke
    )
    return {
        "status": _in_flight_status(
            before_eval_only=before_eval_only, modal_pipeline_smoke=pipeline_smoke
        ),
        "will_start_modal_job": True,
        "debug_before_eval": before_eval_only,
        "pipeline_smoke": pipeline_smoke,
        "eval_profile": eval_profile,
        "wandb_project": config.payload["monitoring"]["wandb_project"],
        "wandb_required_for_modal": config.payload["monitoring"]["wandb_required_for_modal"],
        "wandb_mode": "online",
        "training_started": "scheduled",
        "modal_gpu_or_paid_work_started": True,
        "modal_result_awaited": False,
        "modal_app": MODAL_APP_NAME,
        "modal_app_id": app_id,
        "modal_call_id": call_id,
        "modal_logs_command": logs_command,
        "modal_call_logs_command": call_logs_command,
        "modal_stop_command": _modal_stop_command(app_id),
        "modal_status_command": _modal_status_command(),
        "remote_status_path": str(_remote_status_path(VOLUME_MOUNT / output_stem)),
        "full_launch_command": full_launch_command,
        "resume_command": full_launch_command,
        "volume": config.runtime["modal_volume"],
        "volume_output_dir": str(VOLUME_MOUNT / output_stem),
        "output_dir": str(config.output_dir),
        "projected_cost_usd": config.estimated_full_cost_usd,
        "runtime_spend_stop_usd": config.runtime["full_run_runtime_spend_stop_usd"],
        "timeout_cost_ceiling_usd": _timeout_cost_ceiling_usd(config),
        "report_path": str(config.report_path),
        "doc_path": str(config.doc_path),
    }


def _modal_log_commands(*, app_id: str | None, call_id: str | None) -> tuple[str, str | None]:
    app_identifier = app_id or MODAL_APP_NAME
    logs_command = (
        f"modal app logs {app_identifier} --timestamps --show-function-call-id --show-container-id"
    )
    call_logs_command = f"{logs_command} --function-call {call_id}" if call_id else None
    return logs_command, call_logs_command


def _modal_stop_command(app_id: str | None) -> str | None:
    if not app_id:
        return None
    return f"modal app stop {app_id} --yes"


def _modal_status_command() -> str:
    return "modal app list --json"


def _emit_remote_milestone(stage: str, **fields: Any) -> None:
    payload = {"event": "rlvr_modal_milestone", "stage": stage, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _remote_status_path(output_dir: Path) -> Path:
    return VOLUME_MOUNT / "_rlvr-launch-status" / f"{output_dir.name}.jsonl"


def _record_remote_milestone(path: Path, stage: str, **fields: Any) -> None:
    _emit_remote_milestone(stage, **fields)
    _write_remote_milestone(path, stage, **fields)


def _write_remote_milestone(path: Path, stage: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "rlvr_modal_milestone",
        "stage": stage,
        "monotonic_seconds": time.perf_counter(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _validated_output_stem(value: str) -> str:
    return validate_output_stem(value, env_var="RLVR_MODAL_OUTPUT_STEM")


def _selected_output_stem(*, debug_before_eval: bool, modal_pipeline_smoke: bool) -> str:
    if MODAL_OUTPUT_STEM != DEFAULT_MODAL_OUTPUT_STEM:
        return MODAL_OUTPUT_STEM
    if debug_before_eval:
        return DEFAULT_MODAL_DEBUG_OUTPUT_STEM
    if modal_pipeline_smoke:
        return DEFAULT_MODAL_PIPELINE_SMOKE_OUTPUT_STEM
    return MODAL_OUTPUT_STEM


def _launch_command(
    config: Any,
    output_stem: str,
    *,
    before_eval_only: bool,
    modal_pipeline_smoke: bool,
) -> str:
    command = config.full_launch_command
    if before_eval_only:
        command = command.replace(" --full-run ", " --debug-before-eval ")
    if modal_pipeline_smoke:
        command = command.replace(" --full-run ", " --modal-pipeline-smoke ")
    return _full_launch_command(command, output_stem)


def _full_launch_command(command: str, output_stem: str) -> str:
    return command_with_output_stem(
        command,
        output_stem=output_stem,
        default_stem=DEFAULT_MODAL_OUTPUT_STEM,
        env_var="RLVR_MODAL_OUTPUT_STEM",
    )


def _with_output_stem(
    payload: dict[str, Any],
    output_stem: str,
    *,
    before_eval_only: bool = False,
    modal_pipeline_smoke: bool = False,
) -> dict[str, Any]:
    command = _full_launch_command(str(payload["full_launch_command"]), output_stem)
    if before_eval_only:
        command = command.replace(" --full-run ", " --debug-before-eval ")
    if modal_pipeline_smoke:
        command = command.replace(" --full-run ", " --modal-pipeline-smoke ")
    return {
        **payload,
        "full_launch_command": command,
        "volume_output_dir": str(VOLUME_MOUNT / output_stem),
    }


def _refused_status(*, before_eval_only: bool, modal_pipeline_smoke: bool) -> str:
    if before_eval_only:
        return "debug_before_eval_refused"
    if modal_pipeline_smoke:
        return "modal_pipeline_smoke_refused"
    return "full_run_refused"


def _in_flight_status(*, before_eval_only: bool, modal_pipeline_smoke: bool) -> str:
    if before_eval_only:
        return "modal_rlvr_before_eval_probe_in_flight"
    if modal_pipeline_smoke:
        return "modal_pipeline_smoke_in_flight"
    return "modal_grpo_launch_in_flight"


def _in_flight_reason(*, before_eval_only: bool, modal_pipeline_smoke: bool) -> str:
    if before_eval_only:
        return "Modal RLVR before-eval debug probe spawned; final result intentionally not awaited"
    if modal_pipeline_smoke:
        return "Modal RLVR pipeline_smoke spawned; final result intentionally not awaited"
    return "Modal GRPO call spawned; final result intentionally not awaited by launcher"


def _receipt_eval_profile(
    config: Any, *, before_eval_only: bool, modal_pipeline_smoke: bool
) -> str:
    if before_eval_only:
        return BEFORE_EVAL_DEBUG_PROFILE
    if modal_pipeline_smoke:
        return PIPELINE_SMOKE_PROFILE
    return str(build_eval_profile(config)["profile"])


def _format_payload(payload: dict[str, Any], *, json_output: bool) -> str:
    return format_payload(
        payload,
        json_output=json_output,
        keys=(
            "pipeline_smoke",
            "eval_profile",
            "will_start_modal_job",
            "modal_result_awaited",
            "modal_app",
            "modal_app_id",
            "modal_call_id",
            "modal_logs_command",
            "modal_call_logs_command",
            "full_launch_command",
            "output_dir",
            "bundle_dir",
            "report_path",
            "doc_path",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(launch())
