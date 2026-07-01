#!/usr/bin/env python3
"""Approval-gated Modal launcher for the multi-turn SFT foundation.

Mirrors the Instruct launch contract (``scripts/modal_instruct_sft.py``): the
dry-run never starts Modal, the smoke is capped at $2 with no ``SFT_MODAL_GPU``
spend bypass, the full run refuses without ``--approved`` and bounded-matched
learning-gate evidence, and everything writes to a separate Volume and output
stem (``esme-214m-sft-multiturn``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
from esme_posttrain.sft.full_multiturn import run_full_multi_turn_sft
from esme_posttrain.sft.launch_multiturn import (
    EXPECTED_ARTIFACTS,
    build_multi_turn_dry_run,
    full_launch_blockers,
    load_multi_turn_config,
    smoke_launch_blockers,
    validate_multi_turn_payload,
)
from esme_posttrain.sft.probe_multiturn import (
    DEFAULT_MODAL_PROBE_ROOT,
    PROBE_TIMEOUT_HOURS,
    build_multi_turn_probe_preflight,
    multi_turn_probe_blockers,
    run_multi_turn_throughput_probe,
)
from esme_posttrain.sft.smoke_multiturn import refresh_manifest_files, run_multi_turn_cpu_fixture
from esme_posttrain.sft.sweep_multiturn import (
    DEFAULT_MODAL_SWEEP_ROOT,
    SWEEP_TIMEOUT_HOURS,
    build_multi_turn_sweep_preflight,
    multi_turn_sweep_blockers,
    run_multi_turn_interval_eval_sweep,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_BUNDLE_LOCAL = Path("/Users/adamthuvesen/dev/menti/esme-pretrain/exports/esme-214m-base")
BASE_BUNDLE_REMOTE = Path("/root/esme-214m-base")
SFT_MODAL_GPU = os.environ.get("SFT_MODAL_GPU", "A100")
SFT_TIMEOUT_HOURS = int(float(os.environ.get("SFT_TIMEOUT_HOURS", "1")))
SFT_SWEEP_TIMEOUT_HOURS = int(float(os.environ.get("SFT_SWEEP_TIMEOUT_HOURS", SWEEP_TIMEOUT_HOURS)))
SFT_PROBE_TIMEOUT_HOURS = int(float(os.environ.get("SFT_PROBE_TIMEOUT_HOURS", PROBE_TIMEOUT_HOURS)))
VOLUME_MOUNT = Path("/posttrain")
MODAL_APP_NAME = "esme-posttrain-esme-sft-multiturn"
MODAL_SMOKE_OUTPUT_STEM = "esme-214m-sft-multiturn-modal-smoke"
DEFAULT_MODAL_FULL_OUTPUT_STEM = "esme-214m-sft-multiturn-full"
MODAL_VOLUME_NAME = "esme-posttrain-esme-sft-multiturn"
MODAL_FULL_OUTPUT_STEM = os.environ.get(
    "SFT_MODAL_FULL_OUTPUT_STEM", DEFAULT_MODAL_FULL_OUTPUT_STEM
)
MODAL_SWEEP_OUTPUT_ROOT = DEFAULT_MODAL_SWEEP_ROOT
MODAL_PROBE_OUTPUT_ROOT = DEFAULT_MODAL_PROBE_ROOT

try:
    import modal
except ImportError:  # pragma: no cover - Modal is a runtime dependency for launch.
    modal = None

run_modal_full_sft = None
run_modal_sweep = None
run_modal_throughput_probe = None


if modal is not None:  # pragma: no cover - exercised by Modal, not local unit tests.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PACKAGE_PINS.items()))
        .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(BASE_BUNDLE_LOCAL), remote_path=str(BASE_BUNDLE_REMOTE))
    )
    posttrain_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
    app = modal.App(MODAL_APP_NAME)

    @app.function(
        image=image,
        gpu=SFT_MODAL_GPU,
        timeout=SFT_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): posttrain_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_smoke(config_payload: dict[str, Any], commit: str, dirty: bool) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_smoke_body(
                config_payload, commit=commit, dirty=dirty, started=started
            )
        finally:
            posttrain_volume.commit()

    @app.function(
        image=image,
        gpu=SFT_MODAL_GPU,
        timeout=SFT_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): posttrain_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_full_sft(
        config_payload: dict[str, Any],
        commit: str,
        dirty: bool,
        resume: bool,
        modal_gpu: str,
        output_stem: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_full_sft_body(
                config_payload,
                commit=commit,
                dirty=dirty,
                started=started,
                resume=resume,
                modal_gpu=modal_gpu,
                output_stem=output_stem,
            )
        finally:
            posttrain_volume.commit()

    @app.function(
        image=image,
        gpu=SFT_MODAL_GPU,
        timeout=SFT_SWEEP_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): posttrain_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_sweep(
        config_payload: dict[str, Any],
        commit: str,
        dirty: bool,
        modal_gpu: str,
        timeout_hours: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_sweep_body(
                config_payload,
                commit=commit,
                dirty=dirty,
                modal_gpu=modal_gpu,
                timeout_hours=timeout_hours,
                started=started,
            )
        finally:
            posttrain_volume.commit()

    @app.function(
        image=image,
        gpu=SFT_MODAL_GPU,
        timeout=SFT_PROBE_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): posttrain_volume},
    )
    def run_modal_throughput_probe(
        config_payload: dict[str, Any],
        commit: str,
        dirty: bool,
        modal_gpu: str,
        timeout_hours: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_throughput_probe_body(
                config_payload,
                commit=commit,
                dirty=dirty,
                modal_gpu=modal_gpu,
                timeout_hours=timeout_hours,
                started=started,
            )
        finally:
            posttrain_volume.commit()

    @app.local_entrypoint()
    def main(
        config: str,
        approved: bool = False,
        json: bool = False,
        full_run: bool = False,
        modal_sweep: bool = False,
        throughput_probe: bool = False,
        dry_run: bool = False,
        resume: bool = False,
    ) -> None:
        argv = ["--config", config]
        if dry_run:
            argv.append("--dry-run")
            if modal_sweep:
                argv.append("--modal-sweep")
            if throughput_probe:
                argv.append("--throughput-probe")
            if full_run:
                argv.append("--full-run")
        elif full_run:
            argv.append("--full-run")
        elif throughput_probe:
            argv.append("--throughput-probe")
        elif modal_sweep:
            argv.append("--modal-sweep")
        else:
            argv.append("--modal-smoke")
        if approved:
            argv.append(LAUNCH_APPROVAL_FLAG)
        if resume:
            argv.append("--resume")
        if json:
            argv.append("--json")
        raise SystemExit(launch(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_chat_sft.py",
        description="Validate and launch the approval-gated Esme multi-turn SFT smoke.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Multi-turn SFT config JSON.")
    parser.add_argument(LAUNCH_APPROVAL_FLAG, action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never starts Modal.")
    parser.add_argument(
        "--local-cpu-smoke",
        action="store_true",
        help="Run the no-spend local multi-turn CPU fixture and write evidence.",
    )
    parser.add_argument(
        "--modal-smoke", action="store_true", help="Launch the capped Modal smoke after approval."
    )
    parser.add_argument(
        "--modal-sweep",
        action="store_true",
        help="Launch the approved bounded matched-eval multi-turn Modal LR sweep.",
    )
    parser.add_argument(
        "--throughput-probe",
        action="store_true",
        help="Run the approved bounded 2048-len multi-turn throughput probe.",
    )
    parser.add_argument(
        "--full-run", action="store_true", help="Launch the approved, capped full Modal SFT run."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="For --full-run, resume from the latest checkpoint in the stable Volume output dir.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def launch(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode_count = sum(
        bool(v)
        for v in (
            args.local_cpu_smoke,
            args.modal_smoke,
            args.full_run,
            args.modal_sweep,
            args.throughput_probe,
        )
    )
    if mode_count > 1:
        print("multi-turn SFT launch failed: choose exactly one launch mode", file=sys.stderr)
        return 2
    if args.resume and not args.full_run:
        print("multi-turn SFT launch failed: --resume requires --full-run", file=sys.stderr)
        return 2
    try:
        config = load_multi_turn_config(args.config)
    except LaunchError as error:
        print(f"multi-turn SFT launch failed: {error}", file=sys.stderr)
        return 2
    try:
        full_output_stem = _validated_full_output_stem(MODAL_FULL_OUTPUT_STEM)
    except ValueError as error:
        print(f"multi-turn SFT launch failed: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.throughput_probe:
            payload = build_multi_turn_probe_preflight(
                config, modal_gpu=SFT_MODAL_GPU, timeout_hours=SFT_PROBE_TIMEOUT_HOURS
            )
        elif args.modal_sweep:
            payload = build_multi_turn_sweep_preflight(
                config, timeout_hours=SFT_SWEEP_TIMEOUT_HOURS, modal_gpu=SFT_MODAL_GPU
            )
        else:
            payload = build_multi_turn_dry_run(
                config,
                full_run_approved=args.approved,
                full_run_modal_gpu=SFT_MODAL_GPU if args.full_run else None,
            )
            if args.full_run:
                payload = _with_full_output_stem(payload, config, full_output_stem)
        print(_format_payload(payload, json_output=args.json))
        return 0
    if args.local_cpu_smoke:
        payload = run_multi_turn_cpu_fixture(config)
        print(_format_payload(payload, json_output=args.json))
        return 0
    if args.throughput_probe:
        return _launch_throughput_probe(config, approved=args.approved, json_output=args.json)
    if args.modal_sweep:
        return _launch_modal_sweep(config, approved=args.approved, json_output=args.json)
    if args.full_run:
        return _launch_full_run(
            config,
            approved=args.approved,
            json_output=args.json,
            resume=args.resume,
            output_stem=full_output_stem,
        )
    if not args.modal_smoke:
        print(_format_payload(build_multi_turn_dry_run(config), json_output=args.json))
        return 0
    if not args.approved:
        print(
            f"multi-turn SFT Modal smoke refused: pass {LAUNCH_APPROVAL_FLAG} after chat approval",
            file=sys.stderr,
        )
        return 2
    blockers = smoke_launch_blockers(config)
    if blockers:
        print("multi-turn SFT Modal smoke refused: " + "; ".join(blockers), file=sys.stderr)
        return 2
    if modal is None:
        print("multi-turn SFT Modal smoke failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_smoke.spawn(
            config.payload, local_git_commit(REPO_ROOT), local_git_dirty(REPO_ROOT)
        )
        call_id = modal_call_id(function_call)
        result = function_call.get()
        result.update({"modal_app": MODAL_APP_NAME, "modal_call_id": call_id})
    except Exception as error:
        print(f"multi-turn SFT Modal smoke failed before durable success: {error}", file=sys.stderr)
        return 2
    print(_format_payload(result, json_output=args.json))
    return 0


def _launch_throughput_probe(config: Any, *, approved: bool, json_output: bool) -> int:
    preflight = build_multi_turn_probe_preflight(
        config, modal_gpu=SFT_MODAL_GPU, timeout_hours=SFT_PROBE_TIMEOUT_HOURS
    )
    if not approved:
        payload = {
            **preflight,
            "status": "throughput_probe_refused",
            "launch_blockers": [f"throughput probe requires {LAUNCH_APPROVAL_FLAG}"],
        }
        print(_format_payload(payload, json_output=json_output))
        return 2
    blockers = multi_turn_probe_blockers(
        modal_gpu=SFT_MODAL_GPU, timeout_hours=SFT_PROBE_TIMEOUT_HOURS
    )
    if blockers:
        payload = {**preflight, "status": "throughput_probe_refused", "launch_blockers": blockers}
        print(_format_payload(payload, json_output=json_output))
        return 2
    if modal is None or run_modal_throughput_probe is None:
        print("multi-turn SFT throughput probe failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_throughput_probe.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            SFT_MODAL_GPU,
            SFT_PROBE_TIMEOUT_HOURS,
        )
        call_id = modal_call_id(function_call)
        result = function_call.get()
        result.update(
            {
                "modal_app": MODAL_APP_NAME,
                "modal_call_id": call_id,
                "throughput_probe_command": preflight["throughput_probe_command"],
            }
        )
    except Exception as error:
        print(
            f"multi-turn SFT throughput probe failed before durable success: {error}",
            file=sys.stderr,
        )
        return 2
    print(_format_payload(result, json_output=json_output))
    return 0


def _launch_modal_sweep(config: Any, *, approved: bool, json_output: bool) -> int:
    preflight = build_multi_turn_sweep_preflight(
        config, timeout_hours=SFT_SWEEP_TIMEOUT_HOURS, modal_gpu=SFT_MODAL_GPU
    )
    if not approved:
        payload = {
            **preflight,
            "status": "modal_sweep_refused",
            "launch_blockers": [f"bounded Modal interval sweep requires {LAUNCH_APPROVAL_FLAG}"],
        }
        print(_format_payload(payload, json_output=json_output))
        return 2
    blockers = multi_turn_sweep_blockers(
        config, timeout_hours=SFT_SWEEP_TIMEOUT_HOURS, modal_gpu=SFT_MODAL_GPU
    )
    if blockers:
        payload = {**preflight, "status": "modal_sweep_refused", "launch_blockers": blockers}
        print(_format_payload(payload, json_output=json_output))
        return 2
    if modal is None or run_modal_sweep is None:
        print("multi-turn SFT Modal sweep failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_sweep.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            SFT_MODAL_GPU,
            SFT_SWEEP_TIMEOUT_HOURS,
        )
        call_id = modal_call_id(function_call)
        result = function_call.get()
        result.update(
            {
                "modal_app": MODAL_APP_NAME,
                "modal_call_id": call_id,
                "modal_sweep_command": preflight["modal_sweep_command"],
            }
        )
    except Exception as error:
        print(f"multi-turn SFT Modal sweep failed before durable success: {error}", file=sys.stderr)
        return 2
    print(_format_payload(result, json_output=json_output))
    return 0


def _launch_full_run(
    config: Any, *, approved: bool, json_output: bool, resume: bool, output_stem: str
) -> int:
    full_launch_command = _full_launch_command(config, output_stem)
    blockers = full_launch_blockers(config, approved=approved, modal_gpu=SFT_MODAL_GPU)
    if blockers:
        payload = {
            "status": "full_run_refused",
            "will_start_modal_job": False,
            "full_launch_blockers": blockers,
            "full_launch_command": full_launch_command,
            "volume_output_dir": str(VOLUME_MOUNT / output_stem),
            "resume": resume,
        }
        print(_format_payload(payload, json_output=json_output))
        return 2
    if modal is None or run_modal_full_sft is None:
        print("multi-turn SFT full run failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_full_sft.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            resume,
            SFT_MODAL_GPU,
            output_stem,
        )
    except Exception as error:
        print(f"multi-turn SFT full run failed before Modal spawn: {error}", file=sys.stderr)
        return 2
    call_id = modal_call_id(function_call)
    payload = {
        "status": "modal_full_sft_launched",
        "will_start_modal_job": True,
        "modal_app": MODAL_APP_NAME,
        "modal_call_id": call_id,
        "full_launch_command": full_launch_command,
        "resume": resume,
        "training_mode": "resumed" if resume else "fresh",
        "volume": config.runtime["modal_volume"],
        "volume_output_dir": str(VOLUME_MOUNT / output_stem),
        "projected_cost_usd": config.estimated_full_cost_usd,
        "runtime_spend_stop_usd": config.runtime["full_run_runtime_spend_stop_usd"],
        "wandb_project": config.payload["monitoring"]["wandb_project"],
    }
    print(_format_payload(payload, json_output=json_output))
    return 0


def _run_modal_smoke_body(
    config_payload: dict[str, Any], *, commit: str, dirty: bool, started: float
) -> dict[str, Any]:
    config = validate_multi_turn_payload(
        config_payload,
        Path("configs/esme-214m-sft-multiturn.json"),
        require_base_bundle_exists=False,
    )
    output_dir = fresh_output_dir(VOLUME_MOUNT, MODAL_SMOKE_OUTPUT_STEM)
    payload = run_multi_turn_cpu_fixture(config, output_dir=output_dir, wandb_enabled=True)
    elapsed = time.perf_counter() - started
    selected_profile = config.selected_gpu_profile
    estimated_cost = elapsed * float(selected_profile["usd_per_hour"]) / 3600.0
    cost = {
        "paid_compute": True,
        "elapsed_seconds": elapsed,
        "selected_gpu": config.runtime["selected_gpu"],
        "usd_per_hour": selected_profile["usd_per_hour"],
        "estimated_cost_usd": estimated_cost,
        "runtime_spend_stop_usd": config.runtime["runtime_spend_stop_usd"],
    }
    if estimated_cost > float(config.runtime["smoke_max_cost_usd"]):
        raise RuntimeError("multi-turn Modal smoke exceeded the approved $2 cap")
    (output_dir / "cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    refresh_manifest_files(output_dir, EXPECTED_ARTIFACTS)
    payload.update(
        {
            "status": "modal_smoke_complete",
            "commit": commit,
            "dirty": dirty,
            "cost": cost,
            "output_dir": str(output_dir),
            "paid_compute": True,
            "volume": config.runtime["modal_volume"],
        }
    )
    return payload


def _run_modal_sweep_body(
    config_payload: dict[str, Any],
    *,
    commit: str,
    dirty: bool,
    modal_gpu: str,
    timeout_hours: int,
    started: float,
) -> dict[str, Any]:
    config = validate_multi_turn_payload(
        config_payload,
        Path("configs/esme-214m-sft-multiturn.json"),
        require_base_bundle_exists=False,
    )
    blockers = multi_turn_sweep_blockers(config, timeout_hours=timeout_hours, modal_gpu=modal_gpu)
    if blockers:
        raise RuntimeError("multi-turn interval sweep refused inside Modal: " + "; ".join(blockers))
    return run_multi_turn_interval_eval_sweep(
        config,
        output_root=MODAL_SWEEP_OUTPUT_ROOT,
        base_bundle_path=BASE_BUNDLE_REMOTE,
        allow_remote_download=True,
        require_cuda=True,
        wandb_enabled=True,
        started=started,
        commit=commit,
        dirty=dirty,
    )


def _run_modal_throughput_probe_body(
    config_payload: dict[str, Any],
    *,
    commit: str,
    dirty: bool,
    modal_gpu: str,
    timeout_hours: int,
    started: float,
) -> dict[str, Any]:
    config = validate_multi_turn_payload(
        config_payload,
        Path("configs/esme-214m-sft-multiturn.json"),
        require_base_bundle_exists=False,
    )
    blockers = multi_turn_probe_blockers(modal_gpu=modal_gpu, timeout_hours=timeout_hours)
    if blockers:
        raise RuntimeError(
            "multi-turn throughput probe refused inside Modal: " + "; ".join(blockers)
        )
    return run_multi_turn_throughput_probe(
        config,
        output_root=MODAL_PROBE_OUTPUT_ROOT,
        modal_gpu=modal_gpu,
        base_bundle_path=BASE_BUNDLE_REMOTE,
        allow_remote_download=True,
        require_cuda=True,
        started=started,
        commit=commit,
        dirty=dirty,
    )


def _run_modal_full_sft_body(
    config_payload: dict[str, Any],
    *,
    commit: str,
    dirty: bool,
    started: float,
    resume: bool,
    modal_gpu: str,
    output_stem: str,
) -> dict[str, Any]:
    output_stem = _validated_full_output_stem(output_stem)
    config = validate_multi_turn_payload(
        config_payload,
        Path("configs/esme-214m-sft-multiturn.json"),
        require_base_bundle_exists=False,
    )
    blockers = full_launch_blockers(config, approved=True, modal_gpu=modal_gpu)
    if blockers:
        raise RuntimeError("full multi-turn SFT refused inside Modal: " + "; ".join(blockers))
    output_dir = VOLUME_MOUNT / output_stem
    return run_full_multi_turn_sft(
        config,
        output_dir=output_dir,
        base_bundle_path=BASE_BUNDLE_REMOTE,
        allow_remote_download=True,
        require_cuda=True,
        wandb_enabled=True,
        started=started,
        commit=commit,
        dirty=dirty,
        resume_from_latest=resume,
    )


def _validated_full_output_stem(value: str) -> str:
    return validate_output_stem(value, env_var="SFT_MODAL_FULL_OUTPUT_STEM")


def _full_launch_command(config: Any, output_stem: str) -> str:
    return command_with_output_stem(
        config.full_launch_command,
        output_stem=output_stem,
        default_stem=DEFAULT_MODAL_FULL_OUTPUT_STEM,
        env_var="SFT_MODAL_FULL_OUTPUT_STEM",
    )


def _with_full_output_stem(
    payload: dict[str, Any], config: Any, output_stem: str
) -> dict[str, Any]:
    full_launch_command = _full_launch_command(config, output_stem)
    updated = {
        **payload,
        "full_launch_command": full_launch_command,
        "volume_output_dir": str(VOLUME_MOUNT / output_stem),
    }
    preflight = dict(updated.get("preflight", {}))
    preflight["exact_launch_command"] = full_launch_command
    preflight["volume_output_dir"] = str(VOLUME_MOUNT / output_stem)
    updated["preflight"] = preflight
    return updated


def _format_payload(payload: dict[str, Any], *, json_output: bool) -> str:
    return format_payload(
        payload,
        json_output=json_output,
        keys=(
            "output_dir",
            "output_root",
            "evidence_dir",
            "will_start_modal_job",
            "modal_smoke_command",
            "modal_sweep_command",
            "throughput_probe_command",
            "full_launch_command",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(launch())
