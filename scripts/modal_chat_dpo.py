#!/usr/bin/env python3
"""Approval-gated Modal launcher for the Esme-214M-Chat DPO polish stage.

Mirrors the multi-turn SFT launch contract (``scripts/modal_chat_sft.py``): the
dry-run never starts Modal, the smoke is capped at $2 with no ``DPO_MODAL_GPU``
spend bypass, the full run refuses without ``--approved`` AND bounded beta-sweep
learning-gate evidence, and everything writes to a separate Volume and output
stem (``esme-214m-chat-dpo``). The frozen SFT reference is read from the accepted
SFT foundation's Modal Volume (``esme-posttrain-esme-sft-multiturn``), mounted
read-only; DPO never writes to it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from esme_posttrain.dpo.chat_eval_run import (
    CHAT_EVAL_OUTPUT_STEM,
    CHAT_EVAL_TIMEOUT_HOURS,
    DPO_BEST_CHECKPOINT_REL,
    build_chat_eval_preflight,
    chat_eval_blockers,
    run_chat_eval_job,
)
from esme_posttrain.dpo.full import run_full_dpo
from esme_posttrain.dpo.launch import (
    build_dpo_dry_run,
    full_launch_blockers,
    load_dpo_config,
    smoke_launch_blockers,
    validate_dpo_payload,
)
from esme_posttrain.dpo.sweep import (
    DEFAULT_MODAL_SWEEP_ROOT,
    SWEEP_TIMEOUT_HOURS,
    build_dpo_sweep_preflight,
    dpo_sweep_blockers,
    run_dpo_beta_sweep,
)
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
    with_full_output_stem,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DPO_MODAL_GPU = os.environ.get("DPO_MODAL_GPU", "A100")
# Fractional hours must survive (the emitted chat-eval command sets 0.25); Modal
# wants whole seconds, so hours convert via int(hours * 3600) at the functions.
DPO_TIMEOUT_HOURS = float(os.environ.get("DPO_TIMEOUT_HOURS", "1"))
# Chat eval is generation-only; without an explicit DPO_TIMEOUT_HOURS it keeps the
# documented 0.25h ceiling that holds the worst case under the $1 spend cap.
DPO_CHAT_EVAL_TIMEOUT_HOURS = float(
    os.environ.get("DPO_TIMEOUT_HOURS", str(CHAT_EVAL_TIMEOUT_HOURS))
)
DPO_SWEEP_TIMEOUT_HOURS = int(float(os.environ.get("DPO_SWEEP_TIMEOUT_HOURS", SWEEP_TIMEOUT_HOURS)))
VOLUME_MOUNT = Path("/posttrain")
SFT_VOLUME_MOUNT = Path("/sft-foundation")
MODAL_APP_NAME = "esme-posttrain-esme-chat-dpo"
MODAL_SMOKE_OUTPUT_STEM = "esme-214m-chat-dpo-modal-smoke"
DEFAULT_MODAL_FULL_OUTPUT_STEM = "esme-214m-chat-dpo-full"
MODAL_VOLUME_NAME = "esme-posttrain-esme-chat-dpo"
SFT_VOLUME_NAME = "esme-posttrain-esme-sft-multiturn"
MODAL_FULL_OUTPUT_STEM = os.environ.get(
    "DPO_MODAL_FULL_OUTPUT_STEM", DEFAULT_MODAL_FULL_OUTPUT_STEM
)
MODAL_SWEEP_OUTPUT_ROOT = DEFAULT_MODAL_SWEEP_ROOT

try:
    import modal
except ImportError:  # pragma: no cover - Modal is a runtime dependency for launch.
    modal = None

run_modal_full_dpo = None
run_modal_smoke = None
run_modal_beta_sweep = None
run_modal_chat_eval = None


if modal is not None:  # pragma: no cover - exercised by Modal, not local unit tests.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PACKAGE_PINS.items()))
        .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
    )
    posttrain_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
    # Read-only mount of the accepted SFT foundation; DPO must never write here.
    sft_volume = modal.Volume.from_name(SFT_VOLUME_NAME, create_if_missing=False)
    app = modal.App(MODAL_APP_NAME)

    @app.function(
        image=image,
        gpu=DPO_MODAL_GPU,
        timeout=int(DPO_TIMEOUT_HOURS * 3600),
        volumes={str(VOLUME_MOUNT): posttrain_volume, str(SFT_VOLUME_MOUNT): sft_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_smoke(
        config_payload: dict[str, Any], commit: str, dirty: bool, modal_gpu: str
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_dpo_body(
                config_payload,
                commit=commit,
                dirty=dirty,
                started=started,
                smoke=True,
                modal_gpu=modal_gpu,
                output_stem=MODAL_SMOKE_OUTPUT_STEM,
                fresh=True,
            )
        finally:
            posttrain_volume.commit()

    @app.function(
        image=image,
        gpu=DPO_MODAL_GPU,
        timeout=int(DPO_TIMEOUT_HOURS * 3600),
        volumes={str(VOLUME_MOUNT): posttrain_volume, str(SFT_VOLUME_MOUNT): sft_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_full_dpo(
        config_payload: dict[str, Any], commit: str, dirty: bool, modal_gpu: str, output_stem: str
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_dpo_body(
                config_payload,
                commit=commit,
                dirty=dirty,
                started=started,
                smoke=False,
                modal_gpu=modal_gpu,
                output_stem=output_stem,
                fresh=False,
            )
        finally:
            posttrain_volume.commit()

    @app.function(
        image=image,
        gpu=DPO_MODAL_GPU,
        timeout=DPO_SWEEP_TIMEOUT_HOURS * 60 * 60,
        volumes={str(VOLUME_MOUNT): posttrain_volume, str(SFT_VOLUME_MOUNT): sft_volume},
        secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
    )
    def run_modal_beta_sweep(
        config_payload: dict[str, Any], commit: str, dirty: bool, modal_gpu: str, timeout_hours: int
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_beta_sweep_body(
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
        gpu=DPO_MODAL_GPU,
        timeout=int(DPO_CHAT_EVAL_TIMEOUT_HOURS * 3600),
        volumes={str(VOLUME_MOUNT): posttrain_volume, str(SFT_VOLUME_MOUNT): sft_volume},
    )
    def run_modal_chat_eval(
        config_payload: dict[str, Any],
        commit: str,
        dirty: bool,
        modal_gpu: str,
        timeout_hours: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return _run_modal_chat_eval_body(
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
        beta_sweep: bool = False,
        chat_eval: bool = False,
        dry_run: bool = False,
    ) -> None:
        argv = ["--config", config]
        if dry_run:
            argv.append("--dry-run")
            if beta_sweep:
                argv.append("--beta-sweep")
            if full_run:
                argv.append("--full-run")
            if chat_eval:
                argv.append("--chat-eval")
        elif full_run:
            argv.append("--full-run")
        elif beta_sweep:
            argv.append("--beta-sweep")
        elif chat_eval:
            argv.append("--chat-eval")
        else:
            argv.append("--modal-smoke")
        if approved:
            argv.append(LAUNCH_APPROVAL_FLAG)
        if json:
            argv.append("--json")
        raise SystemExit(launch(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_chat_dpo.py",
        description="Validate and launch the approval-gated Esme-214M-Chat DPO smoke.",
    )
    parser.add_argument("--config", required=True, type=Path, help="DPO config JSON path.")
    parser.add_argument(LAUNCH_APPROVAL_FLAG, action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never starts Modal.")
    parser.add_argument(
        "--local-cpu-smoke",
        action="store_true",
        help="Run the no-spend local DPO CPU fixture and write evidence.",
    )
    parser.add_argument(
        "--modal-smoke", action="store_true", help="Launch the capped Modal smoke after approval."
    )
    parser.add_argument(
        "--beta-sweep",
        action="store_true",
        help="Launch the approved bounded {0.1,0.3,0.5} beta sweep (learning gate).",
    )
    parser.add_argument(
        "--full-run", action="store_true", help="Launch the approved, capped full Modal DPO run."
    )
    parser.add_argument(
        "--chat-eval",
        action="store_true",
        help="Launch the approved SFT-vs-DPO chat-quality eval (generation only, <=$1).",
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
            args.beta_sweep,
            args.chat_eval,
        )
    )
    if mode_count > 1:
        print("DPO launch failed: choose exactly one launch mode", file=sys.stderr)
        return 2
    try:
        config = load_dpo_config(args.config)
    except LaunchError as error:
        print(f"DPO launch failed: {error}", file=sys.stderr)
        return 2
    try:
        full_output_stem = _validated_output_stem(MODAL_FULL_OUTPUT_STEM)
    except ValueError as error:
        print(f"DPO launch failed: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.beta_sweep:
            payload = build_dpo_sweep_preflight(
                config, timeout_hours=DPO_SWEEP_TIMEOUT_HOURS, modal_gpu=DPO_MODAL_GPU
            )
        elif args.chat_eval:
            payload = build_chat_eval_preflight(
                config, modal_gpu=DPO_MODAL_GPU, timeout_hours=DPO_CHAT_EVAL_TIMEOUT_HOURS
            )
        else:
            payload = build_dpo_dry_run(
                config,
                full_run_approved=args.approved,
                full_run_modal_gpu=DPO_MODAL_GPU if args.full_run else None,
            )
            if args.full_run:
                payload = with_full_output_stem(
                    payload,
                    full_launch_command=_full_launch_command(config, full_output_stem),
                    volume_output_dir=str(VOLUME_MOUNT / full_output_stem),
                )
        print(_format_payload(payload, json_output=args.json))
        return 0
    if args.local_cpu_smoke:
        from esme_posttrain.dpo.smoke import run_dpo_cpu_fixture

        payload = run_dpo_cpu_fixture(config)
        print(_format_payload(payload, json_output=args.json))
        return 0
    if args.beta_sweep:
        return _launch_beta_sweep(config, approved=args.approved, json_output=args.json)
    if args.chat_eval:
        return _launch_chat_eval(config, approved=args.approved, json_output=args.json)
    if args.full_run:
        return _launch_full_run(
            config, approved=args.approved, json_output=args.json, output_stem=full_output_stem
        )
    if not args.modal_smoke:
        print(_format_payload(build_dpo_dry_run(config), json_output=args.json))
        return 0
    if not args.approved:
        print(
            f"DPO Modal smoke refused: pass {LAUNCH_APPROVAL_FLAG} after chat approval",
            file=sys.stderr,
        )
        return 2
    blockers = smoke_launch_blockers(config)
    if blockers:
        print("DPO Modal smoke refused: " + "; ".join(blockers), file=sys.stderr)
        return 2
    if modal is None:
        print("DPO Modal smoke failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_smoke.spawn(
            config.payload, local_git_commit(REPO_ROOT), local_git_dirty(REPO_ROOT), DPO_MODAL_GPU
        )
        call_id = modal_call_id(function_call)
        result = function_call.get()
        result.update({"modal_app": MODAL_APP_NAME, "modal_call_id": call_id})
    except Exception as error:
        print(f"DPO Modal smoke failed before durable success: {error}", file=sys.stderr)
        return 2
    print(_format_payload(result, json_output=args.json))
    return 0


def _launch_chat_eval(config: Any, *, approved: bool, json_output: bool) -> int:
    preflight = build_chat_eval_preflight(
        config, modal_gpu=DPO_MODAL_GPU, timeout_hours=DPO_CHAT_EVAL_TIMEOUT_HOURS
    )
    if not approved:
        payload = {
            **preflight,
            "status": "chat_eval_refused",
            "launch_blockers": [f"SFT-vs-DPO chat eval requires {LAUNCH_APPROVAL_FLAG}"],
        }
        print(_format_payload(payload, json_output=json_output))
        return 2
    blockers = chat_eval_blockers(
        config, modal_gpu=DPO_MODAL_GPU, timeout_hours=DPO_CHAT_EVAL_TIMEOUT_HOURS
    )
    if blockers:
        payload = {**preflight, "status": "chat_eval_refused", "launch_blockers": blockers}
        print(_format_payload(payload, json_output=json_output))
        return 2
    if modal is None or run_modal_chat_eval is None:
        print("DPO chat eval failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_chat_eval.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            DPO_MODAL_GPU,
            DPO_CHAT_EVAL_TIMEOUT_HOURS,
        )
        call_id = modal_call_id(function_call)
        result = function_call.get()
        result.update(
            {
                "modal_app": MODAL_APP_NAME,
                "modal_call_id": call_id,
                "chat_eval_command": preflight["chat_eval_command"],
            }
        )
    except Exception as error:
        print(f"DPO chat eval failed before durable success: {error}", file=sys.stderr)
        return 2
    print(_format_payload(result, json_output=json_output))
    return 0


def _launch_beta_sweep(config: Any, *, approved: bool, json_output: bool) -> int:
    preflight = build_dpo_sweep_preflight(
        config, timeout_hours=DPO_SWEEP_TIMEOUT_HOURS, modal_gpu=DPO_MODAL_GPU
    )
    if not approved:
        payload = {
            **preflight,
            "status": "beta_sweep_refused",
            "launch_blockers": [f"bounded DPO beta sweep requires {LAUNCH_APPROVAL_FLAG}"],
        }
        print(_format_payload(payload, json_output=json_output))
        return 2
    blockers = dpo_sweep_blockers(
        config, timeout_hours=DPO_SWEEP_TIMEOUT_HOURS, modal_gpu=DPO_MODAL_GPU
    )
    if blockers:
        payload = {**preflight, "status": "beta_sweep_refused", "launch_blockers": blockers}
        print(_format_payload(payload, json_output=json_output))
        return 2
    if modal is None or run_modal_beta_sweep is None:
        print("DPO beta sweep failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_beta_sweep.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            DPO_MODAL_GPU,
            DPO_SWEEP_TIMEOUT_HOURS,
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
        print(f"DPO beta sweep failed before durable success: {error}", file=sys.stderr)
        return 2
    print(_format_payload(result, json_output=json_output))
    return 0


def _launch_full_run(config: Any, *, approved: bool, json_output: bool, output_stem: str) -> int:
    full_launch_command = _full_launch_command(config, output_stem)
    blockers = full_launch_blockers(config, approved=approved, modal_gpu=DPO_MODAL_GPU)
    if blockers:
        payload = {
            "status": "full_run_refused",
            "will_start_modal_job": False,
            "full_launch_blockers": blockers,
            "full_launch_command": full_launch_command,
            "volume_output_dir": str(VOLUME_MOUNT / output_stem),
        }
        print(_format_payload(payload, json_output=json_output))
        return 2
    if modal is None or run_modal_full_dpo is None:
        print("DPO full run failed: modal is not installed", file=sys.stderr)
        return 2
    try:
        function_call = run_modal_full_dpo.spawn(
            config.payload,
            local_git_commit(REPO_ROOT),
            local_git_dirty(REPO_ROOT),
            DPO_MODAL_GPU,
            output_stem,
        )
    except Exception as error:
        print(f"DPO full run failed before Modal spawn: {error}", file=sys.stderr)
        return 2
    call_id = modal_call_id(function_call)
    payload = {
        "status": "modal_full_dpo_launched",
        "will_start_modal_job": True,
        "modal_app": MODAL_APP_NAME,
        "modal_call_id": call_id,
        "full_launch_command": full_launch_command,
        "volume": config.runtime["modal_volume"],
        "volume_output_dir": str(VOLUME_MOUNT / output_stem),
        "projected_cost_usd": config.estimated_full_cost_usd,
        "runtime_spend_stop_usd": config.runtime["full_run_runtime_spend_stop_usd"],
        "wandb_project": config.payload["monitoring"]["wandb_project"],
    }
    print(_format_payload(payload, json_output=json_output))
    return 0


def _run_modal_dpo_body(
    config_payload: dict[str, Any],
    *,
    commit: str,
    dirty: bool,
    started: float,
    smoke: bool,
    modal_gpu: str,
    output_stem: str,
    fresh: bool,
) -> dict[str, Any]:
    config = validate_dpo_payload(config_payload, Path("configs/esme-214m-chat-dpo.json"))
    if not smoke:
        blockers = full_launch_blockers(config, approved=True, modal_gpu=modal_gpu)
        if blockers:
            raise RuntimeError("full DPO refused inside Modal: " + "; ".join(blockers))
    output_dir = (
        fresh_output_dir(VOLUME_MOUNT, output_stem) if fresh else VOLUME_MOUNT / output_stem
    )
    sft_reference = config.payload["sft_reference"]
    return run_full_dpo(
        config,
        output_dir=output_dir,
        sft_checkpoint_path=SFT_VOLUME_MOUNT / sft_reference["checkpoint_path"],
        sft_tokenizer_path=SFT_VOLUME_MOUNT / sft_reference["tokenizer_path"],
        allow_remote_download=True,
        require_cuda=True,
        smoke=smoke,
        started=started,
        commit=commit,
        dirty=dirty,
    )


def _run_modal_beta_sweep_body(
    config_payload: dict[str, Any],
    *,
    commit: str,
    dirty: bool,
    modal_gpu: str,
    timeout_hours: int,
    started: float,
) -> dict[str, Any]:
    import torch
    from tokenizers import Tokenizer

    from esme_posttrain.dpo.data import build_preference_set
    from esme_posttrain.training.checkpointing import load_training_checkpoint

    config = validate_dpo_payload(config_payload, Path("configs/esme-214m-chat-dpo.json"))
    blockers = dpo_sweep_blockers(config, timeout_hours=timeout_hours, modal_gpu=modal_gpu)
    if blockers:
        raise RuntimeError("DPO beta sweep refused inside Modal: " + "; ".join(blockers))
    from esme_posttrain.dpo.sweep import (
        SWEEP_EVAL_PAIR_CAP,
        SWEEP_EVAL_TOKEN_CAP,
        SWEEP_TRAIN_PAIR_CAP,
        SWEEP_TRAIN_TOKEN_CAP,
    )

    sft_reference = config.payload["sft_reference"]
    checkpoint_path = SFT_VOLUME_MOUNT / sft_reference["checkpoint_path"]
    tokenizer = Tokenizer.from_file(str(SFT_VOLUME_MOUNT / sft_reference["tokenizer_path"]))
    device = torch.device("cuda")
    reference = load_training_checkpoint(checkpoint_path, map_location=device).model.to(device)
    budgets = config.budgets
    train_report = build_preference_set(
        config.preference_source,
        tokenizer,
        max_pairs=SWEEP_TRAIN_PAIR_CAP,
        max_tokens=SWEEP_TRAIN_TOKEN_CAP,
        max_length=int(budgets["max_length"]),
        max_prompt_length=int(budgets["max_prompt_length"]),
        allow_remote_download=True,
    )
    eval_report = build_preference_set(
        config.eval_source,
        tokenizer,
        max_pairs=SWEEP_EVAL_PAIR_CAP,
        max_tokens=SWEEP_EVAL_TOKEN_CAP,
        max_length=int(budgets["max_length"]),
        max_prompt_length=int(budgets["max_prompt_length"]),
        allow_remote_download=True,
    )

    def make_policy() -> Any:
        return load_training_checkpoint(checkpoint_path, map_location=device).model.to(device)

    return run_dpo_beta_sweep(
        config,
        output_root=MODAL_SWEEP_OUTPUT_ROOT,
        reference=reference,
        tokenizer=tokenizer,
        train_pairs=train_report.pairs,
        eval_pairs=eval_report.pairs,
        device=device,
        usd_per_hour=float(config.selected_gpu_profile["usd_per_hour"]),
        wandb_enabled=True,
        started=started,
        commit=commit,
        dirty=dirty,
        make_policy=make_policy,
    )


def _run_modal_chat_eval_body(
    config_payload: dict[str, Any],
    *,
    commit: str,
    dirty: bool,
    modal_gpu: str,
    timeout_hours: float,
    started: float,
) -> dict[str, Any]:
    config = validate_dpo_payload(config_payload, Path("configs/esme-214m-chat-dpo.json"))
    blockers = chat_eval_blockers(config, modal_gpu=modal_gpu, timeout_hours=timeout_hours)
    if blockers:
        raise RuntimeError("chat eval refused inside Modal: " + "; ".join(blockers))
    sft_reference = config.payload["sft_reference"]
    return run_chat_eval_job(
        config,
        output_dir=VOLUME_MOUNT / CHAT_EVAL_OUTPUT_STEM,
        dpo_checkpoint_path=VOLUME_MOUNT / DPO_BEST_CHECKPOINT_REL,
        sft_checkpoint_path=SFT_VOLUME_MOUNT / sft_reference["checkpoint_path"],
        sft_tokenizer_path=SFT_VOLUME_MOUNT / sft_reference["tokenizer_path"],
        require_cuda=True,
        usd_per_hour=float(config.selected_gpu_profile["usd_per_hour"]),
        started=started,
        commit=commit,
        dirty=dirty,
    )


def _validated_output_stem(value: str) -> str:
    return validate_output_stem(value, env_var="DPO_MODAL_FULL_OUTPUT_STEM")


def _full_launch_command(config: Any, output_stem: str) -> str:
    return command_with_output_stem(
        config.full_launch_command,
        output_stem=output_stem,
        default_stem=DEFAULT_MODAL_FULL_OUTPUT_STEM,
        env_var="DPO_MODAL_FULL_OUTPUT_STEM",
    )


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
            "chat_eval_command",
            "markdown_path",
            "json_path",
            "full_launch_command",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(launch())
