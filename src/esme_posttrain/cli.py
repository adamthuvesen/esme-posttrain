from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from esme_posttrain import __version__
from esme_posttrain.dpo.launch import build_dpo_dry_run, load_dpo_config
from esme_posttrain.dpo.smoke import run_dpo_cpu_fixture
from esme_posttrain.export.dense_bundle import ExportRequest, export_dense_bundle
from esme_posttrain.launch.config_guards import LaunchError
from esme_posttrain.rl.countdown_heldout import write_countdown_heldout_dataset
from esme_posttrain.rl.countdown_lite import CountdownLiteError, write_countdown_lite_dataset
from esme_posttrain.rl.countdown_lite_baseline import (
    CountdownBaselineRequest,
    run_countdown_lite_baseline,
)
from esme_posttrain.rl.launch import build_rlvr_dry_run, format_rlvr_dry_run, load_rlvr_config
from esme_posttrain.rl.pipeline_smoke import run_rlvr_pipeline_smoke
from esme_posttrain.sft.launch_instruct import build_sft_dry_run, load_sft_config
from esme_posttrain.sft.launch_multiturn import build_multi_turn_dry_run, load_multi_turn_config
from esme_posttrain.sft.smoke_instruct import run_cpu_fixture_sft
from esme_posttrain.sft.smoke_multiturn import run_multi_turn_cpu_fixture

Payload = dict[str, object]
ParserHandler = Callable[[argparse.ArgumentParser, argparse.Namespace], int]
PayloadFormatter = Callable[[Payload], str]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="esme-posttrain")
    parser.add_argument("--version", action="version", version=f"esme-posttrain {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    rlvr_dry_run = subparsers.add_parser(
        "rlvr-dry-run",
        help="Validate the Esme RLVR prep config without training.",
    )
    rlvr_dry_run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an RLVR dry-run config JSON.",
    )
    rlvr_dry_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    rlvr_dry_run.set_defaults(handler=_handle_rlvr_dry_run)

    rlvr_pipeline_smoke = subparsers.add_parser(
        "rlvr-pipeline-smoke",
        help="Run the no-spend RLVR pipeline_smoke lifecycle gate on CPU.",
    )
    rlvr_pipeline_smoke.add_argument("--config", type=Path, required=True)
    rlvr_pipeline_smoke.add_argument("--output-dir", type=Path)
    rlvr_pipeline_smoke.add_argument("--report-path", type=Path)
    rlvr_pipeline_smoke.add_argument("--doc-path", type=Path)
    rlvr_pipeline_smoke.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    rlvr_pipeline_smoke.set_defaults(handler=_handle_rlvr_pipeline_smoke)

    countdown_build = subparsers.add_parser(
        "rlvr-countdown-lite-build-data",
        help="Generate deterministic local Countdown-Lite RLVR data and manifest.",
    )
    countdown_build.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing data/manifests.",
    )
    countdown_build.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    countdown_build.set_defaults(handler=_handle_countdown_lite_build_data)

    heldout_build = subparsers.add_parser(
        "rlvr-countdown-heldout-build-data",
        help="Generate deterministic held-out Countdown transfer sets and manifest.",
    )
    heldout_build.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root containing data/manifests.",
    )
    heldout_build.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    heldout_build.set_defaults(handler=_handle_countdown_heldout_build_data)

    countdown_baseline = subparsers.add_parser(
        "rlvr-countdown-lite-baseline",
        help="Run a no-spend Countdown-Lite baseline against a local Esme chat bundle.",
    )
    countdown_baseline.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/esme-214m-rl.tasks.json"),
    )
    countdown_baseline.add_argument(
        "--bundle",
        type=Path,
        default=Path("exports/esme-214m-chat"),
    )
    countdown_baseline.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rlvr-countdown-lite"),
    )
    countdown_baseline.add_argument("--split", default="eval")
    countdown_baseline.add_argument("--samples-per-task", type=int, default=32)
    countdown_baseline.add_argument("--max-tasks", type=int)
    countdown_baseline.add_argument("--max-new-tokens", type=int, default=4)
    countdown_baseline.add_argument("--seed", type=int, default=214)
    countdown_baseline.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    countdown_baseline.set_defaults(handler=_handle_countdown_lite_baseline)

    sft_dry_run = subparsers.add_parser(
        "instruct-sft-dry-run",
        help="Validate the approval-gated Esme Instruct SFT pilot config.",
    )
    sft_dry_run.add_argument("--config", type=Path, required=True)
    sft_dry_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    sft_dry_run.set_defaults(handler=_handle_instruct_sft, stage_action="dry-run")

    sft_fixture = subparsers.add_parser(
        "instruct-sft-cpu-fixture",
        help="Run the no-spend CPU SFT fixture and write evidence artifacts.",
    )
    sft_fixture.add_argument("--config", type=Path, required=True)
    sft_fixture.add_argument("--output-dir", type=Path)
    sft_fixture.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    sft_fixture.set_defaults(handler=_handle_instruct_sft, stage_action="cpu-fixture")

    mt_dry_run = subparsers.add_parser(
        "sft-multiturn-dry-run",
        help="Validate the approval-gated Esme multi-turn SFT config.",
    )
    mt_dry_run.add_argument("--config", type=Path, required=True)
    mt_dry_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    mt_dry_run.set_defaults(handler=_handle_multi_turn_sft, stage_action="dry-run")

    mt_fixture = subparsers.add_parser(
        "sft-multiturn-cpu-fixture",
        help="Run the no-spend multi-turn CPU SFT fixture and write evidence artifacts.",
    )
    mt_fixture.add_argument("--config", type=Path, required=True)
    mt_fixture.add_argument("--output-dir", type=Path)
    mt_fixture.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    mt_fixture.set_defaults(handler=_handle_multi_turn_sft, stage_action="cpu-fixture")

    dpo_dry_run = subparsers.add_parser(
        "chat-dpo-dry-run",
        help="Validate the approval-gated Esme-214M-Chat DPO config.",
    )
    dpo_dry_run.add_argument("--config", type=Path, required=True)
    dpo_dry_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    dpo_dry_run.set_defaults(handler=_handle_dpo, stage_action="dry-run")

    dpo_fixture = subparsers.add_parser(
        "chat-dpo-cpu-fixture",
        help="Run the no-spend DPO CPU fixture and write evidence artifacts.",
    )
    dpo_fixture.add_argument("--config", type=Path, required=True)
    dpo_fixture.add_argument("--output-dir", type=Path)
    dpo_fixture.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    dpo_fixture.set_defaults(handler=_handle_dpo, stage_action="cpu-fixture")

    export_bundle = subparsers.add_parser(
        "export-dense-bundle",
        help="Export a posttrain DenseBackbone checkpoint as an llm-infer bundle.",
    )
    export_bundle.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("runs/esme-214m-chat-dpo/esme-214m-chat-dpo-full"),
        help="Local mirror of the completed DPO artifact.",
    )
    export_bundle.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/esme-214m-chat"),
        help="Bundle directory to write.",
    )
    export_bundle.add_argument("--model-id", default="esme-214m-chat")
    export_bundle.add_argument("--source-volume", default="esme-posttrain-esme-chat-dpo")
    export_bundle.add_argument("--source-path", default="esme-214m-chat-dpo-full")
    export_bundle.add_argument(
        "--wandb-run",
        required=True,
        help="W&B run id of the training run being exported (stamped into provenance).",
    )
    export_bundle.add_argument(
        "--dpo-step",
        type=int,
        required=True,
        help="DPO step of the exported checkpoint (stamped into provenance).",
    )
    export_bundle.add_argument("--config-hash")
    export_bundle.add_argument("--max-new-tokens", type=int, default=16)
    export_bundle.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    export_bundle.set_defaults(handler=_handle_export_dense_bundle)

    args = parser.parse_args(argv)
    handler: ParserHandler | None = getattr(args, "handler", None)
    if handler is not None:
        return handler(parser, args)

    print(f"esme-posttrain {__version__}: RLVR, Instruct/multi-turn SFT, and DPO checks.")
    return 0


def _handle_rlvr_dry_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        payload = build_rlvr_dry_run(args.config)
    except LaunchError as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=format_rlvr_dry_run)
    return 0


def _handle_rlvr_pipeline_smoke(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        config = load_rlvr_config(args.config)
        payload = run_rlvr_pipeline_smoke(
            config,
            output_dir=args.output_dir,
            report_path=args.report_path,
            doc_path=args.doc_path,
            launch_command=(
                f"uv run esme-posttrain rlvr-pipeline-smoke --config "
                f"{args.config.as_posix()} --json"
            ),
        )
    except (LaunchError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=_format_pipeline_smoke)
    return 0


def _handle_countdown_lite_build_data(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    try:
        payload = write_countdown_lite_dataset(args.repo_root)
    except CountdownLiteError as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=_format_countdown_lite_build)
    return 0


def _handle_countdown_heldout_build_data(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    try:
        payload = write_countdown_heldout_dataset(args.repo_root)
    except CountdownLiteError as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=_format_countdown_lite_build)
    return 0


def _handle_countdown_lite_baseline(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> int:
    try:
        payload = run_countdown_lite_baseline(
            CountdownBaselineRequest(
                manifest_path=args.manifest,
                bundle_path=args.bundle,
                output_dir=args.output_dir,
                split=args.split,
                samples_per_task=args.samples_per_task,
                max_tasks=args.max_tasks,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
            )
        )
    except (LaunchError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=_format_countdown_lite_baseline)
    return 0


def _handle_instruct_sft(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_config_stage(
        parser,
        args,
        load_config=load_sft_config,
        build_dry_run=build_sft_dry_run,
        run_fixture=run_cpu_fixture_sft,
    )


def _handle_multi_turn_sft(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_config_stage(
        parser,
        args,
        load_config=load_multi_turn_config,
        build_dry_run=build_multi_turn_dry_run,
        run_fixture=run_multi_turn_cpu_fixture,
    )


def _handle_dpo(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_config_stage(
        parser,
        args,
        load_config=load_dpo_config,
        build_dry_run=build_dpo_dry_run,
        run_fixture=run_dpo_cpu_fixture,
    )


def _handle_config_stage(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    load_config: Callable[[Path], Any],
    build_dry_run: Callable[[Any], Payload],
    run_fixture: Callable[..., Payload],
) -> int:
    try:
        config = load_config(args.config)
        payload = (
            build_dry_run(config)
            if args.stage_action == "dry-run"
            else run_fixture(config, output_dir=args.output_dir)
        )
    except (LaunchError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=_format_sft_payload)
    return 0


def _handle_export_dense_bundle(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        payload = export_dense_bundle(
            ExportRequest(
                artifact_dir=args.artifact_dir,
                output_dir=args.output_dir,
                model_id=args.model_id,
                source_volume=args.source_volume,
                source_path=args.source_path,
                wandb_run=args.wandb_run,
                dpo_step=args.dpo_step,
                config_hash=args.config_hash,
                max_new_tokens=args.max_new_tokens,
            )
        )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    _emit_payload(payload, json_output=args.json, formatter=_format_sft_payload)
    return 0


def _emit_payload(payload: Payload, *, json_output: bool, formatter: PayloadFormatter) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(formatter(payload))


def _format_sft_payload(payload: Payload) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"run_id: {payload.get('run_id', 'n/a')}",
        f"artifact: {payload.get('artifact_name', 'n/a')}",
        f"will_start_modal_job: {payload.get('will_start_modal_job', False)}",
    ]
    blockers = payload.get("launch_blockers")
    if isinstance(blockers, list):
        lines.append(f"launch_blockers: {', '.join(str(item) for item in blockers) or 'none'}")
    output_dir = payload.get("output_dir")
    if output_dir:
        lines.append(f"output_dir: {output_dir}")
    return "\n".join(lines)


def _format_countdown_lite_build(payload: Payload) -> str:
    split_counts = payload.get("split_counts")
    return "\n".join(
        [
            "status: countdown_lite_data_written",
            f"manifest_path: {payload.get('manifest_path')}",
            f"data_dir: {payload.get('data_dir')}",
            f"records: {payload.get('records')}",
            f"split_counts: {split_counts}",
        ]
    )


def _format_countdown_lite_baseline(payload: Payload) -> str:
    return "\n".join(
        [
            "status: countdown_lite_baseline_complete",
            f"json_path: {payload.get('json_path')}",
            f"markdown_path: {payload.get('markdown_path')}",
            f"pass@1: {payload.get('pass@1')}",
            f"pass@8: {payload.get('pass@8')}",
            f"pass@32: {payload.get('pass@32')}",
            f"valid_expression_rate: {payload.get('valid_expression_rate')}",
            f"exact_solve_rate: {payload.get('exact_solve_rate')}",
            f"decision: {payload.get('decision')}",
        ]
    )


def _format_pipeline_smoke(payload: Payload) -> str:
    return "\n".join(
        [
            f"status: {payload.get('status')}",
            f"eval_profile: {payload.get('eval_profile')}",
            f"paid_compute: {payload.get('paid_compute')}",
            f"will_start_modal_job: {payload.get('will_start_modal_job')}",
            f"modal_gpu_or_paid_work_started: {payload.get('modal_gpu_or_paid_work_started')}",
            f"report_path: {payload.get('report_path')}",
            f"doc_path: {payload.get('doc_path')}",
        ]
    )
