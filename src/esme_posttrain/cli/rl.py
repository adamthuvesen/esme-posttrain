from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import Payload, emit_payload, exit_with_error
from esme_posttrain.launch.config_guards import LaunchError
from esme_posttrain.rl.countdown_heldout import write_countdown_heldout_dataset
from esme_posttrain.rl.countdown_lite import CountdownLiteError, write_countdown_lite_dataset
from esme_posttrain.rl.countdown_lite_baseline import (
    CountdownBaselineRequest,
    run_countdown_lite_baseline,
)
from esme_posttrain.rl.decomp_emitter import DecompEmitterError, EmitRequest, emit_completion_set
from esme_posttrain.rl.launch import build_rlvr_dry_run, format_rlvr_dry_run, load_rlvr_config
from esme_posttrain.rl.pipeline_smoke import run_rlvr_pipeline_smoke


def add_rl_parsers(subparsers: argparse._SubParsersAction) -> None:
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

    emit_completions = subparsers.add_parser(
        "rlvr-emit-decomp-completions",
        help=(
            "Emit a grpo-decomp CompletionSet (provenance.json + completions.jsonl) for one "
            "arm on an Esme held-out Countdown set."
        ),
    )
    emit_completions.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to the Esme bundle/checkpoint to sample from.",
    )
    emit_completions.add_argument(
        "--heldout-manifest",
        type=Path,
        default=Path("data/manifests/esme-214m-rl-heldout.tasks.json"),
        help="Held-out Countdown task manifest.",
    )
    emit_completions.add_argument(
        "--set", dest="set_name", default="heldout_fresh", help="Held-out split to emit."
    )
    emit_completions.add_argument(
        "--out", dest="output_dir", type=Path, required=True, help="CompletionSet output dir."
    )
    emit_completions.add_argument("--n", type=int, default=1, help="Samples per problem.")
    emit_completions.add_argument("--temperature", type=float, default=0.0)
    emit_completions.add_argument("--max-new-tokens", type=int, default=12)
    emit_completions.add_argument("--seed", type=int, default=0)
    emit_completions.add_argument("--device", default="cpu")
    emit_completions.add_argument(
        "--model-label", default=None, help="Override the provenance model label."
    )
    emit_completions.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    emit_completions.set_defaults(handler=_handle_emit_decomp_completions)


def _handle_rlvr_dry_run(args: argparse.Namespace) -> int:
    try:
        payload = build_rlvr_dry_run(args.config)
    except LaunchError as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=format_rlvr_dry_run)
    return 0


def _handle_rlvr_pipeline_smoke(args: argparse.Namespace) -> int:
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
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=_format_pipeline_smoke)
    return 0


def _handle_countdown_lite_build_data(args: argparse.Namespace) -> int:
    try:
        payload = write_countdown_lite_dataset(args.repo_root)
    except CountdownLiteError as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=_format_countdown_lite_build)
    return 0


def _handle_countdown_heldout_build_data(args: argparse.Namespace) -> int:
    try:
        payload = write_countdown_heldout_dataset(args.repo_root)
    except CountdownLiteError as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=_format_countdown_lite_build)
    return 0


def _handle_countdown_lite_baseline(args: argparse.Namespace) -> int:
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
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=_format_countdown_lite_baseline)
    return 0


def _handle_emit_decomp_completions(args: argparse.Namespace) -> int:
    try:
        payload = emit_completion_set(
            EmitRequest(
                bundle_path=args.bundle,
                heldout_manifest_path=args.heldout_manifest,
                output_dir=args.output_dir,
                set_name=args.set_name,
                n=args.n,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                device=args.device,
                model_label=args.model_label,
            )
        )
    except DecompEmitterError as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=_format_emit_decomp_completions)
    return 0


def _format_emit_decomp_completions(payload: Payload) -> str:
    return "\n".join(
        [
            "status: decomp_completion_set_written",
            f"output_dir: {payload.get('output_dir')}",
            f"set_name: {payload.get('set_name')}",
            f"dataset_name: {payload.get('dataset_name')}",
            f"dataset_revision: {payload.get('dataset_revision')}",
            f"n_problems: {payload.get('n_problems')}",
            f"n_samples: {payload.get('n_samples')}",
            f"model: {payload.get('model')}",
        ]
    )


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
