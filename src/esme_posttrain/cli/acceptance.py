from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import emit_payload, exit_with_error, format_stage_payload
from esme_posttrain.launch.errors import LaunchError
from esme_posttrain.launch.full_path_smoke import (
    load_full_path_smoke_config,
    run_full_path_cpu_smoke,
)


def add_acceptance_parsers(subparsers: argparse._SubParsersAction) -> None:
    full_path = subparsers.add_parser(
        "full-path-cpu-smoke",
        help="Run the no-spend SFT to DPO to RLVR artifact handoff on CPU fixtures.",
    )
    full_path.add_argument(
        "--config",
        type=Path,
        default=Path("fixtures/configs/full-path-cpu-smoke.json"),
    )
    full_path.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/full-path-cpu-smoke"),
    )
    full_path.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    full_path.set_defaults(handler=_handle_full_path_cpu_smoke)


def _handle_full_path_cpu_smoke(args: argparse.Namespace) -> int:
    try:
        config = load_full_path_smoke_config(args.config)
        payload = run_full_path_cpu_smoke(config, output_dir=args.output_dir)
    except (LaunchError, ValueError) as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=format_stage_payload)
    return 0
