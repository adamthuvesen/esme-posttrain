from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from esme_posttrain.launch.config_guards import LaunchError

Payload = dict[str, object]
PayloadFormatter = Callable[[Payload], str]


def exit_with_error(message: str) -> NoReturn:
    """Print ``error: ...`` to stderr and exit 2 (same surface as ``parser.exit``)."""
    sys.stderr.write(f"error: {message}\n")
    raise SystemExit(2)


def emit_payload(payload: Payload, *, json_output: bool, formatter: PayloadFormatter) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(formatter(payload))


def run_config_stage(
    args: argparse.Namespace,
    *,
    load_config: Callable[[Path], Any],
    build_dry_run: Callable[[Any], Payload],
    run_fixture: Callable[..., Payload],
) -> int:
    """Shared dry-run / cpu-fixture command body for the config-gated stages."""
    try:
        config = load_config(args.config)
        payload = (
            build_dry_run(config)
            if args.stage_action == "dry-run"
            else run_fixture(config, output_dir=args.output_dir)
        )
    except (LaunchError, ValueError) as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=format_stage_payload)
    return 0


def format_stage_payload(payload: Payload) -> str:
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
