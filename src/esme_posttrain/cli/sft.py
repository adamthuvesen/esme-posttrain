from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import run_config_stage
from esme_posttrain.sft.launch_multiturn import build_multi_turn_dry_run, load_multi_turn_config
from esme_posttrain.sft.smoke_multiturn import run_multi_turn_cpu_fixture


def add_sft_parsers(subparsers: argparse._SubParsersAction) -> None:
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


def _handle_multi_turn_sft(args: argparse.Namespace) -> int:
    return run_config_stage(
        args,
        load_config=load_multi_turn_config,
        build_dry_run=build_multi_turn_dry_run,
        run_fixture=run_multi_turn_cpu_fixture,
    )
