from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import run_config_stage
from esme_posttrain.dpo.launch import build_dpo_dry_run, load_dpo_config
from esme_posttrain.dpo.smoke import run_dpo_cpu_fixture


def add_dpo_parsers(subparsers: argparse._SubParsersAction) -> None:
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


def _handle_dpo(args: argparse.Namespace) -> int:
    return run_config_stage(
        args,
        load_config=load_dpo_config,
        build_dry_run=build_dpo_dry_run,
        run_fixture=run_dpo_cpu_fixture,
    )
