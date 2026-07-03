from __future__ import annotations

import argparse

from esme_posttrain import __version__
from esme_posttrain.cli.dpo import add_dpo_parsers
from esme_posttrain.cli.export import add_export_parsers
from esme_posttrain.cli.rl import add_rl_parsers
from esme_posttrain.cli.sft import add_sft_parsers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esme-posttrain")
    parser.add_argument("--version", action="version", version=f"esme-posttrain {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    add_rl_parsers(subparsers)
    add_sft_parsers(subparsers)
    add_dpo_parsers(subparsers)
    add_export_parsers(subparsers)

    return parser
