from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from esme_posttrain import __version__
from esme_posttrain.cli.parser import build_parser

__all__ = ("build_parser", "main")

CommandHandler = Callable[[Namespace], int]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler | None = getattr(args, "handler", None)
    if handler is not None:
        return handler(args)

    print(f"esme-posttrain {__version__}: RLVR, multi-turn SFT, and DPO checks.")
    return 0
