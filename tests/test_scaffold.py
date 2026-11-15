from __future__ import annotations

from esme_posttrain import __version__
from esme_posttrain.cli import main


def test_version_is_set() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_cli_runs_clean() -> None:
    assert main([]) == 0
