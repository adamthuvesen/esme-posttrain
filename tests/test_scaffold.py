from __future__ import annotations

from pathlib import Path

import pytest

from esme_posttrain import __version__
from esme_posttrain.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_set() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_cli_runs_clean() -> None:
    assert main([]) == 0


def test_retired_chat_prep_dry_run_is_not_registered(capsys: pytest.CaptureFixture[str]) -> None:
    config_path = REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"

    with pytest.raises(SystemExit) as exc:
        main(["dry-run-chat", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "invalid choice: 'dry-run-chat'" in capsys.readouterr().err


def test_dry_run_instruct_is_not_registered(capsys: pytest.CaptureFixture[str]) -> None:
    config_path = REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"

    with pytest.raises(SystemExit) as exc:
        main(["dry-run-instruct", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "invalid choice: 'dry-run-instruct'" in capsys.readouterr().err


def test_dry_run_rl_is_not_registered(capsys: pytest.CaptureFixture[str]) -> None:
    config_path = REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"

    with pytest.raises(SystemExit) as exc:
        main(["dry-run-rl", "--config", str(config_path)])

    assert exc.value.code == 2
    assert "invalid choice: 'dry-run-rl'" in capsys.readouterr().err
