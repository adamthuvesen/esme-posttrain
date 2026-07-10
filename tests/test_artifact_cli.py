from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from esme_posttrain.bundle import file_sha256
from esme_posttrain.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundle_check_cli_reports_checked_files(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "bundle-check",
                "--bundle-dir",
                str(REPO_ROOT / "fixtures" / "tiny_bundle"),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == 1
    assert payload["format"] == "llm_pretrain_dense_v1"
    assert set(payload["files"]) == {"config", "readme", "tokenizer", "weights"}


def test_bundle_check_cli_fails_loudly_for_missing_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["bundle-check", "--bundle-dir", str(tmp_path / "missing")])

    assert "missing base bundle manifest" in capsys.readouterr().err


def test_bundle_check_cli_rejects_future_weights_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(REPO_ROOT / "fixtures" / "tiny_bundle", bundle_dir)
    weights_path = bundle_dir / "weights.pt"
    weights = torch.load(weights_path, weights_only=False)
    weights["format_version"] = 2
    torch.save(weights, weights_path)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["weights"]["sha256"] = file_sha256(weights_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(["bundle-check", "--bundle-dir", str(bundle_dir)])

    assert "unsupported weights.pt format_version" in capsys.readouterr().err
