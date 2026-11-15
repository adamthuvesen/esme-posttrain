from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def full_rl_recipe_path(tmp_path: Path) -> Path:
    """Use the full launch recipe with a fixture tokenizer and committed tasks."""
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads((repo_root / "configs" / "esme-214m-rl.json").read_text(encoding="utf-8"))
    payload["input_bundle"]["path"] = str(repo_root / "fixtures" / "tiny_bundle")
    payload["dataset"]["manifest_path"] = str(
        repo_root / "data" / "manifests" / "esme-214m-rl.tasks.json"
    )
    config_path = tmp_path / "configs" / "esme-214m-rl.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path
