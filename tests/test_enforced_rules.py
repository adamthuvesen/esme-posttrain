"""Every reference in the AGENTS.md Enforced Rules table must exist.

A cell cites `path` or `path::name`; the name must appear in that file's text.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _enforced_by_refs() -> list[str]:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table = text.split("## Enforced Rules", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    cells = [row.split(" | ")[1] for row in rows[1:]]
    refs = [ref for cell in cells for ref in re.findall(r"`([^`]+)`", cell)]
    return [ref for ref in refs if "::" in ref or "/" in ref]


def test_enforced_rules_table_cites_existing_files_and_names() -> None:
    refs = _enforced_by_refs()
    assert refs, "AGENTS.md Enforced Rules table has no `path` references"
    missing = []
    for ref in refs:
        path, _, name = ref.partition("::")
        target = REPO_ROOT / path
        if not target.is_file() or (name and name not in target.read_text(encoding="utf-8")):
            missing.append(ref)
    assert not missing, f"AGENTS.md Enforced Rules cites missing paths or names: {missing}"
