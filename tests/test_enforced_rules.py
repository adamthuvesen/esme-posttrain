"""Every enforcer cited in the AGENTS.md Enforced Rules table must exist.

Each backticked reference in an "Enforced by" cell is `path`, `path::name`
(the name must appear in that file), or `make <target>` (a Makefile target).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _enforced_by_refs() -> list[str]:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    table = text.split("## Enforced Rules", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ") and "---" not in line]
    cells = [row.split(" | ")[1] for row in rows[1:]]
    return [ref for cell in cells for ref in re.findall(r"`([^`]+)`", cell)]


def _problem(ref: str) -> str | None:
    if ref.startswith("make "):
        target = ref.removeprefix("make ")
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        found = re.search(rf"^{re.escape(target)}:", makefile, re.MULTILINE)
        return None if found else f"`{ref}`: no `{target}:` target in Makefile"
    path, _, name = ref.partition("::")
    target_file = REPO_ROOT / path
    if not target_file.is_file():
        return f"`{ref}`: {path} does not exist"
    if name and name not in target_file.read_text(encoding="utf-8"):
        return f"`{ref}`: {name} not found in {path}"
    return None


def test_enforced_rules_table_cites_existing_enforcers() -> None:
    refs = _enforced_by_refs()
    assert refs, "AGENTS.md Enforced Rules table cites no enforcers"
    problems = [problem for ref in refs if (problem := _problem(ref))]
    assert not problems, (
        "AGENTS.md Enforced Rules cites enforcers that do not exist. Fix the citation "
        "to the real file, test, or make target, or restore the enforcer:\n" + "\n".join(problems)
    )


def test_docs_internal_is_never_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "docs/internal"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert not tracked, f"docs/internal/ is local-only; run `git rm --cached` on: {tracked}"
