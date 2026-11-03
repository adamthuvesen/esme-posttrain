"""Every row of the AGENTS.md Enforced Rules table cites an enforcer that exists.

Each backticked reference in an "Enforced by" cell is `make <target>`, `path`, or
`path::name`. The name must be defined in that file: a Python def, class, or
module constant; a Makefile target; or a TOML key or Import Linter contract id.
"""

from __future__ import annotations

import re
import subprocess  # noqa: TID251
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIX = "Cite the real enforcer in AGENTS.md, or restore the enforcer."


def _definition_patterns(path: Path, name: str) -> list[str]:
    name = re.escape(name)
    if path.suffix == ".py" or path.read_text(encoding="utf-8").startswith("#!/usr/bin/env python"):
        return [rf"^\s*(async\s+)?(def|class)\s+{name}\b", rf"^{name}\s*[:=]"]
    if path.name == "Makefile":
        return [rf"^{name}:"]
    if path.suffix == ".toml":
        return [rf"^\s*{name}\s*=", rf'^\s*id\s*=\s*"{name}"']
    return []


def _problem(ref: str) -> str | None:
    if ref.startswith("make "):
        ref = f"Makefile::{ref.removeprefix('make ')}"
    path_text, _, name = ref.partition("::")
    path = REPO_ROOT / path_text
    if not path.is_file():
        return f"`{ref}`: {path_text} does not exist"
    if not name:
        return None
    text = path.read_text(encoding="utf-8")
    patterns = _definition_patterns(path, name)
    if not any(re.search(pattern, text, re.MULTILINE) for pattern in patterns):
        return f"`{ref}`: {path_text} does not define {name}"
    return None


def table_problems(agents_text: str) -> list[str]:
    if "\n## Enforced Rules\n" not in agents_text:
        return ["AGENTS.md has no `## Enforced Rules` section"]
    section = agents_text.split("\n## Enforced Rules\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    if len(rows) < 2:
        return ["`## Enforced Rules` has no table rows"]
    problems = []
    for row in rows[1:]:
        rule, enforced_by = row.split(" | ")[:2]
        refs = re.findall(r"`([^`]+)`", enforced_by)
        if not refs:
            problems.append(f"row {rule.strip('| ')!r} cites no enforcer")
        problems.extend(problem for ref in refs if (problem := _problem(ref)))
    return problems


def test_enforced_rules_table_cites_defined_enforcers() -> None:
    problems = table_problems((REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    assert not problems, f"{FIX}\n" + "\n".join(problems)


def test_docs_internal_is_never_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "docs/internal"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert not tracked, f"docs/internal/ is local-only; run `git rm --cached` on: {tracked}"


def test_table_check_fails_closed() -> None:
    header = "## Enforced Rules\n\n| Rule | Enforced by | Added |\n| --- | --- | --- |\n"
    assert table_problems("# AGENTS.md\n\n## Rules\n") == [
        "AGENTS.md has no `## Enforced Rules` section"
    ]
    assert table_problems(f"x\n{header}| A rule | prose only | 2026 |\n") == [
        "row 'A rule' cites no enforcer"
    ]
    # Each name appears in its file's text but is not defined there.
    assert table_problems(f"x\n{header}| A rule | `Makefile::uv` | 2026 |\n") == [
        "`Makefile::uv`: Makefile does not define uv"
    ]
    assert table_problems(f"x\n{header}| A rule | `pyproject.toml::esme_posttrain` | 2026 |\n") == [
        "`pyproject.toml::esme_posttrain`: pyproject.toml does not define esme_posttrain"
    ]
