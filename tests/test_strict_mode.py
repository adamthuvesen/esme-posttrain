"""A file-level `# pyright:` mode comment must not turn strict type checking off for that file."""

import ast
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNGRADE = re.compile(
    r"^\s*#\s*pyright:\s*(?:[^#\n]*,\s*)?(?:basic|standard|off|recommended|all|strict\s*=\s*false)\b"
)


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def candidate_python_files() -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (ROOT / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


def test_no_file_downgrades_type_checking_mode() -> None:
    downgrades = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in candidate_python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if DOWNGRADE.match(line)
    ]
    assert not downgrades, "file-level `# pyright:` comment turns strict mode off: " + ", ".join(
        downgrades
    )


REASONLESS_IGNORE = re.compile(r"#\s*pyright:\s*ignore\[[^\]]*\]\s*$")


def test_every_pyright_ignore_gives_a_reason() -> None:
    bare = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in candidate_python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if REASONLESS_IGNORE.search(line)
    ]
    assert not bare, "end each ignore with `  # <reason>`: " + ", ".join(bare)
    assert REASONLESS_IGNORE.search("x = 1  # pyright: ignore[reportAny]")
    assert not REASONLESS_IGNORE.search("x = 1  # pyright: ignore[reportAny]  # untyped JSON")


def test_the_pattern_finds_each_downgrade_and_spares_rule_ignores() -> None:
    for line in ("# pyright: basic", "#pyright: strict=false", "# pyright: reportX=false, off"):
        assert DOWNGRADE.match(line), line
    assert not DOWNGRADE.match("x = 1  # pyright: ignore[reportUnknownMemberType]")


# Each owner file may use only the banned modules listed here; ruff's per-file ignore is
# the mechanism, this table is the limit. A `noqa: TID251` is refused everywhere.
OWNED_BANS = {
    "tests/test_modal_image_pins.py": {"subprocess"},
    "tests/test_strict_mode.py": {"subprocess"},
    "tests/test_rlvr_launch.py": {"subprocess"},
    "tests/test_enforced_rules.py": {"subprocess"},
    "scripts/modal_rlvr_grpo.py": {"modal"},
    "scripts/modal_chat_sft.py": {"modal"},
    "scripts/modal_chat_dpo.py": {"modal"},
    "scripts/modal_emit_completions.py": {"modal"},
    ".agents/skills/verify-esme-posttrain/verify": {"socket", "subprocess"},
    "src/esme_posttrain/launch/modal_cli.py": {"subprocess"},
    "src/esme_posttrain/training/wandb_init.py": {"wandb"},
}


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def banned_uses(source: str, banned: set[str]) -> set[str]:
    """Banned APIs the source imports or reaches by attribute, whatever it names them."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Attribute) and (dotted := _dotted(node)):
            names.add(dotted)
    return {ban for ban in banned for name in names if name == ban or name.startswith(ban + ".")}


NOQA = re.compile(r"#\s*(?:ruff:\s*)?noqa\b(?::\s*([A-Za-z0-9, ]+))?", re.IGNORECASE)


def silences_a_ban(line: str) -> bool:
    """A bare `noqa`, `noqa: ALL`, or any TID code, in a line or file-wide comment."""
    found = NOQA.search(line)
    if not found:
        return False
    codes = (found.group(1) or "").upper().replace(",", " ").split()
    return not codes or any(code == "ALL" or code.startswith("TID") for code in codes)


def ban_problems(root: Path, owned: dict[str, set[str]]) -> list[str]:
    lint = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
    banned = set(lint["flake8-tidy-imports"]["banned-api"])
    lifted = {path for path, codes in lint.get("per-file-ignores", {}).items() if "TID251" in codes}
    problems = [f"{path}: TID251 ignore has no OWNED_BANS entry" for path in lifted - owned.keys()]
    problems += [f"{path}: OWNED_BANS entry has no TID251 ignore" for path in owned.keys() - lifted]
    for path in sorted(lifted & owned.keys()):
        extra = banned_uses((root / path).read_text(), banned) - owned[path]
        if extra:
            problems.append(f"{path}: uses {sorted(extra)}, but owns only {sorted(owned[path])}")
    for path in candidate_python_files():
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if path.name != Path(__file__).name and silences_a_ban(line):
                problems.append(
                    f"{path.relative_to(root)}:{number} silences a module ban with noqa"
                )
    return problems


def test_ban_owners_use_only_the_modules_they_own() -> None:
    assert ban_problems(ROOT, OWNED_BANS) == []


def test_banned_uses_finds_imports_and_attribute_calls() -> None:
    banned = {"subprocess", "os.system", "urllib.request", "modal"}
    assert banned_uses("import subprocess as sp\n", banned) == {"subprocess"}
    assert banned_uses("import os\nos.system('x')\n", banned) == {"os.system"}
    assert banned_uses("from urllib import request\n", banned) == {"urllib.request"}
    assert banned_uses("import json\n", banned) == set()
    for line in ("import modal  # noqa", "# ruff: noqa: TID251", "import x  # noqa: F401, ALL"):
        assert silences_a_ban(line), line
    assert not silences_a_ban("x = 1  # noqa: E501")
