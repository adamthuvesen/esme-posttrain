"""Every path to a paid Modal call passes an approval gate first.

A launcher is any repo module that builds a ``modal.App``. A launch is
``<modal function>.remote/.spawn/.map/...``. A gate is an earlier
``if not approved:`` or ``if blockers:`` whose body returns or raises. A helper
that launches without a gate passes only when every caller gates first.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_METHODS = {"remote", "spawn", "map", "starmap", "for_each", "spawn_map"}
LAUNCHERS = sorted(
    path
    for path in [*REPO_ROOT.glob("scripts/*.py"), *REPO_ROOT.glob("src/**/*.py")]
    if "modal.App(" in path.read_text(encoding="utf-8")
)


def _modal_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr in {"function", "local_entrypoint"}
            for d in node.decorator_list
        )
    }


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _is_gate(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    refuses_unapproved = (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _name(test.operand) == "approved"
    )
    stops_on_blockers = (_name(test) or "").endswith("blockers")
    stops = any(isinstance(n, ast.Return | ast.Raise) for s in node.body for n in ast.walk(s))
    return stops and (refuses_unapproved or stops_on_blockers)


def _gated(function: ast.FunctionDef, line: int) -> bool:
    return any(
        isinstance(n, ast.If) and _is_gate(n) and n.lineno < line for n in ast.walk(function)
    )


def _ungated_paths(path: Path) -> tuple[int, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modal_functions = _modal_function_names(tree)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls(function: ast.FunctionDef) -> list[ast.Call]:
        return [n for n in ast.walk(function) if isinstance(n, ast.Call)]

    def reaches_ungated(name: str, line: int, seen: frozenset[str]) -> bool:
        if _gated(functions[name], line):
            return False
        callers = [
            (caller, call.lineno)
            for caller in functions.values()
            for call in calls(caller)
            if isinstance(call.func, ast.Name) and call.func.id == name
        ]
        if not callers or name in seen:
            return True
        return any(reaches_ungated(c.name, ln, seen | {name}) for c, ln in callers)

    launches, failures = 0, []
    for function in functions.values():
        for call in calls(function):
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in LAUNCH_METHODS
                and isinstance(func.value, ast.Name)
                and func.value.id in modal_functions
            ):
                continue
            launches += 1
            if reaches_ungated(function.name, call.lineno, frozenset()):
                failures.append(
                    f"{path.relative_to(REPO_ROOT)}:{call.lineno} {function.name}() calls "
                    f"{func.value.id}.{func.attr} with no approval gate before it. Refuse first "
                    "with `if not approved: raise SystemExit(...)` naming LAUNCH_APPROVAL_FLAG "
                    "and the exact command, or return when stage `full_launch_blockers` is set."
                )
    return launches, failures


def test_repo_has_modal_launchers() -> None:
    assert LAUNCHERS, "no module builds modal.App; update LAUNCHERS discovery"


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_modal_launches_pass_an_approval_gate(path: Path) -> None:
    launches, failures = _ungated_paths(path)
    assert launches, f"{path.name} builds modal.App but no launch call was found"
    assert not failures, "\n".join(failures)


def test_emit_completions_refuses_without_approval_and_prints_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import modal_emit_completions as emit

    def no_modal_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("emit_job.map ran without approval")

    monkeypatch.setattr(emit.emit_job, "map", no_modal_call)
    jobs = tmp_path / "jobs.json"
    job = {"label": "a", "split": "s", "n": 1, "temperature": 1.0, "max_new_tokens": 8, "seed": 0}
    jobs.write_text(json.dumps([{**job, "bundle": "/root/exports/b"}]), encoding="utf-8")

    with pytest.raises(
        SystemExit, match=r"need --approved\. Ask Adam, then run: uv run .*--approved"
    ):
        emit.main.info.raw_f(str(jobs), str(tmp_path / "out"))  # pyright: ignore[reportOptionalCall]

    jobs.write_text(json.dumps([job]), encoding="utf-8")
    with pytest.raises(SystemExit, match="give exactly one of bundle, checkpoint"):
        emit.main.info.raw_f(str(jobs), str(tmp_path / "out"))  # pyright: ignore[reportOptionalCall]


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_launchers_with_a_parser_have_one_cli_surface(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "build_parser" not in functions:
        return
    entrypoints = [
        f
        for f in functions.values()
        if any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "local_entrypoint"
            for d in f.decorator_list
        )
    ]
    for entrypoint in entrypoints:
        args = entrypoint.args
        assert args.vararg and not (args.args or args.kwonlyargs), (
            f"{path.name}:{entrypoint.lineno} {entrypoint.name}() declares its own flags. "
            "Take `*argv: str` and pass it to launch(), so build_parser() is the only CLI surface."
        )
