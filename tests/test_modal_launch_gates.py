"""Every path to a paid Modal call passes an approval gate first.

A launcher is any repo module that imports ``modal``. A launch is
``<modal function or class>...remote/.spawn/.map/...``, including chains such as
``fn.with_options(gpu=...).spawn()``, and in class methods. A gate is
``if not approved:``, or ``if blockers:`` where ``blockers`` came from a call that
forwards ``approved=``, whose body returns or raises, in a block that encloses the
launch and before it. A gate in a sibling branch does not count. A helper that launches
without a gate passes only when every call to it is gated. A function that writes
files, checkpoints, or a login before its gate, or while refusing, fails.
``approved=True`` near a launch, an ``approved`` parameter that defaults to True,
``modal.Sandbox``, ``Image.run_function``, and ``Function.from_name`` fail: each
one spends or approves without a gate this scan can see.
"""

from __future__ import annotations

import ast
import importlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_METHODS = {"remote", "spawn", "map", "starmap", "for_each", "spawn_map", "remote_gen"}
STATE_WRITES = {
    *("open", "write_text", "write_bytes", "mkdir", "makedirs", "touch", "mkdtemp"),
    *("save", "commit", "init", "login"),
}
UNSCANNED_SPEND = {"Sandbox", "run_function"}


def _imports_modal(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import) and any(
            a.name.split(".")[0] == "modal" for a in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "modal":
            return True
    return False


LAUNCHERS = sorted(
    path
    for path in [*REPO_ROOT.glob("scripts/*.py"), *REPO_ROOT.glob("src/**/*.py")]
    if _imports_modal(path)
)


def _modal_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
        and any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr in {"function", "local_entrypoint", "cls"}
            for d in node.decorator_list
        )
    }


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _launch_root(call: ast.Call) -> str | None:
    """The name a launch chain starts from: `fn` in `fn.with_options(...).spawn()`."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in LAUNCH_METHODS):
        return None
    node: ast.expr = func.value
    while isinstance(node, ast.Attribute | ast.Call):
        node = node.value if isinstance(node, ast.Attribute) else node.func
    return node.id if isinstance(node, ast.Name) else None


def _approval_blockers(function: ast.FunctionDef) -> frozenset[str]:
    """Names assigned from a call that forwards an `approved=` flag, such as stage blockers."""
    return frozenset(
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        for keyword in node.value.keywords
        if keyword.arg == "approved" and not isinstance(keyword.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    )


def _is_gate(node: ast.AST, approval_blockers: frozenset[str]) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    refuses_unapproved = (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _name(test.operand) == "approved"
    )
    stops_on_blockers = isinstance(test, ast.Name) and test.id in approval_blockers
    stops = any(isinstance(n, ast.Return | ast.Raise) for s in node.body for n in ast.walk(s))
    return stops and (refuses_unapproved or stops_on_blockers)


def _gated_calls(function: ast.FunctionDef) -> Iterator[tuple[ast.Call, bool]]:
    """Each call in `function`, and whether a gate in an enclosing block runs first."""
    approval_blockers = _approval_blockers(function)

    def block(statements: list[ast.stmt], gated: bool) -> Iterator[tuple[ast.Call, bool]]:
        for statement in statements:
            yield from walk(statement, gated)
            gated = gated or _is_gate(statement, approval_blockers)

    def walk(node: ast.AST, gated: bool) -> Iterator[tuple[ast.Call, bool]]:
        if isinstance(node, ast.Call):
            yield node, gated
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                inner = gated or (field == "orelse" and _is_gate(node, approval_blockers))
                yield from block(value, inner)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        yield from walk(item, gated)
            elif isinstance(value, ast.AST):
                yield from walk(value, gated)

    yield from block(function.body, False)


def _write_name(call: ast.Call) -> str | None:
    name = _name(call.func) or ""
    writes = name in STATE_WRITES or name.lstrip("_").startswith(("write_", "record_", "save_"))
    return name if writes else None


def _state_writes(function: ast.FunctionDef, label: str) -> list[str]:
    approval_blockers = _approval_blockers(function)
    gates = [
        n for n in ast.walk(function) if isinstance(n, ast.If) and _is_gate(n, approval_blockers)
    ]
    if not gates:
        return []
    first_gate = min(g.lineno for g in gates)
    refusal_lines = {
        n.lineno for g in gates for s in g.body for n in ast.walk(s) if isinstance(n, ast.Call)
    }
    writes = []
    for node in ast.walk(function):
        if not (isinstance(node, ast.Call) and (name := _write_name(node))):
            continue
        if node.lineno < first_gate:
            writes.append(
                f"{label}:{node.lineno} {function.name}() calls {name}() before its approval "
                "gate. Refuse first, then write state."
            )
        elif node.lineno in refusal_lines:
            writes.append(
                f"{label}:{node.lineno} {function.name}() calls {name}() while refusing. "
                "A refusal prints its blockers and writes nothing."
            )
    return writes


def _approval_shortcuts(tree: ast.Module, label: str) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            pairs = [
                *zip(
                    positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
                ),
                *((a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d),
            ]
            for arg, default in pairs:
                if arg.arg == "approved" and isinstance(default, ast.Constant) and default.value:
                    found.append(
                        f"{label}:{node.lineno} {node.name}() defaults approved to True. "
                        "Default to False so a caller that forgets the flag is refused."
                    )
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "from_name"
            and _name(node.value) in {"Function", "Cls"}
        ):
            found.append(
                f"{label}:{node.lineno} looks up a deployed Modal {_name(node.value)}, which this "
                "scan cannot gate. Launch a function this launcher defines instead."
            )
        if isinstance(node, ast.Attribute | ast.Name) and _name(node) in UNSCANNED_SPEND:
            found.append(
                f"{label}:{node.lineno} uses {_name(node)}, which spends with no gate this scan "
                "sees. Run the work in an approval-gated @app.function instead."
            )
    return found


def _ungated_paths(path: Path) -> tuple[int, list[str]]:
    return _ungated_source(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT)))


def _ungated_source(source: str, label: str) -> tuple[int, list[str]]:
    tree = ast.parse(source)
    modal_functions = _modal_function_names(tree)
    functions = {
        n.name: n
        for top in tree.body
        for n in ([top] if isinstance(top, ast.FunctionDef) else getattr(top, "body", []))
        if isinstance(n, ast.FunctionDef)
    }
    gated = {name: list(_gated_calls(f)) for name, f in functions.items()}

    def reaches_ungated(name: str, call_is_gated: bool, seen: frozenset[str]) -> bool:
        if call_is_gated:
            return False
        callers = [
            (caller, is_gated)
            for caller, calls in gated.items()
            for call, is_gated in calls
            if isinstance(call.func, ast.Name) and call.func.id == name
        ]
        if not callers or name in seen:
            return True
        return any(reaches_ungated(c, g, seen | {name}) for c, g in callers)

    def forwards_literal_approval(name: str) -> list[str]:
        return [
            f"{label}:{call.lineno} {name}() passes approved=True on a path to a Modal launch. "
            "Forward the entrypoint's --approved flag instead."
            for call, _ in gated[name]
            for keyword in call.keywords
            if keyword.arg == "approved"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
        ]

    launches, failures = 0, _approval_shortcuts(tree, label)
    launching: set[str] = set()
    for name, function in functions.items():
        failures += _state_writes(function, label)
        for call, is_gated in gated[name]:
            root = _launch_root(call)
            if root not in modal_functions:
                continue
            launches += 1
            launching.add(name)
            if reaches_ungated(name, is_gated, frozenset()):
                failures.append(
                    f"{label}:{call.lineno} {name}() calls {ast.unparse(call.func)} with no "
                    "approval gate before it. Refuse first with `if not approved: raise "
                    "SystemExit(...)` naming LAUNCH_APPROVAL_FLAG and the exact command, or "
                    "return when stage `full_launch_blockers` is set."
                )
    for name in sorted(launching):
        failures += forwards_literal_approval(name)
    return launches, failures


def test_repo_has_modal_launchers() -> None:
    assert LAUNCHERS, "no module builds modal.App; update LAUNCHERS discovery"


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_modal_launches_pass_an_approval_gate(path: Path) -> None:
    launches, failures = _ungated_paths(path)
    assert launches, f"{path.name} builds modal.App but no launch call was found"
    assert not failures, "\n".join(failures)


UNGATED = """
@app.function(gpu="H100")
def train(): ...

@app.cls(gpu="H100")
class Trainer: ...

def launch_early(approved):
    train.remote()
    if not approved:
        raise SystemExit("refused")

def launch_writes_first(approved, out):
    out.mkdir(parents=True)
    if not approved:
        raise SystemExit("refused")
    train.spawn()

def helper():
    train.map([1])

def caller(approved):
    helper()

def launch_ok(config, approved, out):
    blockers = full_launch_blockers(config, approved=approved)
    if blockers:
        return 2
    out.mkdir(parents=True)
    train.spawn()

def launch_sibling_branch(args):
    if args.smoke:
        if not args.approved:
            return 2
    train.with_options(gpu="H100:8").spawn()

def launch_class(approved):
    Trainer().fit.remote()

def launch_nested_ok(args):
    if args.smoke:
        if not args.approved:
            return 2
        train.spawn()

def refuse_with_report(config, approved):
    blockers = full_launch_blockers(config, approved=approved)
    if blockers:
        _write_blocked_report(blockers)
        return 2
    train.spawn()

def launch_default_approved(approved=True):
    if not approved:
        return 2
    train.spawn()

def launch_forged(config):
    blockers = full_launch_blockers(config, approved=True)
    if blockers:
        return 2
    train.spawn()

def launch_without_approval_blockers(config):
    blockers = probe_blockers(config)
    if blockers:
        return 2
    train.spawn()

class Launcher:
    def go(self):
        train.spawn()

def deployed():
    modal.Function.from_name("app", "train").spawn()

def sandbox():
    modal.Sandbox.create(app=app)

image = modal.Image.debian_slim().run_function(train, gpu="H100")
"""


def test_the_scan_finds_each_ungated_launch() -> None:
    launches, failures = _ungated_source(UNGATED, "bad.py")
    assert launches == 12
    assert sorted(failure.split(". ")[0] for failure in failures) == sorted(
        [
            "bad.py:9 launch_early() calls train.remote with no approval gate before it",
            "bad.py:14 launch_writes_first() calls mkdir() before its approval gate",
            "bad.py:20 helper() calls train.map with no approval gate before it",
            "bad.py:36 launch_sibling_branch() calls train.with_options(gpu='H100:8').spawn "
            "with no approval gate before it",
            "bad.py:39 launch_class() calls Trainer().fit.remote with no approval gate before it",
            "bad.py:50 refuse_with_report() calls _write_blocked_report() while refusing",
            "bad.py:54 launch_default_approved() defaults approved to True",
            "bad.py:63 launch_forged() calls train.spawn with no approval gate before it",
            "bad.py:60 launch_forged() passes approved=True on a path to a Modal launch",
            "bad.py:69 launch_without_approval_blockers() calls train.spawn with no approval "
            "gate before it",
            "bad.py:73 go() calls train.spawn with no approval gate before it",
            "bad.py:76 looks up a deployed Modal Function, which this scan cannot gate",
            "bad.py:79 uses Sandbox, which spends with no gate this scan sees",
            "bad.py:81 uses run_function, which spends with no gate this scan sees",
        ]
    )


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


LAUNCHER_CONFIGS = {
    "modal_chat_sft": "configs/esme-214m-sft-multiturn.json",
    "modal_chat_dpo": "configs/esme-214m-chat-dpo.json",
    "modal_rlvr_grpo": "fixtures/configs/esme-214m-rl.fixture.json",
}
# Flags that start no Modal work. Every other launcher flag is driven as a paid mode.
FREE_FLAGS = {
    "--approved",
    "--dry-run",
    "--json",
    "--resume",
    "--local-cpu-smoke",
    "--pipeline-smoke",
}


def _launch_flags(module_name: str) -> list[str]:
    parser = importlib.import_module(f"scripts.{module_name}").build_parser()
    return [
        flag
        for action in parser._actions
        if action.nargs == 0 and action.dest != "help"
        for flag in action.option_strings
        if flag.startswith("--") and flag not in FREE_FLAGS
    ]


PAID_MODES = [
    (module_name, config, flag)
    for module_name, config in LAUNCHER_CONFIGS.items()
    for flag in _launch_flags(module_name)
]


def test_every_launcher_with_a_parser_is_driven() -> None:
    with_parser = {
        path.stem for path in LAUNCHERS if "def build_parser" in path.read_text(encoding="utf-8")
    }
    assert with_parser == set(LAUNCHER_CONFIGS)


class _RecordingModalFunction:
    def __init__(self, calls: list[str], name: str) -> None:
        self._calls, self._name = calls, name

    def __getattr__(self, method: str) -> object:
        def record(*args: object, **kwargs: object) -> None:
            self._calls.append(f"{self._name}.{method}")
            raise AssertionError(f"{self._name}.{method} ran on a refused launch")

        return record


def _record_modal_calls(monkeypatch: pytest.MonkeyPatch, module_name: str) -> list[str]:
    module = importlib.import_module(f"scripts.{module_name}")
    path = REPO_ROOT / "scripts" / f"{module_name}.py"
    calls: list[str] = []
    for name in _modal_function_names(ast.parse(path.read_text(encoding="utf-8"))):
        monkeypatch.setattr(module, name, _RecordingModalFunction(calls, name))
    monkeypatch.setattr(module, "RLVR_TIMEOUT_HOURS", 1, raising=False)
    return calls


@pytest.mark.parametrize(("module_name", "config", "mode"), PAID_MODES)
def test_paid_modes_refuse_without_approval_and_call_nothing(
    module_name: str,
    config: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _record_modal_calls(monkeypatch, module_name)
    launch = importlib.import_module(f"scripts.{module_name}").launch
    assert launch(["--config", str(REPO_ROOT / config), mode, "--json"]) == 2
    assert calls == []
    # Other blockers, such as a missing base bundle, also refuse; the approval gate must too.
    output = "".join(capsys.readouterr())
    assert "requires --approved" in output or "pass --approved" in output, output[-300:]


@pytest.mark.parametrize(
    ("module_name", "config", "gpu_constant"),
    [
        ("modal_chat_sft", "configs/esme-214m-sft-multiturn.json", "SFT_MODAL_GPU"),
        ("modal_chat_dpo", "configs/esme-214m-chat-dpo.json", "DPO_MODAL_GPU"),
    ],
)
def test_approved_smoke_refuses_a_gpu_the_config_does_not_price(
    module_name: str,
    config: str,
    gpu_constant: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _record_modal_calls(monkeypatch, module_name)
    module = importlib.import_module(f"scripts.{module_name}")
    monkeypatch.setattr(module, gpu_constant, "H100:8")
    argv = ["--config", str(REPO_ROOT / config), "--modal-smoke", "--approved"]
    assert module.launch(argv) == 2
    assert calls == []
    assert f"{gpu_constant} must match" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("module_name", "config", "gpu_constant"),
    [
        ("modal_chat_sft", "configs/esme-214m-sft-multiturn.json", "SFT_MODAL_GPU"),
        ("modal_chat_dpo", "configs/esme-214m-chat-dpo.json", "DPO_MODAL_GPU"),
    ],
)
def test_smoke_dry_run_reports_the_gpu_blocker_the_launch_hits(
    module_name: str,
    config: str,
    gpu_constant: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(f"scripts.{module_name}")
    monkeypatch.setattr(module, gpu_constant, "H100:8")
    assert module.launch(["--config", str(REPO_ROOT / config), "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "blocked_by_launch_safety"
    assert any(f"{gpu_constant} must match" in b for b in plan["launch_blockers"])
