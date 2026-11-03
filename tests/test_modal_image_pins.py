"""Modal images install the uv.lock versions through uv, never floating versions through pip.

A floating version makes each image build pull whatever is newest, so a paid GPU run
cannot be reproduced and can drift from the stack the CPU tests ran on. The direct
runtime dependencies are pinned; their transitive dependencies resolve at build time.
Modal re-runs each image definition inside the container, so every file it reads is mounted.
"""

from __future__ import annotations

import ast
import shutil
import subprocess  # noqa: TID251
import sys
from pathlib import Path

import pytest

from esme_posttrain.launch.config_guards import image_package_pins

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = sorted(ROOT.glob("scripts/modal_*.py"))
OTHER_INSTALLERS = {
    *("pip_install", "pip_install_from_requirements", "pip_install_from_pyproject"),
    *("poetry_install_from_file", "micromamba_install", "uv_sync", "run_function"),
}


def install_problems(source: str, label: str) -> list[str]:
    problems: list[str] = []
    for call in (n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)):
        if not isinstance(call.func, ast.Attribute):
            continue
        method, where = call.func.attr, f"{label}:{call.func.end_lineno}"
        if method in OTHER_INSTALLERS:
            problems.append(f"{where} {method}() installs outside uv.lock. Use uv_pip_install.")
        elif method == "run_commands" and "install" in ast.unparse(call):
            problems.append(f"{where} run_commands() installs packages. Use uv_pip_install.")
        elif method == "uv_pip_install":
            packages = [ast.unparse(arg) for arg in call.args]
            locked = len(packages) == 1 and "image_package_pins()" in packages[0]
            pinned_uv = any(k.arg == "uv_version" for k in call.keywords)
            extra = {k.arg for k in call.keywords} - {"uv_version"}
            if not (locked and pinned_uv) or extra:
                problems.append(
                    f"{where} uv_pip_install() must take only `*(... image_package_pins() ...)` "
                    "and uv_version=image_uv_version(), so every package is at its uv.lock version."
                )
    return problems


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_images_install_locked_versions_through_uv(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert "uv_pip_install" in source, f"{path.name} builds no image through uv_pip_install"
    assert not install_problems(source, path.name)


def test_image_pins_cover_every_runtime_dependency_but_modal() -> None:
    assert set(image_package_pins()) == {
        *("datasets", "numpy", "pydantic", "tokenizers", "torch", "wandb"),
    }


def test_the_scan_finds_each_floating_install() -> None:
    bad = """
image = (
    modal.Image.debian_slim()
    .pip_install("torch==2.12.1")
    .uv_pip_install("numpy")
    .uv_pip_install(*(f"{n}=={v}" for n, v in image_package_pins().items()))
    .run_commands("pip install wandb")
)
"""
    found = sorted(" ".join(p.split(" ")[:2]) for p in install_problems(bad, "bad.py"))
    assert found == [
        "bad.py:4 pip_install()",
        "bad.py:5 uv_pip_install()",
        "bad.py:6 uv_pip_install()",
        "bad.py:7 run_commands()",
    ]


def _mounted_files(source: str) -> list[str]:
    return [
        keyword.value.value
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_local_file"
        for keyword in call.keywords
        if keyword.arg == "remote_path"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    ]


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_image_definition_runs_in_the_container_layout(path: Path, tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil.copytree(ROOT / "src", root / "src")
    for remote in _mounted_files(path.read_text(encoding="utf-8")):
        if remote.startswith("/root/") and remote.count("/") == 2:
            shutil.copy(ROOT / Path(remote).name, root / Path(remote).name)
    probe = (
        "from esme_posttrain.launch.config_guards import image_package_pins, image_uv_version\n"
        "from esme_posttrain.launch.modal_cli import LOCAL_PYTHON\n"
        "assert image_package_pins() and image_uv_version() and LOCAL_PYTHON\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, 'src')\n{probe}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{path.name}: the image definition fails in the container, where only mounted files "
        f"exist under /root. Mount each file it reads.\n{result.stderr[-400:]}"
    )


def test_image_pins_match_unnormalized_dependency_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from esme_posttrain.launch import config_guards

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\ndependencies = ["PyYAML>=6.0", "modal>=1"]\n')
    monkeypatch.setattr(config_guards, "PYPROJECT_FILE", pyproject)
    assert list(config_guards.image_package_pins()) == ["pyyaml"]
