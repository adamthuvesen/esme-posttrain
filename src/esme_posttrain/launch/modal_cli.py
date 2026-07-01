from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def format_payload(payload: dict[str, Any], *, json_output: bool, keys: tuple[str, ...]) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [f"status: {payload['status']}"] if "status" in payload else ["status: unknown"]
    for key in keys:
        if key in payload:
            lines.append(f"{key}: {payload[key]}")
    for key in ("launch_blockers", "full_launch_blockers"):
        if isinstance(payload.get(key), list):
            lines.append(f"{key}: {', '.join(payload[key]) or 'none'}")
    return "\n".join(lines)


def modal_call_id(function_call: Any) -> str:
    return (
        getattr(function_call, "object_id", None)
        or getattr(function_call, "function_call_id", None)
        or str(function_call)
    )


def local_git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def local_git_dirty(repo_root: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


def validate_output_stem(
    value: str, *, env_var: str, forbidden_substrings: tuple[str, ...] = ()
) -> str:
    stem = value.strip()
    if not stem:
        raise ValueError(f"{env_var} must be a non-empty output stem")
    path = Path(stem)
    if path.is_absolute() or path.name != stem or stem in {".", ".."} or "\\" in stem:
        raise ValueError(f"{env_var} must be a single relative output stem")
    if any(not (char.islower() or char.isdigit() or char == "-") for char in stem):
        raise ValueError(f"{env_var} must use lowercase letters, digits, and hyphens")
    for forbidden in forbidden_substrings:
        if forbidden in stem:
            raise ValueError(f"{env_var} must not contain {forbidden}")
    return stem


def fresh_output_dir(root: Path, stem: str, *, label: str = "Modal output") -> Path:
    base = root / stem
    for suffix in ("", *[f"-{index}" for index in range(1, 100)]):
        candidate = Path(f"{base}{suffix}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"could not find an empty {label} directory under {root}")


def command_with_output_stem(
    command: str,
    *,
    output_stem: str,
    default_stem: str,
    env_var: str,
) -> str:
    if output_stem == default_stem:
        return command
    return f"{env_var}='{output_stem}' {command}"


def with_full_output_stem(
    payload: dict[str, Any],
    *,
    full_launch_command: str,
    volume_output_dir: str,
) -> dict[str, Any]:
    updated = {
        **payload,
        "full_launch_command": full_launch_command,
        "volume_output_dir": volume_output_dir,
    }
    preflight = dict(updated.get("preflight", {}))
    preflight["exact_launch_command"] = full_launch_command
    preflight["volume_output_dir"] = volume_output_dir
    updated["preflight"] = preflight
    return updated
