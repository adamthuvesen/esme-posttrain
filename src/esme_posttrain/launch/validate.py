from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias, cast

from esme_posttrain.launch.common import LaunchError

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def load_json_file(path: Path, label: str) -> JsonObject:
    if not path.exists():
        raise LaunchError(f"missing {label}: {path}")
    if not path.is_file():
        raise LaunchError(f"{label} must be a file: {path}")

    try:
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise LaunchError(f"malformed {label} JSON at {path}: {exc.msg}") from exc
    return ensure_object(raw, label)


def ensure_object(raw: JsonValue, context: str) -> JsonObject:
    if not isinstance(raw, dict):
        raise LaunchError(f"{context} must be a JSON object")
    return cast(JsonObject, raw)


def require_object(parent: JsonObject, key: str, context: str) -> JsonObject:
    if key not in parent:
        raise LaunchError(f"{context}.{key} is required")
    return ensure_object(parent[key], f"{context}.{key}")


def require_list(parent: JsonObject, key: str, context: str) -> list[JsonValue]:
    if key not in parent:
        raise LaunchError(f"{context}.{key} is required")
    raw = parent[key]
    if not isinstance(raw, list):
        raise LaunchError(f"{context}.{key} must be a list")
    return raw


def require_str_list(parent: JsonObject, key: str, context: str) -> list[str]:
    raw_list = require_list(parent, key, context)
    values: list[str] = []
    for index, raw in enumerate(raw_list):
        if not isinstance(raw, str) or not raw:
            raise LaunchError(f"{context}.{key}[{index}] must be a non-empty string")
        values.append(raw)
    return values


def require_str(parent: JsonObject, key: str, context: str) -> str:
    if key not in parent:
        raise LaunchError(f"{context}.{key} is required")
    raw = parent[key]
    if not isinstance(raw, str) or not raw:
        raise LaunchError(f"{context}.{key} must be a non-empty string")
    return raw


def require_bool(parent: JsonObject, key: str, context: str) -> bool:
    if key not in parent:
        raise LaunchError(f"{context}.{key} is required")
    raw = parent[key]
    if not isinstance(raw, bool):
        raise LaunchError(f"{context}.{key} must be a boolean")
    return raw


def require_positive_int(parent: JsonObject, key: str, context: str) -> int:
    if key not in parent:
        raise LaunchError(f"{context}.{key} is required")
    raw = parent[key]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise LaunchError(f"{context}.{key} must be a positive integer")
    return raw


def require_path(parent: JsonObject, key: str, base_dir: Path, context: str) -> Path:
    raw_path = require_str(parent, key, context)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def require_exact(parent: JsonObject, key: str, expected: str, context: str) -> None:
    actual = require_str(parent, key, context)
    if actual != expected:
        raise LaunchError(f"{context}.{key} must be {expected!r}, got {actual!r}")


def require_schema_version(parent: JsonObject, context: str) -> None:
    version = require_positive_int(parent, "schema_version", context)
    if version != 1:
        raise LaunchError(f"{context}.schema_version must be 1, got {version}")


def require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise LaunchError(f"missing {label}: {path}")
    if not path.is_file():
        raise LaunchError(f"{label} must be a file: {path}")


def require_existing_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise LaunchError(f"missing {label}: {path}")
    if not path.is_dir():
        raise LaunchError(f"{label} must be a directory: {path}")


def iter_jsonl(path: Path) -> Iterator[tuple[str, JsonObject]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise LaunchError(f"{path}:{line_number}: blank JSONL lines are not allowed")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LaunchError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
            yield f"{path}:{line_number}", ensure_object(raw, f"{path}:{line_number}")
