from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, cast

from esme_posttrain.launch.errors import LaunchError

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
