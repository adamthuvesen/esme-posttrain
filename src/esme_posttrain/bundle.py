from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.modeling import BackboneConfig, DenseBackbone

BUNDLE_FORMAT = "llm_pretrain_dense_v1"


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class ValidatedBundle:
    bundle_dir: Path
    manifest_path: Path
    config_path: Path
    tokenizer_path: Path
    weights_path: Path
    manifest: dict[str, Any]
    config: BackboneConfig


@dataclass(frozen=True)
class LoadedBundle:
    bundle: ValidatedBundle
    model: DenseBackbone
    tokenizer: Tokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_base_bundle(bundle_dir: Path) -> ValidatedBundle:
    bundle_dir = bundle_dir.expanduser().resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"missing base bundle manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path, "base bundle manifest")
    if manifest.get("schema_version") != 1:
        raise BundleError("base bundle manifest.schema_version must be 1")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"base bundle format must be {BUNDLE_FORMAT}")
    if manifest.get("model_family") != "DenseBackbone":
        raise BundleError("base bundle model_family must be DenseBackbone")
    if manifest.get("weights_format") != BUNDLE_FORMAT:
        raise BundleError(f"base bundle weights_format must be {BUNDLE_FORMAT}")

    files = _object(manifest.get("files"), "base bundle manifest.files")
    config_path = _verified_file(bundle_dir, files, "config")
    tokenizer_path = _verified_file(bundle_dir, files, "tokenizer")
    weights_path = _verified_file(bundle_dir, files, "weights")

    config_payload = _load_json_object(config_path, "base bundle config")
    config = BackboneConfig.from_dict(config_payload)
    manifest_config = _object(manifest.get("model_config"), "base bundle manifest.model_config")
    if manifest_config != config.to_dict():
        raise BundleError("base bundle config.json does not match manifest.model_config")

    return ValidatedBundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        weights_path=weights_path,
        manifest=manifest,
        config=config,
    )


def load_dense_backbone_bundle(
    bundle_dir: Path, *, map_location: str | torch.device = "cpu"
) -> LoadedBundle:
    bundle = validate_base_bundle(bundle_dir)
    tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))
    # weights_only=False is intentional for the local esme-pretrain export payload. It
    # carries trusted metadata in addition to tensors, and the manifest hashes were
    # verified before loading.
    payload = torch.load(bundle.weights_path, map_location=map_location, weights_only=False)
    weights = _object(payload, "base bundle weights")
    if weights.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"weights.format must be {BUNDLE_FORMAT}")
    if weights.get("key_format") != BUNDLE_FORMAT:
        raise BundleError(f"weights.key_format must be {BUNDLE_FORMAT}")
    if _object(weights.get("model_config"), "weights.model_config") != bundle.config.to_dict():
        raise BundleError("weights.model_config does not match config.json")
    state_dict = _object(weights.get("state_dict"), "weights.state_dict")

    model = DenseBackbone(bundle.config)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedBundle(bundle=bundle, model=model, tokenizer=tokenizer)


def _verified_file(bundle_dir: Path, files: dict[str, Any], key: str) -> Path:
    entry = _object(files.get(key), f"base bundle manifest.files.{key}")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BundleError(f"base bundle manifest.files.{key}.path must be a non-empty string")
    path = (bundle_dir / raw_path).resolve()
    if not path.is_relative_to(bundle_dir):
        raise BundleError(f"base bundle manifest.files.{key}.path escapes the bundle directory")
    if not path.is_file():
        raise BundleError(f"missing base bundle file for {key}: {path}")
    expected = entry.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise BundleError(f"base bundle manifest.files.{key}.sha256 must be a SHA256 hex digest")
    actual = file_sha256(path)
    if actual != expected:
        raise BundleError(f"base bundle hash mismatch for {key}: expected {expected}, got {actual}")
    return path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BundleError(f"malformed {label} JSON at {path}: {error.msg}") from error
    return _object(raw, label)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value
