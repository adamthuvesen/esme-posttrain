from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.modeling import BackboneConfig, DenseBackbone

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FORMAT = "llm_pretrain_dense_v1"
CANONICAL_CONFIG_KEYS = (
    "name",
    "vocab_size",
    "context_length",
    "embedding_dim",
    "layers",
    "heads",
    "feedforward_dim",
    "kv_heads",
    "rope_theta",
    "rms_norm_eps",
    "tie_embeddings",
    "qk_norm",
    "z_loss_weight",
    "attention_kind",
)


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


def canonical_model_config(config: BackboneConfig) -> dict[str, Any]:
    """Project a posttrain config onto the canonical dense-bundle v1 fields."""
    payload = config.to_dict()
    if payload.get("logit_soft_cap", 0.0) != 0.0:
        raise BundleError("dense-bundle v1 cannot represent a non-zero logit_soft_cap")
    if payload.get("mtp_predict_tokens", 0) != 0:
        raise BundleError("dense-bundle v1 cannot represent MTP prediction heads")
    return {key: payload[key] for key in CANONICAL_CONFIG_KEYS}


def llm_infer_model_config(config: dict[str, Any]) -> dict[str, Any]:
    kv_heads = config["heads"] if config["kv_heads"] is None else config["kv_heads"]
    return {
        "format": BUNDLE_FORMAT,
        "name": config["name"],
        "vocab_size": config["vocab_size"],
        "context_length": config["context_length"],
        "hidden_size": config["embedding_dim"],
        "intermediate_size": config["feedforward_dim"],
        "num_hidden_layers": config["layers"],
        "num_attention_heads": config["heads"],
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "tie_word_embeddings": config["tie_embeddings"],
        "attention_kind": config["attention_kind"],
        "qk_norm": config["qk_norm"],
        "z_loss_weight": config["z_loss_weight"],
    }


def validate_base_bundle(bundle_dir: Path) -> ValidatedBundle:
    bundle_dir = bundle_dir.expanduser().resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"missing base bundle manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path, "base bundle manifest")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f"unsupported bundle schema_version: {manifest.get('schema_version')!r} "
            f"(this loader supports {BUNDLE_SCHEMA_VERSION})"
        )
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"base bundle format must be {BUNDLE_FORMAT}")
    if manifest.get("model_family") != "DenseBackbone":
        raise BundleError("base bundle model_family must be DenseBackbone")
    if manifest.get("weights_format") != BUNDLE_FORMAT:
        raise BundleError(f"base bundle weights_format must be {BUNDLE_FORMAT}")

    files = _require_json_object(manifest.get("files"), "base bundle manifest.files")
    config_path = _verified_file(bundle_dir, files, "config")
    tokenizer_path = _verified_file(bundle_dir, files, "tokenizer")
    weights_path = _verified_file(bundle_dir, files, "weights")

    config_payload = _load_json_object(config_path, "base bundle config")
    config = BackboneConfig.from_dict(config_payload)
    manifest_config = _require_json_object(
        manifest.get("model_config"), "base bundle manifest.model_config"
    )
    if manifest_config != config_payload:
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
    weights = _load_and_validate_weights(
        bundle, map_location=map_location, require_canonical_contract=False
    )
    tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))

    model = DenseBackbone(bundle.config)
    model.load_state_dict(_require_json_object(weights.get("state_dict"), "weights.state_dict"))
    model.eval()
    return LoadedBundle(bundle=bundle, model=model, tokenizer=tokenizer)


def validate_bundle_contract(bundle_dir: Path) -> ValidatedBundle:
    """Check manifest, hashes, config, and weights metadata without creating a model."""
    bundle = validate_base_bundle(bundle_dir)
    _validate_canonical_manifest(bundle)
    _load_and_validate_weights(bundle, map_location="cpu", require_canonical_contract=True)
    return bundle


def validate_bundle_compatibility(bundle_dir: Path) -> ValidatedBundle:
    """Reject missing or unsupported versions before expensive consumer setup."""
    bundle = validate_base_bundle(bundle_dir)
    _load_and_validate_weights(bundle, map_location="cpu", require_canonical_contract=False)
    return bundle


def _validate_canonical_manifest(bundle: ValidatedBundle) -> None:
    expected_files = {"README.md", "config.json", "manifest.json", "tokenizer.json", "weights.pt"}
    actual_files = {path.name for path in bundle.bundle_dir.iterdir()}
    if actual_files != expected_files:
        raise BundleError(
            "dense-bundle v1 file set must be exactly "
            f"{sorted(expected_files)}, got {sorted(actual_files)}"
        )

    files = _require_json_object(bundle.manifest.get("files"), "base bundle manifest.files")
    if set(files) != {"config", "tokenizer", "weights", "readme"}:
        raise BundleError(
            "base bundle manifest.files must contain exactly config, tokenizer, weights, and readme"
        )
    readme_path = _verified_file(bundle.bundle_dir, files, "readme")
    if readme_path.name != "README.md":
        raise BundleError("base bundle manifest.files.readme.path must be README.md")

    tokenizer = _require_json_object(
        bundle.manifest.get("tokenizer"), "base bundle manifest.tokenizer"
    )
    if tokenizer != {"path": "tokenizer.json", "format": "tokenizers-json"}:
        raise BundleError(
            "base bundle manifest.tokenizer must declare tokenizer.json in tokenizers-json format"
        )

    config_payload = _load_json_object(bundle.config_path, "base bundle config")
    if set(config_payload) != set(CANONICAL_CONFIG_KEYS):
        raise BundleError(
            f"dense-bundle v1 config.json keys must be exactly {sorted(CANONICAL_CONFIG_KEYS)}"
        )
    expected_infer_config = llm_infer_model_config(config_payload)
    if bundle.manifest.get("llm_infer_config") != expected_infer_config:
        raise BundleError("base bundle manifest.llm_infer_config does not match config.json")

    checkpoint_step = bundle.manifest.get("checkpoint_step")
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise BundleError("base bundle manifest.checkpoint_step must be an integer")
    source_checkpoint = bundle.manifest.get("source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint:
        raise BundleError("base bundle manifest.source_checkpoint must be a non-empty string")
    _require_sha256(
        bundle.manifest.get("source_checkpoint_sha256"),
        "base bundle manifest.source_checkpoint_sha256",
    )


def _load_and_validate_weights(
    bundle: ValidatedBundle,
    *,
    map_location: str | torch.device,
    require_canonical_contract: bool,
) -> dict[str, Any]:
    try:
        payload = torch.load(bundle.weights_path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
        raise BundleError(f"base bundle weights could not be loaded safely: {error}") from error
    weights = _require_json_object(payload, "base bundle weights")
    if weights.get("format_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f"unsupported weights.pt format_version: {weights.get('format_version')!r} "
            f"(this loader supports {BUNDLE_SCHEMA_VERSION})"
        )
    if weights.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"weights.format must be {BUNDLE_FORMAT}")
    if weights.get("key_format") != BUNDLE_FORMAT:
        raise BundleError(f"weights.key_format must be {BUNDLE_FORMAT}")
    if _require_json_object(
        weights.get("model_config"), "weights.model_config"
    ) != _load_json_object(bundle.config_path, "base bundle config"):
        raise BundleError("weights.model_config does not match config.json")
    _require_json_object(weights.get("state_dict"), "weights.state_dict")
    if not require_canonical_contract:
        return weights

    metadata = _require_json_object(weights.get("metadata"), "weights.metadata")
    if metadata.get("key_format") != BUNDLE_FORMAT:
        raise BundleError(f"weights.metadata.key_format must be {BUNDLE_FORMAT}")
    manifest = bundle.manifest
    for field in ("checkpoint_step", "source_checkpoint", "source_checkpoint_sha256"):
        if weights.get(field) != manifest.get(field):
            raise BundleError(f"weights.{field} does not match manifest.{field}")
    if weights.get("llm_infer_config") != manifest.get("llm_infer_config"):
        raise BundleError("weights.llm_infer_config does not match manifest.llm_infer_config")
    return weights


def _verified_file(bundle_dir: Path, files: dict[str, Any], key: str) -> Path:
    entry = _require_json_object(files.get(key), f"base bundle manifest.files.{key}")
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
    return _require_json_object(raw, label)


def _require_json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BundleError(f"{label} must be a lowercase SHA256 hex digest")
    return value
