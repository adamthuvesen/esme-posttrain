"""Turn a GRPO training checkpoint into a dense bundle at that checkpoint's step.

A GRPO run exports only its best-by-``train/reward_mean`` bundle, but it also writes a
training checkpoint every ``checkpoint_interval`` steps. This module rebuilds a bundle
from any of those checkpoints, so arms can be compared at one fixed step. The run's
exported bundle supplies the tokenizer and the bundle identity; the weights and the
step come from the checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.bundle import canonical_model_config, file_sha256, validate_bundle_contract
from esme_posttrain.rl.grpo import RLVRBundleIdentity, write_rlvr_bundle
from esme_posttrain.training.checkpointing import load_training_checkpoint


class CheckpointBundleError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointBundleRequest:
    checkpoint_path: Path
    template_bundle_dir: Path
    output_dir: Path
    expected_step: int | None = None


def export_checkpoint_bundle(request: CheckpointBundleRequest) -> dict[str, Any]:
    template_manifest = _read_template_manifest(request.template_bundle_dir)
    checkpoint = load_training_checkpoint(request.checkpoint_path)
    if request.expected_step is not None and checkpoint.step != request.expected_step:
        raise CheckpointBundleError(
            f"checkpoint {request.checkpoint_path} is step {checkpoint.step}, "
            f"expected {request.expected_step}"
        )
    # Bundles written by older code also carry logit_soft_cap and mtp_predict_tokens at
    # their disabled values. Those keys are no longer canonical, so compare only the
    # canonical ones; canonical_model_config already rejects a non-zero soft cap.
    checkpoint_config = canonical_model_config(checkpoint.config)
    template_config = template_manifest.get("model_config") or {}
    mismatched = sorted(
        key for key in checkpoint_config if template_config.get(key) != checkpoint_config[key]
    )
    if mismatched:
        raise CheckpointBundleError(
            f"checkpoint model config does not match the template bundle on {mismatched}"
        )

    tokenizer = Tokenizer.from_file(str(request.template_bundle_dir / "tokenizer.json"))
    write_rlvr_bundle(
        request.output_dir,
        model=checkpoint.model,
        tokenizer=tokenizer,
        identity=_identity_from_manifest(template_manifest),
        checkpoint_step=checkpoint.step,
        extra_provenance={
            "training_checkpoint": {
                "path": str(request.checkpoint_path),
                "sha256": file_sha256(request.checkpoint_path),
                "step": checkpoint.step,
            },
            "template_bundle_manifest_sha256": file_sha256(
                request.template_bundle_dir / "manifest.json"
            ),
        },
    )
    bundle = validate_bundle_contract(request.output_dir)
    return {
        "bundle_dir": str(bundle.bundle_dir),
        "checkpoint_step": checkpoint.step,
        "weights_sha256": file_sha256(bundle.weights_path),
    }


def _read_template_manifest(template_bundle_dir: Path) -> dict[str, Any]:
    # Only the manifest and tokenizer are read, so a template copied without its
    # weights is enough.
    for name in ("manifest.json", "tokenizer.json"):
        if not (template_bundle_dir / name).is_file():
            raise CheckpointBundleError(f"template bundle is missing {name}: {template_bundle_dir}")
    try:
        manifest = json.loads((template_bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CheckpointBundleError(f"template bundle manifest is not JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise CheckpointBundleError("template bundle manifest must be an object")
    return manifest


def _identity_from_manifest(manifest: dict[str, Any]) -> RLVRBundleIdentity:
    try:
        model = manifest["model"]
        provenance = manifest["provenance"]
        source_manifest = dict(provenance["source_manifest"])
        if "source_checkpoint" in manifest:
            reference_name = str(provenance["starts_from"])
            source_checkpoint = str(manifest["source_checkpoint"])
            source_sha256 = str(manifest["source_checkpoint_sha256"])
        else:
            # Bundles from the 2026-07 multiseed runs predate the top-level source
            # fields. The warm start is still pinned by the embedded source manifest,
            # which is the same weights file current runs record as source_checkpoint.
            source_model_id = str(source_manifest["model"]["id"])
            reference_name = str(source_manifest["model"]["name"])
            source_checkpoint = f"{source_model_id}/weights.pt"
            source_sha256 = str(source_manifest["files"]["weights"]["sha256"])
        return RLVRBundleIdentity(
            artifact_name=str(model["name"]),
            reference_artifact_name=reference_name,
            bundle_model_id=str(model["id"]),
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha256,
            source_manifest=source_manifest,
        )
    except (KeyError, TypeError) as error:
        raise CheckpointBundleError(
            f"template bundle manifest is missing identity field {error}"
        ) from error
