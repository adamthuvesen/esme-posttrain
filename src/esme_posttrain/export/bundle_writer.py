from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import torch

from esme_posttrain.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    canonical_model_config,
    file_sha256,
    llm_infer_model_config,
)
from esme_posttrain.modeling import DenseBackbone


class DenseBundleStageMetadata(TypedDict):
    model: dict[str, str]
    provenance: dict[str, Any]
    chat_template: NotRequired[dict[str, Any]]
    source_manifest: NotRequired[dict[str, Any]]
    source_best_checkpoint: NotRequired[dict[str, Any]]
    source_eval_report: NotRequired[dict[str, Any]]


def write_dense_bundle(
    bundle_dir: Path,
    *,
    model: DenseBackbone,
    checkpoint_step: int,
    source_checkpoint: str,
    source_checkpoint_sha256: str | None,
    eos_token_ids: list[int],
    readme: str,
    stage_metadata: DenseBundleStageMetadata,
) -> None:
    """Write a dense bundle after its stage prepares ``tokenizer.json``."""
    model_config = canonical_model_config(model.config)
    (bundle_dir / "config.json").write_text(
        json.dumps(model_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    weights_payload = {
        "format_version": BUNDLE_SCHEMA_VERSION,
        "format": BUNDLE_FORMAT,
        "key_format": BUNDLE_FORMAT,
        "metadata": {"key_format": BUNDLE_FORMAT, "state_dict_key": "state_dict"},
        "state_dict_key": "state_dict",
        "model_config": model_config,
        "llm_infer_config": llm_infer_model_config(model_config),
        "state_dict": model.state_dict(),
        "checkpoint_step": checkpoint_step,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    weights_path = bundle_dir / "weights.pt"
    tmp_path = weights_path.with_suffix(".pt.tmp")
    torch.save(weights_payload, tmp_path)
    tmp_path.replace(weights_path)
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        **stage_metadata,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "format": BUNDLE_FORMAT,
        "weights_format": BUNDLE_FORMAT,
        "model_family": "DenseBackbone",
        "model_config": model_config,
        "llm_infer_config": llm_infer_model_config(model_config),
        "checkpoint_step": checkpoint_step,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "files": {
            key: {"path": name, "sha256": file_sha256(bundle_dir / name)}
            for key, name in (
                ("config", "config.json"),
                ("tokenizer", "tokenizer.json"),
                ("weights", "weights.pt"),
                ("readme", "README.md"),
            )
        },
        "tokenizer": {"path": "tokenizer.json", "format": "tokenizers-json"},
        "eos_token_ids": eos_token_ids,
        "decoding": {
            "eos_token_ids": eos_token_ids,
            "default_add_special_tokens": False,
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
