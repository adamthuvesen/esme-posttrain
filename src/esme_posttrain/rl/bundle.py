from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.export.bundle_writer import write_dense_bundle
from esme_posttrain.modeling import DenseBackbone


@dataclass(frozen=True)
class RLVRBundleIdentity:
    artifact_name: str
    reference_artifact_name: str
    bundle_model_id: str
    source_checkpoint: str
    source_checkpoint_sha256: str | None
    source_manifest: dict[str, Any]


def write_rlvr_bundle(
    bundle_dir: Path,
    *,
    model: DenseBackbone,
    tokenizer: Tokenizer,
    identity: RLVRBundleIdentity,
    checkpoint_step: int,
    extra_provenance: dict[str, Any] | None = None,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(bundle_dir / "tokenizer.json"))
    eos_id = tokenizer.token_to_id("<eos>")
    write_dense_bundle(
        bundle_dir,
        model=model,
        checkpoint_step=checkpoint_step,
        source_checkpoint=identity.source_checkpoint,
        source_checkpoint_sha256=identity.source_checkpoint_sha256,
        eos_token_ids=[int(eos_id)] if eos_id is not None else [],
        readme=(
            f"# {identity.artifact_name} DenseBackbone Bundle\n\n"
            f"- Model id: `{identity.bundle_model_id}`\n"
            f"- Stage: `rlvr`; checkpoint step: `{checkpoint_step}`\n"
            f"- Starts from: `{identity.reference_artifact_name}`\n\n"
            "Format contract: `llm_pretrain_dense_v1`.\n"
        ),
        stage_metadata={
            "model": {
                "id": identity.bundle_model_id,
                "name": identity.artifact_name,
                "stage": "rlvr",
            },
            "provenance": {
                "source": "countdown_lite_grpo",
                "starts_from": identity.reference_artifact_name,
                "checkpoint_step": checkpoint_step,
                "source_manifest": identity.source_manifest,
                **(extra_provenance or {}),
            },
        },
    )
