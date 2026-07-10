from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from esme_posttrain.bundle import BUNDLE_FORMAT, canonical_model_config, validate_base_bundle
from esme_posttrain.export.dense_bundle import (
    CHAT_TEMPLATE_ID,
    ExportError,
    ExportRequest,
    export_dense_bundle,
)
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.smoke_multiturn import tiny_backbone_config, tiny_chat_tokenizer
from esme_posttrain.training.checkpointing import save_training_checkpoint


def test_export_dense_bundle_writes_validator_shape_and_chat_metadata(tmp_path: Path) -> None:
    artifact_dir = _write_tiny_dpo_artifact(tmp_path / "artifact")
    output_dir = tmp_path / "bundle"

    smoke = export_dense_bundle(
        ExportRequest(
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            model_id="esme-214m-chat",
            source_volume="esme-posttrain-esme-chat-dpo",
            source_path="esme-214m-chat-dpo-full",
            wandb_run="pgt1zlpq",
            dpo_step=600,
            config_hash="cfg-hash",
            max_new_tokens=3,
        )
    )

    validate_base_bundle(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = torch.load(output_dir / "weights.pt", weights_only=True)

    assert manifest["schema_version"] == 1
    assert manifest["format"] == BUNDLE_FORMAT
    assert manifest["model"]["id"] == "esme-214m-chat"
    assert manifest["model"]["stage"] == "dpo"
    provenance = manifest["provenance"]
    assert provenance["modal_volume"] == "esme-posttrain-esme-chat-dpo"
    assert provenance["wandb_run"] == "pgt1zlpq"
    assert provenance["dpo_step"] == 600
    assert provenance["dpo_config_hash"] == "cfg-hash"
    assert manifest["decoding"]["eos_token_ids"] == [1]
    chat_template = manifest["chat_template"]
    assert chat_template["id"] == CHAT_TEMPLATE_ID
    assert chat_template["example"] == "user\n...\nassistant\n"
    assert chat_template["add_special_tokens"] is False
    assert weights["format"] == BUNDLE_FORMAT
    assert weights["key_format"] == BUNDLE_FORMAT
    assert weights["model_config"] == canonical_model_config(tiny_backbone_config())
    assert smoke["status"] == "ok"
    assert smoke["finish_reason"] in {"eos", "length"}
    assert (
        (output_dir / "README.md")
        .read_text(encoding="utf-8")
        .startswith("# Esme-214M-Chat DenseBackbone Bundle")
    )


def test_export_dense_bundle_requires_completed_artifact_files(tmp_path: Path) -> None:
    artifact_dir = _write_tiny_dpo_artifact(tmp_path / "artifact")
    (artifact_dir / "eval-report.json").unlink()

    with pytest.raises(ExportError, match="eval-report.json"):
        export_dense_bundle(
            ExportRequest(
                artifact_dir=artifact_dir,
                output_dir=tmp_path / "bundle",
                model_id="esme-214m-chat",
                source_volume="esme-posttrain-esme-chat-dpo",
                source_path="esme-214m-chat-dpo-full",
                wandb_run="pgt1zlpq",
                dpo_step=600,
            )
        )


def _write_tiny_dpo_artifact(artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True)
    config = tiny_backbone_config()
    model = DenseBackbone(config)
    tokenizer = tiny_chat_tokenizer()
    (artifact_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    tokenizer.save(str(artifact_dir / "tokenizer.json"))
    save_training_checkpoint(
        artifact_dir / "best-checkpoint.pt",
        model=model,
        step=600,
        metrics={"eval/preference_accuracy": 0.625},
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "esme_214m_chat_dpo",
                "artifact_name": "Esme-214M-Chat",
                "stage": "dpo",
                "dpo_config_hash": "source-hash",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "eval-report.json").write_text(
        json.dumps({"status": "accepted", "metrics": {"preference_accuracy": 0.625}}),
        encoding="utf-8",
    )
    (artifact_dir / "best-checkpoint.json").write_text(
        json.dumps({"selected_step": 600, "metrics": {"eval/preference_accuracy": 0.625}}),
        encoding="utf-8",
    )
    return artifact_dir
