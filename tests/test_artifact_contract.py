"""Producer and consumer checks for the canonical dense-bundle v1 contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from esme_posttrain.bundle import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    file_sha256,
    load_dense_backbone_bundle,
    validate_bundle_contract,
)
from esme_posttrain.export.dense_bundle import ExportRequest, export_dense_bundle
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.smoke_multiturn import tiny_backbone_config, tiny_chat_tokenizer
from esme_posttrain.training.checkpointing import save_training_checkpoint

EXPECTED_BUNDLE_FILES = {
    "README.md",
    "config.json",
    "manifest.json",
    "tokenizer.json",
    "weights.pt",
}
EXPECTED_CONFIG_KEYS = {
    "attention_kind",
    "context_length",
    "embedding_dim",
    "feedforward_dim",
    "heads",
    "kv_heads",
    "layers",
    "name",
    "qk_norm",
    "rms_norm_eps",
    "rope_theta",
    "tie_embeddings",
    "vocab_size",
    "z_loss_weight",
}
EXPECTED_STATE_DICT_KEYS = {
    "token_embedding.weight",
    "lm_head.weight",
    "final_norm.weight",
} | {
    f"blocks.0.{suffix}"
    for suffix in (
        "attention_norm.weight",
        "attention.wq.weight",
        "attention.wk.weight",
        "attention.wv.weight",
        "attention.wo.weight",
        "attention.q_norm.weight",
        "attention.k_norm.weight",
        "feedforward_norm.weight",
        "feedforward.w_gate.weight",
        "feedforward.w_up.weight",
        "feedforward.w_down.weight",
    )
}


@pytest.fixture()
def exported_chat_bundle(tmp_path: Path) -> Path:
    artifact_dir = _write_tiny_dpo_artifact(tmp_path / "artifact")
    output_dir = tmp_path / "bundle"
    export_dense_bundle(
        ExportRequest(
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            model_id="contract-chat",
            source_volume="local-cpu",
            source_path="contract-dpo",
            wandb_run="disabled",
            dpo_step=2,
            config_hash="contract-config",
            max_new_tokens=2,
        )
    )
    return output_dir


def test_chat_bundle_file_set_is_pinned(exported_chat_bundle: Path) -> None:
    assert {path.name for path in exported_chat_bundle.iterdir()} == EXPECTED_BUNDLE_FILES


def test_chat_manifest_matches_canonical_v1(exported_chat_bundle: Path) -> None:
    manifest = _read_json(exported_chat_bundle / "manifest.json")

    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION == 1
    assert manifest["format"] == BUNDLE_FORMAT == "llm_pretrain_dense_v1"
    assert manifest["weights_format"] == BUNDLE_FORMAT
    assert manifest["model_family"] == "DenseBackbone"
    assert manifest["tokenizer"] == {
        "path": "tokenizer.json",
        "format": "tokenizers-json",
    }
    assert manifest["checkpoint_step"] == 2
    assert manifest["source_checkpoint"] == "contract-dpo/best-checkpoint.pt"
    assert len(manifest["source_checkpoint_sha256"]) == 64
    assert manifest["model_config"] == _read_json(exported_chat_bundle / "config.json")
    assert set(manifest["files"]) == {"config", "tokenizer", "weights", "readme"}
    for entry in manifest["files"].values():
        assert file_sha256(exported_chat_bundle / entry["path"]) == entry["sha256"]


def test_chat_config_and_weights_keys_are_pinned(exported_chat_bundle: Path) -> None:
    config = _read_json(exported_chat_bundle / "config.json")
    weights = torch.load(exported_chat_bundle / "weights.pt", weights_only=False)

    assert set(config) == EXPECTED_CONFIG_KEYS
    assert set(weights) == {
        "checkpoint_step",
        "format",
        "format_version",
        "key_format",
        "llm_infer_config",
        "metadata",
        "model_config",
        "source_checkpoint",
        "source_checkpoint_sha256",
        "state_dict",
        "state_dict_key",
    }
    assert weights["format_version"] == BUNDLE_SCHEMA_VERSION
    assert weights["format"] == weights["key_format"] == BUNDLE_FORMAT
    assert weights["metadata"]["key_format"] == BUNDLE_FORMAT
    assert weights["model_config"] == config
    assert set(weights["state_dict"]) == EXPECTED_STATE_DICT_KEYS


def test_future_manifest_version_fails_before_tokenizer_or_weights_load(
    exported_chat_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = exported_chat_bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "esme_posttrain.bundle.Tokenizer.from_file",
        lambda _path: pytest.fail("tokenizer loaded before manifest check"),
    )
    monkeypatch.setattr(
        "esme_posttrain.bundle.torch.load",
        lambda *_args, **_kwargs: pytest.fail("weights loaded before manifest check"),
    )

    with pytest.raises(BundleError, match="unsupported bundle schema_version"):
        load_dense_backbone_bundle(exported_chat_bundle)


def test_future_weights_version_is_rejected(exported_chat_bundle: Path) -> None:
    weights_path = exported_chat_bundle / "weights.pt"
    weights = torch.load(weights_path, weights_only=False)
    weights["format_version"] = 2
    torch.save(weights, weights_path)
    manifest_path = exported_chat_bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["files"]["weights"]["sha256"] = file_sha256(weights_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleError, match="unsupported weights.pt format_version"):
        load_dense_backbone_bundle(exported_chat_bundle)


def test_producer_contract_rejects_incomplete_v1_bundle(exported_chat_bundle: Path) -> None:
    (exported_chat_bundle / "README.md").unlink()

    with pytest.raises(BundleError, match="file set must be exactly"):
        validate_bundle_contract(exported_chat_bundle)


def test_contract_check_uses_safe_tensor_loading(
    exported_chat_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_load = torch.load
    calls: list[bool | None] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("weights_only"))
        return torch_load(*args, **kwargs)

    monkeypatch.setattr("esme_posttrain.bundle.torch.load", recording_load)

    validate_bundle_contract(exported_chat_bundle)

    assert calls == [True]


def _write_tiny_dpo_artifact(artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True)
    config = tiny_backbone_config()
    model = DenseBackbone(config)
    tokenizer = tiny_chat_tokenizer()
    (artifact_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    tokenizer.save(str(artifact_dir / "tokenizer.json"))
    save_training_checkpoint(
        artifact_dir / "best-checkpoint.pt",
        model=model,
        step=2,
        metrics={"eval/preference_accuracy": 0.5},
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "contract-dpo",
                "artifact_name": "Contract Chat",
                "stage": "dpo",
                "dpo_config_hash": "contract-config",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "eval-report.json").write_text(
        json.dumps({"status": "accepted"}), encoding="utf-8"
    )
    (artifact_dir / "best-checkpoint.json").write_text(
        json.dumps({"selected_step": 2}), encoding="utf-8"
    )
    return artifact_dir


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
