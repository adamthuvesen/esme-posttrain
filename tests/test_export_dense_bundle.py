from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from esme_posttrain.bundle import (
    BUNDLE_FORMAT,
    BundleError,
    canonical_model_config,
    file_sha256,
    validate_base_bundle,
)
from esme_posttrain.chat.data import _render_chat_segments, tokenize_multi_turn
from esme_posttrain.chat.data_types import ChatTurn, MultiTurnExample
from esme_posttrain.chat.fixtures import tiny_backbone_config, tiny_chat_tokenizer
from esme_posttrain.export.chat_template import CHAT_TEMPLATE_ID, ChatTemplateError
from esme_posttrain.export.chat_template_upgrade import V1_CHAT_TEMPLATE, upgrade_chat_template
from esme_posttrain.export.dense_bundle import (
    ExportError,
    ExportRequest,
    export_dense_bundle,
)
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.training.checkpointing import save_training_checkpoint


def test_export_dense_bundle_writes_validator_shape_and_chat_metadata(tmp_path: Path) -> None:
    artifact_dir = _write_tiny_dpo_artifact(tmp_path / "artifact")
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(json.loads(tokenizer_path.read_text(encoding="utf-8"))) + "\n",
        encoding="utf-8",
    )
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
    assert (output_dir / "tokenizer.json").read_bytes() == tokenizer_path.read_bytes()
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

    with pytest.raises(ExportError, match=r"eval-report.json"):
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


def test_export_dense_bundle_requires_eos_token(tmp_path: Path) -> None:
    artifact_dir = _write_tiny_dpo_artifact(tmp_path / "artifact")
    tokenizer_path = artifact_dir / "tokenizer.json"
    tokenizer_path.write_text(
        tokenizer_path.read_text(encoding="utf-8").replace('"<eos>"', '"<end>"'),
        encoding="utf-8",
    )

    with pytest.raises(ExportError, match="missing required <eos> token"):
        export_dense_bundle(
            ExportRequest(
                artifact_dir=artifact_dir,
                output_dir=tmp_path / "bundle",
                model_id="esme-214m-chat",
                source_volume="local-cpu",
                source_path="dpo-artifact",
                wandb_run="disabled",
                dpo_step=600,
            )
        )


def test_exported_chat_template_renders_the_training_prompt_the_way_llm_infer_does(
    tmp_path: Path,
) -> None:
    manifest = _export_tiny_bundle_manifest(tmp_path)
    template, add_special_tokens = _llm_infer_tokenizer_metadata(manifest)
    example = MultiTurnExample(
        turns=(
            ChatTurn("system", "helpful"),
            ChatTurn("user", "say red"),
            ChatTurn("assistant", "red"),
        ),
        source="fixture",
        row_id="parity-1",
    )
    messages = [{"role": "system", "content": "helpful"}, {"role": "user", "content": "say red"}]

    served_prompt = _llm_infer_render(template, messages, add_generation_prompt=True)

    assert served_prompt == _training_text_before_last_reply(example)
    tokenizer = tiny_chat_tokenizer()
    trained = tokenize_multi_turn(tokenizer, example, max_sequence_tokens=64)
    served_ids = tokenizer.encode(served_prompt, add_special_tokens=add_special_tokens).ids
    assert tuple(served_ids) == trained.input_ids[: trained.prompt_tokens]


def test_served_multi_turn_history_tokenizes_exactly_like_training(tmp_path: Path) -> None:
    manifest = _export_tiny_bundle_manifest(tmp_path)
    template, add_special_tokens = _llm_infer_tokenizer_metadata(manifest)
    example = MultiTurnExample(
        turns=(
            ChatTurn("system", "helpful"),
            ChatTurn("user", "say red"),
            ChatTurn("assistant", "red"),
            ChatTurn("user", "say blue again"),
            ChatTurn("assistant", "blue"),
        ),
        source="fixture",
        row_id="parity-2",
    )
    messages = [{"role": turn.role, "content": turn.content} for turn in example.turns[:-1]]
    tokenizer = tiny_chat_tokenizer()

    served_prompt = _llm_infer_render(template, messages, add_generation_prompt=True)
    served_ids = tuple(tokenizer.encode(served_prompt, add_special_tokens=add_special_tokens).ids)

    trained = tokenize_multi_turn(tokenizer, example, max_sequence_tokens=64)
    last_reply_ids = (*tokenizer.encode("blue", add_special_tokens=False).ids, 1)
    assert trained.input_ids[-len(last_reply_ids) :] == last_reply_ids
    assert served_ids == trained.input_ids[: -len(last_reply_ids)]
    assert served_ids.count(1) == 1


def test_export_refuses_a_tokenizer_that_splits_the_eos_text(tmp_path: Path) -> None:
    artifact_dir = _write_tiny_dpo_artifact(tmp_path / "artifact")
    _unregister_eos(artifact_dir / "tokenizer.json")
    output_dir = tmp_path / "bundle"

    with pytest.raises(ExportError, match=r"Register <eos> as a special token"):
        export_dense_bundle(
            ExportRequest(
                artifact_dir=artifact_dir,
                output_dir=output_dir,
                model_id="esme-214m-chat",
                source_volume="local-cpu",
                source_path="dpo-artifact",
                wandb_run="disabled",
                dpo_step=600,
            )
        )
    assert not output_dir.exists()


def test_upgrade_rewrites_a_v1_bundle_that_still_loads_the_way_llm_infer_does(
    tmp_path: Path,
) -> None:
    bundle_dir = _export_tiny_bundle(tmp_path)
    _edit_bundle_manifest(bundle_dir, chat_template=V1_CHAT_TEMPLATE)
    hashed_before = {name: file_sha256(bundle_dir / name) for name in _HASHED_FILES}
    v1_text = (bundle_dir / "manifest.json").read_text(encoding="utf-8")

    preview = upgrade_chat_template(bundle_dir, dry_run=True)

    assert preview["status"] == "dry_run"
    assert '-      "assistant": "assistant\\n{content}\\n",' in preview["diff"]
    assert '+      "assistant": "assistant\\n{content}<eos>",' in preview["diff"]
    assert (bundle_dir / "manifest.json").read_text(encoding="utf-8") == v1_text

    result = upgrade_chat_template(bundle_dir, dry_run=False)

    assert result["status"] == "upgraded"
    assert result["diff"] == preview["diff"]
    assert {name: file_sha256(bundle_dir / name) for name in _HASHED_FILES} == hashed_before
    assert sorted(path.name for path in bundle_dir.iterdir()) == sorted(
        (*_HASHED_FILES, "manifest.json")
    )
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    _llm_infer_verify_file_hashes(bundle_dir, manifest)
    template, add_special_tokens = _llm_infer_tokenizer_metadata(manifest)
    assert template["id"] == CHAT_TEMPLATE_ID
    messages = [
        {"role": "user", "content": "say red"},
        {"role": "assistant", "content": "red"},
        {"role": "user", "content": "say blue"},
    ]
    served = _llm_infer_render(template, messages, add_generation_prompt=True)
    tokenizer = tiny_chat_tokenizer()
    assert tokenizer.encode(served, add_special_tokens=add_special_tokens).ids == [
        2, 5, 6, 3, 6, 1, 2, 5, 7, 3,
    ]  # fmt: skip

    again = upgrade_chat_template(bundle_dir, dry_run=False)

    assert again["status"] == "already_current"
    assert again["diff"] == ""
    assert json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_upgrade_refuses_a_bundle_without_a_chat_template(tmp_path: Path) -> None:
    bundle_dir = _export_tiny_bundle(tmp_path)
    _edit_bundle_manifest(bundle_dir, chat_template=None)

    with pytest.raises(ChatTemplateError, match="has no chat_template"):
        upgrade_chat_template(bundle_dir, dry_run=False)


def test_upgrade_refuses_a_template_that_is_not_v1(tmp_path: Path) -> None:
    bundle_dir = _export_tiny_bundle(tmp_path)
    _edit_bundle_manifest(bundle_dir, chat_template={**V1_CHAT_TEMPLATE, "id": "custom"})

    with pytest.raises(ChatTemplateError, match="not the exact esme_newline_v1 object"):
        upgrade_chat_template(bundle_dir, dry_run=False)


def test_upgrade_refuses_a_tokenizer_that_splits_the_eos_text(tmp_path: Path) -> None:
    bundle_dir = _export_tiny_bundle(tmp_path)
    _unregister_eos(bundle_dir / "tokenizer.json")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["tokenizer"]["sha256"] = file_sha256(bundle_dir / "tokenizer.json")
    manifest["chat_template"] = V1_CHAT_TEMPLATE
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    before = (bundle_dir / "manifest.json").read_bytes()

    with pytest.raises(ChatTemplateError, match=r"encodes the text '<eos>' as \[0, 0, 0\]"):
        upgrade_chat_template(bundle_dir, dry_run=False)
    assert (bundle_dir / "manifest.json").read_bytes() == before


def test_upgrade_rewrites_a_bundle_exported_before_the_readme_contract(tmp_path: Path) -> None:
    bundle_dir = _export_tiny_bundle(tmp_path)
    _edit_bundle_manifest(bundle_dir, chat_template=V1_CHAT_TEMPLATE)
    _to_pre_contract_shape(bundle_dir)

    preview = upgrade_chat_template(bundle_dir, dry_run=True)
    result = upgrade_chat_template(bundle_dir, dry_run=False)

    assert preview["status"] == "dry_run"
    assert result["status"] == "upgraded"
    assert result["diff"] == preview["diff"]
    assert sorted(path.name for path in bundle_dir.iterdir()) == [
        "config.json", "manifest.json", "tokenizer.json", "weights.pt",
    ]  # fmt: skip
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    _llm_infer_verify_file_hashes(bundle_dir, manifest)
    assert manifest["chat_template"]["id"] == CHAT_TEMPLATE_ID
    assert manifest["tokenizer"]["chat_template"] == manifest["chat_template"]
    assert upgrade_chat_template(bundle_dir, dry_run=False)["status"] == "already_current"


def test_upgrade_refuses_a_pre_contract_bundle_with_a_hash_mismatch(tmp_path: Path) -> None:
    bundle_dir = _export_tiny_bundle(tmp_path)
    _edit_bundle_manifest(bundle_dir, chat_template=V1_CHAT_TEMPLATE)
    _to_pre_contract_shape(bundle_dir)
    with (bundle_dir / "weights.pt").open("ab") as handle:
        handle.write(b"tampered")
    before = (bundle_dir / "manifest.json").read_bytes()

    with pytest.raises(BundleError, match="hash mismatch for weights"):
        upgrade_chat_template(bundle_dir, dry_run=False)
    assert (bundle_dir / "manifest.json").read_bytes() == before


_HASHED_FILES = ("config.json", "tokenizer.json", "weights.pt", "README.md")


def _export_tiny_bundle(tmp_path: Path) -> Path:
    _export_tiny_bundle_manifest(tmp_path)
    return tmp_path / "bundle"


def _edit_bundle_manifest(bundle_dir: Path, *, chat_template: dict[str, Any] | None) -> None:
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if chat_template is None:
        del manifest["chat_template"]
    else:
        manifest["chat_template"] = chat_template
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _to_pre_contract_shape(bundle_dir: Path) -> None:
    """Match exports made before the canonical v1 contract: no README, template nested too."""
    (bundle_dir / "README.md").unlink()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"]["readme"]
    manifest["tokenizer"]["chat_template"] = V1_CHAT_TEMPLATE
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _unregister_eos(tokenizer_path: Path) -> None:
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    payload["added_tokens"] = []
    tokenizer_path.write_text(json.dumps(payload), encoding="utf-8")


def _export_tiny_bundle_manifest(tmp_path: Path) -> dict[str, Any]:
    output_dir = tmp_path / "bundle"
    export_dense_bundle(
        ExportRequest(
            artifact_dir=_write_tiny_dpo_artifact(tmp_path / "artifact"),
            output_dir=output_dir,
            model_id="esme-214m-chat",
            source_volume="local-cpu",
            source_path="dpo-artifact",
            wandb_run="disabled",
            dpo_step=600,
            max_new_tokens=1,
        )
    )
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def _training_text_before_last_reply(example: MultiTurnExample) -> str:
    segments = _render_chat_segments(example)
    last_reply = max(index for index, (_, supervised) in enumerate(segments) if supervised)
    return "".join(
        text + "<eos>" if supervised else text for text, supervised in segments[:last_reply]
    )


# The two helpers below reproduce llm-infer's dense-bundle loader
# (llm_infer/model/runtime.py: _dense_tokenizer_metadata, _render_bundle_chat_template).
# Sibling repos exchange artifacts, never imports, so keep them in step by hand.
def _llm_infer_tokenizer_metadata(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    tokenizer_config = manifest.get("tokenizer")
    if not isinstance(tokenizer_config, dict):
        tokenizer_config = {}
    template = next(
        value
        for value in (manifest.get("chat_template"), tokenizer_config.get("chat_template"))
        if value is not None
    )
    assert isinstance(template, dict)
    for value in (
        tokenizer_config.get("add_special_tokens"),
        manifest.get("add_special_tokens"),
        template.get("add_special_tokens"),
    ):
        if value is not None:
            assert isinstance(value, bool)
            return template, value
    return template, True


def _llm_infer_render(
    template: Mapping[str, Any],
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    roles = template["roles"]
    parts = [roles[message["role"]].format(content=message["content"]) for message in messages]
    if add_generation_prompt:
        parts.append(template["generation_prompt"])
    return "".join(parts)


# Reproduces llm-infer's pretrain_bundle_loader.resolve_bundle_file hash check.
def _llm_infer_verify_file_hashes(bundle_dir: Path, manifest: Mapping[str, Any]) -> None:
    for name in ("config", "tokenizer", "weights"):
        entry = manifest["files"][name]
        path = (bundle_dir / entry["path"]).resolve()
        assert path.is_relative_to(bundle_dir.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"].lower()


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
