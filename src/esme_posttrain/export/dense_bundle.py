from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.bundle import BUNDLE_FORMAT, file_sha256, load_dense_backbone_bundle
from esme_posttrain.training.checkpointing import load_training_checkpoint

CHAT_TEMPLATE_ID = "esme_newline_v1"
DEFAULT_CHAT_TEMPLATE = {
    "id": CHAT_TEMPLATE_ID,
    "format": "role_newline_content_newline",
    "roles": {
        "system": "system\n{content}\n",
        "user": "user\n{content}\n",
        "assistant": "assistant\n{content}\n",
    },
    "generation_prompt": "assistant\n",
    "example": "user\n...\nassistant\n",
    "add_special_tokens": False,
}


class ExportError(ValueError):
    pass


@dataclass(frozen=True)
class ExportRequest:
    artifact_dir: Path
    output_dir: Path
    model_id: str
    source_volume: str
    source_path: str
    wandb_run: str
    dpo_step: int
    config_hash: str | None = None
    max_new_tokens: int = 16


def export_dense_bundle(request: ExportRequest) -> dict[str, Any]:
    artifact_dir = request.artifact_dir.expanduser().resolve()
    output_dir = request.output_dir.expanduser().resolve()
    _require_artifact_files(artifact_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_file_atomically(artifact_dir / "tokenizer.json", output_dir / "tokenizer.json")

    checkpoint = load_training_checkpoint(artifact_dir / "best-checkpoint.pt", map_location="cpu")
    (output_dir / "config.json").write_text(
        json.dumps(checkpoint.config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    weights_payload = {
        "format_version": 1,
        "format": BUNDLE_FORMAT,
        "key_format": BUNDLE_FORMAT,
        "state_dict_key": "state_dict",
        "model_config": checkpoint.config.to_dict(),
        "state_dict": checkpoint.model.state_dict(),
    }
    _save_torch_payload_atomically(weights_payload, output_dir / "weights.pt")

    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    eos_token_ids = _eos_token_ids(tokenizer)
    manifest = _manifest(
        request=request,
        artifact_dir=artifact_dir,
        output_dir=output_dir,
        model_config=checkpoint.config.to_dict(),
        checkpoint_step=checkpoint.step,
        eos_token_ids=eos_token_ids,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    loaded = load_dense_backbone_bundle(output_dir)
    smoke = _smoke_payload(
        request=request,
        loaded_model=loaded.model,
        tokenizer=loaded.tokenizer,
        eos_token_ids=eos_token_ids,
        output_dir=output_dir,
        checkpoint_step=checkpoint.step,
    )
    (output_dir / "smoke.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        _readme(request=request, checkpoint_step=checkpoint.step, eos_token_ids=eos_token_ids),
        encoding="utf-8",
    )
    return smoke


def _require_artifact_files(artifact_dir: Path) -> None:
    required = (
        "best-checkpoint.pt",
        "tokenizer.json",
        "manifest.json",
        "eval-report.json",
        "best-checkpoint.json",
        "config.json",
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise ExportError(f"artifact is missing required files: {', '.join(missing)}")


def _manifest(
    *,
    request: ExportRequest,
    artifact_dir: Path,
    output_dir: Path,
    model_config: dict[str, Any],
    checkpoint_step: int,
    eos_token_ids: list[int],
) -> dict[str, Any]:
    source_manifest = _read_json_object(artifact_dir / "manifest.json")
    best_checkpoint = _read_json_object(artifact_dir / "best-checkpoint.json")
    eval_report = _read_json_object(artifact_dir / "eval-report.json")
    config_hash = (
        request.config_hash
        or _hash_from(source_manifest)
        or file_sha256(artifact_dir / "config.json")
    )
    return {
        "schema_version": 1,
        "format": BUNDLE_FORMAT,
        "weights_format": BUNDLE_FORMAT,
        "model_family": "DenseBackbone",
        "model": {
            "id": request.model_id,
            "name": "Esme-214M-Chat",
            "stage": "dpo",
        },
        "model_config": model_config,
        "files": {
            "config": {"path": "config.json", "sha256": file_sha256(output_dir / "config.json")},
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": file_sha256(output_dir / "tokenizer.json"),
            },
            "weights": {"path": "weights.pt", "sha256": file_sha256(output_dir / "weights.pt")},
        },
        "tokenizer": {
            "path": "tokenizer.json",
            "format": "tokenizers-json",
            "add_special_tokens": False,
            "chat_template": DEFAULT_CHAT_TEMPLATE,
        },
        "chat_template": DEFAULT_CHAT_TEMPLATE,
        "eos_token_ids": eos_token_ids,
        "decoding": {
            "eos_token_ids": eos_token_ids,
            "default_add_special_tokens": False,
        },
        "provenance": {
            "source": "modal_volume",
            "modal_volume": request.source_volume,
            "modal_path": request.source_path,
            "checkpoint_file": "best-checkpoint.pt",
            "checkpoint_sha256": file_sha256(artifact_dir / "best-checkpoint.pt"),
            "checkpoint_step": checkpoint_step,
            "dpo_step": request.dpo_step,
            "wandb_run": request.wandb_run,
            "dpo_config_hash": config_hash,
            "source_manifest_sha256": file_sha256(artifact_dir / "manifest.json"),
            "source_config_sha256": file_sha256(artifact_dir / "config.json"),
            "eval_report_sha256": file_sha256(artifact_dir / "eval-report.json"),
            "best_checkpoint_json_sha256": file_sha256(artifact_dir / "best-checkpoint.json"),
        },
        "source_manifest": _selected_source_fields(source_manifest),
        "source_best_checkpoint": _selected_source_fields(best_checkpoint),
        "source_eval_report": _selected_source_fields(eval_report),
    }


def _smoke_payload(
    *,
    request: ExportRequest,
    loaded_model: Any,
    tokenizer: Tokenizer,
    eos_token_ids: list[int],
    output_dir: Path,
    checkpoint_step: int,
) -> dict[str, Any]:
    prompt = "user\nSay hello in one short sentence.\nassistant\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise ExportError("smoke prompt produced zero tokens")
    eos_id = eos_token_ids[0] if eos_token_ids else None
    generated = loaded_model.generate(
        torch.tensor([prompt_ids], dtype=torch.long),
        request.max_new_tokens,
        eos_token_id=eos_id,
    )
    generated_ids = [int(token_id) for token_id in generated[0, len(prompt_ids) :].tolist()]
    hit_eos = bool(generated_ids and generated_ids[-1] in eos_token_ids)
    text_ids = generated_ids[:-1] if hit_eos else generated_ids
    text = tokenizer.decode(text_ids, skip_special_tokens=False)
    logits = loaded_model(torch.tensor([prompt_ids], dtype=torch.long))
    if not torch.isfinite(logits).all():
        raise ExportError("smoke logits contain non-finite values")
    return {
        "status": "ok",
        "model_id": request.model_id,
        "checkpoint_step": checkpoint_step,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "generated_ids": generated_ids,
        "generated_text": text,
        "finish_reason": "eos" if hit_eos else "length",
        "logits_shape": list(logits.shape),
        "logits_last_finite": True,
        "files": {
            name: file_sha256(output_dir / name)
            for name in ("config.json", "tokenizer.json", "weights.pt", "manifest.json")
        },
    }


def _readme(*, request: ExportRequest, checkpoint_step: int, eos_token_ids: list[int]) -> str:
    return (
        "# Esme-214M-Chat DenseBackbone Bundle\n\n"
        f"- Model id: `{request.model_id}`\n"
        f"- Source: Modal Volume `{request.source_volume}` path `{request.source_path}`\n"
        f"- W&B run: `{request.wandb_run}`\n"
        f"- DPO step: `{request.dpo_step}`; checkpoint step: `{checkpoint_step}`\n"
        f"- EOS token ids: `{eos_token_ids}`\n"
        "- Chat template: `user\\n...\\nassistant\\n`, encoded with "
        "`add_special_tokens=false`.\n\n"
        'Load through `llm_infer.model.runtime.load_model_runtime("dense", bundle_path=...)`.\n'
    )


def _eos_token_ids(tokenizer: Tokenizer) -> list[int]:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ExportError("tokenizer is missing required <eos> token")
    return [int(eos_id)]


def _copy_file_atomically(source: Path, target: Path) -> None:
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp_path)
    tmp_path.replace(target)


def _save_torch_payload_atomically(payload: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"{path.name} is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ExportError(f"{path.name} must contain a JSON object")
    return payload


def _hash_from(payload: dict[str, Any]) -> str | None:
    for key in ("dpo_config_hash", "config_hash", "run_config_hash"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _selected_source_fields(payload: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for key in (
        "run_id",
        "artifact_name",
        "stage",
        "step",
        "best_step",
        "selected_step",
        "metrics",
        "acceptance",
        "status",
    ):
        if key in payload:
            selected[key] = payload[key]
    return selected
