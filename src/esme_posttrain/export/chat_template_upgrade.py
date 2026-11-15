"""Upgrade an exported chat bundle's `chat_template` from esme_newline_v1 to v2.

Only `manifest.json` changes. The manifest's `files.*.sha256` entries cover the
other bundle files, never the manifest itself, so the bundle still verifies
after the upgrade. Bundles exported before the canonical v1 contract (no
README.md, no `readme` files entry) are accepted: the upgrade checks only the
hashes of the files the manifest lists, before and after the rewrite.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from esme_posttrain.bundle import validate_listed_files
from esme_posttrain.export.chat_template import (
    CHAT_TEMPLATE_ID,
    ChatTemplateError,
    chat_template_manifest,
    require_eos_text_is_eos_id,
)

V1_CHAT_TEMPLATE: dict[str, Any] = {
    "id": "esme_newline_v1",
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


def upgrade_chat_template(bundle_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    """Rewrite a v1 bundle's `chat_template` to the current template.

    Refuses a bundle with no `chat_template` (RL bundles omit it), a template
    that is not the exact v1 object, or a tokenizer that fails the `<eos>` guard.
    """
    bundle = validate_listed_files(bundle_dir)
    template = bundle.manifest.get("chat_template")
    if template is None:
        raise ChatTemplateError(
            f"{bundle.manifest_path} has no chat_template; only chat bundles carry one "
            "(RL bundles omit it on purpose), so there is nothing to upgrade"
        )
    if not isinstance(template, dict):
        raise ChatTemplateError(f"{bundle.manifest_path} chat_template must be an object")
    template_id = template.get("id")
    if template_id == CHAT_TEMPLATE_ID:
        return _payload(bundle.manifest_path, "already_current", template_id, diff="")
    if template != V1_CHAT_TEMPLATE:
        raise ChatTemplateError(
            f"{bundle.manifest_path} chat_template is not the exact esme_newline_v1 object "
            f"(id {template_id!r}); upgrade only rewrites a v1 template, so re-export this bundle"
        )
    require_eos_text_is_eos_id(Tokenizer.from_file(str(bundle.tokenizer_path)))

    old_text = bundle.manifest_path.read_text(encoding="utf-8")
    new_manifest = {**bundle.manifest, "chat_template": chat_template_manifest()}
    tokenizer_section = bundle.manifest.get("tokenizer")
    if isinstance(tokenizer_section, dict) and tokenizer_section.get("chat_template") is not None:
        # Pre-contract exports also nest the template under `tokenizer`; leaving a v1 copy
        # there would make the manifest carry two contradicting templates.
        if tokenizer_section["chat_template"] != V1_CHAT_TEMPLATE:
            raise ChatTemplateError(
                f"{bundle.manifest_path} tokenizer.chat_template is not the exact "
                "esme_newline_v1 object; upgrade only rewrites a v1 template, so re-export this bundle"
            )
        new_manifest["tokenizer"] = {**tokenizer_section, "chat_template": chat_template_manifest()}
    new_text = json.dumps(new_manifest, indent=2, sort_keys=True) + "\n"
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"{bundle.manifest_path} (v1)",
            tofile=f"{bundle.manifest_path} ({CHAT_TEMPLATE_ID})",
        )
    )
    if dry_run:
        return _payload(bundle.manifest_path, "dry_run", template_id, diff=diff)
    _write_atomically(bundle.manifest_path, new_text)
    validate_listed_files(bundle.bundle_dir)
    return _payload(bundle.manifest_path, "upgraded", template_id, diff=diff)


def _payload(manifest_path: Path, status: str, from_id: object, *, diff: str) -> dict[str, Any]:
    return {
        "status": status,
        "manifest_path": str(manifest_path),
        "from_template_id": from_id,
        "to_template_id": CHAT_TEMPLATE_ID,
        "diff": diff,
    }


def _write_atomically(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
