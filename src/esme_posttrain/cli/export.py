from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import (
    Payload,
    Subparsers,
    emit_payload,
    exit_with_error,
    format_stage_payload,
)
from esme_posttrain.export.chat_template_upgrade import upgrade_chat_template
from esme_posttrain.export.dense_bundle import ExportRequest, export_dense_bundle


def add_export_parsers(subparsers: Subparsers) -> None:
    export_bundle = subparsers.add_parser(
        "export-dense-bundle",
        help="Export a posttrain DenseBackbone checkpoint as an llm-infer bundle.",
    )
    export_bundle.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("runs/esme-214m-chat-dpo/esme-214m-chat-dpo-full"),
        help="Local mirror of the completed DPO artifact.",
    )
    export_bundle.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/esme-214m-chat"),
        help="Bundle directory to write.",
    )
    export_bundle.add_argument("--model-id", default="esme-214m-chat")
    export_bundle.add_argument("--source-volume", default="esme-posttrain-esme-chat-dpo")
    export_bundle.add_argument("--source-path", default="esme-214m-chat-dpo-full")
    export_bundle.add_argument(
        "--wandb-run",
        required=True,
        help="W&B run id of the training run being exported (stamped into provenance).",
    )
    export_bundle.add_argument(
        "--dpo-step",
        type=int,
        required=True,
        help="DPO step of the exported checkpoint (stamped into provenance).",
    )
    export_bundle.add_argument("--config-hash")
    export_bundle.add_argument("--max-new-tokens", type=int, default=16)
    export_bundle.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    export_bundle.set_defaults(handler=_handle_export_dense_bundle)

    upgrade = subparsers.add_parser(
        "upgrade-chat-template",
        help="Rewrite an exported chat bundle's chat_template from esme_newline_v1 to v2.",
    )
    upgrade.add_argument("--bundle-dir", type=Path, required=True)
    upgrade.add_argument(
        "--dry-run", action="store_true", help="Print the manifest diff and write nothing."
    )
    upgrade.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    upgrade.set_defaults(handler=_handle_upgrade_chat_template)


def _handle_export_dense_bundle(args: argparse.Namespace) -> int:
    try:
        payload = export_dense_bundle(
            ExportRequest(
                artifact_dir=args.artifact_dir,
                output_dir=args.output_dir,
                model_id=args.model_id,
                source_volume=args.source_volume,
                source_path=args.source_path,
                wandb_run=args.wandb_run,
                dpo_step=args.dpo_step,
                config_hash=args.config_hash,
                max_new_tokens=args.max_new_tokens,
            )
        )
    except ValueError as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=format_stage_payload)
    return 0


def _handle_upgrade_chat_template(args: argparse.Namespace) -> int:
    try:
        payload = upgrade_chat_template(args.bundle_dir, dry_run=args.dry_run)
    except ValueError as exc:
        exit_with_error(str(exc))
    emit_payload(payload, json_output=args.json, formatter=_format_upgrade_payload)
    return 0


def _format_upgrade_payload(payload: Payload) -> str:
    lines = [
        f"status: {payload['status']}",
        f"manifest: {payload['manifest_path']}",
        f"chat_template: {payload['from_template_id']} -> {payload['to_template_id']}",
    ]
    diff = payload["diff"]
    return "\n".join(lines) + (f"\n{diff}" if diff else "")
