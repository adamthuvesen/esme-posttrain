from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import emit_payload, exit_with_error, format_stage_payload
from esme_posttrain.export.dense_bundle import ExportRequest, export_dense_bundle


def add_export_parsers(subparsers: argparse._SubParsersAction) -> None:
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
