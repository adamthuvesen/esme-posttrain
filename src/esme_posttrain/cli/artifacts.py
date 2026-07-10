from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.bundle import BundleError, validate_bundle_contract
from esme_posttrain.cli.output import emit_payload, exit_with_error, format_stage_payload


def add_artifact_parsers(subparsers: argparse._SubParsersAction) -> None:
    bundle_check = subparsers.add_parser(
        "bundle-check",
        help="Check bundle versions, hashes, and weights metadata without creating a model.",
    )
    bundle_check.add_argument("--bundle-dir", type=Path, required=True)
    bundle_check.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    bundle_check.set_defaults(handler=_handle_bundle_check)


def _handle_bundle_check(args: argparse.Namespace) -> int:
    try:
        bundle = validate_bundle_contract(args.bundle_dir)
    except BundleError as exc:
        exit_with_error(str(exc))
    payload = {
        "status": "ok",
        "bundle_dir": str(bundle.bundle_dir),
        "manifest_path": str(bundle.manifest_path),
        "schema_version": bundle.manifest["schema_version"],
        "format": bundle.manifest["format"],
        "files": {
            "config": str(bundle.config_path),
            "readme": str(bundle.bundle_dir / "README.md"),
            "tokenizer": str(bundle.tokenizer_path),
            "weights": str(bundle.weights_path),
        },
    }
    emit_payload(payload, json_output=args.json, formatter=format_stage_payload)
    return 0
