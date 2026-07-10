from __future__ import annotations

import argparse
from pathlib import Path

from esme_posttrain.cli.output import emit_payload, exit_with_error
from esme_posttrain.studies.report import StudyReportError, generate_study_report


def add_study_parsers(subparsers: argparse._SubParsersAction) -> None:
    study_report = subparsers.add_parser(
        "study-report",
        help="Generate checked JSON and Markdown from a hashed study specification.",
    )
    study_report.add_argument("--study", type=Path, required=True)
    study_report.add_argument("--output-dir", type=Path)
    study_report.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Allowed root for artifact paths in the study specification.",
    )
    study_report.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    study_report.set_defaults(handler=_handle_study_report)


def _handle_study_report(args: argparse.Namespace) -> int:
    try:
        generated = generate_study_report(
            args.study,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
        )
    except StudyReportError as exc:
        exit_with_error(str(exc))
    payload = {
        **generated.payload,
        "json_path": str(generated.json_path),
        "markdown_path": str(generated.markdown_path),
    }
    emit_payload(payload, json_output=args.json, formatter=_format_study_report)
    return 0


def _format_study_report(payload: dict[str, object]) -> str:
    return "\n".join(
        (
            f"verdict: {payload.get('verdict')}",
            f"complete: {payload.get('complete')}",
            f"compatible: {payload.get('compatible')}",
            f"json_path: {payload.get('json_path')}",
            f"markdown_path: {payload.get('markdown_path')}",
        )
    )
