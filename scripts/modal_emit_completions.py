#!/usr/bin/env python3
# ruff: noqa: E402
"""Rebuild GRPO checkpoints as bundles and sample CompletionSets on a remote GPU.

Each job names a training checkpoint and template bundle on the RLVR volume (or a
ready bundle baked into the image), a held-out split, and decoding settings. The
remote function converts the checkpoint with ``export_checkpoint_bundle``, samples
it with ``emit_completion_set``, and returns the small CompletionSet files, which
are written locally under ``--out-root``. Checkpoints never leave the volume.

Usage:
    uv run --with modal==1.5.1 modal run scripts/modal_emit_completions.py \\
        --jobs <jobs.json> --out-root runs/<dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from esme_posttrain.launch.config_guards import IMAGE_PACKAGE_PINS
from esme_posttrain.launch.modal_cli import local_git_commit, local_git_dirty
from esme_posttrain.rl.launch import MODAL_VOLUME

VOLUME_MOUNT = Path("/posttrain")
# The account shares a small GPU quota with other work; keep this runner narrow.
MAX_CONTAINERS = 3
JOB_KEYS = {"label", "split", "n", "temperature", "max_new_tokens", "seed"}

try:
    import modal
except ImportError:  # pragma: no cover - Modal is a runtime dependency for launch.
    modal = None

if modal is not None:  # pragma: no cover - exercised by Modal, not local unit tests.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PACKAGE_PINS.items()))
        .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
        .add_local_dir(str(REPO_ROOT / "data"), remote_path="/root/data")
        .add_local_dir(
            str(REPO_ROOT / "exports" / "esme-214m-chat"),
            remote_path="/root/exports/esme-214m-chat",
        )
    )
    volume = modal.Volume.from_name(MODAL_VOLUME)
    app = modal.App("esme-posttrain-emit-completions")

    @app.function(
        image=image,
        gpu="A10G",
        timeout=60 * 60,
        volumes={str(VOLUME_MOUNT): volume},
        max_containers=MAX_CONTAINERS,
    )
    def emit_job(job: dict[str, Any]) -> dict[str, Any]:
        import tempfile

        from esme_posttrain.rl.checkpoint_bundle import (
            CheckpointBundleRequest,
            export_checkpoint_bundle,
        )
        from esme_posttrain.rl.decomp_emitter import EmitRequest, emit_completion_set

        work = Path(tempfile.mkdtemp())
        if "bundle" in job:
            bundle_dir = Path(job["bundle"])
        else:
            bundle_dir = work / "bundle"
            export_checkpoint_bundle(
                CheckpointBundleRequest(
                    checkpoint_path=VOLUME_MOUNT / job["checkpoint"],
                    template_bundle_dir=VOLUME_MOUNT / job["template"],
                    output_dir=bundle_dir,
                    expected_step=job.get("expected_step"),
                )
            )
        out = work / "completions"
        emit_completion_set(
            EmitRequest(
                bundle_path=bundle_dir,
                heldout_manifest_path=Path("/root/data/manifests/esme-214m-rl-heldout.tasks.json"),
                output_dir=out,
                set_name=job["split"],
                n=int(job["n"]),
                temperature=float(job["temperature"]),
                max_new_tokens=int(job["max_new_tokens"]),
                seed=int(job["seed"]),
                device="cuda",
                model_label=job["label"],
            )
        )
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        # The container has no git checkout; record the commit of the shipped source.
        provenance["commit"] = job["commit"]
        provenance["dirty"] = job["dirty"]
        provenance["device"] = "cuda"
        bundle_manifest = (bundle_dir / "manifest.json").read_text(encoding="utf-8")
        return {
            "label": job["label"],
            "provenance": provenance,
            "completions": (out / "completions.jsonl").read_text(encoding="utf-8"),
            "bundle_manifest": bundle_manifest,
        }

    @app.local_entrypoint()
    def main(jobs: str, out_root: str) -> None:
        job_list = json.loads(Path(jobs).read_text(encoding="utf-8"))
        commit = local_git_commit(REPO_ROOT)
        dirty = local_git_dirty(REPO_ROOT)
        for job in job_list:
            missing = JOB_KEYS - set(job)
            if missing or ("bundle" in job) == ("checkpoint" in job):
                raise SystemExit(f"invalid job {job.get('label')}: missing {sorted(missing)}")
            job.update(commit=commit, dirty=dirty)
        root = Path(out_root)
        for result in emit_job.map(job_list, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"FAILED {result!r}", flush=True)
                continue
            target = root / result["label"]
            target.mkdir(parents=True, exist_ok=True)
            (target / "provenance.json").write_text(
                json.dumps(result["provenance"], sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            (target / "completions.jsonl").write_text(result["completions"], encoding="utf-8")
            (target / "bundle-manifest.json").write_text(
                result["bundle_manifest"], encoding="utf-8"
            )
            print(f"DONE {result['label']}", flush=True)
