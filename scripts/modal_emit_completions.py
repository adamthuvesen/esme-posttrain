#!/usr/bin/env python3
# ruff: noqa: E402
"""Rebuild GRPO checkpoints as bundles and sample CompletionSets on a remote GPU.

Each job names a training checkpoint and template bundle on the RLVR volume (or a
ready bundle baked into the image), a held-out split, and decoding settings. The
remote function converts the checkpoint with ``export_checkpoint_bundle``, samples
it with ``emit_completion_set``, and returns the small CompletionSet files, which
are written locally under ``--out-root``. Checkpoints never leave the volume.

Usage (spends GPU time; the refusal prints the approved command to ask Adam about):
    uv run --with modal==1.5.1 modal run scripts/modal_emit_completions.py \\
        --jobs <jobs.json> --out-root runs/<dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Self

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pydantic import StrictBool, StrictFloat, StrictInt, StrictStr, ValidationError, model_validator

from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    PYPROJECT_FILE,
    UV_LOCK_FILE,
    image_package_pins,
    image_uv_version,
)
from esme_posttrain.launch.config_models import FrozenConfig
from esme_posttrain.launch.modal_cli import (
    LOCAL_PYTHON,
    PYTHON_VERSION_FILE,
    local_git_commit,
    local_git_dirty,
)
from esme_posttrain.rl.config import MODAL_VOLUME

VOLUME_MOUNT = Path("/posttrain")
# The account shares a small GPU quota with other work; keep this runner narrow.
MAX_CONTAINERS = 3


class EmitJob(FrozenConfig):
    label: StrictStr
    split: StrictStr
    n: StrictInt
    temperature: StrictFloat | StrictInt
    max_new_tokens: StrictInt
    seed: StrictInt
    bundle: StrictStr | None = None
    checkpoint: StrictStr | None = None
    template: StrictStr | None = None
    expected_step: StrictInt | None = None
    commit: StrictStr
    dirty: StrictBool

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        if (self.bundle is None) == (self.checkpoint is None):
            raise ValueError("give exactly one of bundle, checkpoint")
        if self.checkpoint is not None and self.template is None:
            raise ValueError("a checkpoint job needs a template bundle")
        return self


def load_jobs(jobs_path: Path, *, commit: str, dirty: bool) -> tuple[EmitJob, ...]:
    raw_jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise SystemExit(f"{jobs_path}: jobs must be a non-empty JSON list")
    jobs = []
    for index, raw in enumerate(raw_jobs):
        try:
            jobs.append(EmitJob.model_validate({**raw, "commit": commit, "dirty": dirty}))
        except (TypeError, ValidationError) as error:
            raise SystemExit(f"{jobs_path}: invalid job {index}: {error}") from error
    return tuple(jobs)


def launch_command(jobs: str, out_root: str) -> str:
    return (
        f"uv run --with modal=={MODAL_CLIENT_VERSION} modal run "
        f"scripts/modal_emit_completions.py --jobs {jobs} --out-root {out_root} "
        f"{LAUNCH_APPROVAL_FLAG}"
    )


import modal  # noqa: TID251

image = (
    modal.Image.debian_slim(python_version=LOCAL_PYTHON)
    .uv_pip_install(
        *(f"{name}=={version}" for name, version in image_package_pins().items()),
        uv_version=image_uv_version(),
    )
    .env({"PYTHONPATH": "/root/src", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(str(REPO_ROOT / "src"), remote_path="/root/src")
    .add_local_file(str(PYTHON_VERSION_FILE), remote_path="/root/.python-version")
    .add_local_file(str(PYPROJECT_FILE), remote_path="/root/pyproject.toml")
    .add_local_file(str(UV_LOCK_FILE), remote_path="/root/uv.lock")
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
def emit_job(job: EmitJob) -> dict[str, Any]:
    import tempfile

    from esme_posttrain.rl.checkpoint_bundle import (
        CheckpointBundleRequest,
        export_checkpoint_bundle,
    )
    from esme_posttrain.rl.decomp_emitter import EmitRequest, emit_completion_set

    work = Path(tempfile.mkdtemp())
    if job.bundle is not None:
        bundle_dir = Path(job.bundle)
    else:
        assert job.checkpoint is not None and job.template is not None
        bundle_dir = work / "bundle"
        export_checkpoint_bundle(
            CheckpointBundleRequest(
                checkpoint_path=VOLUME_MOUNT / job.checkpoint,
                template_bundle_dir=VOLUME_MOUNT / job.template,
                output_dir=bundle_dir,
                expected_step=job.expected_step,
            )
        )
    out = work / "completions"
    emit_completion_set(
        EmitRequest(
            bundle_path=bundle_dir,
            heldout_manifest_path=Path("/root/data/manifests/esme-214m-rl-heldout.tasks.json"),
            output_dir=out,
            set_name=job.split,
            n=job.n,
            temperature=float(job.temperature),
            max_new_tokens=job.max_new_tokens,
            seed=job.seed,
            device="cuda",
            model_label=job.label,
        )
    )
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    # The container has no git checkout; record the commit of the shipped source.
    provenance["commit"] = job.commit
    provenance["dirty"] = job.dirty
    provenance["device"] = "cuda"
    bundle_manifest = (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    return {
        "label": job.label,
        "provenance": provenance,
        "completions": (out / "completions.jsonl").read_text(encoding="utf-8"),
        "bundle_manifest": bundle_manifest,
    }


@app.local_entrypoint()
def main(jobs: str, out_root: str, approved: bool = False) -> None:
    job_list = load_jobs(
        Path(jobs), commit=local_git_commit(REPO_ROOT), dirty=local_git_dirty(REPO_ROOT)
    )
    if not approved:
        raise SystemExit(
            f"emit-completions refused: {len(job_list)} GPU jobs need {LAUNCH_APPROVAL_FLAG}. "
            f"Ask Adam, then run: {launch_command(jobs, out_root)}"
        )
    root = Path(out_root)
    failures = 0
    for result in emit_job.map(job_list, return_exceptions=True):
        if isinstance(result, Exception):
            failures += 1
            print(f"FAILED {result!r}", flush=True)
            continue
        target = root / result["label"]
        target.mkdir(parents=True, exist_ok=True)
        (target / "provenance.json").write_text(
            json.dumps(result["provenance"], sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (target / "completions.jsonl").write_text(result["completions"], encoding="utf-8")
        (target / "bundle-manifest.json").write_text(result["bundle_manifest"], encoding="utf-8")
        print(f"DONE {result['label']}", flush=True)
    if failures:
        raise SystemExit(f"{failures} of {len(job_list)} jobs failed")
