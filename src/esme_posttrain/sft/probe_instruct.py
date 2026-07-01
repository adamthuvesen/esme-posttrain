from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.launch.config_guards import LAUNCH_APPROVAL_FLAG, MODAL_CLIENT_VERSION
from esme_posttrain.run_artifacts import RuntimeSpendTracker, write_json
from esme_posttrain.sft.data import build_eval_set, build_training_mix
from esme_posttrain.sft.launch_instruct import SFTLaunchConfig
from esme_posttrain.sft.sweep_instruct import SWEEP_OUTPUT_STEM
from esme_posttrain.sft.sweep_shared import SFTSweepError
from esme_posttrain.sft.sweep_shared import select_sweep_device as _select_sweep_device
from esme_posttrain.sft.trainer import SFTTrainerConfig, run_sft_training
from esme_posttrain.training.wandb_init import WandbConfig

PROBE_OUTPUT_STEM = "esme-instruct-sft-throughput-probe"
PROBE_STEPS = 80
PROBE_TIMEOUT_HOURS = 2
PROBE_TRAIN_SAMPLE_CAP = 384
PROBE_TRAIN_TOKEN_CAP = 393_216
PROBE_EVAL_SAMPLE_CAP = 32
PROBE_EVAL_TOKEN_CAP = 65_536
PROBE_TARGET_TRAIN_TOKENS = 50_000_000
PROBE_RECIPE = {
    "learning_rate": 3e-5,
    "micro_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "effective_batch_size": 2,
}
PROBE_GPU_USD_PER_HOUR = {
    "A100": 2.0988,
    "H100!": 3.9492,
    "H200": 4.5396,
    "B200": 6.2496,
}
DEFAULT_MODAL_PROBE_ROOT = Path("/posttrain") / PROBE_OUTPUT_STEM


def build_throughput_probe_preflight(
    config: SFTLaunchConfig, *, modal_gpu: str, timeout_hours: int = PROBE_TIMEOUT_HOURS
) -> dict[str, Any]:
    blockers = throughput_probe_blockers(modal_gpu=modal_gpu, timeout_hours=timeout_hours)
    return {
        "status": "ready_for_throughput_probe" if not blockers else "blocked_by_launch_safety",
        "mode": "throughput_probe",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "will_start_modal_job": False,
        "will_download_data": False,
        "modal_run_will_download_real_data": True,
        "uses_full_run_output_dir": False,
        "uses_sweep_output_dir": False,
        "volume": config.runtime["modal_volume"],
        "volume_output_root": str(DEFAULT_MODAL_PROBE_ROOT),
        "gpu": modal_gpu,
        "recipe": PROBE_RECIPE,
        "steps": PROBE_STEPS,
        "data_caps": {
            "train_samples": PROBE_TRAIN_SAMPLE_CAP,
            "train_tokens": PROBE_TRAIN_TOKEN_CAP,
            "eval_samples": PROBE_EVAL_SAMPLE_CAP,
            "eval_tokens": PROBE_EVAL_TOKEN_CAP,
            "no_robots_train_allowed": False,
        },
        "runtime": {
            "provider": "modal",
            "gpu": modal_gpu,
            "timeout_hours": timeout_hours,
            "usd_per_hour": PROBE_GPU_USD_PER_HOUR.get(modal_gpu),
            "timeout_cost_ceiling_usd": PROBE_GPU_USD_PER_HOUR.get(modal_gpu, 0.0) * timeout_hours,
            "runtime_spend_stop_usd": 15.0,
        },
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION},
        "launch_blockers": blockers,
        "throughput_probe_command": throughput_probe_command(
            config.config_path, gpu=modal_gpu, timeout_hours=timeout_hours
        ),
    }


def throughput_probe_blockers(*, modal_gpu: str, timeout_hours: int) -> list[str]:
    blockers: list[str] = []
    if modal_gpu not in PROBE_GPU_USD_PER_HOUR:
        blockers.append(
            "SFT_MODAL_GPU must be one of A100, H100!, H200, or B200 for throughput probe"
        )
    if timeout_hours <= 0 or timeout_hours > 4:
        blockers.append("SFT_PROBE_TIMEOUT_HOURS must be between 1 and 4")
    if "-excellence" in str(DEFAULT_MODAL_PROBE_ROOT):
        blockers.append("throughput probe output root must not use an excellence public name")
    if DEFAULT_MODAL_PROBE_ROOT.name == SWEEP_OUTPUT_STEM:
        blockers.append("throughput probe output root must not reuse interval sweep root")
    return blockers


def throughput_probe_command(config_path: Path, *, gpu: str, timeout_hours: int) -> str:
    return (
        f"SFT_MODAL_GPU='{gpu}' SFT_PROBE_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} modal run "
        f"scripts/modal_instruct_sft.py --config {config_path} "
        f"--throughput-probe --approved --json"
    )


def run_throughput_probe(
    config: SFTLaunchConfig,
    *,
    output_root: Path,
    modal_gpu: str,
    base_bundle_path: Path | None = None,
    allow_remote_download: bool,
    require_cuda: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
) -> dict[str, Any]:
    blockers = throughput_probe_blockers(modal_gpu=modal_gpu, timeout_hours=PROBE_TIMEOUT_HOURS)
    if blockers:
        raise SFTSweepError("throughput probe refused: " + "; ".join(blockers))
    output_root = output_root.expanduser().resolve()
    if output_root.name != PROBE_OUTPUT_STEM:
        raise SFTSweepError(f"throughput probe output root must end with {PROBE_OUTPUT_STEM}")
    if "-excellence" in str(output_root):
        raise SFTSweepError("throughput probe output root must not use an excellence public name")
    output_root.mkdir(parents=True, exist_ok=True)

    started = started or time.perf_counter()
    usd_per_hour = PROBE_GPU_USD_PER_HOUR[modal_gpu]
    spend_tracker = RuntimeSpendTracker(
        started=started,
        usd_per_hour=usd_per_hour,
        stop_usd=15.0,
        output_dir=output_root,
    )
    device = _select_sweep_device(require_cuda=require_cuda)
    bundle_path = (base_bundle_path or config.base_bundle_path).expanduser().resolve()
    loaded = load_dense_backbone_bundle(bundle_path, map_location="cpu")
    budgets = config.budgets
    train_report = build_training_mix(
        config.train_sources,
        loaded.tokenizer,
        max_samples=PROBE_TRAIN_SAMPLE_CAP,
        max_tokens=PROBE_TRAIN_TOKEN_CAP,
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    eval_report = build_eval_set(
        config.eval_source,
        loaded.tokenizer,
        max_samples=PROBE_EVAL_SAMPLE_CAP,
        max_tokens=PROBE_EVAL_TOKEN_CAP,
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    if train_report.shortfalls:
        raise SFTSweepError("training data shortfall: " + "; ".join(train_report.shortfalls))
    if eval_report.shortfalls and not eval_report.examples:
        raise SFTSweepError("eval data shortfall: " + "; ".join(eval_report.shortfalls))

    launch_id = _fresh_probe_id(output_root, modal_gpu)
    output_dir = output_root / launch_id
    output_dir.mkdir()
    write_json(
        output_dir / "config.json",
        {
            "mode": "throughput_probe",
            "gpu": modal_gpu,
            "recipe": PROBE_RECIPE,
            "steps": PROBE_STEPS,
            "source_config": config.payload,
            "commit": commit,
            "dirty": dirty,
            "approval": {
                "approved_by": "Adam",
                "approved_on": "2026-06-27",
                "condition": "bounded throughput probe only; not full-data SFT",
            },
        },
    )
    write_json(
        output_dir / "data-report.json",
        {
            "mode": "throughput_probe",
            "remote_dataset_download": allow_remote_download,
            "train": train_report.to_dict(),
            "eval": eval_report.to_dict(),
            "no_robots_training": False,
        },
    )
    optimizer_config = config.payload["optimizer"]
    sequence_config = config.payload["sequence"]
    training_started = time.perf_counter()
    result = run_sft_training(
        loaded.model,
        loaded.tokenizer,
        train_report.examples,
        eval_report.examples,
        SFTTrainerConfig(
            max_steps=PROBE_STEPS,
            micro_batch_size=int(PROBE_RECIPE["micro_batch_size"]),
            gradient_accumulation_steps=int(PROBE_RECIPE["gradient_accumulation_steps"]),
            learning_rate=float(PROBE_RECIPE["learning_rate"]),
            scheduler=str(optimizer_config["scheduler"]),
            warmup_steps=10,
            weight_decay=float(optimizer_config["weight_decay"]),
            precision=str(config.runtime["precision"]),
            pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
            seed=int(optimizer_config["seed"]),
            output_dir=output_dir,
            assistant_only_loss=bool(config.payload["loss"]["assistant_only_loss"]),
            completion_only_loss=bool(config.payload["loss"]["completion_only_loss"]),
            tuning_mode=str(config.payload["tuning"]["mode"]),
            grad_clip=float(optimizer_config["grad_clip"]),
            log_interval=20,
            eval_interval=0,
            checkpoint_interval=0,
            retain_last_checkpoints=0,
            sample_new_tokens=1,
            device=device.type,
            wandb=WandbConfig(enabled=False),
        ),
        base_bundle_manifest=loaded.bundle.manifest,
        step_callback=lambda step: spend_tracker.check_cap(
            step,
            label="throughput probe",
            error_type=SFTSweepError,
        ),
    )
    training_elapsed = time.perf_counter() - training_started
    trained_tokens_per_second = result.trained_tokens / max(1e-9, training_elapsed)
    supervised_tokens_per_second = result.supervised_tokens / max(1e-9, training_elapsed)
    projected_full_seconds = PROBE_TARGET_TRAIN_TOKENS / max(1e-9, trained_tokens_per_second)
    projected_full_cost = projected_full_seconds * usd_per_hour / 3600.0
    cost = spend_tracker.write_cost(step=result.steps_completed, status="throughput_probe_complete")
    payload = {
        "status": "throughput_probe_complete",
        "mode": "throughput_probe",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "gpu": modal_gpu,
        "launch_id": launch_id,
        "output_dir": str(output_dir),
        "volume": config.runtime["modal_volume"],
        "commit": commit,
        "dirty": dirty,
        "device": device.type,
        "recipe": PROBE_RECIPE,
        "steps_completed": result.steps_completed,
        "trained_tokens": result.trained_tokens,
        "supervised_tokens": result.supervised_tokens,
        "training_elapsed_seconds": training_elapsed,
        "trained_tokens_per_second": trained_tokens_per_second,
        "supervised_tokens_per_second": supervised_tokens_per_second,
        "usd_per_hour": usd_per_hour,
        "estimated_probe_cost_usd": cost["estimated_cost_usd"],
        "projected_full_train_tokens": PROBE_TARGET_TRAIN_TOKENS,
        "projected_full_seconds": projected_full_seconds,
        "projected_full_hours": projected_full_seconds / 3600.0,
        "projected_full_cost_usd": projected_full_cost,
        "eval": result.instruct_eval.to_dict(),
        "cost": cost,
    }
    write_json(output_dir / "throughput-probe.json", payload)
    return payload


def _fresh_probe_id(output_root: Path, modal_gpu: str) -> str:
    safe_gpu = modal_gpu.replace("!", "-strict").lower()
    base = time.strftime(f"probe-%Y%m%dT%H%M%SZ-{safe_gpu}", time.gmtime())
    for suffix in ("", *[f"-{index}" for index in range(1, 100)]):
        candidate = f"{base}{suffix}"
        if not (output_root / candidate).exists():
            return candidate
    raise SFTSweepError(f"could not find an isolated throughput probe id under {output_root}")
