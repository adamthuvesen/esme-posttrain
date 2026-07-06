"""Real full-run path for the multi-turn SFT foundation (Modal/GPU only).

Builds multi-turn conversations, supervises every assistant turn, evaluates a
matched held-out covering single-turn instruction (tulu) and multi-turn chat
(smol-smoltalk), and writes ``multi-turn-samples.md``. Guarded by the runtime
spend tracker and refused unless the learning gate has passed.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.launch.config_guards import LAUNCH_APPROVAL_FLAG, MODAL_CLIENT_VERSION
from esme_posttrain.run_artifacts import (
    RuntimeSpendTracker,
    refresh_manifest_files,
    write_environment,
    write_json,
    write_selected_row_manifest,
)
from esme_posttrain.sft.data import TokenizedExample, sequence_efficiency_report
from esme_posttrain.sft.full_shared import (
    SFTFullRunError,
)
from esme_posttrain.sft.full_shared import (
    assert_full_run_data_safe as _assert_full_run_data_safe,
)
from esme_posttrain.sft.full_shared import (
    assert_required_artifacts as _assert_required_artifacts,
)
from esme_posttrain.sft.full_shared import (
    select_full_run_device as _select_full_run_device,
)
from esme_posttrain.sft.full_shared import (
    steps_for_target_tokens as _steps_for_target_tokens,
)
from esme_posttrain.sft.full_shared import (
    write_eval_suite_manifests as _write_eval_suite_manifests,
)
from esme_posttrain.sft.launch_multiturn import EXPECTED_ARTIFACTS, MultiTurnLaunchConfig
from esme_posttrain.sft.multiturn_data import (
    build_multi_turn_eval_set,
    build_multi_turn_matched_eval_sets,
    build_multi_turn_mix,
    turn_distribution,
)
from esme_posttrain.sft.multiturn_judge import run_multi_turn_judge
from esme_posttrain.sft.sample_artifacts import write_multi_turn_samples
from esme_posttrain.sft.trainer import (
    EvalSplit,
    SFTTrainerConfig,
    WandbConfig,
    load_sft_checkpoint,
    run_sft_training,
)
from esme_posttrain.training.checkpointing import latest_checkpoint_path

# Bounded generation-only evidence resample from the completed full-run
# checkpoint. The $1 cap keeps the 0.25h pinned-A100 timeout below the limit.
RESAMPLE_SPEND_CAP_USD = 1.0
RESAMPLE_TIMEOUT_HOURS = 0.25
ORIGINAL_SAMPLES_ARTIFACT = "multi-turn-samples.md"
RESAMPLE_SAMPLES_ARTIFACT = "multi-turn-samples-v2.md"


def run_full_multi_turn_sft(
    config: MultiTurnLaunchConfig,
    *,
    output_dir: Path,
    base_bundle_path: Path | None = None,
    allow_remote_download: bool,
    require_cuda: bool,
    wandb_enabled: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
    resume_from_latest: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not resume_from_latest:
        raise SFTFullRunError(f"full-run output_dir must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = latest_checkpoint_path(output_dir) if resume_from_latest else None
    if resume_from_latest and resume_checkpoint is None:
        raise SFTFullRunError(f"--resume requested but no checkpoint exists in {output_dir}")

    profile = config.selected_gpu_profile
    spend_tracker = RuntimeSpendTracker(
        started=started or time.perf_counter(),
        usd_per_hour=float(profile["usd_per_hour"]),
        stop_usd=float(config.runtime["full_run_runtime_spend_stop_usd"]),
        output_dir=output_dir,
    )
    check_spend = partial(
        spend_tracker.check_cap,
        label="full multi-turn SFT",
        error_type=SFTFullRunError,
    )
    device = _select_full_run_device(require_cuda=require_cuda)
    if config.estimated_full_cost_usd > float(config.runtime["full_run_max_cost_usd"]):
        raise SFTFullRunError("projected full-run cost exceeds runtime.full_run_max_cost_usd")

    bundle_path = (base_bundle_path or config.base_bundle_path).expanduser().resolve()
    loaded = load_dense_backbone_bundle(bundle_path, map_location="cpu")

    budgets = config.budgets
    train_report = build_multi_turn_mix(
        config.train_sources,
        loaded.tokenizer,
        max_samples=int(budgets["max_train_samples"]),
        max_tokens=int(budgets["max_train_tokens"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    eval_report = build_multi_turn_eval_set(
        config.eval_source,
        loaded.tokenizer,
        max_samples=int(budgets["max_eval_samples"]),
        max_tokens=int(budgets["max_eval_tokens"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    matched_eval_reports = build_multi_turn_matched_eval_sets(
        config.train_sources,
        loaded.tokenizer,
        skip_selected_by_source={
            name: counts.selected for name, counts in train_report.counts_by_source.items()
        },
        max_samples_per_source=int(budgets["matched_eval_samples_per_source"]),
        max_tokens_per_source=int(budgets["matched_eval_tokens_per_source"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )
    _assert_full_run_data_safe(config.budgets, train_report.to_dict(), eval_report.to_dict())

    optimizer_config = config.payload["optimizer"]
    monitoring_config = config.payload["monitoring"]
    sequence_config = config.payload["sequence"]
    train_steps = _steps_for_target_tokens(
        train_report.examples,
        target_train_tokens=int(budgets["target_train_tokens"]),
        micro_batch_size=int(optimizer_config["micro_batch_size"]),
        gradient_accumulation_steps=int(optimizer_config["gradient_accumulation_steps"]),
        max_steps=int(optimizer_config["max_steps"]),
    )

    write_json(
        output_dir / "config.json",
        _config_evidence(
            config,
            commit=commit,
            dirty=dirty,
            resume_from_latest=resume_from_latest,
            resume_checkpoint=resume_checkpoint,
        ),
    )
    write_selected_row_manifest(output_dir / "selected-row-manifest.jsonl", train_report.examples)
    _write_eval_suite_manifests(output_dir, matched_eval_reports, eval_report)
    write_json(
        output_dir / "data-report.json",
        {
            "mode": "full_multi_turn_sft",
            "remote_dataset_download": allow_remote_download,
            "train": train_report.to_dict(),
            "turn_distribution": turn_distribution(train_report.examples).to_dict(),
            "eval": {
                "matched": {
                    name: report.to_dict() for name, report in matched_eval_reports.items()
                },
                "no_robots": eval_report.to_dict(),
            },
            "train_sources": [source.__dict__ for source in config.train_sources],
            "eval_source": config.eval_source.__dict__,
            "token_accounting": {
                "max_train_tokens": budgets["max_train_tokens"],
                "target_train_tokens": budgets["target_train_tokens"],
                "selected_train_tokens": train_report.selected_tokens,
                "selected_supervised_tokens": train_report.selected_supervised_tokens,
                "selected_examples": len(train_report.examples),
                "used_optimizer_steps": train_steps,
                "eval_examples": len(eval_report.examples),
                "matched_eval_examples": {
                    name: len(report.examples) for name, report in matched_eval_reports.items()
                },
            },
            "sequence_efficiency": {
                "train": sequence_efficiency_report(
                    train_report.examples,
                    max_sequence_tokens=int(budgets["max_sequence_tokens"]),
                    micro_batch_size=int(optimizer_config["micro_batch_size"]),
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
                "eval": sequence_efficiency_report(
                    eval_report.examples,
                    max_sequence_tokens=int(budgets["max_sequence_tokens"]),
                    micro_batch_size=int(optimizer_config["micro_batch_size"]),
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
            },
            "multi_turn_masking_asserted": all(
                all(label == -100 for label in example.labels[: example.prompt_tokens])
                for example in train_report.examples
            ),
            "no_robots_training": False,
            "no_robots_eval_only_reason": (
                "HuggingFaceH4/no_robots is CC-BY-NC-4.0 and this run card does not approve "
                "non-commercial training data."
            ),
        },
    )
    write_environment(output_dir / "environment.txt", device=device)

    result = run_sft_training(
        loaded.model,
        loaded.tokenizer,
        train_report.examples,
        eval_report.examples,
        SFTTrainerConfig(
            max_steps=train_steps,
            micro_batch_size=int(optimizer_config["micro_batch_size"]),
            gradient_accumulation_steps=int(optimizer_config["gradient_accumulation_steps"]),
            learning_rate=float(optimizer_config["learning_rate"]),
            scheduler=str(optimizer_config["scheduler"]),
            warmup_steps=int(optimizer_config["warmup_steps"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            precision=str(config.runtime["precision"]),
            pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
            seed=int(optimizer_config["seed"]),
            output_dir=output_dir,
            artifact_name=config.artifact_name,
            assistant_only_loss=bool(config.payload["loss"]["assistant_only_loss"]),
            completion_only_loss=bool(config.payload["loss"]["completion_only_loss"]),
            tuning_mode=str(config.payload["tuning"]["mode"]),
            grad_clip=float(optimizer_config["grad_clip"]),
            log_interval=int(monitoring_config["log_interval"]),
            eval_interval=int(monitoring_config["eval_interval"]),
            checkpoint_interval=int(monitoring_config["checkpoint_interval"]),
            retain_last_checkpoints=int(monitoring_config["retain_last_checkpoints"]),
            early_stopping_patience=int(monitoring_config["early_stopping_patience"]),
            no_robots_catastrophic_regression_multiplier=float(
                monitoring_config["no_robots_catastrophic_regression_multiplier"]
            ),
            sample_new_tokens=int(monitoring_config["sample_new_tokens"]),
            device=device.type,
            resume_from_latest=resume_from_latest,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(monitoring_config["wandb_project"]),
                run_name=f"{config.run_id}-{'resume' if resume_from_latest else 'full-sft'}",
                tags=(
                    "Esme-214M-Instruct",
                    "sft",
                    "multi-turn",
                    "resume" if resume_from_latest else "full",
                    "smol-smoltalk",
                    "tulu-personas",
                    "no-robots-eval",
                    config.runtime["selected_gpu"],
                ),
                group=config.run_id,
                job_type="full-sft",
                notes="Full multi-turn supervised SFT foundation; no RL, DPO, LoRA, or QLoRA.",
                extra_config={
                    "model": config.artifact_name,
                    "stage": "sft",
                    "run_type": "full",
                    "dataset_mix": {
                        source.name: source.mix_ratio for source in config.train_sources
                    },
                    "eval_holdout": config.eval_source.source,
                    "gpu": config.runtime["selected_gpu"],
                    "precision": config.runtime["precision"],
                    "scheduler": optimizer_config["scheduler"],
                    "max_sequence_tokens": budgets["max_sequence_tokens"],
                },
            ),
        ),
        eval_splits=_eval_splits(config, matched_eval_reports, eval_report),
        base_bundle_manifest=loaded.bundle.manifest,
        step_callback=check_spend,
    )
    if not result.instruct_beats_base:
        raise SFTFullRunError("weighted matched response loss did not beat Base")

    write_multi_turn_samples(
        output_dir / "multi-turn-samples.md",
        loaded.model,
        loaded.tokenizer,
        _multi_turn_sample_pool(matched_eval_reports),
        sample_new_tokens=int(monitoring_config["sample_new_tokens"]),
        selected_step=result.selected_step,
    )
    judge_report = run_multi_turn_judge(
        lambda prompt: _generate_chat_continuation(
            loaded.model,
            loaded.tokenizer,
            prompt,
            max_new_tokens=int(monitoring_config["sample_new_tokens"]),
        ),
        judge=None,
        passes=int(monitoring_config["judge_repeat_passes"]),
    )
    eval_payload = result.to_dict()
    eval_payload["multi_turn_judge"] = judge_report.to_dict()
    write_json(output_dir / "eval-report.json", eval_payload)
    cost = spend_tracker.write_cost(step=result.steps_completed, status="complete")
    refresh_manifest_files(output_dir, EXPECTED_ARTIFACTS)
    _assert_required_artifacts(output_dir, EXPECTED_ARTIFACTS)

    return {
        "status": "modal_full_multi_turn_sft_complete",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "output_dir": str(output_dir),
        "volume": config.runtime["modal_volume"],
        "commit": commit,
        "dirty": dirty,
        "device": device.type,
        "paid_compute": True,
        "training_mode": result.training_mode,
        "resumed_from_checkpoint": result.resumed_from_checkpoint,
        "cost": cost,
        "wandb_run": result.wandb_run_url,
        "result": result.to_dict(),
        "required_artifacts_present": {
            name: (output_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


def resample_evidence_blockers(
    config: MultiTurnLaunchConfig,
    *,
    modal_gpu: str,
    timeout_hours: float = RESAMPLE_TIMEOUT_HOURS,
) -> list[str]:
    blockers: list[str] = []
    if timeout_hours <= 0 or timeout_hours > 24:
        blockers.append("resample-evidence timeout_hours must be between 0 and 24")
    if modal_gpu != config.runtime["selected_gpu"]:
        blockers.append(
            "SFT_MODAL_GPU must match runtime.selected_gpu for resample-evidence cost accounting"
        )
    if timeout_hours * float(config.selected_gpu_profile["usd_per_hour"]) > RESAMPLE_SPEND_CAP_USD:
        blockers.append("resample-evidence timeout cost ceiling exceeds the $1 resample spend cap")
    return blockers


def resample_evidence_command(config_path: Path, *, gpu: str, timeout_hours: float) -> str:
    return (
        f"SFT_MODAL_GPU='{gpu}' SFT_RESAMPLE_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} modal run --detach "
        f"scripts/modal_chat_sft.py --config {config_path.as_posix()} "
        f"--resample-evidence {LAUNCH_APPROVAL_FLAG} --json"
    )


def build_resample_evidence_preflight(
    config: MultiTurnLaunchConfig,
    *,
    modal_gpu: str,
    output_stem: str,
    timeout_hours: float = RESAMPLE_TIMEOUT_HOURS,
) -> dict[str, Any]:
    timeout_cost_ceiling = float(config.selected_gpu_profile["usd_per_hour"]) * timeout_hours
    blockers = resample_evidence_blockers(config, modal_gpu=modal_gpu, timeout_hours=timeout_hours)
    return {
        "status": "ready_for_resample_evidence" if not blockers else "blocked_by_launch_safety",
        "mode": "multi_turn_resample_evidence",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "will_start_modal_job": False,
        "will_download_data": False,
        "generation_only": True,
        "volume": config.runtime["modal_volume"],
        "checkpoint": f"{output_stem}/best-checkpoint.pt",
        "tokenizer": f"{output_stem}/tokenizer.json",
        "data_report": f"{output_stem}/data-report.json",
        "outputs": {
            "resampled_markdown": f"{output_stem}/{RESAMPLE_SAMPLES_ARTIFACT}",
            "original_preserved": f"{output_stem}/{ORIGINAL_SAMPLES_ARTIFACT}",
        },
        "runtime": {
            "provider": "modal",
            "selected_gpu": config.runtime["selected_gpu"],
            "modal_gpu": modal_gpu,
            "timeout_hours": timeout_hours,
            "spend_cap_usd": RESAMPLE_SPEND_CAP_USD,
            "timeout_cost_ceiling_usd": round(timeout_cost_ceiling, 4),
            "sample_new_tokens": int(config.payload["monitoring"]["sample_new_tokens"]),
        },
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION},
        "launch_blockers": blockers,
        "resample_evidence_command": resample_evidence_command(
            config.config_path, gpu=modal_gpu, timeout_hours=timeout_hours
        ),
    }


def resample_multi_turn_evidence(
    config: MultiTurnLaunchConfig,
    *,
    output_dir: Path,
    allow_remote_download: bool,
    require_cuda: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
    sample_pool: tuple[TokenizedExample, ...] | None = None,
) -> dict[str, Any]:
    """Regenerate the multi-turn sample evidence from a completed full run.

    Loads ``best-checkpoint.pt`` and ``tokenizer.json`` from ``output_dir``,
    rebuilds the matched held-out eval examples exactly like the full run
    (skip counts come from the persisted ``data-report.json``), and writes the
    resampled markdown without modifying ``multi-turn-samples.md``.
    ``sample_pool`` bypasses the dataset rebuild for the no-download CPU fixture
    path.
    """
    started = started or time.perf_counter()
    output_dir = output_dir.expanduser().resolve()
    checkpoint_path = output_dir / "best-checkpoint.pt"
    tokenizer_path = output_dir / "tokenizer.json"
    for required, label in ((checkpoint_path, "best checkpoint"), (tokenizer_path, "tokenizer")):
        if not required.is_file():
            raise SFTFullRunError(f"resample requires the completed full-run {label}: {required}")

    device = _select_full_run_device(require_cuda=require_cuda)
    checkpoint = load_sft_checkpoint(checkpoint_path, map_location=device)
    model = checkpoint.model.to(device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if sample_pool is None:
        matched_eval_reports = _rebuild_matched_eval_reports(
            config, tokenizer, output_dir, allow_remote_download=allow_remote_download
        )
        sample_pool = _multi_turn_sample_pool(matched_eval_reports)
    if not sample_pool:
        raise SFTFullRunError("resample produced no multi-turn eval examples to sample from")

    resampled_path = output_dir / RESAMPLE_SAMPLES_ARTIFACT
    write_multi_turn_samples(
        resampled_path,
        model,
        tokenizer,
        sample_pool,
        sample_new_tokens=int(config.payload["monitoring"]["sample_new_tokens"]),
        selected_step=checkpoint.step,
    )
    resampled_markdown = resampled_path.read_text(encoding="utf-8")

    elapsed = time.perf_counter() - started
    usd_per_hour = float(config.selected_gpu_profile["usd_per_hour"])
    estimated_cost = elapsed * usd_per_hour / 3600.0
    if estimated_cost > RESAMPLE_SPEND_CAP_USD:
        raise SFTFullRunError(
            f"resample-evidence exceeded the ${RESAMPLE_SPEND_CAP_USD:.0f} cap "
            f"(estimated ${estimated_cost:.4f})"
        )
    original_path = output_dir / ORIGINAL_SAMPLES_ARTIFACT
    return {
        "status": "resample_evidence_complete",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "output_dir": str(output_dir),
        "volume": config.runtime["modal_volume"],
        "device": device.type,
        "paid_compute": True,
        "commit": commit,
        "dirty": dirty,
        "selected_step": checkpoint.step,
        "sample_new_tokens": int(config.payload["monitoring"]["sample_new_tokens"]),
        "original_samples_path": str(original_path),
        "original_samples_preserved": original_path.is_file(),
        "resampled_samples_path": str(resampled_path),
        "resampled_markdown": resampled_markdown,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": estimated_cost,
        "spend_cap_usd": RESAMPLE_SPEND_CAP_USD,
    }


def _rebuild_matched_eval_reports(
    config: MultiTurnLaunchConfig,
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    allow_remote_download: bool,
) -> dict[str, Any]:
    data_report_path = output_dir / "data-report.json"
    if not data_report_path.is_file():
        raise SFTFullRunError(
            f"resample requires the completed full-run data report: {data_report_path}"
        )
    data_report = json.loads(data_report_path.read_text(encoding="utf-8"))
    try:
        counts_by_source = data_report["train"]["counts_by_source"]
        skip_selected = {name: int(counts["selected"]) for name, counts in counts_by_source.items()}
    except (KeyError, TypeError, ValueError) as error:
        raise SFTFullRunError(
            "full-run data report is missing train.counts_by_source selected counts: "
            f"{data_report_path}"
        ) from error
    budgets = config.budgets
    return build_multi_turn_matched_eval_sets(
        config.train_sources,
        tokenizer,
        skip_selected_by_source=skip_selected,
        max_samples_per_source=int(budgets["matched_eval_samples_per_source"]),
        max_tokens_per_source=int(budgets["matched_eval_tokens_per_source"]),
        max_sequence_tokens=int(budgets["max_sequence_tokens"]),
        allow_remote_download=allow_remote_download,
    )


def _generate_chat_continuation(
    model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int
) -> str:
    eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
    if was_training:
        model.train()
    new_ids = generated[0].detach().cpu().tolist()[len(prompt_ids) :]
    if eos_id is not None and eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def _multi_turn_sample_pool(matched_eval_reports: dict[str, Any]) -> tuple[TokenizedExample, ...]:
    smol = matched_eval_reports.get("smol-smoltalk")
    pool = list(smol.examples) if smol is not None else []
    multi = [example for example in pool if example.assistant_turns > 1]
    return tuple(multi[:3] or pool[:3])


def _eval_splits(
    config: MultiTurnLaunchConfig, matched_eval_reports: dict[str, Any], no_robots_report: Any
) -> tuple[EvalSplit, ...]:
    splits: list[EvalSplit] = []
    for source in config.train_sources:
        report = matched_eval_reports[source.name]
        splits.append(EvalSplit(source.name, report.examples, selector_weight=source.mix_ratio))
    splits.append(EvalSplit("no_robots", no_robots_report.examples))
    return tuple(splits)


def _config_evidence(
    config: MultiTurnLaunchConfig,
    *,
    commit: str,
    dirty: bool,
    resume_from_latest: bool,
    resume_checkpoint: Path | None,
) -> dict[str, Any]:
    return {
        "mode": "full_multi_turn_sft",
        "training_mode": "resumed" if resume_from_latest else "fresh",
        "resume_from_latest": resume_from_latest,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "source_config": config.payload,
        "commit": commit,
        "dirty": dirty,
        "projected_full_cost_usd": config.estimated_full_cost_usd,
        "target_train_tokens": config.budgets["target_train_tokens"],
        "max_train_tokens": config.budgets["max_train_tokens"],
    }
