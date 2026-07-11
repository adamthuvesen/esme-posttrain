"""Real full/smoke DPO run path (Modal/GPU only) for Esme-214M-Chat.

Loads the accepted SFT foundation checkpoint from the SFT Modal Volume as BOTH
the warm-started policy and the frozen reference, builds UltraFeedback preference
data, runs the real decoding pre-check on that checkpoint, trains one vanilla DPO
pass, and writes proxy-metric evidence (held-out preference accuracy, response
length, n-gram repetition, chosen/rejected logps) plus chat samples. Guarded by
the runtime spend tracker and refused unless the beta-sweep
learning gate passed (enforced in the launcher). CUDA-required.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.dpo.chat_eval import FIXED_MULTI_TURN_PROMPTS
from esme_posttrain.dpo.data import build_preference_set
from esme_posttrain.dpo.decoding_precheck import run_decoding_precheck
from esme_posttrain.dpo.launch import EXPECTED_ARTIFACTS, DPOLaunchConfig
from esme_posttrain.dpo.sample_artifacts import write_chat_samples
from esme_posttrain.dpo.trainer import DPOTrainerConfig, run_dpo_training
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.run_artifacts import (
    RuntimeSpendTracker,
    refresh_manifest_files,
    write_environment,
    write_json,
    write_selected_row_manifest,
)
from esme_posttrain.training.checkpointing import load_training_checkpoint


class DPOFullRunError(RuntimeError):
    pass


def run_full_dpo(
    config: DPOLaunchConfig,
    *,
    output_dir: Path,
    sft_checkpoint_path: Path,
    sft_tokenizer_path: Path,
    allow_remote_download: bool,
    require_cuda: bool,
    smoke: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DPOFullRunError(f"DPO output_dir must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(require_cuda=require_cuda)
    profile = config.selected_gpu_profile
    runtime = config.runtime
    spend_stop = float(
        runtime["runtime_spend_stop_usd"] if smoke else runtime["full_run_runtime_spend_stop_usd"]
    )
    max_cost = float(runtime["smoke_max_cost_usd"] if smoke else runtime["full_run_max_cost_usd"])
    spend_tracker = RuntimeSpendTracker(
        started=started or time.perf_counter(),
        usd_per_hour=float(profile["usd_per_hour"]),
        stop_usd=spend_stop,
        output_dir=output_dir,
    )
    check_spend = partial(
        spend_tracker.check_cap,
        label="DPO",
        error_type=DPOFullRunError,
    )
    estimated = config.estimated_smoke_cost_usd if smoke else config.estimated_full_cost_usd
    if estimated > max_cost:
        raise DPOFullRunError("projected DPO cost exceeds the approved cap")

    tokenizer = Tokenizer.from_file(str(sft_tokenizer_path))
    reference = _load_sft_backbone(sft_checkpoint_path, device=device)
    policy = _load_sft_backbone(sft_checkpoint_path, device=device)

    budgets = config.budgets
    pair_caps = _pair_caps(config, smoke=smoke)
    train_report = build_preference_set(
        config.preference_source,
        tokenizer,
        max_pairs=pair_caps["train_pairs"],
        min_pairs=pair_caps["min_train_pairs"],
        max_tokens=pair_caps["train_tokens"],
        max_length=int(budgets["max_length"]),
        max_prompt_length=int(budgets["max_prompt_length"]),
        allow_remote_download=allow_remote_download,
    )
    eval_report = build_preference_set(
        config.eval_source,
        tokenizer,
        max_pairs=pair_caps["eval_pairs"],
        min_pairs=pair_caps["min_eval_pairs"],
        max_tokens=int(budgets["max_eval_tokens"]),
        max_length=int(budgets["max_length"]),
        max_prompt_length=int(budgets["max_prompt_length"]),
        allow_remote_download=allow_remote_download,
    )
    _assert_data_safe(train_report.to_dict(), eval_report.to_dict(), pair_caps)
    prompt_masking_asserted = _assert_prompt_masking(train_report.pairs)

    write_json(
        output_dir / "config.json",
        {
            "mode": "modal_smoke_dpo" if smoke else "modal_full_dpo",
            "source_config": config.payload,
            "commit": commit,
            "dirty": dirty,
            "sft_checkpoint_path": str(sft_checkpoint_path),
            "projected_cost_usd": estimated,
        },
    )
    write_json(
        output_dir / "data-report.json",
        {
            "mode": "modal_smoke_dpo" if smoke else "modal_full_dpo",
            "remote_dataset_download": allow_remote_download,
            "train": train_report.to_dict(),
            "eval": eval_report.to_dict(),
            "preference_source": config.preference_source.__dict__,
            "eval_source": config.eval_source.__dict__,
            "prompt_masking_asserted": prompt_masking_asserted,
            "caps": pair_caps,
        },
    )
    write_selected_row_manifest(output_dir / "selected-pair-manifest.jsonl", train_report.pairs)
    write_selected_row_manifest(output_dir / "eval-pair-manifest.jsonl", eval_report.pairs)

    # Real decoding pre-check on the SFT checkpoint, before DPO touches the policy.
    precheck = run_decoding_precheck(
        reference,
        tokenizer,
        FIXED_MULTI_TURN_PROMPTS,
        is_real_checkpoint=True,
        note="pre-DPO decoding baseline on the accepted Esme SFT foundation checkpoint",
    )
    write_json(output_dir / "decoding-precheck.json", precheck.to_dict())

    optimizer = config.optimizer
    dpo = config.payload["dpo"]
    monitoring = config.payload["monitoring"]
    max_steps = int(optimizer["smoke_max_steps"] if smoke else optimizer["max_steps"])
    warmup_steps = int(round(float(optimizer["warmup_ratio"]) * max_steps))
    result = run_dpo_training(
        policy,
        reference,
        tokenizer,
        train_report.pairs,
        eval_report.pairs,
        DPOTrainerConfig(
            max_steps=max_steps,
            micro_batch_size=int(optimizer["micro_batch_size"]),
            gradient_accumulation_steps=int(optimizer["gradient_accumulation_steps"]),
            learning_rate=float(optimizer["learning_rate"]),
            beta=float(dpo["beta"]),
            length_normalized=bool(dpo["length_normalized"]),
            scheduler=str(optimizer["scheduler"]),
            warmup_steps=warmup_steps,
            weight_decay=float(optimizer["weight_decay"]),
            precision=str(runtime["precision"]),
            pad_to_multiple_of=config.payload["sequence"]["pad_to_multiple_of"],
            seed=int(optimizer["seed"]),
            output_dir=output_dir,
            artifact_name=config.artifact_name,
            grad_clip=float(optimizer["grad_clip"]),
            log_interval=int(monitoring["log_interval"]),
            eval_interval=int(monitoring["eval_interval"]),
            checkpoint_interval=int(monitoring["checkpoint_interval"]),
            device=device.type,
            wandb=_wandb_config(config, smoke=smoke),
        ),
        reference_bundle_manifest={
            "sft_checkpoint_path": str(sft_checkpoint_path),
            "wandb_run": config.payload["sft_reference"]["wandb_run"],
            "best_step": config.payload["sft_reference"]["best_step"],
        },
        step_callback=check_spend,
    )
    _assert_accepted_dpo_result(result)

    write_chat_samples(
        output_dir / "chat-samples.md",
        policy,
        tokenizer,
        eval_report.pairs,
        selected_step=result.selected_step,
    )
    eval_payload = result.to_dict()
    eval_payload["decoding_precheck"] = precheck.to_dict()
    write_json(output_dir / "eval-report.json", eval_payload)
    cost = spend_tracker.write_cost(step=result.steps_completed, status="complete")
    if cost["estimated_cost_usd"] > max_cost:
        raise DPOFullRunError("DPO run exceeded the approved cost cap")
    write_environment(output_dir / "environment.txt", device=device)
    refresh_manifest_files(output_dir, EXPECTED_ARTIFACTS)
    _assert_required_artifacts(output_dir)
    return {
        "status": "modal_smoke_dpo_complete" if smoke else "modal_full_dpo_complete",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "output_dir": str(output_dir),
        "volume": runtime["modal_volume"],
        "commit": commit,
        "dirty": dirty,
        "device": device.type,
        "paid_compute": True,
        "cost": cost,
        "margin_increased": result.margin_increased,
        "chosen_logp_collapsed": result.chosen_logp_collapsed,
        "wandb_run": result.wandb_run_url,
        "result": result.to_dict(),
        "required_artifacts_present": {
            name: (output_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


def _load_sft_backbone(checkpoint_path: Path, *, device: torch.device) -> DenseBackbone:
    loaded = load_training_checkpoint(checkpoint_path, map_location=device)
    return loaded.model.to(device)


def _pair_caps(config: DPOLaunchConfig, *, smoke: bool) -> dict[str, int]:
    budgets = config.budgets
    if smoke:
        # The smoke is a tiny tightly-bounded correctness check: cap == floor.
        return {
            "train_pairs": int(budgets["smoke_train_pairs"]),
            "train_tokens": int(budgets["smoke_train_tokens"]),
            "eval_pairs": int(budgets["smoke_eval_pairs"]),
            "eval_tokens": int(budgets["max_eval_tokens"]),
            "min_train_pairs": int(budgets["smoke_train_pairs"]),
            "min_eval_pairs": int(budgets["smoke_eval_pairs"]),
        }
    return {
        "train_pairs": int(budgets["max_train_pairs"]),
        "train_tokens": int(budgets["max_train_tokens"]),
        "eval_pairs": int(budgets["max_eval_pairs"]),
        "eval_tokens": int(budgets["max_eval_tokens"]),
        # max_*_pairs are caps; min_*_pairs are the sufficiency floors. UltraFeedback
        # responses are long, so far fewer than the cap survive max_length=1024 --
        # which is fine as long as we clear the floor.
        "min_train_pairs": int(budgets["min_train_pairs"]),
        "min_eval_pairs": int(budgets["min_eval_pairs"]),
    }


def _wandb_config(config: DPOLaunchConfig, *, smoke: bool) -> Any:
    from esme_posttrain.sft.trainer import WandbConfig

    monitoring = config.payload["monitoring"]
    return WandbConfig(
        enabled=True,
        project=str(monitoring["wandb_project"]),
        run_name=f"{config.run_id}-{'smoke' if smoke else 'full-dpo'}",
        tags=tuple(monitoring["wandb_tags"]) + (("smoke",) if smoke else ("full",)),
        group=config.run_id,
        job_type="smoke" if smoke else "full-dpo",
        notes="Vanilla offline DPO chat polish; frozen SFT reference; no SimPO, no RL.",
        extra_config={
            "model": config.artifact_name,
            "stage": "dpo",
            "run_type": "smoke" if smoke else "full",
            "beta": float(config.payload["dpo"]["beta"]),
            "length_normalized": bool(config.payload["dpo"]["length_normalized"]),
        },
    )


def _assert_data_safe(
    train_report: dict[str, Any], eval_report: dict[str, Any], caps: dict[str, int]
) -> None:
    if int(train_report["selected_pairs"]) > caps["train_pairs"]:
        raise DPOFullRunError("train pair cap exceeded")
    if int(train_report["selected_tokens"]) > caps["train_tokens"]:
        raise DPOFullRunError("train token cap exceeded")
    if train_report["shortfalls"]:
        raise DPOFullRunError(
            "preference training data shortfall: " + "; ".join(train_report["shortfalls"])
        )
    if int(eval_report["selected_pairs"]) > caps["eval_pairs"]:
        raise DPOFullRunError("eval pair cap exceeded")
    if int(eval_report["selected_tokens"]) > caps["eval_tokens"]:
        raise DPOFullRunError("eval token cap exceeded")
    if eval_report["shortfalls"]:
        raise DPOFullRunError(
            "preference eval data shortfall: " + "; ".join(eval_report["shortfalls"])
        )
    if int(eval_report["selected_pairs"]) == 0:
        raise DPOFullRunError("preference eval set is empty")


def _assert_accepted_dpo_result(result: Any) -> None:
    if result.selected_eval.preference_accuracy <= result.base_eval.preference_accuracy:
        raise DPOFullRunError(
            "held-out preference accuracy did not improve versus the SFT reference"
        )
    if not result.margin_increased:
        raise DPOFullRunError(
            "held-out preference margin did not increase versus the SFT reference"
        )
    if result.chosen_logp_collapsed:
        raise DPOFullRunError("chosen response log-prob collapsed versus the SFT reference")


def _assert_prompt_masking(pairs: tuple[Any, ...]) -> bool:
    if not pairs:
        raise DPOFullRunError("no selected preference pairs to assert prompt masking on")
    for pair in pairs:
        for completion in (pair.chosen, pair.rejected):
            if not all(label == -100 for label in completion.labels[: completion.prompt_tokens]):
                raise DPOFullRunError(f"{pair.row_id}: prompt span leaked into a completion loss")
            if completion.response_supervised_tokens <= 0:
                raise DPOFullRunError(
                    f"{pair.row_id}: completion has no supervised response tokens"
                )
    return True


def _assert_required_artifacts(output_dir: Path) -> None:
    missing = [name for name in EXPECTED_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        raise DPOFullRunError("missing required DPO artifacts: " + ", ".join(missing))


def _select_device(*, require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise DPOFullRunError("Modal DPO requires CUDA, but torch.cuda.is_available() is false")
    return torch.device("cpu")
