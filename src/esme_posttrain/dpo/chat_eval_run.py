"""Approval-gated driver for the SFT-vs-DPO chat-quality eval.

Generation-only (no training), so it is cheap -- the spend cap is a hard $1. The
preflight never starts Modal (`will_start_modal_job:false`); the actual run loads
the DPO best checkpoint and the SFT reference from their Volumes (read-only),
generates side by side, and writes the comparison markdown + JSON to the DPO
Volume.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.dpo.chat_eval import (
    run_chat_eval,
    write_chat_eval_json,
    write_chat_eval_markdown,
)
from esme_posttrain.dpo.launch import DPOLaunchConfig
from esme_posttrain.launch.config_guards import LAUNCH_APPROVAL_FLAG, MODAL_CLIENT_VERSION

CHAT_EVAL_SPEND_CAP_USD = 1.0
CHAT_EVAL_OUTPUT_STEM = "esme-214m-chat-dpo-full"
CHAT_EVAL_MARKDOWN = "chat-eval-sft-vs-dpo.md"
CHAT_EVAL_JSON = "chat-eval-sft-vs-dpo.json"
# Best checkpoint produced by the full DPO run (step 600), on the DPO Volume.
DPO_BEST_CHECKPOINT_REL = "esme-214m-chat-dpo-full/best-checkpoint.pt"
CHAT_EVAL_MAX_NEW_TOKENS = 96
# Generation over ~8 short prompts x 2 decoders x 2 models is a few minutes; a
# 0.25h (15 min) Modal timeout keeps the worst-case cost ceiling under the $1 cap
# (15 min x $2.10/h ~= $0.52). The actual elapsed-cost guard still enforces $1.
CHAT_EVAL_TIMEOUT_HOURS = 0.25


class ChatEvalError(RuntimeError):
    pass


def build_chat_eval_preflight(
    config: DPOLaunchConfig, *, modal_gpu: str, timeout_hours: float = CHAT_EVAL_TIMEOUT_HOURS
) -> dict[str, Any]:
    selected_profile = config.selected_gpu_profile
    timeout_cost_ceiling = float(selected_profile["usd_per_hour"]) * timeout_hours
    blockers = chat_eval_blockers(config, modal_gpu=modal_gpu, timeout_hours=timeout_hours)
    sft_reference = config.payload["sft_reference"]
    return {
        "status": "ready_for_chat_eval" if not blockers else "blocked_by_launch_safety",
        "mode": "dpo_chat_eval",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "will_start_modal_job": False,
        "will_download_data": False,
        "generation_only": True,
        "volume": config.runtime["modal_volume"],
        "sft_reference_volume": sft_reference["volume"],
        "checkpoints": {
            "dpo_best": DPO_BEST_CHECKPOINT_REL,
            "sft_reference": sft_reference["checkpoint_path"],
        },
        "outputs": {
            "markdown": f"{CHAT_EVAL_OUTPUT_STEM}/{CHAT_EVAL_MARKDOWN}",
            "json": f"{CHAT_EVAL_OUTPUT_STEM}/{CHAT_EVAL_JSON}",
        },
        "runtime": {
            "provider": "modal",
            "selected_gpu": config.runtime["selected_gpu"],
            "modal_gpu": modal_gpu,
            "timeout_hours": timeout_hours,
            "spend_cap_usd": CHAT_EVAL_SPEND_CAP_USD,
            "timeout_cost_ceiling_usd": round(timeout_cost_ceiling, 4),
            "max_new_tokens": CHAT_EVAL_MAX_NEW_TOKENS,
        },
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION},
        "launch_blockers": blockers,
        "chat_eval_command": chat_eval_command(
            config.config_path, gpu=modal_gpu, timeout_hours=timeout_hours
        ),
    }


def chat_eval_blockers(
    config: DPOLaunchConfig, *, modal_gpu: str, timeout_hours: float = CHAT_EVAL_TIMEOUT_HOURS
) -> list[str]:
    blockers: list[str] = []
    if timeout_hours <= 0 or timeout_hours > 24:
        blockers.append("chat-eval timeout_hours must be between 0 and 24")
    if modal_gpu != config.runtime["selected_gpu"]:
        blockers.append(
            "DPO_MODAL_GPU must match runtime.selected_gpu for chat-eval cost accounting"
        )
    selected_profile = config.selected_gpu_profile
    if timeout_hours * float(selected_profile["usd_per_hour"]) > CHAT_EVAL_SPEND_CAP_USD:
        blockers.append("chat-eval timeout cost ceiling exceeds the $1 chat-eval spend cap")
    return blockers


def chat_eval_command(config_path: Path, *, gpu: str, timeout_hours: float) -> str:
    return (
        f"DPO_MODAL_GPU='{gpu}' DPO_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} modal run "
        f"scripts/modal_chat_dpo.py --config {config_path.as_posix()} "
        f"--chat-eval {LAUNCH_APPROVAL_FLAG} --json"
    )


def run_chat_eval_job(
    config: DPOLaunchConfig,
    *,
    output_dir: Path,
    dpo_checkpoint_path: Path,
    sft_checkpoint_path: Path,
    sft_tokenizer_path: Path,
    require_cuda: bool,
    usd_per_hour: float,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
) -> dict[str, Any]:
    started = started or time.perf_counter()
    device = _select_device(require_cuda=require_cuda)
    if not dpo_checkpoint_path.is_file():
        raise ChatEvalError(f"DPO checkpoint not found: {dpo_checkpoint_path}")
    if not sft_checkpoint_path.is_file():
        raise ChatEvalError(f"SFT checkpoint not found: {sft_checkpoint_path}")
    tokenizer = Tokenizer.from_file(str(sft_tokenizer_path))
    comparison = run_chat_eval(
        sft_checkpoint_path,
        dpo_checkpoint_path,
        tokenizer,
        device=device,
        max_new_tokens=CHAT_EVAL_MAX_NEW_TOKENS,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / CHAT_EVAL_MARKDOWN
    json_path = output_dir / CHAT_EVAL_JSON
    write_chat_eval_markdown(comparison, markdown_path)
    write_chat_eval_json(comparison, json_path)

    elapsed = time.perf_counter() - started
    estimated_cost = elapsed * usd_per_hour / 3600.0
    if estimated_cost > CHAT_EVAL_SPEND_CAP_USD:
        raise ChatEvalError(
            f"chat-eval exceeded the ${CHAT_EVAL_SPEND_CAP_USD:.0f} cap "
            f"(estimated ${estimated_cost:.4f})"
        )
    (output_dir / "chat-eval-environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version.split()[0]}",
                f"platform={platform.platform()}",
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
                f"selected_device={device.type}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "chat_eval_complete",
        "run_id": config.run_id,
        "device": device.type,
        "paid_compute": True,
        "commit": commit,
        "dirty": dirty,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": estimated_cost,
        "spend_cap_usd": CHAT_EVAL_SPEND_CAP_USD,
        "markdown_path": str(markdown_path),
        "json_path": str(json_path),
        "summary_comparison": comparison.to_dict()["summary_comparison"],
        "degeneration_flags": comparison.to_dict()["degeneration_flags"],
    }


def _select_device(*, require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise ChatEvalError("Modal chat-eval requires CUDA, but torch.cuda.is_available() is false")
    return torch.device("cpu")
