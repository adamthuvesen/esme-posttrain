"""Bounded beta sweep {0.1, 0.3, 0.5} for the Esme-214M-Chat DPO stage.

The 0.5 beta in the SmolLM2-360M anchor rests on a single SmolLM2-vs-Zephyr data
point, so the full-run learning gate requires a real sweep: train one short DPO
arm per beta on UltraFeedback, evaluate held-out preference accuracy, and select
the beta whose held-out preference accuracy beats the SFT reference AND whose
chosen-logp does not collapse (meaningful likelihood displacement, i.e. a drop
beyond the relative tolerance — not eval noise) — never a single anchored point.
Emits the ``bounded_beta_sweep`` evidence the launcher's full-run gate checks.
Structurally mirrors ``sft_multiturn_sweep``.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.dpo.launch import EXPECTED_SWEEP_BETAS, DPOLaunchConfig
from esme_posttrain.dpo.trainer import (
    CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE,
    DPOTrainerConfig,
    run_dpo_training,
)
from esme_posttrain.launch.config_guards import (
    LAUNCH_APPROVAL_FLAG,
    MODAL_CLIENT_VERSION,
    estimate_cost_usd,
)
from esme_posttrain.run_artifacts import RuntimeSpendTracker, write_environment, write_json
from esme_posttrain.training.wandb_init import WandbConfig

SWEEP_OUTPUT_STEM = "esme-chat-dpo-beta-sweep"
SWEEP_GROUP = "esme_214m_chat_dpo_beta_sweep"
SWEEP_SPEND_CAP_USD = 8.0
SWEEP_TIMEOUT_HOURS = 3
SWEEP_TRAIN_PAIR_CAP = 512
SWEEP_TRAIN_TOKEN_CAP = 1_572_864
SWEEP_EVAL_PAIR_CAP = 128
SWEEP_EVAL_TOKEN_CAP = 393_216
SWEEP_MAX_STEPS = 120
DEFAULT_MODAL_SWEEP_ROOT = Path("/posttrain") / SWEEP_OUTPUT_STEM


class DPOSweepError(ValueError):
    pass


@dataclass(frozen=True)
class BetaArm:
    beta: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    eval_interval: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 60

    @property
    def name(self) -> str:
        return f"beta{self.beta}".replace(".", "p")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def planned_token_upper_bound(self, *, max_length: int) -> int:
        # Each optimizer step scores both chosen and rejected for the whole batch.
        return self.max_steps * self.effective_batch_size * max_length * 2

    def to_dict(self, *, max_length: int) -> dict[str, Any]:
        return {
            "arm_name": self.name,
            "beta": self.beta,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "eval_interval": self.eval_interval,
            "planned_token_upper_bound": self.planned_token_upper_bound(max_length=max_length),
        }


def beta_arms(*, betas: tuple[float, ...] = EXPECTED_SWEEP_BETAS) -> tuple[BetaArm, ...]:
    return tuple(
        BetaArm(
            beta=beta,
            micro_batch_size=2,
            gradient_accumulation_steps=8,
            max_steps=SWEEP_MAX_STEPS,
            warmup_steps=12,
        )
        for beta in betas
    )


SWEEP_ARMS: tuple[BetaArm, ...] = beta_arms()


def build_dpo_sweep_preflight(
    config: DPOLaunchConfig,
    *,
    timeout_hours: int = SWEEP_TIMEOUT_HOURS,
    modal_gpu: str | None = None,
) -> dict[str, Any]:
    selected_profile = config.selected_gpu_profile
    max_length = int(config.budgets["max_length"])
    projected_tokens = sum(
        arm.planned_token_upper_bound(max_length=max_length) for arm in SWEEP_ARMS
    )
    projected_cost = estimate_cost_usd(
        tokens=projected_tokens,
        projected_tokens_per_second=float(selected_profile["projected_tokens_per_second"]),
        usd_per_hour=float(selected_profile["usd_per_hour"]),
    )
    blockers = dpo_sweep_blockers(
        config,
        timeout_hours=timeout_hours,
        modal_gpu=modal_gpu or str(config.runtime["selected_gpu"]),
    )
    return {
        "status": "ready_for_modal_sweep" if not blockers else "blocked_by_launch_safety",
        "mode": "dpo_beta_sweep",
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "starts_from": config.payload["starts_from"],
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "will_start_modal_job": False,
        "will_download_data": False,
        "modal_run_will_download_real_data": True,
        "volume": config.runtime["modal_volume"],
        "volume_output_root": str(DEFAULT_MODAL_SWEEP_ROOT),
        "swept_betas": list(EXPECTED_SWEEP_BETAS),
        "arms": [arm.to_dict(max_length=max_length) for arm in SWEEP_ARMS],
        "datasets": config.payload["datasets"],
        "data_caps": {
            "train_pairs": SWEEP_TRAIN_PAIR_CAP,
            "train_tokens": SWEEP_TRAIN_TOKEN_CAP,
            "eval_pairs": SWEEP_EVAL_PAIR_CAP,
            "eval_tokens": SWEEP_EVAL_TOKEN_CAP,
            "max_length": max_length,
        },
        "runtime": {
            "provider": "modal",
            "selected_gpu": config.runtime["selected_gpu"],
            "modal_gpu": modal_gpu or config.runtime["selected_gpu"],
            "precision": config.runtime["precision"],
            "timeout_hours": timeout_hours,
            "sweep_spend_cap_usd": SWEEP_SPEND_CAP_USD,
            "timeout_cost_ceiling_usd": float(selected_profile["usd_per_hour"]) * timeout_hours,
            "projected_train_token_upper_bound": projected_tokens,
            "projected_cost_usd": round(projected_cost, 4),
        },
        "acceptance": {
            "gate": (
                "held-out preference accuracy beats the SFT reference for the best beta "
                "and chosen-logp does not collapse"
            ),
            "selector_metric": "eval/preference_accuracy",
            "displacement_guard": (
                "best_chosen_logp_collapsed must be false; collapse fires only on a "
                f">{CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE:.0%} relative drop in chosen "
                "log-prob, not eval-noise jitter"
            ),
        },
        "monitoring": {
            "wandb_project": config.payload["monitoring"]["wandb_project"],
            "wandb_tags": config.payload["monitoring"]["wandb_tags"],
            "group": SWEEP_GROUP,
        },
        "dependency_pins": {"modal": MODAL_CLIENT_VERSION},
        "launch_blockers": blockers,
        "modal_sweep_command": dpo_sweep_command(
            config.config_path, timeout_hours=timeout_hours, gpu=str(config.runtime["selected_gpu"])
        ),
    }


def dpo_sweep_blockers(config: DPOLaunchConfig, *, timeout_hours: int, modal_gpu: str) -> list[str]:
    blockers: list[str] = []
    runtime = config.runtime
    if timeout_hours <= 0 or timeout_hours > 24:
        blockers.append("DPO_SWEEP_TIMEOUT_HOURS must be between 1 and 24")
    if modal_gpu != runtime["selected_gpu"]:
        blockers.append("DPO_MODAL_GPU must match runtime.selected_gpu for sweep cost accounting")
    selected_profile = config.selected_gpu_profile
    if timeout_hours * float(selected_profile["usd_per_hour"]) > SWEEP_SPEND_CAP_USD:
        blockers.append("sweep timeout cost ceiling exceeds the approved $8 sweep spend cap")
    max_length = int(config.budgets["max_length"])
    projected_tokens = sum(
        arm.planned_token_upper_bound(max_length=max_length) for arm in SWEEP_ARMS
    )
    projected_cost = estimate_cost_usd(
        tokens=projected_tokens,
        projected_tokens_per_second=float(selected_profile["projected_tokens_per_second"]),
        usd_per_hour=float(selected_profile["usd_per_hour"]),
    )
    if projected_cost > SWEEP_SPEND_CAP_USD:
        blockers.append("projected beta sweep cost exceeds the approved $8 sweep spend cap")
    if tuple(arm.beta for arm in SWEEP_ARMS) != EXPECTED_SWEEP_BETAS:
        blockers.append("sweep arms must cover exactly the {0.1, 0.3, 0.5} betas")
    for arm in SWEEP_ARMS:
        if arm.eval_interval <= 0:
            blockers.append(f"{arm.name} must enable interval eval")
        if arm.max_steps > 200:
            blockers.append(f"{arm.name} exceeds the bounded sweep max_steps cap")
        if arm.effective_batch_size > 16:
            blockers.append(f"{arm.name} exceeds the bounded effective batch cap")
    return blockers


def dpo_sweep_command(config_path: Path, *, timeout_hours: int, gpu: str) -> str:
    return (
        f"DPO_MODAL_GPU='{gpu}' DPO_SWEEP_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} modal run "
        f"scripts/modal_chat_dpo.py --config {config_path.as_posix()} "
        f"--beta-sweep {LAUNCH_APPROVAL_FLAG} --json"
    )


def select_best_arm(arm_payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the beta whose held-out preference accuracy beats the reference and whose
    chosen-logp did not collapse; tie-break on higher preference accuracy."""
    eligible = [
        arm
        for arm in arm_payloads
        if arm["status"] == "complete"
        and arm["selected_eval"]["preference_accuracy"]
        > arm["reference_eval"]["preference_accuracy"]
        and not arm["chosen_logp_collapsed"]
    ]
    return max(eligible, key=lambda arm: arm["selected_eval"]["preference_accuracy"], default=None)


def learning_gate_payload(*, best_arm: dict[str, Any] | None, evidence_uri: str) -> dict[str, Any]:
    if best_arm is None:
        return {
            "status": "fail",
            "kind": "bounded_beta_sweep",
            "selector_metric": "eval/preference_accuracy",
            "swept_betas": list(EXPECTED_SWEEP_BETAS),
            "evidence_uri": evidence_uri,
            "blocker": (
                "no swept beta beat the SFT reference on held-out preference accuracy "
                "without chosen-logp collapse"
            ),
        }
    return {
        "status": "pass",
        "bounded_beta_sweep": {
            "kind": "bounded_beta_sweep",
            "selector_metric": "eval/preference_accuracy",
            "swept_betas": list(EXPECTED_SWEEP_BETAS),
            "best_beta": best_arm["arm"]["beta"],
            "reference_preference_accuracy": best_arm["reference_eval"]["preference_accuracy"],
            "best_preference_accuracy": best_arm["selected_eval"]["preference_accuracy"],
            "best_chosen_logp_collapsed": best_arm["chosen_logp_collapsed"],
            "evidence_uri": evidence_uri,
        },
    }


def run_dpo_beta_sweep(
    config: DPOLaunchConfig,
    *,
    output_root: Path,
    reference: Any,
    tokenizer: Any,
    train_pairs: tuple[Any, ...],
    eval_pairs: tuple[Any, ...],
    device: torch.device,
    usd_per_hour: float,
    wandb_enabled: bool,
    started: float | None = None,
    commit: str = "unknown",
    dirty: bool = True,
    make_policy: Any,
) -> dict[str, Any]:
    """Run the bounded beta sweep. ``make_policy`` returns a fresh warm-started
    policy (a copy of the SFT reference) per arm so arms are independent.
    """
    output_root = output_root.expanduser().resolve()
    if output_root.name != SWEEP_OUTPUT_STEM:
        raise DPOSweepError(f"sweep output root must end with {SWEEP_OUTPUT_STEM}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = started or time.perf_counter()
    spend_tracker = RuntimeSpendTracker(
        started=started,
        usd_per_hour=usd_per_hour,
        stop_usd=SWEEP_SPEND_CAP_USD,
        output_dir=output_root,
    )
    launch_id = _fresh_launch_id(output_root)
    evidence_dir = output_root / f"{launch_id}-evidence"
    evidence_dir.mkdir()
    max_length = int(config.budgets["max_length"])

    arm_payloads: list[dict[str, Any]] = []
    for arm in SWEEP_ARMS:
        arm_id = f"{launch_id}-{arm.name}"
        arm_output_dir = output_root / arm_id
        arm_output_dir.mkdir()
        try:
            arm_payloads.append(
                _run_sweep_arm(
                    config,
                    arm,
                    arm_id=arm_id,
                    output_dir=arm_output_dir,
                    policy=make_policy(),
                    reference=reference,
                    tokenizer=tokenizer,
                    train_pairs=train_pairs,
                    eval_pairs=eval_pairs,
                    device=device,
                    spend_tracker=spend_tracker,
                    wandb_enabled=wandb_enabled,
                    commit=commit,
                    dirty=dirty,
                )
            )
        except DPOSweepError as error:
            arm_payloads.append(
                _arm_failure_payload(arm, arm_id=arm_id, output_dir=arm_output_dir, error=error)
            )
            break
        except Exception as error:  # noqa: BLE001 - record and continue to the next arm.
            arm_payloads.append(
                _arm_failure_payload(arm, arm_id=arm_id, output_dir=arm_output_dir, error=error)
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    best_arm = select_best_arm(arm_payloads)
    status = "beta_sweep_passed" if best_arm is not None else "beta_sweep_failed"
    cost = spend_tracker.write_cost(
        step=max((int(arm.get("steps_completed", 0)) for arm in arm_payloads), default=0),
        status=status,
    )
    gate = learning_gate_payload(
        best_arm=best_arm, evidence_uri=str(evidence_dir / "beta-sweep.json")
    )
    payload = {
        "status": status,
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "mode": "bounded_beta_sweep",
        "launch_id": launch_id,
        "output_root": str(output_root),
        "evidence_dir": str(evidence_dir),
        "beta_sweep_path": str(evidence_dir / "beta-sweep.json"),
        "learning_gate_path": str(evidence_dir / "learning-gate.json"),
        "volume": config.runtime["modal_volume"],
        "commit": commit,
        "dirty": dirty,
        "device": device.type,
        "paid_compute": True,
        "cost": cost,
        "spend_cap_usd": SWEEP_SPEND_CAP_USD,
        "swept_betas": list(EXPECTED_SWEEP_BETAS),
        "arms": arm_payloads,
        "selected_best_arm": best_arm["arm_id"] if best_arm is not None else None,
        "selected_best_beta": best_arm["arm"]["beta"] if best_arm is not None else None,
        "max_length": max_length,
        "learning_gate": gate,
    }
    write_json(evidence_dir / "beta-sweep.json", payload)
    write_json(evidence_dir / "learning-gate.json", gate)
    return payload


def _run_sweep_arm(
    config: DPOLaunchConfig,
    arm: BetaArm,
    *,
    arm_id: str,
    output_dir: Path,
    policy: Any,
    reference: Any,
    tokenizer: Any,
    train_pairs: tuple[Any, ...],
    eval_pairs: tuple[Any, ...],
    device: torch.device,
    spend_tracker: RuntimeSpendTracker,
    wandb_enabled: bool,
    commit: str,
    dirty: bool,
) -> dict[str, Any]:
    optimizer_config = config.optimizer
    monitoring_config = config.payload["monitoring"]
    write_environment(output_dir / "environment.txt", device=device)
    arm_started_cost = spend_tracker.estimated_cost_usd()
    result = run_dpo_training(
        policy,
        reference,
        tokenizer,
        train_pairs,
        eval_pairs,
        DPOTrainerConfig(
            max_steps=arm.max_steps,
            micro_batch_size=arm.micro_batch_size,
            gradient_accumulation_steps=arm.gradient_accumulation_steps,
            learning_rate=float(optimizer_config["learning_rate"]),
            beta=arm.beta,
            length_normalized=bool(config.payload["dpo"]["length_normalized"]),
            scheduler=str(optimizer_config["scheduler"]),
            warmup_steps=arm.warmup_steps,
            weight_decay=float(optimizer_config["weight_decay"]),
            precision=str(config.runtime["precision"]),
            pad_to_multiple_of=config.payload["sequence"]["pad_to_multiple_of"],
            seed=int(optimizer_config["seed"]),
            output_dir=output_dir,
            artifact_name=config.artifact_name,
            grad_clip=float(optimizer_config["grad_clip"]),
            log_interval=arm.log_interval,
            eval_interval=arm.eval_interval,
            checkpoint_interval=arm.checkpoint_interval,
            device=device.type,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(monitoring_config["wandb_project"]),
                run_name=f"{SWEEP_GROUP}-{arm_id}",
                tags=tuple(monitoring_config["wandb_tags"]) + ("beta-sweep", f"beta:{arm.beta}"),
                group=SWEEP_GROUP,
                job_type="sweep",
                notes="Bounded DPO beta sweep; no full launch, no SimPO, no RL.",
                extra_config={
                    "model": config.artifact_name,
                    "stage": "dpo",
                    "run_type": "beta_sweep",
                    "beta": arm.beta,
                    "length_normalized": bool(config.payload["dpo"]["length_normalized"]),
                    "arm_id": arm_id,
                },
            ),
        ),
        reference_bundle_manifest={"mode": "modal_dpo_beta_sweep", "arm_id": arm_id},
        step_callback=lambda step: spend_tracker.check_cap(
            step,
            label="DPO beta sweep",
            error_type=DPOSweepError,
        ),
    )
    cost = spend_tracker.write_cost(step=result.steps_completed, status="arm_complete")
    estimated_arm_cost = max(0.0, cost["estimated_cost_usd"] - arm_started_cost)
    payload = {
        "status": "complete",
        "arm_id": arm_id,
        "arm": arm.to_dict(max_length=int(config.budgets["max_length"])),
        "output_dir": str(output_dir),
        "metrics_path": str(result.metrics_path),
        "wandb_run": result.wandb_run_url,
        "steps_completed": result.steps_completed,
        "reference_eval": result.base_eval.to_dict(),
        "selected_eval": result.selected_eval.to_dict(),
        "selected_step": result.selected_step,
        "margin_increased": result.margin_increased,
        "chosen_logp_collapsed": result.chosen_logp_collapsed,
        "cost": {**cost, "estimated_arm_cost_usd": estimated_arm_cost},
    }
    write_json(output_dir / "arm-summary.json", payload)
    return payload


def _arm_failure_payload(
    arm: BetaArm, *, arm_id: str, output_dir: Path, error: Exception
) -> dict[str, Any]:
    payload = {
        "status": "failed",
        "arm_id": arm_id,
        "arm": arm.to_dict(max_length=1024),
        "output_dir": str(output_dir),
        "error": str(error),
        "chosen_logp_collapsed": True,
    }
    write_json(output_dir / "arm-summary.json", payload)
    return payload


def _fresh_launch_id(output_root: Path) -> str:
    base = time.strftime("sweep-%Y%m%dT%H%M%SZ", time.gmtime())
    for suffix in ("", *[f"-{index}" for index in range(1, 100)]):
        candidate = f"{base}{suffix}"
        paths = [output_root / f"{candidate}-evidence"] + [
            output_root / f"{candidate}-{arm.name}" for arm in SWEEP_ARMS
        ]
        if all(not path.exists() for path in paths):
            return candidate
    raise DPOSweepError(f"could not find an isolated sweep launch id under {output_root}")
