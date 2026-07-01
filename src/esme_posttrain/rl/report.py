"""Countdown-Lite GRPO decision report writers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from esme_posttrain.rl.launch import (
    FULL_EVAL_PROFILE,
    FULL_RUN_SPEND_CAP_USD,
    RLVRLaunchConfig,
    build_eval_profile,
)


def write_grpo_report_artifacts(
    config: RLVRLaunchConfig,
    run_payload: dict[str, Any],
    *,
    modal_app: str | None = None,
    modal_call_id: str | None = None,
    launch_command: str | None = None,
) -> dict[str, Any]:
    report = build_grpo_report(
        config,
        run_payload,
        modal_app=modal_app,
        modal_call_id=modal_call_id,
        launch_command=launch_command,
    )
    write_grpo_report(config, report)
    return report


def write_grpo_report(config: RLVRLaunchConfig, report: dict[str, Any]) -> None:
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config.doc_path.parent.mkdir(parents=True, exist_ok=True)
    config.doc_path.write_text(_markdown_report(report), encoding="utf-8")


def build_blocked_grpo_report(
    config: RLVRLaunchConfig,
    *,
    reason: str,
    spend_evidence: dict[str, Any],
    modal_evidence: dict[str, Any],
    launch_command: str | None = None,
    status: str = "blocked-with-evidence",
    ready_for_hq_inspection: bool = True,
) -> dict[str, Any]:
    _validate_spend_evidence(spend_evidence)
    _validate_modal_evidence(modal_evidence)
    payload = {
        "status": status,
        "grpo_result": "blocked-with-evidence",
        "blocker": reason,
        "before": _baseline_from_acceptance(config),
        "after": None,
        "cost": {
            "paid_compute": bool(spend_evidence["paid_compute"]),
            "estimated_cost_usd": float(spend_evidence["actual_or_estimated_cost_usd"]),
            "cost_basis": spend_evidence["cost_basis"],
            "timeout_cost_ceiling_usd": spend_evidence.get("timeout_cost_ceiling_usd"),
        },
        "paid_compute": bool(spend_evidence["paid_compute"]),
        "modal_app": modal_evidence.get("app"),
        "modal_app_id": modal_evidence.get("app_id"),
        "modal_call_id": modal_evidence.get("call_id"),
        "modal_logs_command": modal_evidence.get("logs_command"),
        "modal_call_logs_command": modal_evidence.get("call_logs_command"),
        "modal_stop_command": modal_evidence.get("stop_command"),
        "modal_status_command": modal_evidence.get("status_command"),
        "modal_status_basis": modal_evidence["status_basis"],
        "post_stop_status": modal_evidence.get("post_stop_status"),
        "ready_for_hq_inspection": ready_for_hq_inspection,
        "gsm8k_lite": {
            "status": "not_run",
            "reason": "GRPO run did not complete, so transfer eval was not meaningful",
        },
    }
    return build_grpo_report(config, payload, launch_command=launch_command)


def build_grpo_report(
    config: RLVRLaunchConfig,
    run_payload: dict[str, Any],
    *,
    modal_app: str | None = None,
    modal_call_id: str | None = None,
    launch_command: str | None = None,
) -> dict[str, Any]:
    before = run_payload.get("before") or _baseline_from_acceptance(config)
    after = run_payload.get("after")
    eval_profile = str(
        run_payload.get("eval_profile")
        or (before.get("eval_profile") if isinstance(before, dict) else None)
        or FULL_EVAL_PROFILE
    )
    result = str(run_payload.get("grpo_result", "blocked-with-evidence"))
    if result == "blocked-with-evidence" and not isinstance(run_payload.get("cost"), dict):
        raise ValueError("blocked GRPO reports require cost evidence")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": config.run_id,
        "artifact_name": config.artifact_name,
        "mode": str(run_payload.get("mode") or _mode_from_payload(run_payload, eval_profile)),
        "status": run_payload.get("status", result),
        "pipeline_smoke": bool(run_payload.get("pipeline_smoke", False)),
        "eval_profile": eval_profile,
        "lifecycle_milestones": tuple(run_payload.get("lifecycle_milestones", ())),
        "will_start_modal_job": run_payload.get("will_start_modal_job"),
        "modal_gpu_or_paid_work_started": run_payload.get("modal_gpu_or_paid_work_started"),
        "online_wandb": run_payload.get("online_wandb"),
        "paid_api": run_payload.get("paid_api"),
        "remote_dataset_download": run_payload.get("remote_dataset_download"),
        "grpo_result": result,
        "recommendation": _recommendation(result, before, after),
        "ready_for_hq_inspection": _ready_for_hq_inspection(result, run_payload),
        "blocker": run_payload.get("blocker"),
        "input_bundle_path": str(config.input_bundle_path),
        "dataset_manifest_path": str(config.dataset_manifest_path),
        "sample_budget": config.budgets["dataset_sample_budget"],
        "token_budget": config.budgets["max_rollout_tokens"],
        "estimated_train_tokens": config.estimated_train_tokens,
        "hardware": {
            "provider": config.runtime["provider"],
            "selected_gpu": config.runtime["selected_gpu"],
            "modal_gpu": config.selected_gpu_profile["modal_gpu"],
            "expected_duration_minutes": round(
                config.estimated_train_tokens
                / float(config.selected_gpu_profile["projected_tokens_per_second"])
                / 60.0,
                2,
            ),
        },
        "spend": {
            "cap_usd": FULL_RUN_SPEND_CAP_USD,
            "projected_cost_usd": config.estimated_full_cost_usd,
            "actual_or_estimated_cost_usd": _actual_cost(run_payload),
            "runtime_spend_stop_usd": config.runtime["full_run_runtime_spend_stop_usd"],
            "paid_compute": bool(run_payload.get("paid_compute", False)),
            "cost_basis": _cost_basis(run_payload),
            "timeout_cost_ceiling_usd": _timeout_cost_ceiling(run_payload),
        },
        "modal": {
            "app": modal_app or run_payload.get("modal_app"),
            "app_id": run_payload.get("modal_app_id"),
            "call_id": modal_call_id or run_payload.get("modal_call_id"),
            "logs_command": run_payload.get("modal_logs_command"),
            "call_logs_command": run_payload.get("modal_call_logs_command"),
            "launch_command": launch_command or config.full_launch_command,
            "stop_command": run_payload.get("modal_stop_command"),
            "status_command": run_payload.get("modal_status_command"),
            "status_basis": run_payload.get("modal_status_basis"),
            "post_stop_status": run_payload.get("post_stop_status"),
        },
        "resume_command": run_payload.get("resume_command") or launch_command,
        "wandb": {
            "project": config.payload["monitoring"].get("wandb_project"),
            "required_for_modal": config.payload["monitoring"].get("wandb_required_for_modal"),
            "tags": config.payload["monitoring"].get("wandb_tags"),
            "eval_profile": eval_profile,
            "run": run_payload.get("wandb_run"),
        },
        "paths": {
            "output_dir": run_payload.get("output_dir"),
            "bundle_dir": run_payload.get("bundle_dir"),
            "report_path": str(config.report_path),
            "doc_path": str(config.doc_path),
            "before_report": run_payload.get("before_report"),
            "after_report": run_payload.get("after_report"),
        },
        "before": before,
        "after": after,
        "gsm8k_lite": run_payload.get(
            "gsm8k_lite",
            {
                "status": "not_run",
                "reason": "no checked-in GSM8K-lite fixture or evaluator exists in esme-posttrain",
            },
        ),
        "trainer": run_payload.get("trainer"),
        "required_artifacts_present": run_payload.get("required_artifacts_present"),
    }


def _baseline_from_acceptance(config: RLVRLaunchConfig) -> dict[str, Any]:
    acceptance = config.payload["acceptance"]
    eval_profile = build_eval_profile(config)
    return {
        "eval_profile": eval_profile["profile"],
        "pass@1": acceptance["baseline_pass_at_1"],
        "pass@8": acceptance["baseline_pass_at_8"],
        "pass@32": acceptance["baseline_pass_at_32"],
        "valid_expression_rate": acceptance["baseline_valid_expression_rate"],
        "exact_solve_rate": acceptance["baseline_exact_solve_rate"],
        "task_count": eval_profile["tasks"],
        "samples_per_task": eval_profile["samples_per_task"],
    }


def _mode_from_payload(run_payload: dict[str, Any], eval_profile: str) -> str:
    if bool(run_payload.get("pipeline_smoke")):
        return "pipeline_smoke"
    if (
        run_payload.get("debug_before_eval")
        or run_payload.get("grpo_result") == "before-eval-probe"
    ):
        return "before_eval_probe"
    if eval_profile == FULL_EVAL_PROFILE:
        return "full_acceptance"
    return "grpo"


def _actual_cost(run_payload: dict[str, Any]) -> float:
    cost = run_payload.get("cost")
    if isinstance(cost, dict):
        value = cost.get("estimated_cost_usd")
        if isinstance(value, int | float):
            return float(value)
    value = run_payload.get("projected_cost_usd")
    return float(value) if isinstance(value, int | float) else 0.0


def _cost_basis(run_payload: dict[str, Any]) -> str | None:
    cost = run_payload.get("cost")
    if isinstance(cost, dict) and isinstance(cost.get("cost_basis"), str):
        return str(cost["cost_basis"])
    value = run_payload.get("cost_basis")
    return str(value) if isinstance(value, str) else None


def _timeout_cost_ceiling(run_payload: dict[str, Any]) -> float | None:
    cost = run_payload.get("cost")
    if isinstance(cost, dict):
        value = cost.get("timeout_cost_ceiling_usd")
        if isinstance(value, int | float):
            return float(value)
    return None


def _validate_spend_evidence(spend_evidence: dict[str, Any]) -> None:
    required = {"paid_compute", "actual_or_estimated_cost_usd", "cost_basis"}
    missing = sorted(required - set(spend_evidence))
    if missing:
        raise ValueError("blocked GRPO reports require spend evidence: " + ", ".join(missing))
    if not isinstance(spend_evidence["paid_compute"], bool):
        raise ValueError("spend_evidence.paid_compute must be a boolean")
    cost = spend_evidence["actual_or_estimated_cost_usd"]
    if isinstance(cost, bool) or not isinstance(cost, int | float) or cost < 0:
        raise ValueError("spend_evidence.actual_or_estimated_cost_usd must be non-negative")
    if not isinstance(spend_evidence["cost_basis"], str) or not spend_evidence["cost_basis"]:
        raise ValueError("spend_evidence.cost_basis must be a non-empty string")
    timeout_ceiling = spend_evidence.get("timeout_cost_ceiling_usd")
    if timeout_ceiling is not None and (
        isinstance(timeout_ceiling, bool)
        or not isinstance(timeout_ceiling, int | float)
        or timeout_ceiling < 0
    ):
        raise ValueError("spend_evidence.timeout_cost_ceiling_usd must be non-negative")


def _validate_modal_evidence(modal_evidence: dict[str, Any]) -> None:
    required = {"app", "app_id", "call_id", "stop_command", "post_stop_status", "status_basis"}
    missing = sorted(required - set(modal_evidence))
    if missing:
        raise ValueError(
            "blocked GRPO reports require Modal/status evidence: " + ", ".join(missing)
        )
    if not isinstance(modal_evidence["app"], str) or not modal_evidence["app"]:
        raise ValueError("modal_evidence.app must be a non-empty string")
    for key in ("app_id", "call_id", "logs_command", "call_logs_command", "stop_command"):
        value = modal_evidence.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"modal_evidence.{key} must be a string or null")
    post_stop_status = modal_evidence["post_stop_status"]
    if post_stop_status is not None and not isinstance(post_stop_status, dict):
        raise ValueError("modal_evidence.post_stop_status must be an object or null")
    if not isinstance(modal_evidence["status_basis"], str) or not modal_evidence["status_basis"]:
        raise ValueError("modal_evidence.status_basis must be a non-empty string")


def _ready_for_hq_inspection(result: str, run_payload: dict[str, Any]) -> bool:
    if "ready_for_hq_inspection" in run_payload:
        return bool(run_payload["ready_for_hq_inspection"])
    if result in {
        "RLVR-improved",
        "no-improvement",
        "not-acceptance-evidence",
        "pipeline_smoke_passed",
    }:
        return True
    return bool(run_payload.get("blocker") and run_payload.get("cost"))


def _recommendation(result: str, before: Any, after: Any) -> str:
    if result == "pipeline_smoke_passed":
        return "review smoke evidence, then decide full acceptance"
    if result == "not-acceptance-evidence":
        return "rerun the full acceptance eval profile before claiming improvement"
    if result == "RLVR-improved":
        return "continue curriculum"
    if result == "no-improvement":
        if isinstance(before, dict) and isinstance(after, dict):
            before_valid = float(before.get("valid_expression_rate", 0.0))
            after_valid = float(after.get("valid_expression_rate", 0.0))
            if after_valid <= before_valid:
                return "add tiny SFT/hint warm-start"
        return "continue curriculum"
    return "pause RL capstone"


def _markdown_report(report: dict[str, Any]) -> str:
    before = report["before"]
    after = report.get("after")
    after = after if isinstance(after, dict) else {}
    is_pipeline_check = bool(report.get("pipeline_smoke"))
    title = (
        "# RLVR Countdown-Lite Pipeline Check"
        if is_pipeline_check
        else "# RLVR Countdown-Lite GRPO"
    )
    lines = [
        title,
        "",
        f"- Result: `{_display_result(report['grpo_result'])}`",
        f"- Eval profile: `{report['eval_profile']}`",
        f"- Recommendation: `{_display_recommendation(report['recommendation'])}`",
        (
            f"- Spend: `${float(report['spend']['actual_or_estimated_cost_usd']):.4f}` "
            f"actual/estimated against `${float(report['spend']['cap_usd']):.2f}` cap"
        ),
        f"- GSM8K-lite: `{report['gsm8k_lite']['status']}` - {report['gsm8k_lite']['reason']}",
        "",
        "## Countdown-Lite Before/After",
        "",
        "| Metric | Before | After |",
        "| --- | ---: | ---: |",
        f"| pass@1 | {_percent(before.get('pass@1'))} | {_percent(after.get('pass@1'))} |",
        f"| pass@8 | {_percent(before.get('pass@8'))} | {_percent(after.get('pass@8'))} |",
        f"| pass@32 | {_percent(before.get('pass@32'))} | {_percent(after.get('pass@32'))} |",
        "| valid-expression rate | "
        f"{_percent(before.get('valid_expression_rate'))} | "
        f"{_percent(after.get('valid_expression_rate'))} |",
        "| exact-solve rate | "
        f"{_percent(before.get('exact_solve_rate'))} | "
        f"{_percent(after.get('exact_solve_rate'))} |",
        "",
        "## Bounds",
        "",
        f"- Dataset: `{report['dataset_manifest_path']}`",
        f"- Sample budget: `{report['sample_budget']}`",
        f"- Token budget: `{report['token_budget']}`",
        f"- Estimated train tokens: `{report['estimated_train_tokens']}`",
        f"- Hardware: `{report['hardware']['modal_gpu']}`",
        f"- Runtime hard stop: `${float(report['spend']['runtime_spend_stop_usd']):.2f}`",
        "",
    ]
    if is_pipeline_check:
        lines.extend(
            [
                "## Scope",
                "",
                (
                    "This is an internal no-spend lifecycle check. "
                    "It is not acceptance evidence for `Esme-214M-RL`."
                ),
                "",
            ]
        )
    if report.get("blocker"):
        lines.extend(["## Blocker", "", str(report["blocker"]), ""])
    return "\n".join(lines)


def _display_recommendation(value: Any) -> str:
    if value == "review smoke evidence, then decide full acceptance":
        return "review internal check, then decide full acceptance"
    return str(value)


def _display_result(value: Any) -> str:
    if value == "pipeline_smoke_passed":
        return "pipeline check passed"
    return str(value)


def _percent(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value) * 100:.2f}%"
    return "n/a"
