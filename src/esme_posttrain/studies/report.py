"""Build JSON and Markdown study reports from immutable completion artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from esme_posttrain.rl.countdown_lite import (
    VerificationResult,
    verify_countdown_lite_expression,
)
from esme_posttrain.studies.models import (
    ArmRole,
    StudyMetric,
    StudyRunSpec,
    StudySpecification,
)


class StudyReportError(ValueError):
    """A study input is missing, corrupt, malformed, or unsafe to compare."""


@dataclass(frozen=True)
class GeneratedStudyReport:
    json_path: Path
    markdown_path: Path
    payload: dict[str, Any]


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
# 2**20 sign assignments is about a million; exact enumeration stays cheap below that.
MAX_PERMUTATION_SEEDS = 20

ProblemMetrics = dict[str, dict[str, float]]


def load_study_specification(path: Path) -> StudySpecification:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return StudySpecification.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise StudyReportError(f"invalid study specification {path}: {error}") from error


def generate_study_report(
    study_path: Path,
    *,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> GeneratedStudyReport:
    study_path = study_path.expanduser().resolve()
    root = (repo_root or Path.cwd()).expanduser().resolve()
    specification = load_study_specification(study_path)
    output_dir = (output_dir or study_path.parent).expanduser().resolve()

    arm_results: dict[str, dict[str, Any]] = {}
    compatibility_records: list[tuple[str, int, dict[str, Any]]] = []
    artifact_records: list[dict[str, Any]] = []
    missing_seeds: dict[str, list[int]] = {}
    problem_metrics: dict[str, dict[int, ProblemMetrics]] = {}

    for arm in specification.arms:
        run_results = []
        problem_metrics[arm.name] = {}
        for run in arm.runs:
            result, per_problem, compatibility, artifacts = _read_run(run, root)
            run_results.append(result)
            problem_metrics[arm.name][run.seed] = per_problem
            compatibility_records.append((arm.name, run.seed, compatibility))
            artifact_records.extend(artifacts)
        expected = set(specification.included_seeds)
        actual = {run.seed for run in arm.runs}
        if arm.role in {ArmRole.TREATMENT, ArmRole.CONTROL} and actual != expected:
            missing_seeds[arm.name] = sorted(expected - actual)
        arm_results[arm.name] = _aggregate_arm(arm.role, run_results)

    comparisons = _build_comparisons(specification, arm_results, problem_metrics)
    compatibility_errors = _compatibility_errors(specification, compatibility_records, arm_results)
    reference_errors, reference_artifacts = _reference_summary_errors(
        specification,
        root,
        arm_results,
        comparisons,
    )
    compatibility_errors.extend(reference_errors)
    artifact_records.extend(reference_artifacts)
    evidence_warnings = _evidence_warnings(specification)
    verdict, verdict_reasons = _verdict(
        specification,
        comparisons,
        missing_seeds=missing_seeds,
        compatibility_errors=compatibility_errors,
    )
    payload = {
        "schema_version": 1,
        "study_id": specification.study_id,
        "study_specification_sha256": hashlib.sha256(study_path.read_bytes()).hexdigest(),
        "title": specification.title,
        "hypothesis": specification.hypothesis,
        "primary_metric": specification.primary_metric.value,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "complete": not missing_seeds,
        "missing_seeds": missing_seeds,
        "compatible": not compatibility_errors,
        "compatibility_errors": compatibility_errors,
        "evidence_warnings": evidence_warnings,
        "confidence_interval_method": specification.confidence_interval_method.value,
        "inference": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "permutation_test": "exact two-sided sign-flip over seed-level paired effects",
        },
        "sample_budget": specification.sample_budget,
        "decoding": specification.decoding.model_dump(mode="json"),
        "task_manifest_ids": list(specification.task_manifest_ids),
        "arms": arm_results,
        "comparisons": comparisons,
        "allowed_claims": list(specification.allowed_claims),
        "supported_claims": list(specification.allowed_claims) if verdict == "accepted" else [],
        "excluded_runs": [run.model_dump(mode="json") for run in specification.excluded_runs],
        "provenance": artifact_records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = study_path.stem
    json_path = output_dir / f"{stem}.report.json"
    markdown_path = output_dir / f"{stem}.report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(
        _render_markdown(payload, json_filename=json_path.name), encoding="utf-8"
    )
    return GeneratedStudyReport(json_path=json_path, markdown_path=markdown_path, payload=payload)


def _read_run(
    run: StudyRunSpec, root: Path
) -> tuple[dict[str, Any], ProblemMetrics, dict[str, Any], list[dict[str, Any]]]:
    completions_path = _artifact_path(run.completions.path, root)
    provenance_path = _artifact_path(run.provenance.path, root)
    artifacts = [
        _check_artifact(completions_path, run.completions.sha256, root),
        _check_artifact(provenance_path, run.provenance.sha256, root),
    ]
    provenance = _read_json(provenance_path)
    rows = _read_jsonl(completions_path)
    metrics, per_problem = _score_completion_rows(rows)
    training = {
        "status": None,
        "selected_step": None,
        "selected_metric_name": None,
        "selected_metric_value": None,
        "elapsed_seconds": None,
        "estimated_cost_usd": None,
        "training_token_budget": run.training_token_budget,
    }
    if run.training_manifest is not None:
        path = _artifact_path(run.training_manifest.path, root)
        artifacts.append(_check_artifact(path, run.training_manifest.sha256, root))
        manifest = _read_json(path)
        training.update(
            {
                "selected_step": manifest.get("selected_step"),
                "selected_metric_name": manifest.get("selected_metric_name"),
                "selected_metric_value": manifest.get("selected_metric_value"),
            }
        )
    if run.cost is not None:
        path = _artifact_path(run.cost.path, root)
        artifacts.append(_check_artifact(path, run.cost.sha256, root))
        cost = _read_json(path)
        training.update(
            {
                "status": cost.get("status"),
                "elapsed_seconds": cost.get("elapsed_seconds"),
                "estimated_cost_usd": cost.get("estimated_cost_usd"),
            }
        )
    return (
        {"seed": run.seed, "metrics": metrics, "training": training},
        per_problem,
        _compatibility_projection(provenance),
        artifacts,
    )


def _score_completion_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float | int], ProblemMetrics]:
    if not rows:
        raise StudyReportError("completion artifact contains no rows")
    valid = 0
    exact = 0
    any_exact = 0
    sample_count: int | None = None
    seen_ids: set[str] = set()
    per_problem: ProblemMetrics = {}
    for row in rows:
        try:
            problem = row["problem"]
            problem_id = problem["id"]
            gold_answer = problem["gold_answer"]
            samples = row["samples"]
        except (KeyError, TypeError) as error:
            raise StudyReportError(f"malformed completion row: {error}") from error
        if not isinstance(problem, dict):
            raise StudyReportError("completion row problem must be an object")
        if not isinstance(problem_id, str) or not problem_id:
            raise StudyReportError("completion row problem.id must be a non-empty string")
        if not isinstance(gold_answer, str):
            raise StudyReportError(f"problem {problem_id} gold_answer must be a string")
        target, numbers = _parse_countdown_key(gold_answer)
        if problem_id in seen_ids:
            raise StudyReportError(f"duplicate problem ID in completion artifact: {problem_id}")
        seen_ids.add(problem_id)
        if not isinstance(samples, list) or not samples:
            raise StudyReportError(f"problem {problem_id} has no samples")
        if any(not isinstance(sample, str) for sample in samples):
            raise StudyReportError(f"problem {problem_id} samples must all be strings")
        if sample_count is None:
            sample_count = len(samples)
        elif len(samples) != sample_count:
            raise StudyReportError("completion artifact has inconsistent samples per problem")
        problem_valid = 0
        problem_exact = 0
        for sample in samples:
            result = _score_countdown_sample(sample, numbers=numbers, target=target)
            problem_valid += int(result.is_valid_expression)
            problem_exact += int(result.is_exact_solve)
        valid += problem_valid
        exact += problem_exact
        any_exact += int(problem_exact > 0)
        per_problem[problem_id] = {
            StudyMetric.VALID_EXPRESSION_RATE.value: problem_valid / len(samples),
            StudyMetric.EXACT_SOLVE_RATE.value: problem_exact / len(samples),
            StudyMetric.ANY_EXACT_SOLVE_RATE.value: float(problem_exact > 0),
        }
    assert sample_count is not None
    total_samples = len(rows) * sample_count
    return {
        "problem_count": len(rows),
        "samples_per_problem": sample_count,
        "sample_count": total_samples,
        "valid_expression_rate": valid / total_samples,
        "exact_solve_rate": exact / total_samples,
        "any_exact_solve_rate": any_exact / len(rows),
        "any_exact_solved": any_exact,
    }, per_problem


def _score_countdown_sample(
    sample: str, *, numbers: tuple[int, ...], target: int
) -> VerificationResult:
    # CompletionSet artifacts contain the already-extracted answer in a box. The
    # decomposition verifier treats decimal tokens with leading zeroes as malformed;
    # pin that stricter artifact contract here before using the in-repo expression
    # verifier for number use and arithmetic.
    expression = sample[7:-1] if sample.startswith("\\boxed{") and sample.endswith("}") else sample
    if re.search(r"(?<!\d)0\d+", expression):
        return VerificationResult(False, False, None, "leading zero in decimal token")
    return verify_countdown_lite_expression(expression, numbers=numbers, target=target)


def _parse_countdown_key(value: str) -> tuple[int, tuple[int, ...]]:
    try:
        target_part, numbers_part = value.split(";", maxsplit=1)
        target = int(target_part.removeprefix("target="))
        numbers = tuple(int(item) for item in numbers_part.removeprefix("numbers=").split(","))
    except (ValueError, TypeError) as error:
        raise StudyReportError(f"invalid Countdown answer key: {value!r}") from error
    if (
        not numbers
        or not target_part.startswith("target=")
        or not numbers_part.startswith("numbers=")
    ):
        raise StudyReportError(f"invalid Countdown answer key: {value!r}")
    return target, numbers


def _compatibility_projection(provenance: dict[str, Any]) -> dict[str, Any]:
    try:
        sampling = provenance["sampling"]
        return {
            "dataset": provenance["dataset"],
            "prompt_strategy": provenance["prompt_strategy"],
            "n_problems": provenance["n_problems"],
            "sampling": {
                "max_new_tokens": sampling["max_new_tokens"],
                "n": sampling["n"],
                "seed": sampling["seed"],
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
            },
        }
    except (KeyError, TypeError) as error:
        raise StudyReportError(f"malformed completion provenance: missing {error}") from error


def _compatibility_errors(
    specification: StudySpecification,
    records: list[tuple[str, int, dict[str, Any]]],
    arm_results: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    reference_arm, reference_seed, reference = records[0]
    for arm, seed, record in records[1:]:
        if record != reference:
            errors.append(
                f"{arm} seed {seed} manifest differs from {reference_arm} seed {reference_seed}"
            )
    expected_decoding = specification.decoding.model_dump(mode="json")
    actual_decoding = reference.get("sampling", {})
    for key, value in expected_decoding.items():
        if actual_decoding.get(key) != value:
            errors.append(f"decoding {key!r} is {actual_decoding.get(key)!r}, expected {value!r}")
    expected_manifest_ids = set(specification.task_manifest_ids)
    dataset = reference.get("dataset", {})
    actual_manifest_ids = {str(dataset.get("revision"))}
    if expected_manifest_ids != actual_manifest_ids:
        errors.append(
            f"task manifest IDs are {sorted(actual_manifest_ids)}, "
            f"expected {sorted(expected_manifest_ids)}"
        )
    for name, result in arm_results.items():
        for run in result["runs"]:
            if run["metrics"]["sample_count"] != specification.sample_budget:
                errors.append(
                    f"{name} seed {run['seed']} has {run['metrics']['sample_count']} samples, "
                    f"expected {specification.sample_budget}"
                )
            status = run["training"]["status"]
            if status is not None and status != "complete":
                errors.append(f"{name} seed {run['seed']} training status is {status!r}")
    return errors


def _evidence_warnings(specification: StudySpecification) -> list[str]:
    warnings = []
    for arm in specification.arms:
        if arm.role == ArmRole.BASELINE:
            continue
        for run in arm.runs:
            if run.training_manifest is None:
                warnings.append(f"{arm.name} seed {run.seed} has no training manifest")
            if run.cost is None:
                warnings.append(f"{arm.name} seed {run.seed} has no comparable cost artifact")
    return warnings


def _reference_summary_errors(
    specification: StudySpecification,
    root: Path,
    arm_results: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    reference = specification.reference_summary
    if reference is None:
        return [], []
    path = _artifact_path(reference.path, root)
    artifact = _check_artifact(path, reference.sha256, root)
    payload = _read_json(path)
    expected = payload.get("report_projection")
    if not isinstance(expected, dict):
        return ["reference summary is missing report_projection"], [artifact]
    actual = {
        "seeds": list(specification.included_seeds),
        "arms": {
            name: {
                "mean_metrics": arm["mean_metrics"],
                "per_seed": {str(run["seed"]): run["metrics"] for run in arm["runs"]},
            }
            for name, arm in arm_results.items()
        },
        "comparisons": {
            comparison["comparison_id"]: {"mean_effect": comparison["mean_effect"]}
            for comparison in comparisons
        },
    }
    errors: list[str] = []
    _compare_projection(expected, actual, "reference_summary", errors)
    return errors, [artifact]


def _compare_projection(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path} type differs from committed reference summary")
            return
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key} is missing from generated report")
                continue
            _compare_projection(value, actual[key], f"{path}.{key}", errors)
        return
    if isinstance(expected, list):
        if expected != actual:
            errors.append(f"{path} differs from committed reference summary")
        return
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int | float)
            or not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12)
        ):
            errors.append(f"{path} differs from committed reference summary")
        return
    if expected != actual:
        errors.append(f"{path} differs from committed reference summary")


def _aggregate_arm(role: ArmRole, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(metric.value for metric in StudyMetric)
    means = {
        name: statistics.fmean(float(run["metrics"][name]) for run in runs) for name in metric_names
    }
    costs = [
        float(run["training"]["estimated_cost_usd"])
        for run in runs
        if run["training"]["estimated_cost_usd"] is not None
    ]
    tokens = [
        int(run["training"]["training_token_budget"])
        for run in runs
        if run["training"]["training_token_budget"] is not None
    ]
    return {
        "role": role.value,
        "runs": runs,
        "mean_metrics": means,
        "training_summary": {
            "runs_with_cost": len(costs),
            "estimated_cost_usd": sum(costs),
            "runs_with_token_budget": len(tokens),
            "training_token_budget": sum(tokens),
        },
    }


def _build_comparisons(
    specification: StudySpecification,
    arm_results: dict[str, dict[str, Any]],
    problem_metrics: dict[str, dict[int, ProblemMetrics]],
) -> list[dict[str, Any]]:
    results = []
    for comparison in specification.comparisons:
        metric = comparison.metric.value
        treatment = {
            int(run["seed"]): float(run["metrics"][metric])
            for run in arm_results[comparison.treatment_arm]["runs"]
        }
        reference = {
            int(run["seed"]): float(run["metrics"][metric])
            for run in arm_results[comparison.reference_arm]["runs"]
        }
        paired_seeds = sorted(treatment.keys() & reference.keys())
        effects = [treatment[seed] - reference[seed] for seed in paired_seeds]
        ci = None
        permutation_p = None
        if len(effects) >= 2:
            problem_effects = _paired_problem_effects(
                comparison.comparison_id,
                [
                    (
                        problem_metrics[comparison.treatment_arm][seed],
                        problem_metrics[comparison.reference_arm][seed],
                    )
                    for seed in paired_seeds
                ],
                metric,
            )
            ci = _seed_problem_bootstrap_95(problem_effects)
            permutation_p = _sign_flip_permutation_p(effects)
        results.append(
            {
                "comparison_id": comparison.comparison_id,
                "treatment_arm": comparison.treatment_arm,
                "reference_arm": comparison.reference_arm,
                "metric": comparison.metric.value,
                "paired_seeds": paired_seeds,
                "seed_effects": [
                    {"seed": seed, "effect": treatment[seed] - reference[seed]}
                    for seed in paired_seeds
                ],
                "mean_effect": statistics.fmean(effects) if effects else None,
                "confidence_interval_95": ci,
                "permutation_p_value": permutation_p,
            }
        )
    return results


def _paired_problem_effects(
    comparison_id: str,
    seed_pairs: list[tuple[ProblemMetrics, ProblemMetrics]],
    metric: str,
) -> np.ndarray:
    """Return a (seed, problem) matrix of treatment-minus-reference effects.

    Arms are paired by training seed and, within a seed, by problem ID, so both
    resampling levels keep the treatment and reference rows aligned.
    """
    problem_ids = sorted(seed_pairs[0][0])
    rows = []
    for treatment, reference in seed_pairs:
        if sorted(treatment) != problem_ids or sorted(reference) != problem_ids:
            raise StudyReportError(
                f"comparison {comparison_id!r} arms do not share one problem set across seeds"
            )
        rows.append([treatment[pid][metric] - reference[pid][metric] for pid in problem_ids])
    return np.asarray(rows, dtype=np.float64)


def _seed_problem_bootstrap_95(problem_effects: np.ndarray) -> dict[str, float]:
    # Two-level percentile bootstrap: resample training seeds, then problems within
    # each drawn seed. Rates stay inside [-1, 1] by construction, unlike a t interval.
    seed_count, problem_count = problem_effects.shape
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_draws = rng.integers(0, seed_count, size=(BOOTSTRAP_RESAMPLES, seed_count))
    problem_draws = rng.integers(
        0, problem_count, size=(BOOTSTRAP_RESAMPLES, seed_count, problem_count)
    )
    resampled = problem_effects[seed_draws[:, :, None], problem_draws]
    means = resampled.mean(axis=(1, 2))
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {"lower": float(lower), "upper": float(upper)}


def _sign_flip_permutation_p(effects: list[float]) -> float:
    # Under the null of no treatment effect, each seed's paired effect is equally
    # likely to carry either sign. Enumerate every assignment exactly.
    if len(effects) > MAX_PERMUTATION_SEEDS:
        raise StudyReportError(
            f"exact sign-flip test supports at most {MAX_PERMUTATION_SEEDS} paired seeds"
        )
    values = np.asarray(effects, dtype=np.float64)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(values))))
    permuted = np.abs((signs * values).mean(axis=1))
    observed = abs(values.mean())
    return float(np.mean(permuted >= observed - 1e-12))


def _verdict(
    specification: StudySpecification,
    comparisons: list[dict[str, Any]],
    *,
    missing_seeds: dict[str, list[int]],
    compatibility_errors: list[str],
) -> tuple[str, list[str]]:
    reasons = []
    if missing_seeds:
        reasons.append("required treatment or control seeds are missing")
    if compatibility_errors:
        reasons.append("run manifests or sample budgets are incompatible")
    comparison = next(
        item
        for item in comparisons
        if item["comparison_id"] == specification.acceptance_rule.comparison_id
    )
    mean_effect = comparison["mean_effect"]
    if mean_effect is None or mean_effect <= specification.acceptance_rule.minimum_effect:
        reasons.append(
            f"primary effect does not exceed {specification.acceptance_rule.minimum_effect}"
        )
    required_lower = specification.acceptance_rule.require_ci_lower_above
    interval = comparison["confidence_interval_95"]
    if required_lower is not None and (interval is None or interval["lower"] <= required_lower):
        reasons.append(f"95% confidence interval lower bound does not exceed {required_lower}")
    comparison_by_id = {item["comparison_id"]: item for item in comparisons}
    for rule in specification.acceptance_rule.supporting_comparisons:
        supporting = comparison_by_id[rule.comparison_id]
        supporting_effect = supporting["mean_effect"]
        if rule.minimum_effect is not None and (
            supporting_effect is None or supporting_effect <= rule.minimum_effect
        ):
            reasons.append(
                f"supporting comparison {rule.comparison_id!r} effect does not exceed "
                f"{rule.minimum_effect}"
            )
        upper_id = rule.maximum_effect_below_comparison
        if upper_id is not None:
            upper_effect = comparison_by_id[upper_id]["mean_effect"]
            if (
                supporting_effect is None
                or upper_effect is None
                or supporting_effect >= upper_effect
            ):
                reasons.append(
                    f"supporting comparison {rule.comparison_id!r} effect is not below {upper_id!r}"
                )
    return ("rejected", reasons) if reasons else ("accepted", [])


def _render_markdown(payload: dict[str, Any], *, json_filename: str) -> str:
    lines = [
        "<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->",
        f"# {payload['title']}",
        "",
        f"Machine-readable report: [{json_filename}]({json_filename})",
        "",
        f"**Verdict:** {payload['verdict']}",
        "",
        payload["hypothesis"],
        "",
        "## Completeness and compatibility",
        "",
        f"- Complete: {'yes' if payload['complete'] else 'no'}",
        f"- Compatible: {'yes' if payload['compatible'] else 'no'}",
        f"- Confidence interval: `{payload['confidence_interval_method']}`",
        f"- Sample budget per run: {payload['sample_budget']}",
    ]
    for reason in payload["verdict_reasons"]:
        lines.append(f"- Rejection reason: {reason}")
    for error in payload["compatibility_errors"]:
        lines.append(f"- Compatibility error: {error}")
    for warning in payload["evidence_warnings"]:
        lines.append(f"- Evidence warning: {warning}")

    lines.extend(["", "## Per-seed results", ""])
    for arm_name, arm in payload["arms"].items():
        lines.extend(
            [
                f"### {arm_name} ({arm['role']})",
                "",
                "| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in arm["runs"]:
            metrics = run["metrics"]
            cost = run["training"]["estimated_cost_usd"]
            lines.append(
                f"| {run['seed']} | {_percent(metrics['valid_expression_rate'])} | "
                f"{_percent(metrics['exact_solve_rate'])} | "
                f"{metrics['any_exact_solved']}/{metrics['problem_count']} | "
                f"{cost:.4f} |"
                if cost is not None
                else f"| {run['seed']} | {_percent(metrics['valid_expression_rate'])} | "
                f"{_percent(metrics['exact_solve_rate'])} | "
                f"{metrics['any_exact_solved']}/{metrics['problem_count']} | n/a |"
            )
        lines.extend(
            [
                "",
                f"[Table data]({json_filename})",
                "",
                "| Seed | Train status | Selected step | Selected metric | Token budget |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for run in arm["runs"]:
            training = run["training"]
            status = training["status"] or "n/a"
            step = training["selected_step"] if training["selected_step"] is not None else "n/a"
            selected_metric = (
                f"{training['selected_metric_name']}={training['selected_metric_value']:.4f}"
                if training["selected_metric_name"] is not None
                and training["selected_metric_value"] is not None
                else "n/a"
            )
            token_budget = (
                training["training_token_budget"]
                if training["training_token_budget"] is not None
                else "n/a"
            )
            lines.append(
                f"| {run['seed']} | {status} | {step} | {selected_metric} | {token_budget} |"
            )
        lines.extend(["", f"[Table data]({json_filename})", ""])

    lines.extend(
        [
            "",
            "## Aggregate arms",
            "",
            "| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm_name, arm in payload["arms"].items():
        means = arm["mean_metrics"]
        lines.append(
            f"| {arm_name} | {arm['role']} | {_percent(means['valid_expression_rate'])} | "
            f"{_percent(means['exact_solve_rate'])} | "
            f"{_percent(means['any_exact_solve_rate'])} | "
            f"{arm['training_summary']['estimated_cost_usd']:.4f} |"
        )
    lines.extend(["", f"[Table data]({json_filename})"])

    lines.extend(["", "## Planned comparisons", ""])
    for comparison in payload["comparisons"]:
        interval = comparison["confidence_interval_95"]
        interval_text = (
            f"[{_percent(interval['lower'], signed=True)}, "
            f"{_percent(interval['upper'], signed=True)}]"
            if interval is not None
            else "n/a"
        )
        lines.extend(
            [
                f"### {comparison['comparison_id']}",
                "",
                f"- Metric: `{comparison['metric']}`",
                f"- Mean effect: {_percent(comparison['mean_effect'], signed=True)}",
                f"- 95% CI: {interval_text}",
                f"- Sign-flip permutation p (two-sided): {_p_value(comparison)}",
                f"- Paired seeds: {', '.join(str(seed) for seed in comparison['paired_seeds'])}",
                f"- [Comparison data]({json_filename})",
                "",
            ]
        )

    lines.extend(["", "## Supported claims", ""])
    if payload["supported_claims"]:
        lines.extend(f"- {claim}" for claim in payload["supported_claims"])
    else:
        lines.append("No claims are accepted from this report.")
    lines.extend(["", "## Excluded runs", ""])
    if payload["excluded_runs"]:
        lines.extend(f"- `{run['run_id']}`: {run['reason']}" for run in payload["excluded_runs"])
    else:
        lines.append("None.")
    lines.extend(["", "## Artifact provenance", ""])
    for artifact in payload["provenance"]:
        lines.append(f"- `{artifact['path']}` — `{artifact['sha256']}`")
    return "\n".join(lines) + "\n"


def _p_value(comparison: dict[str, Any]) -> str:
    p_value = comparison["permutation_p_value"]
    if p_value is None:
        return "n/a"
    floor = 2 / 2 ** len(comparison["paired_seeds"])
    return f"{p_value:.4f} (smallest possible with these seeds: {floor:.4f})"


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value * 100:.1f}%"


def _artifact_path(value: str, root: Path) -> Path:
    path = (root / value).resolve()
    _confine_path(path, root, label="artifact")
    return path


def _confine_path(path: Path, root: Path, *, label: str) -> None:
    if not path.is_relative_to(root):
        raise StudyReportError(f"{label} path escapes repository root: {path}")


def _check_artifact(path: Path, expected_sha256: str, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StudyReportError(f"study artifact does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise StudyReportError(
            f"study artifact hash mismatch for {path}: expected {expected_sha256}, got {digest}"
        )
    return {"path": str(path.relative_to(root)), "sha256": digest}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyReportError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise StudyReportError(f"JSON artifact must contain an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise StudyReportError(
                        f"JSONL artifact {path}:{line_number} must contain an object"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise StudyReportError(f"invalid JSONL artifact {path}: {error}") from error
    return rows
