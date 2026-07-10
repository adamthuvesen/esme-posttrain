"""Strict records for reproducible, artifact-backed studies."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArmRole(StrEnum):
    TREATMENT = "treatment"
    CONTROL = "control"
    BASELINE = "baseline"


class StudyMetric(StrEnum):
    VALID_EXPRESSION_RATE = "valid_expression_rate"
    EXACT_SOLVE_RATE = "exact_solve_rate"
    ANY_EXACT_SOLVE_RATE = "any_exact_solve_rate"


class ConfidenceIntervalMethod(StrEnum):
    PAIRED_STUDENT_T_95 = "paired_student_t_95"


class HashedArtifact(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecodingSettings(_StrictModel):
    max_new_tokens: int = Field(gt=0)
    n: int = Field(gt=0)
    seed: int = Field(ge=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)


class StudyRunSpec(_StrictModel):
    seed: int = Field(ge=0)
    completions: HashedArtifact
    provenance: HashedArtifact
    training_manifest: HashedArtifact | None = None
    cost: HashedArtifact | None = None
    training_token_budget: int | None = Field(default=None, ge=0)


class StudyArmSpec(_StrictModel):
    name: str = Field(min_length=1)
    role: ArmRole
    runs: tuple[StudyRunSpec, ...]

    @model_validator(mode="after")
    def check_unique_seeds(self) -> StudyArmSpec:
        seeds = [run.seed for run in self.runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"arm {self.name!r} has duplicate seeds")
        if not self.runs:
            raise ValueError(f"arm {self.name!r} must contain at least one run")
        return self


class PlannedComparison(_StrictModel):
    comparison_id: str = Field(min_length=1)
    treatment_arm: str = Field(min_length=1)
    reference_arm: str = Field(min_length=1)
    metric: StudyMetric

    @model_validator(mode="after")
    def check_distinct_arms(self) -> PlannedComparison:
        if self.treatment_arm == self.reference_arm:
            raise ValueError("a comparison must name two different arms")
        return self


class AcceptanceRule(_StrictModel):
    comparison_id: str = Field(min_length=1)
    minimum_effect: float = 0.0
    require_ci_lower_above: float | None = None
    supporting_comparisons: tuple[SupportingComparisonRule, ...] = ()


class SupportingComparisonRule(_StrictModel):
    comparison_id: str = Field(min_length=1)
    minimum_effect: float | None = None
    maximum_effect_below_comparison: str | None = Field(default=None, min_length=1)


class ExcludedRun(_StrictModel):
    run_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class StudySpecification(_StrictModel):
    schema_version: Literal[1]
    study_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    primary_metric: StudyMetric
    arms: tuple[StudyArmSpec, ...]
    included_seeds: tuple[int, ...]
    task_manifest_ids: tuple[str, ...]
    sample_budget: int = Field(gt=0)
    decoding: DecodingSettings
    comparisons: tuple[PlannedComparison, ...]
    confidence_interval_method: ConfidenceIntervalMethod
    acceptance_rule: AcceptanceRule
    allowed_claims: tuple[str, ...]
    excluded_runs: tuple[ExcludedRun, ...] = ()
    reference_summary: HashedArtifact | None = None

    @model_validator(mode="after")
    def check_study_shape(self) -> StudySpecification:
        if len(self.included_seeds) != len(set(self.included_seeds)):
            raise ValueError("included_seeds contains duplicates")
        if not self.included_seeds:
            raise ValueError("included_seeds must not be empty")
        if not self.task_manifest_ids:
            raise ValueError("task_manifest_ids must not be empty")
        if not self.allowed_claims:
            raise ValueError("allowed_claims must not be empty")

        arm_names = [arm.name for arm in self.arms]
        if len(arm_names) != len(set(arm_names)):
            raise ValueError("arm names must be unique")
        roles = [arm.role for arm in self.arms]
        if roles.count(ArmRole.TREATMENT) != 1 or roles.count(ArmRole.CONTROL) != 1:
            raise ValueError("a study needs exactly one treatment and one control arm")
        if roles.count(ArmRole.BASELINE) > 1:
            raise ValueError("a study may have at most one baseline arm")

        expected_seeds = set(self.included_seeds)
        for arm in self.arms:
            if arm.role in {ArmRole.TREATMENT, ArmRole.CONTROL}:
                seeds = {run.seed for run in arm.runs}
                if seeds - expected_seeds:
                    raise ValueError(
                        f"arm {arm.name!r} contains seeds not named in included_seeds: "
                        f"{sorted(seeds - expected_seeds)}"
                    )

        comparison_ids = [comparison.comparison_id for comparison in self.comparisons]
        if not comparison_ids:
            raise ValueError("comparisons must not be empty")
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("comparison IDs must be unique")
        known_arms = set(arm_names)
        arm_roles = {arm.name: arm.role for arm in self.arms}
        for comparison in self.comparisons:
            unknown = {comparison.treatment_arm, comparison.reference_arm} - known_arms
            if unknown:
                raise ValueError(f"comparison names unknown arms: {sorted(unknown)}")
            if arm_roles[comparison.treatment_arm] != ArmRole.TREATMENT:
                raise ValueError("comparison treatment_arm must name the treatment arm")
            if arm_roles[comparison.reference_arm] == ArmRole.TREATMENT:
                raise ValueError("comparison reference_arm must name a control or baseline arm")
        if self.acceptance_rule.comparison_id not in set(comparison_ids):
            raise ValueError("acceptance_rule names an unknown comparison")
        for rule in self.acceptance_rule.supporting_comparisons:
            if rule.comparison_id not in set(comparison_ids):
                raise ValueError("supporting comparison rule names an unknown comparison")
            if (
                rule.maximum_effect_below_comparison is not None
                and rule.maximum_effect_below_comparison not in set(comparison_ids)
            ):
                raise ValueError("supporting comparison rule names an unknown upper comparison")
        if self.primary_metric != next(
            comparison.metric
            for comparison in self.comparisons
            if comparison.comparison_id == self.acceptance_rule.comparison_id
        ):
            raise ValueError("acceptance comparison must use the primary metric")
        return self
