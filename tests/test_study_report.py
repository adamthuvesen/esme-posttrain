from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from esme_posttrain.cli import main
from esme_posttrain.studies.models import StudySpecification
from esme_posttrain.studies.report import (
    StudyReportError,
    _p_value,
    _paired_problem_effects,
    _score_countdown_sample,
    _seed_problem_bootstrap_95,
    _sign_flip_permutation_p,
    generate_study_report,
    load_study_specification,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SPEC = REPO_ROOT / "fixtures" / "studies" / "study.json"


def _payload() -> dict[str, object]:
    return json.loads(FIXTURE_SPEC.read_text(encoding="utf-8"))


def _write_spec(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_generates_accepted_json_and_markdown_from_hashed_artifacts(tmp_path: Path) -> None:
    generated = generate_study_report(FIXTURE_SPEC, output_dir=tmp_path, repo_root=REPO_ROOT)

    assert generated.payload["verdict"] == "accepted"
    assert generated.payload["complete"] is True
    assert generated.payload["compatible"] is True
    comparison = generated.payload["comparisons"][0]
    assert comparison["mean_effect"] == pytest.approx(0.5)
    assert comparison["confidence_interval_95"] == {"lower": 0.5, "upper": 0.5}
    assert generated.payload["arms"]["treatment"]["training_summary"] == {
        "runs_with_cost": 2,
        "estimated_cost_usd": 0.02,
        "runs_with_token_budget": 2,
        "training_token_budget": 200,
    }

    markdown = generated.markdown_path.read_text(encoding="utf-8")
    assert "GENERATED FILE" in markdown
    assert "[study.report.json](study.report.json)" in markdown
    assert "fixture-preempted" in markdown
    assert json.loads(generated.json_path.read_text(encoding="utf-8")) == generated.payload


def test_bootstrap_interval_stays_inside_rate_bounds_with_an_outlier_seed() -> None:
    # The shape of the six-seed placebo study: five seeds near +96 pp, one at +26 pp.
    # A paired t interval on these seed effects reaches past +100 pp.
    seed_effects = [0.26, 0.97, 0.96, 0.96, 0.96, 0.96]
    problem_effects = np.array([[effect] * 30 for effect in seed_effects])

    interval = _seed_problem_bootstrap_95(problem_effects)

    assert -1.0 <= interval["lower"] < np.mean(seed_effects) < interval["upper"] <= 1.0


def test_sign_flip_p_value_is_exact() -> None:
    assert _sign_flip_permutation_p([0.1] * 6) == pytest.approx(2 / 64)
    assert _sign_flip_permutation_p([0.1, -0.1]) == pytest.approx(1.0)


def test_p_value_floor_ignores_zero_seed_effects() -> None:
    comparison = {
        "permutation_p_value": _sign_flip_permutation_p([0.1] * 5 + [0.0]),
        "paired_seeds": list(range(6)),
        "seed_effects": [{"effect": effect} for effect in [0.1] * 5 + [0.0]],
    }

    assert _p_value(comparison) == "0.0625 (smallest possible with these seeds: 0.0625)"


def test_evaluated_bundle_must_match_the_sampled_weights(tmp_path: Path) -> None:
    shutil.copytree(REPO_ROOT / "fixtures" / "studies", tmp_path / "fixtures" / "studies")
    manifest_path = tmp_path / "bundle-manifest.json"
    manifest_path.write_text(
        json.dumps({"checkpoint_step": 240, "files": {"weights": {"sha256": "b" * 64}}}),
        encoding="utf-8",
    )
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[0]["runs"][0]["evaluated_bundle_manifest"] = {
        "path": "bundle-manifest.json",
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    spec_path = _write_spec(tmp_path, payload)

    # The fixture provenance names no model_revision, so it cannot match these weights.
    with pytest.raises(StudyReportError, match="were not sampled from the weights"):
        generate_study_report(spec_path, output_dir=tmp_path / "output", repo_root=tmp_path)


def test_comparison_rejects_arms_scored_on_different_problems() -> None:
    treatment = {"p1": {"valid_expression_rate": 1.0}}
    reference = {"p2": {"valid_expression_rate": 0.0}}

    with pytest.raises(StudyReportError, match="do not share one problem set"):
        _paired_problem_effects("c", [(treatment, reference)], "valid_expression_rate")


def test_specification_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["surprise"] = True

    with pytest.raises(ValidationError, match="surprise"):
        StudySpecification.model_validate(payload)


def test_completion_scoring_matches_decomposition_leading_zero_rule() -> None:
    score = _score_countdown_sample("\\boxed{3 - 06}", numbers=(3, 6), target=18)

    assert score.is_valid_expression is False
    assert score.is_exact_solve is False


def test_completion_artifact_rejects_non_string_samples(tmp_path: Path) -> None:
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[0]["runs"][0]["completions"] = {
        "path": "fixtures/studies/malformed-sample.jsonl",
        "sha256": "49fac3aaa19f8fb3af9c5134b43a57ff981a63805d3f1d5f9490024636fff7cd",
    }
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(StudyReportError, match="samples must all be strings"):
        generate_study_report(spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT)


def test_missing_control_seed_forces_rejected_verdict(tmp_path: Path) -> None:
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[1]["runs"] = arms[1]["runs"][:1]
    spec_path = _write_spec(tmp_path, payload)

    generated = generate_study_report(
        spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT
    )

    assert generated.payload["verdict"] == "rejected"
    assert generated.payload["complete"] is False
    assert generated.payload["missing_seeds"] == {"control": [2]}
    assert generated.payload["supported_claims"] == []


def test_supporting_claim_comparison_can_force_rejected_verdict(tmp_path: Path) -> None:
    payload = _payload()
    acceptance_rule = payload["acceptance_rule"]
    assert isinstance(acceptance_rule, dict)
    acceptance_rule["supporting_comparisons"] = [
        {"comparison_id": "treatment-vs-control-exact", "minimum_effect": 1.0}
    ]
    spec_path = _write_spec(tmp_path, payload)

    generated = generate_study_report(
        spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT
    )

    assert generated.payload["verdict"] == "rejected"
    assert generated.payload["supported_claims"] == []
    assert any("supporting comparison" in reason for reason in generated.payload["verdict_reasons"])


def test_incompatible_provenance_forces_rejected_verdict(tmp_path: Path) -> None:
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[1]["runs"][1]["provenance"] = {
        "path": "fixtures/studies/incompatible-provenance.json",
        "sha256": "6636aa6e0a3e931e023489cfde0e0ce9b6bb8550e40b28f2077d3b6f7adb130c",
    }
    spec_path = _write_spec(tmp_path, payload)

    generated = generate_study_report(
        spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT
    )

    assert generated.payload["verdict"] == "rejected"
    assert generated.payload["compatible"] is False
    assert "manifest differs" in generated.payload["compatibility_errors"][0]


def test_reference_summary_drift_forces_rejected_verdict(tmp_path: Path) -> None:
    payload = _payload()
    payload["reference_summary"] = {
        "path": "fixtures/studies/reference-summary-drift.json",
        "sha256": "468baa484eb5e58a0b96694b9135d384dc73ce1868408727bc6d0e6ad686b7c7",
    }
    spec_path = _write_spec(tmp_path, payload)

    generated = generate_study_report(
        spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT
    )

    assert generated.payload["verdict"] == "rejected"
    assert generated.payload["compatible"] is False
    assert any(
        "reference_summary.arms.treatment.mean_metrics.valid_expression_rate" in error
        for error in generated.payload["compatibility_errors"]
    )


def test_failed_training_status_forces_rejected_verdict(tmp_path: Path) -> None:
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[0]["runs"][0]["cost"] = {
        "path": "fixtures/studies/cost-failed.json",
        "sha256": "36f9e5f865c23d8dcd1de90b07001d72ad71160aa32ec979df670ad83967b825",
    }
    spec_path = _write_spec(tmp_path, payload)

    generated = generate_study_report(
        spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT
    )

    assert generated.payload["verdict"] == "rejected"
    assert "training status is 'failed'" in generated.payload["compatibility_errors"][0]


def test_hash_mismatch_fails_before_reporting(tmp_path: Path) -> None:
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[0]["runs"][0]["completions"]["sha256"] = "0" * 64
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(StudyReportError, match="hash mismatch"):
        generate_study_report(spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT)


def test_artifact_path_cannot_escape_repository_root(tmp_path: Path) -> None:
    payload = _payload()
    arms = payload["arms"]
    assert isinstance(arms, list)
    arms[0]["runs"][0]["completions"]["path"] = "../outside.jsonl"
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(StudyReportError, match="escapes repository root"):
        generate_study_report(spec_path, output_dir=tmp_path / "output", repo_root=REPO_ROOT)


def test_load_rejects_malformed_specification(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(StudyReportError, match="invalid study specification"):
        load_study_specification(path)


def test_study_report_cli_writes_both_formats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "study-report",
                "--study",
                str(FIXTURE_SPEC),
                "--output-dir",
                str(tmp_path),
                "--repo-root",
                str(REPO_ROOT),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "accepted"
    assert Path(payload["json_path"]).is_file()
    assert Path(payload["markdown_path"]).is_file()


@pytest.mark.parametrize(
    "study_path",
    sorted(
        path
        for path in (REPO_ROOT / "studies").glob("*.json")
        if "report" not in path.name and "summary" not in path.name
    ),
    ids=lambda path: path.stem,
)
def test_committed_report_matches_its_study_specification(study_path: Path) -> None:
    report_path = study_path.with_name(f"{study_path.stem}.report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    study = json.loads(study_path.read_text(encoding="utf-8"))

    assert (
        report["study_specification_sha256"] == hashlib.sha256(study_path.read_bytes()).hexdigest()
    )
    expected_claims = study["allowed_claims"] if report["verdict"] == "accepted" else []
    assert report["supported_claims"] == expected_claims
