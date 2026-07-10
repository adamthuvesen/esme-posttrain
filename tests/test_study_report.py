from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from esme_posttrain.cli import main
from esme_posttrain.studies.models import StudySpecification
from esme_posttrain.studies.report import (
    StudyReportError,
    _compare_projection,
    _score_countdown_sample,
    generate_study_report,
    load_study_specification,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SPEC = REPO_ROOT / "fixtures" / "studies" / "study.json"
REAL_REPORT = REPO_ROOT / "studies" / "rlvr-placebo.report.json"
REAL_REFERENCE = REPO_ROOT / "studies" / "references" / "grpo-decomp-sampled-multiseed-summary.json"


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


def test_checked_real_report_matches_grpo_decomp_projection() -> None:
    report = json.loads(REAL_REPORT.read_text(encoding="utf-8"))
    reference = json.loads(REAL_REFERENCE.read_text(encoding="utf-8"))
    study_path = REPO_ROOT / "studies" / "rlvr-placebo.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    actual = {
        "seeds": [run["seed"] for run in report["arms"]["real_reward"]["runs"]],
        "arms": {
            name: {
                "mean_metrics": arm["mean_metrics"],
                "per_seed": {str(run["seed"]): run["metrics"] for run in arm["runs"]},
            }
            for name, arm in report["arms"].items()
        },
        "comparisons": {
            comparison["comparison_id"]: {
                "mean_effect": comparison["mean_effect"],
                "confidence_interval_95": comparison["confidence_interval_95"],
            }
            for comparison in report["comparisons"]
        },
    }
    errors: list[str] = []

    _compare_projection(reference["report_projection"], actual, "reference_summary", errors)

    assert errors == []
    assert (
        report["study_specification_sha256"] == hashlib.sha256(study_path.read_bytes()).hexdigest()
    )
    assert report["supported_claims"] == study["allowed_claims"]
