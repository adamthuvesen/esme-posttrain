from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts import modal_instruct_sft

from esme_posttrain.bundle import file_sha256
from esme_posttrain.cli import main
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft import sweep_instruct as sft_sweep
from esme_posttrain.sft.data import (
    DatasetSource,
)
from esme_posttrain.sft.full_instruct import SFTFullRunError, run_full_instruct_sft
from esme_posttrain.sft.launch_instruct import (
    EXPECTED_ARTIFACTS as SFT_EXPECTED_ARTIFACTS,
)
from esme_posttrain.sft.launch_instruct import (
    LaunchError,
    SFTLaunchConfig,
    build_sft_dry_run,
    full_launch_blockers,
    load_sft_config,
)
from esme_posttrain.sft.probe_instruct import (
    DEFAULT_MODAL_PROBE_ROOT,
    PROBE_RECIPE,
    build_throughput_probe_preflight,
    throughput_probe_blockers,
)
from esme_posttrain.sft.smoke_instruct import tiny_backbone_config, tiny_tokenizer
from esme_posttrain.sft.sweep_instruct import (
    SWEEP_OUTPUT_STEM,
    SFTSweepArm,
    build_interval_sweep_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "esme-214m-instruct.json"
WEIGHTS_FIELD = "key_format"


def test_sft_config_pins_mix_caps_and_dry_run_never_launches() -> None:
    config = load_sft_config(CONFIG_PATH)

    assert [source.name for source in config.train_sources] == [
        "smol-smoltalk",
        "tulu-3-personas",
    ]
    assert [source.mix_ratio for source in config.train_sources] == [0.8, 0.2]
    assert config.eval_source.source == "HuggingFaceH4/no_robots"
    assert config.eval_source.train_allowed is False

    dry_run = build_sft_dry_run(config)
    assert dry_run["will_start_modal_job"] is False
    assert dry_run["will_download_data"] is False
    assert dry_run["runtime"]["estimated_smoke_cost_usd"] <= 2
    assert "full Esme-214M-Instruct SFT launch requires --approved" in "; ".join(
        dry_run["full_launch_blockers"]
    )
    assert "bounded_matched_interval_eval_sweep" not in "; ".join(dry_run["full_launch_blockers"])
    approved_preflight = build_sft_dry_run(config, full_run_approved=True)
    assert approved_preflight["preflight"]["will_start_modal_job"] is False
    assert approved_preflight["preflight"]["dataset_revisions"] == {
        "smol-smoltalk": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "tulu-3-personas": "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
        "no_robots": "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
    }
    assert (
        approved_preflight["learning_gate"]["evidence"]["stopped_run_reconciliation"][
            "showcase_best_step"
        ]
        == 600
    )
    sweep_evidence = approved_preflight["learning_gate"]["evidence"][
        "bounded_matched_interval_eval_sweep"
    ]
    assert sweep_evidence["eval_metric"] == "eval/matched/response_loss"
    assert sweep_evidence["step0_response_loss"] == pytest.approx(2.187998926732106)
    assert sweep_evidence["best_response_loss"] == pytest.approx(1.7372412149933547)
    assert sweep_evidence["best_response_loss"] < sweep_evidence["step0_response_loss"]
    assert sweep_evidence["best_arm_id"] == "sweep-20260627T143203Z-lr3e-5-mb2-ga8-eb16"
    assert approved_preflight["preflight"]["blockers"] == []
    assert approved_preflight["runtime"]["projected_train_tokens"] == 30_000_000
    assert approved_preflight["budgets"]["target_train_tokens"] == 30_000_000
    assert approved_preflight["optimizer"]["learning_rate"] == 3e-5
    assert approved_preflight["optimizer"]["micro_batch_size"] == 2
    assert approved_preflight["optimizer"]["gradient_accumulation_steps"] == 8
    assert approved_preflight["optimizer"]["effective_batch_size"] == 16
    assert approved_preflight["optimizer"]["scheduler"] == "cosine_decay"
    assert approved_preflight["optimizer"]["warmup_steps"] == 700
    assert approved_preflight["optimizer"]["weight_decay"] == 0.1
    assert approved_preflight["loss"]["assistant_only_loss"] is True
    assert approved_preflight["tuning"]["mode"] == "full"
    assert approved_preflight["sequence"]["sequence_packing"] is False
    assert approved_preflight["sequence"]["pad_to_multiple_of"] == 8
    assert approved_preflight["runtime"]["precision"] == "bf16"
    assert approved_preflight["runtime"]["selected_gpu"] == "A100"
    assert approved_preflight["runtime"]["gpu_profiles"]["A100"]["measured"] is True
    assert approved_preflight["runtime"]["estimated_full_cost_usd"] < 25
    assert (
        "SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=10"
        in approved_preflight["preflight"]["exact_launch_command"]
    )
    assert set(approved_preflight["runtime"]["gpu_profiles"]) >= {"L4", "A10G", "A100"}


def test_modal_smoke_body_refreshes_manifest_with_expected_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "modal-smoke"
    refresh_calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_run_cpu_fixture_sft(config: object, *, output_dir: Path, wandb_enabled: bool) -> dict:
        output_dir.mkdir(parents=True)
        return {"status": "local_cpu_fixture_complete", "wandb_enabled": wandb_enabled}

    def fake_refresh_manifest_files(output_dir: Path, expected_artifacts: tuple[str, ...]) -> None:
        refresh_calls.append((output_dir, expected_artifacts))

    monkeypatch.setattr(modal_instruct_sft, "fresh_output_dir", lambda root, stem: output_dir)
    monkeypatch.setattr(modal_instruct_sft, "run_cpu_fixture_sft", fake_run_cpu_fixture_sft)
    monkeypatch.setattr(modal_instruct_sft, "refresh_manifest_files", fake_refresh_manifest_files)

    payload = modal_instruct_sft._run_modal_smoke_body(
        _load_config_payload(),
        commit="abc123",
        dirty=False,
        started=modal_instruct_sft.time.perf_counter(),
    )

    assert refresh_calls == [(output_dir, SFT_EXPECTED_ARTIFACTS)]
    assert (output_dir / "cost.json").is_file()
    assert payload["status"] == "modal_smoke_complete"


def test_sft_config_rejects_noncommercial_training_and_caps(tmp_path: Path) -> None:
    payload = _load_config_payload()
    payload["datasets"]["eval_holdout"]["train_allowed"] = True
    path = _write_json(tmp_path / "bad.json", payload)

    with pytest.raises(LaunchError, match="train_allowed"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["budgets"]["max_train_samples"] = 50001
    path = _write_json(tmp_path / "too_many_samples.json", payload)

    with pytest.raises(LaunchError, match="max_train_samples"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["budgets"]["target_train_tokens"] = payload["budgets"]["max_train_tokens"] + 1
    path = _write_json(tmp_path / "too_many_target_tokens.json", payload)

    with pytest.raises(LaunchError, match="target_train_tokens"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["optimizer"]["effective_batch_size"] = 99
    path = _write_json(tmp_path / "bad_batch.json", payload)

    with pytest.raises(LaunchError, match="effective_batch_size"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["tuning"]["mode"] = "qlora"
    path = _write_json(tmp_path / "bad_tuning.json", payload)

    with pytest.raises(LaunchError, match="tuning.mode"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["optimizer"]["scheduler"] = "cosine"
    path = _write_json(tmp_path / "bad_scheduler.json", payload)

    with pytest.raises(LaunchError, match="optimizer.scheduler"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["runtime"]["precision"] = "fp16"
    path = _write_json(tmp_path / "bad_precision.json", payload)

    with pytest.raises(LaunchError, match="runtime.precision"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["sequence"]["sequence_packing"] = True
    path = _write_json(tmp_path / "bad_packing.json", payload)

    with pytest.raises(LaunchError, match="sequence.sequence_packing"):
        load_sft_config(path)

    payload = _load_config_payload()
    payload["runtime"]["gpu_profiles"] = {"L4": payload["runtime"]["gpu_profiles"]["L4"]}
    path = _write_json(tmp_path / "one_gpu.json", payload)

    with pytest.raises(LaunchError, match="at least two GPU profiles"):
        load_sft_config(path)


def test_sft_cost_blockers_cover_full_run_projection(tmp_path: Path) -> None:
    payload = _load_config_payload()
    payload["runtime"]["gpu_profiles"]["A100"]["projected_tokens_per_second"] = 1
    path = _write_json(tmp_path / "costly.json", payload)

    config = load_sft_config(path)

    assert "projected full-run cost exceeds runtime.full_run_max_cost_usd" in (
        full_launch_blockers(config, approved=True)
    )


def test_full_run_learning_gate_accepts_reconciliation_and_matched_sweep_evidence(
    tmp_path: Path,
) -> None:
    payload = _load_config_payload()
    payload["learning_gate"]["evidence"] = _complete_learning_gate_evidence()
    path = _write_json(tmp_path / "learning_gate.json", payload)

    config = load_sft_config(path)

    assert full_launch_blockers(config, approved=True) == []


def test_full_run_learning_gate_rejects_sweep_only_evidence(tmp_path: Path) -> None:
    payload = _load_config_payload()
    payload["learning_gate"]["evidence"] = _bounded_interval_eval_sweep_evidence()
    path = _write_json(tmp_path / "sweep_only.json", payload)

    config = load_sft_config(path)

    blockers = full_launch_blockers(config, approved=True)
    assert "learning_gate.evidence is missing stopped_run_reconciliation evidence" in blockers


def test_full_run_learning_gate_rejects_missing_or_flat_sweep_evidence(tmp_path: Path) -> None:
    payload = _load_config_payload()
    evidence = _complete_learning_gate_evidence()
    del evidence["bounded_matched_interval_eval_sweep"]
    payload["learning_gate"]["evidence"] = evidence
    path = _write_json(tmp_path / "missing_sweep.json", payload)

    config = load_sft_config(path)

    assert full_launch_blockers(config, approved=True) == [
        "learning_gate.evidence is missing bounded_matched_interval_eval_sweep evidence"
    ]


def test_full_run_learning_gate_rejects_sweep_without_eval_loss_drop(tmp_path: Path) -> None:
    payload = _load_config_payload()
    evidence = _complete_learning_gate_evidence()
    evidence["bounded_matched_interval_eval_sweep"]["best_response_loss"] = 4.2
    payload["learning_gate"]["evidence"] = evidence
    path = _write_json(tmp_path / "no_eval_loss_drop.json", payload)

    config = load_sft_config(path)

    assert full_launch_blockers(config, approved=True) == [
        "learning_gate.evidence.bounded_matched_interval_eval_sweep must show "
        "best_response_loss < step0_response_loss"
    ]


def test_full_run_learning_gate_rejects_wrong_stopped_run_path_counts(
    tmp_path: Path,
) -> None:
    payload = _load_config_payload()
    evidence = _complete_learning_gate_evidence()
    evidence["stopped_run_reconciliation"]["showcase_eval_rows"] = 2
    payload["learning_gate"]["evidence"] = evidence
    path = _write_json(tmp_path / "bad_reconciliation.json", payload)

    config = load_sft_config(path)

    assert full_launch_blockers(config, approved=True) == [
        "learning_gate.evidence.stopped_run_reconciliation must record 98 eval rows"
    ]


def test_instruct_release_docs_keep_approval_history_internal() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    run_card = (REPO_ROOT / "run_cards" / "esme-214m-instruct.md").read_text(encoding="utf-8")
    internal_recipe = (REPO_ROOT / "docs" / "internal" / "instruct-sft-recipe.md").read_text(
        encoding="utf-8"
    )

    required = "records stopped-run reconciliation and bounded matched interval-eval sweep evidence"
    public_text = " ".join(f"{readme}\n{run_card}".split())
    internal_text = " ".join(internal_recipe.split())

    assert required not in " ".join(readme.split())
    assert required not in " ".join(run_card.split())
    assert required in internal_text
    assert "new full-data rerun requires explicit chat approval" not in public_text
    assert "Adam approved the full A100 launch" not in public_text


def test_instruct_base_bundle_resolves_env_or_sibling_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_bundle = tmp_path / "base-bundle"
    monkeypatch.setenv(modal_instruct_sft.BASE_BUNDLE_ENV, str(env_bundle))
    assert modal_instruct_sft._resolve_base_bundle_local() == env_bundle

    monkeypatch.delenv(modal_instruct_sft.BASE_BUNDLE_ENV, raising=False)
    assert modal_instruct_sft._resolve_base_bundle_local() == (
        REPO_ROOT.parent / "esme-pretrain" / "exports" / "esme-214m-base"
    )


def test_instruct_training_launches_block_when_base_bundle_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_bundle = tmp_path / "missing-base"

    class RefusingRunner:
        def spawn(self, *args: object) -> object:
            del args
            raise AssertionError("missing base bundle must block before Modal spawn")

    monkeypatch.setattr(modal_instruct_sft, "BASE_BUNDLE_LOCAL", missing_bundle)
    blocker = modal_instruct_sft._base_bundle_blocker()
    assert blocker is not None
    assert str(missing_bundle) in blocker
    assert modal_instruct_sft.BASE_BUNDLE_ENV in blocker
    assert "sibling-repo fallback" in blocker
    assert "re-export Esme-214M-Base" in blocker

    monkeypatch.setattr(modal_instruct_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_instruct_sft, "modal", object())
    monkeypatch.setattr(modal_instruct_sft, "run_modal_full_sft", RefusingRunner())

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--full-run", "--approved", "--json"]
        )
        == 2
    )
    refused = json.loads(capsys.readouterr().out)
    assert refused["will_start_modal_job"] is False
    assert any(
        str(missing_bundle) in blocker and modal_instruct_sft.BASE_BUNDLE_ENV in blocker
        for blocker in refused["full_launch_blockers"]
    )


def test_full_run_refuses_without_approval_and_preflight_never_spawns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(modal_instruct_sft, "SFT_MODAL_GPU", "A100")

    assert modal_instruct_sft.launch(["--config", str(CONFIG_PATH), "--full-run", "--json"]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["will_start_modal_job"] is False
    assert "requires --approved" in "; ".join(refused["full_launch_blockers"])

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--dry-run", "--full-run", "--approved", "--json"]
        )
        == 0
    )
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["will_start_modal_job"] is False
    assert preflight["preflight"]["blockers"] == []
    assert preflight["full_launch_blockers"] == preflight["preflight"]["blockers"]
    assert "--full-run --approved" in preflight["preflight"]["exact_launch_command"]
    assert (
        "SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=10"
        in preflight["preflight"]["exact_launch_command"]
    )


def test_full_run_output_stem_override_updates_dry_run_and_launch_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fresh_stem = "esme-instruct-sft-showcase-full-20260627-a100-matched"
    captured: dict[str, object] = {}

    class FakeCall:
        object_id = "fc-full-test"

    class FakeRunner:
        def spawn(self, *args: object) -> FakeCall:
            captured["args"] = args
            return FakeCall()

    def fake_mirror_launch_evidence(target_dir: Path, payload: dict[str, object]) -> Path:
        captured["receipt_target_dir"] = target_dir
        captured["receipt_payload"] = payload
        receipt = tmp_path / "launch-receipt.json"
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        return receipt

    monkeypatch.setattr(modal_instruct_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_instruct_sft, "MODAL_FULL_OUTPUT_STEM", fresh_stem)
    monkeypatch.setattr(modal_instruct_sft, "BASE_BUNDLE_LOCAL", tmp_path)

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--dry-run", "--full-run", "--approved", "--json"]
        )
        == 0
    )
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run["will_start_modal_job"] is False
    assert dry_run["full_launch_blockers"] == []
    assert dry_run["volume_output_dir"] == f"/posttrain/{fresh_stem}"
    assert dry_run["preflight"]["volume_output_dir"] == f"/posttrain/{fresh_stem}"
    assert (
        f"SFT_MODAL_FULL_OUTPUT_STEM='{fresh_stem}'" in dry_run["preflight"]["exact_launch_command"]
    )
    assert dry_run["resume_launch_command"].endswith("--json --resume")

    monkeypatch.setattr(modal_instruct_sft, "modal", object())
    monkeypatch.setattr(modal_instruct_sft, "run_modal_full_sft", FakeRunner())
    monkeypatch.setattr(modal_instruct_sft, "mirror_launch_evidence", fake_mirror_launch_evidence)

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--full-run", "--approved", "--json"]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert captured["args"][3:] == (False, "A100", fresh_stem)
    assert receipt["modal_call_id"] == "fc-full-test"
    assert receipt["volume_output_dir"] == f"/posttrain/{fresh_stem}"
    assert f"SFT_MODAL_FULL_OUTPUT_STEM='{fresh_stem}'" in receipt["full_launch_command"]
    assert receipt["resume_launch_command"].endswith("--json --resume")
    assert receipt["local_launch_receipt"] == str(tmp_path / "launch-receipt.json")
    assert captured["receipt_payload"]["volume_output_dir"] == f"/posttrain/{fresh_stem}"


def test_full_run_dry_run_blocks_modal_gpu_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(modal_instruct_sft, "SFT_MODAL_GPU", "B200")

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--full-run", "--approved", "--dry-run", "--json"]
        )
        == 0
    )
    preflight = json.loads(capsys.readouterr().out)

    assert preflight["status"] == "blocked_by_launch_safety"
    assert preflight["will_start_modal_job"] is False
    assert preflight["preflight"]["will_start_modal_job"] is False
    assert preflight["full_launch_blockers"] == [
        "SFT_MODAL_GPU must match runtime.gpu_profiles[runtime.selected_gpu].modal_gpu "
        "for full-run cost accounting",
    ]
    assert preflight["preflight"]["blockers"] == preflight["full_launch_blockers"]


def test_interval_sweep_preflight_is_bounded_real_data_and_isolated() -> None:
    config = load_sft_config(CONFIG_PATH)

    preflight = build_interval_sweep_preflight(config, timeout_hours=2, modal_gpu="A100")

    assert preflight["status"] == "ready_for_modal_sweep"
    assert preflight["will_start_modal_job"] is False
    assert preflight["will_download_data"] is False
    assert preflight["modal_run_will_download_real_data"] is True
    assert preflight["uses_cpu_fixture_path"] is False
    assert preflight["uses_full_run_output_dir"] is False
    assert preflight["uses_showcase_output_dir"] is False
    assert preflight["volume_output_root"] == "/posttrain/esme-instruct-sft-interval-sweep"
    assert "-excellence" not in preflight["volume_output_root"]
    assert "showcase-full" not in preflight["volume_output_root"]
    assert preflight["data_caps"]["eval_samples"] == 128
    assert preflight["data_caps"]["matched_eval_samples_per_source"] == 64
    assert len(preflight["arms"]) == 4
    assert {arm["gradient_accumulation_steps"] for arm in preflight["arms"]} == {8}
    assert {arm["micro_batch_size"] for arm in preflight["arms"]} == {2}
    assert {arm["effective_batch_size"] for arm in preflight["arms"]} == {16}
    assert {arm["learning_rate"] for arm in preflight["arms"]} == {1e-5, 2e-5, 3e-5, 5e-5}
    assert all(arm["warmup_steps"] == 16 for arm in preflight["arms"])
    assert all(arm["max_steps"] <= 200 for arm in preflight["arms"])
    assert all(arm["eval_interval"] > 0 for arm in preflight["arms"])
    assert preflight["data_caps"]["train_samples"] < config.budgets["max_train_samples"]
    assert preflight["data_caps"]["train_tokens"] < config.budgets["max_train_tokens"]
    assert preflight["runtime"]["sweep_spend_cap_usd"] == 5
    assert preflight["runtime"]["timeout_cost_ceiling_usd"] <= 5
    assert preflight["monitoring"]["job_type"] == "sweep"
    assert preflight["monitoring"]["group"] == "esme_214m_instruct_sft_interval_sweep"
    assert "eval/matched/response_loss" in preflight["monitoring"]["required_eval_metrics"]
    assert preflight["acceptance"]["metric"] == "eval/matched/response_loss"
    assert "--modal-sweep --approved" in preflight["modal_sweep_command"]
    assert "--full-run" not in preflight["modal_sweep_command"]
    assert preflight["launch_blockers"] == []


def test_modal_sweep_dry_run_and_mode_conflict_never_spawn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--dry-run", "--modal-sweep", "--json"]
        )
        == 0
    )
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["will_start_modal_job"] is False
    assert preflight["mode"] == "modal_interval_eval_sweep"
    assert preflight["uses_full_run_output_dir"] is False
    assert preflight["volume_output_root"].endswith(SWEEP_OUTPUT_STEM)

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--dry-run", "--modal-sweep", "--full-run", "--json"]
        )
        == 2
    )
    assert "choose exactly one launch mode" in capsys.readouterr().err


def test_modal_sweep_launch_passes_gpu_and_timeout_into_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    class FakeCall:
        object_id = "fc-test"

        def get(self) -> dict[str, object]:
            return {"status": "interval_eval_sweep_passed"}

    class FakeRunner:
        def spawn(self, *args: object) -> FakeCall:
            captured["args"] = args
            return FakeCall()

    monkeypatch.setattr(modal_instruct_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_instruct_sft, "SFT_SWEEP_TIMEOUT_HOURS", 2)
    monkeypatch.setattr(modal_instruct_sft, "BASE_BUNDLE_LOCAL", tmp_path)
    monkeypatch.setattr(modal_instruct_sft, "modal", object())
    monkeypatch.setattr(modal_instruct_sft, "run_modal_sweep", FakeRunner())

    assert (
        modal_instruct_sft.launch(
            ["--config", str(CONFIG_PATH), "--modal-sweep", "--approved", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "interval_eval_sweep_passed"
    assert captured["args"][3:] == ("A100", 2)


def test_throughput_probe_preflight_covers_high_end_gpus() -> None:
    config = load_sft_config(CONFIG_PATH)

    for gpu in ("A100", "H100!", "H200", "B200"):
        preflight = build_throughput_probe_preflight(config, modal_gpu=gpu, timeout_hours=2)

        assert preflight["status"] == "ready_for_throughput_probe"
        assert preflight["will_start_modal_job"] is False
        assert preflight["modal_run_will_download_real_data"] is True
        assert preflight["uses_full_run_output_dir"] is False
        assert preflight["uses_sweep_output_dir"] is False
        assert preflight["volume_output_root"] == str(DEFAULT_MODAL_PROBE_ROOT)
        assert preflight["gpu"] == gpu
        assert preflight["recipe"] == PROBE_RECIPE
        assert preflight["steps"] == 80
        assert preflight["launch_blockers"] == []
        assert "--throughput-probe --approved" in preflight["throughput_probe_command"]
        assert "--full-run" not in preflight["throughput_probe_command"]


def test_throughput_probe_rejects_unsupported_gpu() -> None:
    assert throughput_probe_blockers(modal_gpu="L4", timeout_hours=2) == [
        "SFT_MODAL_GPU must be one of A100, H100!, H200, or B200 for throughput probe"
    ]


def test_interval_sweep_writes_learning_gate_evidence_with_tiny_local_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sft_sweep,
        "SWEEP_ARMS",
        (
            SFTSweepArm(
                name="tiny-lr5e-2-b1",
                learning_rate=0.05,
                micro_batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=20,
                warmup_steps=0,
                eval_interval=5,
                log_interval=5,
                checkpoint_interval=10,
            ),
        ),
    )
    monkeypatch.setattr(sft_sweep, "SWEEP_TRAIN_SAMPLE_CAP", 5)
    monkeypatch.setattr(sft_sweep, "SWEEP_TRAIN_TOKEN_CAP", 500)
    monkeypatch.setattr(sft_sweep, "SWEEP_EVAL_SAMPLE_CAP", 2)
    monkeypatch.setattr(sft_sweep, "SWEEP_EVAL_TOKEN_CAP", 200)

    config = _tiny_sweep_config(tmp_path)
    output_root = tmp_path / SWEEP_OUTPUT_STEM

    payload = sft_sweep.run_interval_eval_sweep(
        config,
        output_root=output_root,
        allow_remote_download=False,
        require_cuda=False,
        wandb_enabled=False,
        dirty=False,
    )

    assert payload["status"] == "interval_eval_sweep_passed"
    assert payload["selected_best_arm"].endswith("tiny-lr5e-2-b1")
    assert Path(payload["interval_eval_sweep_path"]).is_file()
    learning_gate = json.loads(Path(payload["learning_gate_path"]).read_text(encoding="utf-8"))
    sweep_evidence = learning_gate["bounded_matched_interval_eval_sweep"]
    assert sweep_evidence["eval_metric"] == "eval/matched/response_loss"
    assert sweep_evidence["best_response_loss"] < sweep_evidence["step0_response_loss"]
    assert all(str(path).startswith(str(output_root)) for path in output_root.iterdir())
    arm = payload["arms"][0]
    assert arm["wandb_run"] is None
    assert arm["step0_eval"]["step"] == 0
    assert "eval/no_robots/response_loss" in arm["step0_eval"]
    assert arm["best_eval"]["step"] in arm["interval_eval_steps"]
    assert arm["train_sanity"]["finite_loss"] is True


def test_full_run_resume_requires_latest_checkpoint_in_chosen_output_dir(tmp_path: Path) -> None:
    config = load_sft_config(CONFIG_PATH)
    output_dir = tmp_path / "full-run"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("not a checkpoint\n", encoding="utf-8")

    with pytest.raises(SFTFullRunError, match="empty or absent"):
        run_full_instruct_sft(
            config,
            output_dir=output_dir,
            allow_remote_download=False,
            require_cuda=False,
            wandb_enabled=False,
        )

    with pytest.raises(SFTFullRunError, match="--resume requested but no checkpoint exists"):
        run_full_instruct_sft(
            config,
            output_dir=output_dir,
            allow_remote_download=False,
            require_cuda=False,
            wandb_enabled=False,
            resume_from_latest=True,
        )


def test_full_run_resume_flag_is_full_run_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert modal_instruct_sft.launch(["--config", str(CONFIG_PATH), "--resume", "--json"]) == 2
    assert "--resume requires --full-run" in capsys.readouterr().err


def test_cli_sft_dry_run_json_proves_no_modal_job(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["instruct-sft-dry-run", "--config", str(CONFIG_PATH), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["will_start_modal_job"] is False
    assert payload["launch_blockers"] == []


def _load_config_payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _tiny_sweep_config(tmp_path: Path) -> SFTLaunchConfig:
    bundle_dir = _write_tiny_bundle(tmp_path / "bundle")
    smol_path = tmp_path / "smol.jsonl"
    tulu_path = tmp_path / "tulu.jsonl"
    eval_path = tmp_path / "no_robots.jsonl"
    _write_jsonl(
        smol_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                ]
            }
            for _ in range(8)
        ],
    )
    _write_jsonl(
        tulu_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say blue"},
                    {"role": "assistant", "content": "blue"},
                ],
                "constraints": ["answer with one token"],
            },
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                ],
                "constraints": ["answer with one token"],
            },
            {
                "messages": [
                    {"role": "user", "content": "say blue"},
                    {"role": "assistant", "content": "blue"},
                ],
                "constraints": ["answer with one token"],
            },
        ],
    )
    _write_jsonl(
        eval_path,
        [
            {"instruction": "say red", "response": "red"},
            {"instruction": "say blue", "response": "blue"},
        ],
    )
    payload = _load_config_payload()
    payload["runtime"]["precision"] = "fp32"
    payload["base_bundle"]["path"] = str(bundle_dir)
    return SFTLaunchConfig(
        payload=payload,
        config_path=tmp_path / "config.json",
        base_bundle_path=bundle_dir,
        train_sources=(
            DatasetSource(
                name="smol-smoltalk",
                source="HuggingFaceTB/smol-smoltalk",
                revision="f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
                license="apache-2.0",
                split="train",
                role="train",
                mix_ratio=0.8,
                path=smol_path,
            ),
            DatasetSource(
                name="tulu-3-personas",
                source="allenai/tulu-3-sft-personas-instruction-following",
                revision="fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e",
                license="odc-by",
                split="train",
                role="train",
                mix_ratio=0.2,
                path=tulu_path,
            ),
        ),
        eval_source=DatasetSource(
            name="no_robots",
            source="HuggingFaceH4/no_robots",
            revision="e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b",
            license="cc-by-nc-4.0",
            split="test",
            role="eval",
            path=eval_path,
            train_allowed=False,
        ),
        output_dir=tmp_path / "unused",
        train_steps=20,
        tokens_per_step=1,
        estimated_full_cost_usd=0.0,
        estimated_smoke_cost_usd=0.0,
        smoke_launch_command="smoke",
        full_launch_command="full",
    )


def _complete_learning_gate_evidence() -> dict[str, object]:
    return {
        "stopped_run_reconciliation": {
            "kind": "stopped_run_reconciliation",
            "showcase_metrics_uri": "/posttrain/esme-instruct-sft-showcase-full/metrics.jsonl",
            "stopped_full_metrics_uri": "/posttrain/esme-instruct-sft-full/metrics.jsonl",
            "showcase_eval_rows": 98,
            "showcase_best_step": 600,
            "showcase_latest_step": 19400,
            "notes": "showcase-full and stopped full metrics are distinct",
        },
        "bounded_matched_interval_eval_sweep": _bounded_interval_eval_sweep_evidence(),
    }


def _bounded_interval_eval_sweep_evidence() -> dict[str, object]:
    return {
        "kind": "bounded_matched_interval_eval_sweep",
        "eval_metric": "eval/matched/response_loss",
        "baseline_step": 0,
        "step0_response_loss": 4.2,
        "best_response_loss": 3.9,
        "interval_eval_steps": [10, 20, 40],
        "evidence_uri": "runs/esme-214m-instruct-sft-pilot/interval-eval-sweep.json",
    }


def _write_tiny_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir()
    config = tiny_backbone_config()
    model = DenseBackbone(config)
    tokenizer = tiny_tokenizer()
    (bundle_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    tokenizer.save(str(bundle_dir / "tokenizer.json"))
    torch.save(
        {
            "format_version": 1,
            "format": "llm_pretrain_dense_v1",
            WEIGHTS_FIELD: "llm_pretrain_dense_v1",
            "state_dict_key": "dense_backbone",
            "state_dict": model.state_dict(),
            "model_config": config.to_dict(),
        },
        bundle_dir / "weights.pt",
    )
    manifest = {
        "schema_version": 1,
        "format": "llm_pretrain_dense_v1",
        "weights_format": "llm_pretrain_dense_v1",
        "model_family": "DenseBackbone",
        "model_config": config.to_dict(),
        "files": {
            "config": {
                "path": "config.json",
                "sha256": file_sha256(bundle_dir / "config.json"),
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": file_sha256(bundle_dir / "tokenizer.json"),
            },
            "weights": {
                "path": "weights.pt",
                "sha256": file_sha256(bundle_dir / "weights.pt"),
            },
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle_dir
