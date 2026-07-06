from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
from scripts import modal_chat_sft

from esme_posttrain.bundle import BundleError, file_sha256, load_dense_backbone_bundle
from esme_posttrain.cli import main
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.data import (
    IGNORE_INDEX,
    ChatTurn,
    DataError,
    DatasetSource,
    LossSemantics,
    MultiTurnExample,
    measure_multi_turn_lengths,
    tokenize_multi_turn,
)
from esme_posttrain.sft.full_multiturn import (
    RESAMPLE_SPEND_CAP_USD,
    SFTFullRunError,
    build_resample_evidence_preflight,
    resample_evidence_blockers,
    resample_multi_turn_evidence,
    run_full_multi_turn_sft,
)
from esme_posttrain.sft.launch_instruct import LaunchError
from esme_posttrain.sft.launch_multiturn import (
    EXPECTED_ARTIFACTS as MULTITURN_EXPECTED_ARTIFACTS,
)
from esme_posttrain.sft.launch_multiturn import (
    MULTITURN_FULL_RUN_SPEND_CAP_USD,
    build_multi_turn_dry_run,
    full_launch_blockers,
    load_multi_turn_config,
    smoke_launch_blockers,
    validate_multi_turn_payload,
)
from esme_posttrain.sft.multiturn_data import (
    build_multi_turn_eval_set,
    build_multi_turn_matched_eval_sets,
    build_multi_turn_mix,
    turn_distribution,
)
from esme_posttrain.sft.multiturn_judge import JudgeError, run_multi_turn_judge
from esme_posttrain.sft.probe_multiturn import (
    PROBE_SPEND_CAP_USD,
    build_multi_turn_probe_preflight,
    multi_turn_probe_blockers,
)
from esme_posttrain.sft.sample_artifacts import write_multi_turn_samples
from esme_posttrain.sft.smoke_multiturn import (
    run_multi_turn_cpu_fixture,
    tiny_backbone_config,
    tiny_chat_tokenizer,
)
from esme_posttrain.sft.sweep_multiturn import (
    SWEEP_ARMS,
    SWEEP_SPEND_CAP_USD,
    build_multi_turn_sweep_preflight,
    multi_turn_sweep_blockers,
)
from esme_posttrain.sft.trainer import load_sft_checkpoint
from esme_posttrain.training.collate import collate_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "esme-214m-sft-multiturn.json"
WEIGHTS_FIELD = "key_format"


# --- chat template + multi-turn masking ---------------------------------------


def test_multi_turn_masking_supervises_every_assistant_turn_and_masks_others() -> None:
    tokenizer = tiny_chat_tokenizer()
    example = MultiTurnExample(
        turns=(
            ChatTurn("system", "helpful"),
            ChatTurn("user", "say red"),
            ChatTurn("assistant", "red"),
            ChatTurn("user", "again"),
            ChatTurn("assistant", "red"),
            ChatTurn("user", "say blue"),
            ChatTurn("assistant", "blue"),
        ),
        source="fixture",
        row_id="mt-1",
    )
    tokenized = tokenize_multi_turn(tokenizer, example, max_sequence_tokens=64)

    assert tokenized.turns == 7
    assert tokenized.assistant_turns == 3
    assert tokenized.is_multi_turn

    eos_id = tokenizer.token_to_id("<eos>")
    assistant_marker = tokenizer.token_to_id("assistant")
    user_marker = tokenizer.token_to_id("user")

    # Supervised labels must equal their input ids; every other position is masked.
    for token, label in zip(tokenized.input_ids, tokenized.labels, strict=True):
        if label != IGNORE_INDEX:
            assert label == token

    # One supervised <eos> per assistant turn -> three assistant turns supervised.
    supervised_ids = [
        token
        for token, label in zip(tokenized.input_ids, tokenized.labels, strict=True)
        if label != IGNORE_INDEX
    ]
    assert supervised_ids.count(eos_id) == 3

    # Role markers and user/system content never enter the loss.
    for token, label in zip(tokenized.input_ids, tokenized.labels, strict=True):
        if token in {assistant_marker, user_marker}:
            assert label == IGNORE_INDEX

    # The leading system/user prompt span is fully masked.
    assert all(label == IGNORE_INDEX for label in tokenized.labels[: tokenized.prompt_tokens])
    assert tokenized.prompt_tokens > 0


def test_multi_turn_samples_prompt_ends_at_final_assistant_turn(tmp_path: Path) -> None:
    tokenizer = tiny_chat_tokenizer()
    model = DenseBackbone(tiny_backbone_config())
    example = MultiTurnExample(
        turns=(
            ChatTurn("system", "helpful"),
            ChatTurn("user", "say red"),
            ChatTurn("assistant", "blue"),  # unique to the first assistant turn
            ChatTurn("user", "again"),
            ChatTurn("assistant", "green"),  # unique to the final assistant turn
        ),
        source="fixture",
        row_id="mt-samples",
    )
    tokenized = tokenize_multi_turn(tokenizer, example, max_sequence_tokens=48)
    path = tmp_path / "multi-turn-samples.md"
    write_multi_turn_samples(
        path, model, tokenizer, (tokenized,), sample_new_tokens=4, selected_step=0
    )

    text = path.read_text(encoding="utf-8")
    assert "truncated before the final assistant" in text
    prompt_block = text.split("Prompt:")[1].split("Generation:")[0]
    prompt = prompt_block.split("```text\n")[1].split("\n```")[0]
    # Earlier assistant turns stay in the prompt; it ends with the final
    # assistant header and never leaks the final turn's content.
    assert "blue" in prompt
    assert prompt.rstrip().endswith("assistant")
    assert "green" not in prompt


def test_multi_turn_rejects_disabled_assistant_only_loss() -> None:
    tokenizer = tiny_chat_tokenizer()
    example = MultiTurnExample(
        turns=(ChatTurn("user", "say red"), ChatTurn("assistant", "red")),
        source="fixture",
        row_id="1",
    )
    with pytest.raises(DataError, match="assistant_only_loss"):
        tokenize_multi_turn(
            tokenizer,
            example,
            max_sequence_tokens=24,
            loss_semantics=LossSemantics(assistant_only_loss=False),
        )


def test_multi_turn_example_requires_an_assistant_turn() -> None:
    with pytest.raises(DataError, match="assistant turn"):
        MultiTurnExample(
            turns=(ChatTurn("user", "hi"),),
            source="fixture",
            row_id="1",
        )


def test_multi_turn_length_measurement_flags_truncation() -> None:
    tokenizer = tiny_chat_tokenizer()
    example = MultiTurnExample(
        turns=(
            ChatTurn("user", "say"),  # first prompt span: "user say assistant" = 3 tokens
            ChatTurn("assistant", "red"),
            ChatTurn("user", "say blue"),
            ChatTurn("assistant", "blue"),
        ),
        source="fixture",
        row_id="trunc",
    )
    # max length sits above the first-assistant offset (3) but below the full
    # conversation, so a later assistant turn is what overflows.
    lengths = measure_multi_turn_lengths(tokenizer, example, max_sequence_tokens=4)
    assert lengths["input_tokens"] > 4
    assert lengths["prompt_truncated"] is False
    assert lengths["assistant_target_truncated"] is True


# --- parsing, capacity filtering, mixing --------------------------------------


def test_multi_turn_parsing_capacity_filtering_and_turn_distribution(tmp_path: Path) -> None:
    smol_path = tmp_path / "smol.jsonl"
    rows = [
        # A 3-turn conversation with two assistant turns.
        {
            "source": "smol-magpie-ultra",
            "messages": [
                {"role": "user", "content": "say red"},
                {"role": "assistant", "content": "red"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "red"},
            ],
        },
        # A single-turn conversation.
        {
            "source": "smol-magpie-ultra",
            "messages": [
                {"role": "user", "content": "say blue"},
                {"role": "assistant", "content": "blue"},
            ],
        },
        # Capacity-filtered: function calling subset must be dropped.
        {
            "source": "xlam-function-calling-60k",
            "messages": [
                {"role": "user", "content": "call a tool"},
                {"role": "assistant", "content": "red"},
            ],
        },
    ]
    _write_jsonl(smol_path, rows)
    tulu_path = tmp_path / "tulu.jsonl"
    _write_jsonl(
        tulu_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "repeat one"},
                    {"role": "assistant", "content": "one"},
                ],
                "constraints": ["answer with one token"],
            }
        ],
    )
    sources = (
        DatasetSource(
            name="smol-smoltalk",
            source="local-smol",
            revision="0" * 40,
            license="apache-2.0",
            split="train",
            role="train",
            mix_ratio=0.85,
            path=smol_path,
        ),
        DatasetSource(
            name="tulu-3-personas",
            source="local-tulu",
            revision="1" * 40,
            license="odc-by",
            split="train",
            role="train",
            mix_ratio=0.15,
            path=tulu_path,
        ),
    )

    result = build_multi_turn_mix(
        sources, tiny_chat_tokenizer(), max_samples=10, max_tokens=1000, max_sequence_tokens=48
    )

    smol_counts = result.counts_by_source["smol-smoltalk"]
    assert smol_counts.rejected_capacity_filtered == 1
    assert smol_counts.selected == 2  # function-calling row dropped, two kept
    assert smol_counts.selected_multi_turn == 1
    assert smol_counts.selected_single_turn == 1

    distribution = turn_distribution(result.examples).to_dict()
    assert distribution["multi_turn_examples"] >= 1
    assert distribution["single_turn_examples"] >= 1
    assert distribution["mean_assistant_turns"] > 1.0

    # Tulu constraints are folded into the user turn (still single-turn, supervised).
    tulu_example = next(e for e in result.examples if e.source == "tulu-3-personas")
    assert tulu_example.assistant_turns == 1
    assert tulu_example.supervised_tokens > 0


def test_multi_turn_mix_accepts_single_source(tmp_path: Path) -> None:
    smol_path = tmp_path / "smol.jsonl"
    _write_jsonl(
        smol_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "red"},
                ]
            }
            for _ in range(3)
        ],
    )
    sources = (
        DatasetSource(
            name="smol-smoltalk",
            source="local-smol",
            revision="0" * 40,
            license="apache-2.0",
            split="train",
            role="train",
            mix_ratio=1.0,
            path=smol_path,
        ),
    )
    result = build_multi_turn_mix(
        sources, tiny_chat_tokenizer(), max_samples=3, max_tokens=1000, max_sequence_tokens=48
    )
    assert len(result.examples) == 3
    assert all(example.assistant_turns == 2 for example in result.examples)


def test_multi_turn_mix_rejects_too_many_sources(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "a"}]}])
    sources = tuple(
        DatasetSource(
            name="smol-smoltalk",
            source="s",
            revision="0" * 40,
            license="apache-2.0",
            split="train",
            role="train",
            mix_ratio=1 / 3,
            path=path,
        )
        for _ in range(3)
    )
    with pytest.raises(DataError, match="one or two train sources"):
        build_multi_turn_mix(
            sources, tiny_chat_tokenizer(), max_samples=3, max_tokens=10, max_sequence_tokens=24
        )


def test_multi_turn_modal_smoke_body_refreshes_manifest_with_expected_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "modal-smoke"
    refresh_calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_run_multi_turn_cpu_fixture(
        config: object, *, output_dir: Path, wandb_enabled: bool
    ) -> dict:
        output_dir.mkdir(parents=True)
        return {"status": "local_cpu_fixture_complete", "wandb_enabled": wandb_enabled}

    def fake_refresh_manifest_files(output_dir: Path, expected_artifacts: tuple[str, ...]) -> None:
        refresh_calls.append((output_dir, expected_artifacts))

    monkeypatch.setattr(modal_chat_sft, "fresh_output_dir", lambda root, stem: output_dir)
    monkeypatch.setattr(
        modal_chat_sft,
        "run_multi_turn_cpu_fixture",
        fake_run_multi_turn_cpu_fixture,
    )
    monkeypatch.setattr(modal_chat_sft, "refresh_manifest_files", fake_refresh_manifest_files)

    payload = modal_chat_sft._run_modal_smoke_body(
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        commit="abc123",
        dirty=False,
        started=modal_chat_sft.time.perf_counter(),
    )

    assert refresh_calls == [(output_dir, MULTITURN_EXPECTED_ARTIFACTS)]
    assert (output_dir / "cost.json").is_file()
    assert payload["status"] == "modal_smoke_complete"


# --- matched eval split -------------------------------------------------------


def test_matched_eval_splits_are_disjoint_from_train(tmp_path: Path) -> None:
    smol_path = tmp_path / "smol.jsonl"
    _write_jsonl(
        smol_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "red"},
                ]
            }
            for _ in range(6)
        ],
    )
    tulu_path = tmp_path / "tulu.jsonl"
    _write_jsonl(
        tulu_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "repeat one"},
                    {"role": "assistant", "content": "one"},
                ]
            }
            for _ in range(4)
        ],
    )
    sources = (
        DatasetSource(
            name="smol-smoltalk",
            source="local-smol",
            revision="0" * 40,
            license="apache-2.0",
            split="train",
            role="train",
            mix_ratio=0.5,
            path=smol_path,
        ),
        DatasetSource(
            name="tulu-3-personas",
            source="local-tulu",
            revision="1" * 40,
            license="odc-by",
            split="train",
            role="train",
            mix_ratio=0.5,
            path=tulu_path,
        ),
    )
    tokenizer = tiny_chat_tokenizer()
    train = build_multi_turn_mix(
        sources, tokenizer, max_samples=4, max_tokens=1000, max_sequence_tokens=48
    )
    matched = build_multi_turn_matched_eval_sets(
        sources,
        tokenizer,
        skip_selected_by_source={
            name: counts.selected for name, counts in train.counts_by_source.items()
        },
        max_samples_per_source=1,
        max_tokens_per_source=500,
        max_sequence_tokens=48,
    )
    train_keys = {(e.source, e.row_id) for e in train.examples}
    heldout_keys = {(e.source, e.row_id) for report in matched.values() for e in report.examples}
    assert heldout_keys.isdisjoint(train_keys)
    assert set(matched) == {"smol-smoltalk", "tulu-3-personas"}


def test_multi_turn_eval_set_rejects_train_allowed_source(tmp_path: Path) -> None:
    path = tmp_path / "no_robots.jsonl"
    _write_jsonl(path, [{"instruction": "say red", "response": "red"}])
    bad = DatasetSource(
        name="no_robots",
        source="HuggingFaceH4/no_robots",
        revision="2" * 40,
        license="cc-by-nc-4.0",
        split="train",
        role="eval",
        train_allowed=True,
        path=path,
    )
    with pytest.raises(DataError, match="eval holdout must be false"):
        build_multi_turn_eval_set(
            bad, tiny_chat_tokenizer(), max_samples=1, max_tokens=50, max_sequence_tokens=24
        )


# --- config + dry-run + guards ------------------------------------------------


def test_config_pins_multi_turn_recipe_and_dry_run_never_launches() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    assert config.run_id == "esme_214m_sft_multiturn"
    assert config.artifact_name == "Esme-214M-Instruct"
    assert config.budgets["max_sequence_tokens"] == 1024
    assert [s.name for s in config.train_sources] == ["smol-smoltalk", "tulu-3-personas"]
    assert config.runtime["modal_volume"] == "esme-posttrain-esme-sft-multiturn"

    dry_run = build_multi_turn_dry_run(config)
    assert dry_run["will_start_modal_job"] is False
    assert dry_run["will_download_data"] is False
    assert dry_run["preflight"]["will_start_modal_job"] is False
    assert dry_run["preflight"]["dataset_revisions"]["smol-smoltalk"]
    assert dry_run["preflight"]["projected_cost_usd"] > 0
    assert "--full-run" in dry_run["full_launch_command"]
    assert "modal_chat_sft.py" in dry_run["modal_smoke_command"]


def test_smoke_cost_is_under_two_dollars() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    assert config.estimated_smoke_cost_usd <= 2
    assert smoke_launch_blockers(config) == []


def test_full_run_refused_without_approval() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=False, modal_gpu="A100")
    assert any("requires --approved" in b for b in blockers)


def test_full_run_refused_without_learning_gate_evidence() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["learning_gate"]["evidence"] = None
    config = validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert any("bounded_matched_interval_eval_sweep" in b for b in blockers)


def test_shipped_config_satisfies_learning_gate() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    assert config.payload["learning_gate"]["evidence"] is not None
    assert full_launch_blockers(config, approved=True, modal_gpu="A100") == []


def test_full_run_blocks_modal_gpu_mismatch() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="L4")
    assert any("SFT_MODAL_GPU must match" in b for b in blockers)


def test_full_run_accepts_matched_sweep_evidence(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["learning_gate"]["evidence"] = {
        "bounded_matched_interval_eval_sweep": {
            "kind": "bounded_matched_interval_eval_sweep",
            "eval_metric": "eval/matched/response_loss",
            "baseline_step": 0,
            "step0_response_loss": 2.2,
            "best_response_loss": 1.6,
            "interval_eval_steps": [20, 40, 60],
            "evidence_uri": "/posttrain/esme-214m-sft-multiturn-sweep/evidence.json",
        }
    }
    config = validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert blockers == []


def test_full_run_rejects_flat_sweep_evidence(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["learning_gate"]["evidence"] = {
        "bounded_matched_interval_eval_sweep": {
            "kind": "bounded_matched_interval_eval_sweep",
            "eval_metric": "eval/matched/response_loss",
            "baseline_step": 0,
            "step0_response_loss": 2.0,
            "best_response_loss": 2.0,
            "interval_eval_steps": [20],
            "evidence_uri": "x",
        }
    }
    config = validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert any("best_response_loss < step0_response_loss" in b for b in blockers)


def test_config_rejects_non_1024_sequence_length() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["budgets"]["max_sequence_tokens"] = 2048
    with pytest.raises(LaunchError, match="max_sequence_tokens must be 1024"):
        validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)


def test_config_rejects_noncommercial_training_flag() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["datasets"]["non_commercial_training_approved"] = True
    with pytest.raises(LaunchError, match="non_commercial_training_approved"):
        validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)


def test_config_requires_judge_passes_at_least_five() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["monitoring"]["judge_repeat_passes"] = 3
    with pytest.raises(LaunchError, match="judge_repeat_passes must be >= 5"):
        validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)


# --- LLM judge (reported, not selector) ---------------------------------------


def test_multi_turn_judge_reports_mean_and_spread_over_k_passes() -> None:
    rng = random.Random(7)
    report = run_multi_turn_judge(
        lambda prompt: "blue",
        lambda prompt, generation: 6.0 + rng.uniform(-1.0, 1.0),
        passes=5,
    )
    payload = report.to_dict()
    assert payload["available"] is True
    assert payload["is_selector"] is False
    assert payload["passes"] == 5
    assert len(payload["per_pass_mean_scores"]) == 5
    assert payload["score_min"] <= payload["mean_score"] <= payload["score_max"]
    assert payload["score_stdev"] >= 0.0


def test_multi_turn_judge_requires_at_least_five_passes() -> None:
    with pytest.raises(JudgeError, match="K>=5"):
        run_multi_turn_judge(lambda prompt: "x", lambda prompt, generation: 5.0, passes=4)


def test_multi_turn_judge_reports_unavailable_without_a_judge() -> None:
    report = run_multi_turn_judge(lambda prompt: "x", None, passes=5)
    payload = report.to_dict()
    assert payload["available"] is False
    assert payload["mean_score"] is None


# --- base bundle hash validation ----------------------------------------------


def test_base_bundle_hash_validation_and_dense_load(tmp_path: Path) -> None:
    bundle_dir = _write_tiny_chat_bundle(tmp_path / "bundle")
    loaded = load_dense_backbone_bundle(bundle_dir)
    assert isinstance(loaded.model, DenseBackbone)

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["weights"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(BundleError, match="hash mismatch"):
        load_dense_backbone_bundle(bundle_dir)


# --- CPU fixture: loss decrease, masking, checkpoint round-trip ----------------


def test_cpu_fixture_multi_turn_loss_decreases_and_masking_asserted(tmp_path: Path) -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    out = tmp_path / "fixture"
    result = run_multi_turn_cpu_fixture(config, output_dir=out)

    assert result["status"] == "local_cpu_fixture_multi_turn_complete"
    assert result["assistant_only_multi_turn_masking_asserted"] is True
    assert result["turn_distribution"]["multi_turn_examples"] >= 1
    assert result["result"]["response_loss_decreased"] is True
    assert result["result"]["instruct_beats_base"] is True
    assert all(result["required_artifacts_present"].values())
    assert (out / "multi-turn-samples.md").is_file()


def test_cpu_fixture_checkpoint_reload_reproduces_logits(tmp_path: Path) -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    out = tmp_path / "fixture"
    run_multi_turn_cpu_fixture(config, output_dir=out)

    checkpoint = load_sft_checkpoint(out / "best-checkpoint.pt")
    tokenizer = tiny_chat_tokenizer()
    example = tokenize_multi_turn(
        tokenizer,
        MultiTurnExample(
            turns=(ChatTurn("user", "say red"), ChatTurn("assistant", "red")),
            source="fixture",
            row_id="1",
        ),
        max_sequence_tokens=48,
    )
    input_ids, _ = collate_batch((example,), device="cpu")
    model = checkpoint.model
    model.eval()
    with torch.no_grad():
        first = model(input_ids[:, :-1])
        second = model(input_ids[:, :-1])
    assert torch.equal(first, second)


def test_full_run_requires_cuda_when_required(tmp_path: Path) -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    if torch.cuda.is_available():
        pytest.skip("CUDA available; the loud-failure path is not exercised")
    with pytest.raises(SFTFullRunError, match="requires CUDA"):
        run_full_multi_turn_sft(
            config,
            output_dir=tmp_path / "full",
            allow_remote_download=False,
            require_cuda=True,
            wandb_enabled=False,
        )


# --- CLI ----------------------------------------------------------------------


def test_cli_multi_turn_dry_run_proves_no_modal_job(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["sft-multiturn-dry-run", "--config", str(CONFIG_PATH), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["will_start_modal_job"] is False
    assert payload["run_id"] == "esme_214m_sft_multiturn"


def test_cli_multi_turn_cpu_fixture_writes_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fixture"
    assert (
        main(
            [
                "sft-multiturn-cpu-fixture",
                "--config",
                str(CONFIG_PATH),
                "--output-dir",
                str(out),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "local_cpu_fixture_multi_turn_complete"
    assert (out / "metrics.jsonl").is_file()


# --- bounded evidence resample (multi-turn-samples-v2.md) ----------------------


FULL_OUTPUT_STEM = "esme-214m-sft-multiturn-full"


def test_multi_turn_base_bundle_resolves_env_or_sibling_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_bundle = tmp_path / "base-bundle"
    monkeypatch.setenv(modal_chat_sft.BASE_BUNDLE_ENV, str(env_bundle))
    assert modal_chat_sft._resolve_base_bundle_local() == env_bundle

    monkeypatch.delenv(modal_chat_sft.BASE_BUNDLE_ENV, raising=False)
    assert modal_chat_sft._resolve_base_bundle_local() == (
        REPO_ROOT.parent / "esme-pretrain" / "exports" / "esme-214m-base"
    )


def test_training_launches_block_when_base_bundle_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_bundle = tmp_path / "missing-base"
    monkeypatch.setattr(modal_chat_sft, "BASE_BUNDLE_LOCAL", missing_bundle)
    blocker = modal_chat_sft._base_bundle_blocker()
    assert blocker is not None
    assert str(missing_bundle) in blocker
    assert modal_chat_sft.BASE_BUNDLE_ENV in blocker
    assert "sibling-repo fallback" in blocker
    assert "re-export Esme-214M-Base" in blocker

    monkeypatch.setattr(modal_chat_sft, "BASE_BUNDLE_LOCAL", tmp_path)
    assert modal_chat_sft._base_bundle_blocker() is None

    monkeypatch.setattr(modal_chat_sft, "BASE_BUNDLE_LOCAL", missing_bundle)
    monkeypatch.setattr(modal_chat_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_chat_sft, "modal", object())
    monkeypatch.setattr(modal_chat_sft, "run_modal_full_sft", _RefusingRunner())

    assert (
        modal_chat_sft.launch(["--config", str(CONFIG_PATH), "--full-run", "--approved", "--json"])
        == 2
    )
    refused = json.loads(capsys.readouterr().out)
    assert refused["will_start_modal_job"] is False
    assert any(
        str(missing_bundle) in blocker and modal_chat_sft.BASE_BUNDLE_ENV in blocker
        for blocker in refused["full_launch_blockers"]
    )


def test_resample_evidence_preflight_never_launches_and_fits_cap() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    preflight = build_resample_evidence_preflight(
        config, modal_gpu="A100", output_stem=FULL_OUTPUT_STEM
    )
    assert preflight["status"] == "ready_for_resample_evidence"
    assert preflight["will_start_modal_job"] is False
    assert preflight["will_download_data"] is False
    assert preflight["generation_only"] is True
    assert preflight["launch_blockers"] == []
    assert preflight["checkpoint"] == f"{FULL_OUTPUT_STEM}/best-checkpoint.pt"
    assert preflight["outputs"]["resampled_markdown"].endswith("multi-turn-samples-v2.md")
    assert preflight["outputs"]["original_preserved"].endswith("multi-turn-samples.md")
    assert preflight["runtime"]["spend_cap_usd"] == RESAMPLE_SPEND_CAP_USD == 1.0
    assert preflight["runtime"]["timeout_cost_ceiling_usd"] <= RESAMPLE_SPEND_CAP_USD
    assert preflight["runtime"]["sample_new_tokens"] == 256
    assert "modal_chat_sft.py" in preflight["resample_evidence_command"]
    assert "--resample-evidence" in preflight["resample_evidence_command"]
    assert "modal run --detach" in preflight["resample_evidence_command"]
    assert "SFT_RESAMPLE_TIMEOUT_HOURS=0.25" in preflight["resample_evidence_command"]


def test_resample_evidence_blocks_gpu_mismatch_and_over_cap_timeout() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    assert any(
        "SFT_MODAL_GPU must match runtime.selected_gpu" in b
        for b in resample_evidence_blockers(config, modal_gpu="L4")
    )
    # 1h on the $2.0988/h A100 breaks the $1 resample cap.
    blockers = resample_evidence_blockers(config, modal_gpu="A100", timeout_hours=1.0)
    assert any("exceeds the $1 resample spend cap" in b for b in blockers)
    assert resample_evidence_blockers(config, modal_gpu="A100", timeout_hours=0.25) == []


class _RefusingRunner:
    def spawn(self, *args: object) -> object:
        del args
        raise AssertionError("resample launcher must not spawn a Modal function here")


def test_launcher_resample_evidence_dry_run_never_starts_modal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(modal_chat_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_chat_sft, "SFT_RESAMPLE_TIMEOUT_HOURS", 0.25)
    monkeypatch.setattr(modal_chat_sft, "run_modal_resample_evidence", _RefusingRunner())

    assert (
        modal_chat_sft.launch(
            ["--config", str(CONFIG_PATH), "--resample-evidence", "--dry-run", "--json"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready_for_resample_evidence"
    assert payload["will_start_modal_job"] is False
    assert payload["runtime"]["timeout_hours"] == 0.25
    assert payload["runtime"]["timeout_cost_ceiling_usd"] <= 1.0


def test_launcher_resample_evidence_refused_without_approval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(modal_chat_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_chat_sft, "run_modal_resample_evidence", _RefusingRunner())

    assert (
        modal_chat_sft.launch(["--config", str(CONFIG_PATH), "--resample-evidence", "--json"]) == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "resample_evidence_refused"
    assert any("requires --approved" in b for b in payload["launch_blockers"])


def test_launcher_resample_evidence_blocks_over_cap_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(modal_chat_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_chat_sft, "SFT_RESAMPLE_TIMEOUT_HOURS", 1.0)
    monkeypatch.setattr(modal_chat_sft, "run_modal_resample_evidence", _RefusingRunner())

    assert (
        modal_chat_sft.launch(
            ["--config", str(CONFIG_PATH), "--resample-evidence", "--approved", "--json"]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "resample_evidence_refused"
    assert any("exceeds the $1 resample spend cap" in b for b in payload["launch_blockers"])


def test_launcher_resample_evidence_spawn_returns_receipt_without_get(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    class FakeCall:
        object_id = "fc-resample-test"

        def get(self) -> dict[str, object]:
            raise AssertionError("resample launcher must not wait for the remote result")

    class FakeRunner:
        def spawn(self, *args: object) -> FakeCall:
            captured["args"] = args
            return FakeCall()

    monkeypatch.setattr(modal_chat_sft, "SFT_MODAL_GPU", "A100")
    monkeypatch.setattr(modal_chat_sft, "SFT_RESAMPLE_TIMEOUT_HOURS", 0.25)
    monkeypatch.setattr(modal_chat_sft, "modal", object())
    monkeypatch.setattr(modal_chat_sft, "run_modal_resample_evidence", FakeRunner())

    assert (
        modal_chat_sft.launch(
            ["--config", str(CONFIG_PATH), "--resample-evidence", "--approved", "--json"]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    args = captured["args"]
    assert args[0] == load_multi_turn_config(CONFIG_PATH).payload
    assert args[3] == "A100"
    assert args[4] == 0.25
    assert args[5] == FULL_OUTPUT_STEM
    assert receipt["status"] == "modal_resample_evidence_launched"
    assert receipt["will_start_modal_job"] is True
    assert receipt["modal_result_awaited"] is False
    assert receipt["modal_call_id"] == "fc-resample-test"
    assert receipt["volume"] == "esme-posttrain-esme-sft-multiturn"
    assert receipt["volume_output_dir"] == f"/posttrain/{FULL_OUTPUT_STEM}"
    assert receipt["timeout_cost_ceiling_usd"] <= receipt["spend_cap_usd"] == 1.0


def test_resample_multi_turn_evidence_writes_resample_artifact(
    tmp_path: Path,
) -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    out = tmp_path / "full-run"
    run_multi_turn_cpu_fixture(config, output_dir=out)
    original_text = (out / "multi-turn-samples.md").read_text(encoding="utf-8")

    tokenizer = tiny_chat_tokenizer()
    pool_example = tokenize_multi_turn(
        tokenizer,
        MultiTurnExample(
            turns=(
                ChatTurn("system", "helpful"),
                ChatTurn("user", "say red"),
                ChatTurn("assistant", "blue"),  # unique to the first assistant turn
                ChatTurn("user", "again"),
                ChatTurn("assistant", "green"),  # unique to the final assistant turn
            ),
            source="fixture",
            row_id="resample-mt-1",
        ),
        max_sequence_tokens=48,
    )
    result = resample_multi_turn_evidence(
        config,
        output_dir=out,
        allow_remote_download=False,
        require_cuda=False,
        commit="abc123",
        dirty=False,
        sample_pool=(pool_example,),
    )

    assert result["status"] == "resample_evidence_complete"
    assert result["original_samples_preserved"] is True
    assert result["resampled_samples_path"] == str(out / "multi-turn-samples-v2.md")
    assert result["estimated_cost_usd"] <= result["spend_cap_usd"] == 1.0
    # Existing sample evidence is untouched byte for byte.
    assert (out / "multi-turn-samples.md").read_text(encoding="utf-8") == original_text

    text = (out / "multi-turn-samples-v2.md").read_text(encoding="utf-8")
    assert result["resampled_markdown"] == text
    assert "truncated before the final assistant" in text
    prompt_block = text.split("Prompt:")[1].split("Generation:")[0]
    prompt = prompt_block.split("```text\n")[1].split("\n```")[0]
    # Fixed truncation: earlier assistant turns stay in the prompt; it ends at
    # the final assistant header and never leaks the final turn's content.
    assert "blue" in prompt
    assert prompt.rstrip().endswith("assistant")
    assert "green" not in prompt


def test_resample_multi_turn_evidence_requires_completed_run(tmp_path: Path) -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    with pytest.raises(SFTFullRunError, match="best checkpoint"):
        resample_multi_turn_evidence(
            config,
            output_dir=tmp_path / "missing",
            allow_remote_download=False,
            require_cuda=False,
        )


# --- helpers ------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_tiny_chat_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir()
    config = tiny_backbone_config()
    model = DenseBackbone(config)
    tokenizer = tiny_chat_tokenizer()
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
            "config": {"path": "config.json", "sha256": file_sha256(bundle_dir / "config.json")},
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": file_sha256(bundle_dir / "tokenizer.json"),
            },
            "weights": {"path": "weights.pt", "sha256": file_sha256(bundle_dir / "weights.pt")},
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle_dir


# --- raised $40 full-run cap (multi-turn only) --------------------------------


def test_config_pins_forty_dollar_full_run_cap() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    assert MULTITURN_FULL_RUN_SPEND_CAP_USD == 40.0
    assert float(config.runtime["full_run_max_cost_usd"]) == 40.0
    assert float(config.runtime["full_run_runtime_spend_stop_usd"]) == 40.0


def test_full_run_cap_above_forty_is_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["full_run_max_cost_usd"] = 41
    with pytest.raises(LaunchError, match="full_run_max_cost_usd must be <= 40"):
        validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)


def test_full_run_blocks_projected_cost_over_cap() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    # A very slow projected rate pushes projected full-run cost above the $40 cap.
    payload["runtime"]["gpu_profiles"]["A100"]["projected_tokens_per_second"] = 100.0
    payload["learning_gate"]["evidence"] = {
        "bounded_matched_interval_eval_sweep": {
            "kind": "bounded_matched_interval_eval_sweep",
            "eval_metric": "eval/matched/response_loss",
            "baseline_step": 0,
            "step0_response_loss": 2.2,
            "best_response_loss": 1.6,
            "interval_eval_steps": [20, 40, 60],
            "evidence_uri": "x",
        }
    }
    config = validate_multi_turn_payload(payload, CONFIG_PATH, require_base_bundle_exists=False)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert any("exceeds runtime.full_run_max_cost_usd" in b for b in blockers)


# --- multi-turn throughput probe ----------------------------------------------


def test_multi_turn_probe_preflight_never_launches_and_is_bounded() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    preflight = build_multi_turn_probe_preflight(config, modal_gpu="A100", timeout_hours=1)
    assert preflight["status"] == "ready_for_throughput_probe"
    assert preflight["will_start_modal_job"] is False
    assert preflight["launch_blockers"] == []
    assert preflight["max_sequence_tokens"] == 1024
    assert preflight["runtime"]["spend_cap_usd"] == PROBE_SPEND_CAP_USD == 3.0
    assert preflight["runtime"]["timeout_cost_ceiling_usd"] <= PROBE_SPEND_CAP_USD
    assert "modal_chat_sft.py" in preflight["throughput_probe_command"]


def test_multi_turn_probe_blocks_unknown_gpu_and_long_timeout() -> None:
    assert any(
        "SFT_MODAL_GPU must be one of" in b
        for b in multi_turn_probe_blockers(modal_gpu="L4", timeout_hours=1)
    )
    assert any(
        "SFT_PROBE_TIMEOUT_HOURS" in b
        for b in multi_turn_probe_blockers(modal_gpu="A100", timeout_hours=5)
    )


# --- multi-turn matched-eval LR sweep -----------------------------------------


def test_multi_turn_sweep_preflight_never_launches_and_fits_cap() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    preflight = build_multi_turn_sweep_preflight(config, timeout_hours=3, modal_gpu="A100")
    assert preflight["status"] == "ready_for_modal_sweep"
    assert preflight["will_start_modal_job"] is False
    assert preflight["launch_blockers"] == []
    assert preflight["runtime"]["projected_cost_usd"] <= SWEEP_SPEND_CAP_USD == 8.0
    assert preflight["acceptance"]["metric"] == "eval/matched/response_loss"
    assert "modal_chat_sft.py" in preflight["modal_sweep_command"]


def test_multi_turn_sweep_arms_anchor_around_config_lr_and_stay_bounded() -> None:
    learning_rates = [arm.learning_rate for arm in SWEEP_ARMS]
    assert 1e-4 in learning_rates
    assert max(learning_rates) > 1e-4  # explores higher than the 1e-4 anchor
    for arm in SWEEP_ARMS:
        assert arm.effective_batch_size == 16
        assert arm.max_steps <= 200
        assert arm.eval_interval > 0


def test_multi_turn_sweep_blocks_gpu_mismatch() -> None:
    config = load_multi_turn_config(CONFIG_PATH)
    blockers = multi_turn_sweep_blockers(config, timeout_hours=3, modal_gpu="L4")
    assert any("SFT_MODAL_GPU must match runtime.selected_gpu" in b for b in blockers)
