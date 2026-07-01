from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from esme_posttrain.bundle import BundleError, file_sha256, load_dense_backbone_bundle
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.checkpointing import latest_checkpoint_path
from esme_posttrain.sft.data import (
    DatasetSource,
    SingleTurnExample,
    sequence_efficiency_report,
    tokenize_single_turn,
)
from esme_posttrain.sft.launch_instruct import (
    SFTLaunchConfig,
)
from esme_posttrain.sft.metrics import EVAL_METRIC_NAMES, TRAIN_METRIC_NAMES
from esme_posttrain.sft.smoke_instruct import tiny_backbone_config, tiny_tokenizer
from esme_posttrain.sft.trainer import (
    EvalMetrics,
    EvalSplit,
    EvalSuiteResult,
    SFTTrainerConfig,
    _generate_samples,
    _markdown_fenced_text,
    _no_robots_catastrophic_regression,
    collate_batch,
    load_sft_checkpoint,
    resolve_torch_device,
    run_sft_training,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "esme-214m-instruct.json"
WEIGHTS_FIELD = "key" "_format"


def test_collate_batch_honors_device() -> None:
    tokenized = tokenize_single_turn(
        tiny_tokenizer(),
        SingleTurnExample("say red", "red", "fixture", "1"),
        max_sequence_tokens=24,
    )

    input_ids, labels = collate_batch((tokenized,), device="cpu")

    assert input_ids.device.type == "cpu"
    assert labels.device.type == "cpu"


def test_collate_batch_can_pad_to_multiple_and_reports_padding_efficiency() -> None:
    tokenizer = tiny_tokenizer()
    examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=24)
        for example in (
            SingleTurnExample("say red", "red", "fixture", "train-1"),
            SingleTurnExample("repeat one", "one", "fixture", "train-2"),
        )
    )

    input_ids, labels = collate_batch(examples, device="cpu", pad_to_multiple_of=8)
    report = sequence_efficiency_report(
        examples,
        max_sequence_tokens=24,
        micro_batch_size=2,
        sequence_packing=False,
        pad_to_multiple_of=8,
        no_packing_rationale="measured no-packing fixture",
    )

    assert input_ids.shape[1] % 8 == 0
    assert labels.shape == input_ids.shape
    assert report["sequence_packing"] is False
    assert report["padding_tokens"] > 0
    assert 0 < report["padding_efficiency"] < 1


def test_cuda_device_request_fails_loudly_when_unavailable() -> None:
    if torch.cuda.is_available():
        assert resolve_torch_device("cuda").type == "cuda"
        return

    with pytest.raises(ValueError, match="CUDA was requested"):
        resolve_torch_device("cuda")


def test_bf16_precision_requires_supported_cuda(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only validation test")
    tokenizer = tiny_tokenizer()
    example = tokenize_single_turn(
        tokenizer,
        SingleTurnExample("say red", "red", "fixture", "train-1"),
        max_sequence_tokens=24,
    )

    with pytest.raises(ValueError, match="bf16 precision requires CUDA"):
        run_sft_training(
            DenseBackbone(tiny_backbone_config()),
            tokenizer,
            (example,),
            (example,),
            SFTTrainerConfig(
                max_steps=1,
                micro_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=0.05,
                seed=214,
                output_dir=tmp_path / "bf16",
                precision="bf16",
            ),
        )


def test_bundle_hash_validation_and_dense_load(tmp_path: Path) -> None:
    bundle_dir = _write_tiny_bundle(tmp_path / "bundle")

    loaded = load_dense_backbone_bundle(bundle_dir)

    assert loaded.bundle.config.name == "tiny-sft-fixture"
    assert loaded.tokenizer.token_to_id("<eos>") == 1

    (bundle_dir / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleError, match="hash mismatch"):
        load_dense_backbone_bundle(bundle_dir)


def test_dense_generate_stops_after_eos(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DenseBackbone(tiny_backbone_config())
    next_tokens = iter((5, 1, 6))

    def fake_forward(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            model.config.vocab_size,
            dtype=torch.float32,
        )
        logits[:, -1, next(next_tokens)] = 10.0
        return logits

    monkeypatch.setattr(model, "forward", fake_forward)
    model.train()

    generated = model.generate(torch.tensor([[2, 3]]), max_new_tokens=5, eos_token_id=1)

    assert generated.tolist() == [[2, 3, 5, 1]]
    assert model.training is True


def test_sample_generation_trims_eos_marker_and_trailing_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = tiny_tokenizer()
    model = DenseBackbone(tiny_backbone_config())
    example = tokenize_single_turn(
        tokenizer,
        SingleTurnExample("say red", "red", "fixture", "eval-1"),
        max_sequence_tokens=model.config.context_length,
    )

    def fake_generate(
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        assert max_new_tokens == 4
        assert eos_token_id == 1
        return torch.tensor([input_ids[0].tolist() + [5, 1, 6]], dtype=torch.long)

    monkeypatch.setattr(model, "generate", fake_generate)

    (sample,) = _generate_samples(model, tokenizer, (example,), sample_new_tokens=4)

    assert sample.endswith("assistant red")
    assert "<eos>" not in sample
    assert "blue" not in sample


def test_markdown_sample_fence_handles_generated_code_blocks() -> None:
    lines = _markdown_fenced_text("```python\nprint('hi')\n```")

    assert lines[0] == "````text"
    assert lines[-1] == "````"


def test_cpu_fixture_sft_loss_decreases_and_checkpoint_reloads_logits(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    model = DenseBackbone(tiny_backbone_config())
    train_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=model.config.context_length)
        for example in (
            SingleTurnExample("say red", "red", "fixture", "train-1"),
            SingleTurnExample("say blue", "blue", "fixture", "train-2"),
            SingleTurnExample("repeat one", "one", "fixture", "train-3"),
            SingleTurnExample("repeat two", "two", "fixture", "train-4"),
        )
    )
    eval_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=model.config.context_length)
        for example in (
            SingleTurnExample("say red", "red", "fixture", "eval-1"),
            SingleTurnExample("repeat two", "two", "fixture", "eval-2"),
        )
    )

    result = run_sft_training(
        model,
        tokenizer,
        train_examples,
        eval_examples,
        SFTTrainerConfig(
            max_steps=40,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=0.05,
            seed=214,
            output_dir=tmp_path / "sft",
            eval_interval=10,
            checkpoint_interval=10,
            retain_last_checkpoints=2,
            log_interval=10,
            sample_new_tokens=4,
        ),
    )
    loaded = load_sft_checkpoint(result.checkpoint_path)
    input_ids, _labels = collate_batch(eval_examples)

    assert result.instruct_eval.response_loss < result.base_eval.response_loss
    assert result.trained_tokens > 0
    assert result.supervised_tokens > 0
    assert result.effective_epochs > 1
    assert loaded.step == result.selected_step
    assert result.best_checkpoint_path.is_file()
    best_metadata = json.loads(result.best_checkpoint_metadata_path.read_text(encoding="utf-8"))
    assert best_metadata["selected_metric"] == "eval/heldout/response_loss"
    assert best_metadata["selected_step"] == result.selected_step
    assert torch.allclose(model(input_ids), loaded.model(input_ids), atol=1e-6)
    metrics = [
        json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {
        key for payload in metrics if payload["event"] == "train" for key in payload
    } >= TRAIN_METRIC_NAMES
    legacy_eval_names = {
        "eval/response_loss",
        "eval/perplexity",
        "eval/supervised_tokens",
        "eval/examples",
    }
    assert {
        key for payload in metrics if payload["event"] == "eval" for key in payload
    } >= legacy_eval_names
    checkpoints = sorted((result.output_dir / "checkpoints").glob("step-*/checkpoint.pt"))
    assert len(checkpoints) == 2
    assert checkpoints[-1].parent.name == "step-000040"


def test_matched_eval_selector_best_checkpoint_and_no_robots_guardrail(
    tmp_path: Path,
) -> None:
    tokenizer = tiny_tokenizer()
    train_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=24)
        for example in (
            SingleTurnExample("say red", "red", "smol-smoltalk", "train-smol-1"),
            SingleTurnExample("say blue", "blue", "smol-smoltalk", "train-smol-2"),
            SingleTurnExample("repeat one", "one", "tulu-3-personas", "train-tulu-1"),
            SingleTurnExample("repeat two", "two", "tulu-3-personas", "train-tulu-2"),
        )
    )
    smol_eval = (
        tokenize_single_turn(
            tokenizer,
            SingleTurnExample("say red", "red", "smol-smoltalk", "eval-smol-1"),
            max_sequence_tokens=24,
        ),
    )
    tulu_eval = (
        tokenize_single_turn(
            tokenizer,
            SingleTurnExample("repeat one", "one", "tulu-3-personas", "eval-tulu-1"),
            max_sequence_tokens=24,
        ),
    )
    no_robots_eval = (
        tokenize_single_turn(
            tokenizer,
            SingleTurnExample("say blue", "blue", "no_robots", "eval-ood-1"),
            max_sequence_tokens=24,
        ),
    )

    result = run_sft_training(
        DenseBackbone(tiny_backbone_config()),
        tokenizer,
        train_examples,
        no_robots_eval,
        SFTTrainerConfig(
            max_steps=20,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=0.05,
            seed=214,
            output_dir=tmp_path / "matched-selector",
            eval_interval=5,
            checkpoint_interval=10,
            log_interval=5,
            sample_new_tokens=4,
        ),
        eval_splits=(
            EvalSplit("smol-smoltalk", smol_eval, selector_weight=0.8),
            EvalSplit("tulu-3-personas", tulu_eval, selector_weight=0.2),
            EvalSplit("no_robots", no_robots_eval, guardrail=True),
        ),
    )

    assert result.selected_metric_name == "eval/matched/response_loss"
    assert result.selected_eval_suite.selector_weights == {
        "smol-smoltalk": 0.8,
        "tulu-3-personas": 0.2,
    }
    metadata = json.loads(result.best_checkpoint_metadata_path.read_text(encoding="utf-8"))
    assert metadata["selected_metric"] == "eval/matched/response_loss"
    assert metadata["selected_step"] == result.selected_step
    assert set(metadata["component_eval_losses"]) == {
        "smol-smoltalk",
        "tulu-3-personas",
        "no_robots",
    }
    rows = [
        json.loads(line) for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    eval_keys = {key for row in rows if row["event"] == "eval" for key in row}
    assert eval_keys >= EVAL_METRIC_NAMES
    assert "no_robots" not in result.selected_eval_suite.selector_weights

    safe_suite = EvalSuiteResult(
        selector_metric_name="eval/matched/response_loss",
        selector_response_loss=1.0,
        selector_weights={"smol-smoltalk": 0.8, "tulu-3-personas": 0.2},
        split_metrics={
            "smol-smoltalk": EvalMetrics(1.0, 2.7, 10, 1),
            "tulu-3-personas": EvalMetrics(1.0, 2.7, 10, 1),
            "no_robots": EvalMetrics(1.4, 4.0, 10, 1),
        },
    )
    bad_suite = EvalSuiteResult(
        selector_metric_name="eval/matched/response_loss",
        selector_response_loss=1.0,
        selector_weights={"smol-smoltalk": 0.8, "tulu-3-personas": 0.2},
        split_metrics={
            "smol-smoltalk": EvalMetrics(1.0, 2.7, 10, 1),
            "tulu-3-personas": EvalMetrics(1.0, 2.7, 10, 1),
            "no_robots": EvalMetrics(1.6, 5.0, 10, 1),
        },
    )
    assert not _no_robots_catastrophic_regression(safe_suite, baseline=1.0, multiplier=1.5)
    assert _no_robots_catastrophic_regression(bad_suite, baseline=1.0, multiplier=1.5)


def test_sft_resume_from_latest_checkpoint_continues_training(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    train_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=24)
        for example in (
            SingleTurnExample("say red", "red", "fixture", "train-1"),
            SingleTurnExample("say blue", "blue", "fixture", "train-2"),
        )
    )
    eval_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=24)
        for example in (SingleTurnExample("say red", "red", "fixture", "eval-1"),)
    )
    output_dir = tmp_path / "resume"
    torch.manual_seed(214)
    first = run_sft_training(
        DenseBackbone(tiny_backbone_config()),
        tokenizer,
        train_examples,
        eval_examples,
        SFTTrainerConfig(
            max_steps=5,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=0.05,
            seed=214,
            output_dir=output_dir,
            checkpoint_interval=5,
        ),
    )
    torch.manual_seed(214)
    resumed_without_extra_steps = run_sft_training(
        DenseBackbone(tiny_backbone_config()),
        tokenizer,
        train_examples,
        eval_examples,
        SFTTrainerConfig(
            max_steps=first.steps_completed,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=0.05,
            seed=214,
            output_dir=output_dir,
            checkpoint_interval=5,
            resume_from_latest=True,
        ),
    )

    assert first.steps_completed == 5
    assert first.instruct_beats_base is True
    assert resumed_without_extra_steps.training_mode == "resumed"
    assert resumed_without_extra_steps.start_step == first.steps_completed
    assert resumed_without_extra_steps.base_eval.response_loss == pytest.approx(
        first.base_eval.response_loss
    )
    assert resumed_without_extra_steps.instruct_beats_base is True

    torch.manual_seed(214)
    second = run_sft_training(
        DenseBackbone(tiny_backbone_config()),
        tokenizer,
        train_examples,
        eval_examples,
        SFTTrainerConfig(
            max_steps=8,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=0.05,
            seed=214,
            output_dir=output_dir,
            checkpoint_interval=4,
            resume_from_latest=True,
        ),
    )

    assert second.steps_completed == 8
    assert second.trained_tokens > first.trained_tokens
    assert second.training_mode == "resumed"
    assert second.start_step == 5
    assert second.resumed_from_checkpoint is not None
    assert load_sft_checkpoint(second.checkpoint_path).step == 8


def test_sft_failure_saves_latest_checkpoint_and_preserves_error(tmp_path: Path) -> None:
    tokenizer = tiny_tokenizer()
    train_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=24)
        for example in (
            SingleTurnExample("say red", "red", "fixture", "train-1"),
            SingleTurnExample("say blue", "blue", "fixture", "train-2"),
        )
    )
    eval_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=24)
        for example in (SingleTurnExample("say red", "red", "fixture", "eval-1"),)
    )
    output_dir = tmp_path / "failure"

    def fail_after_step(step: int) -> None:
        if step == 3:
            raise RuntimeError("boom after optimizer state exists")

    with pytest.raises(RuntimeError, match="boom after optimizer state exists"):
        run_sft_training(
            DenseBackbone(tiny_backbone_config()),
            tokenizer,
            train_examples,
            eval_examples,
            SFTTrainerConfig(
                max_steps=5,
                micro_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=0.05,
                seed=214,
                output_dir=output_dir,
            ),
            step_callback=fail_after_step,
        )

    failure_report = json.loads((output_dir / "failure-report.json").read_text())
    checkpoint = latest_checkpoint_path(output_dir)

    assert failure_report["status"] == "failed"
    assert failure_report["step"] == 3
    assert failure_report["message"] == "boom after optimizer state exists"
    assert failure_report["checkpoint_error"] is None
    assert checkpoint is not None
    assert failure_report["checkpoint_path"] == str(checkpoint)
    assert load_sft_checkpoint(checkpoint).step == 3


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
            "older_full_metrics_uri": "/posttrain/esme-instruct-sft-full/metrics.jsonl",
            "showcase_eval_rows": 98,
            "showcase_best_step": 600,
            "showcase_latest_step": 19400,
            "notes": "showcase-full and older full metrics are distinct",
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
