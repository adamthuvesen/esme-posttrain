"""No-spend CPU fixture for the multi-turn SFT path.

Mirrors ``sft_smoke`` but trains on tiny multi-turn conversations so the fixture
exercises all-assistant-turn masking, multi-turn loss decrease, checkpoint
round-trip, and the extra ``multi-turn-samples.md`` artifact.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from esme_posttrain.modeling import BackboneConfig, DenseBackbone
from esme_posttrain.run_artifacts import (
    refresh_manifest_files,
    write_environment,
    write_json,
    write_selected_row_manifest,
)
from esme_posttrain.sft.data import (
    ChatTurn,
    MultiTurnExample,
    TokenizedExample,
    sequence_efficiency_report,
    tokenize_multi_turn,
)
from esme_posttrain.sft.launch_multiturn import EXPECTED_ARTIFACTS, MultiTurnLaunchConfig
from esme_posttrain.sft.multiturn_data import turn_distribution
from esme_posttrain.sft.sample_artifacts import write_multi_turn_samples
from esme_posttrain.sft.trainer import (
    EvalSplit,
    SFTTrainerConfig,
    run_sft_training,
)
from esme_posttrain.training.wandb_init import WandbConfig


def run_multi_turn_cpu_fixture(
    config: MultiTurnLaunchConfig,
    *,
    output_dir: Path | None = None,
    wandb_enabled: bool = False,
) -> dict[str, Any]:
    evidence_dir = _prepare_evidence_dir(config, output_dir)

    tokenizer = tiny_chat_tokenizer()
    model = DenseBackbone(tiny_backbone_config())
    train_examples = tuple(
        tokenize_multi_turn(tokenizer, example, max_sequence_tokens=model.config.context_length)
        for example in _tiny_train_examples()
    )
    eval_examples = tuple(
        tokenize_multi_turn(tokenizer, example, max_sequence_tokens=model.config.context_length)
        for example in _tiny_eval_examples()
    )
    sequence_config = config.payload["sequence"]

    multi_turn_masking = _assert_multi_turn_masking(train_examples)
    distribution = turn_distribution(train_examples)

    write_json(
        evidence_dir / "config.json",
        {"mode": "local_cpu_fixture_multi_turn", "source_config": config.run_id},
    )
    write_json(
        evidence_dir / "data-report.json",
        {
            "mode": "local_cpu_fixture_multi_turn",
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "assistant_only_multi_turn_masking_asserted": multi_turn_masking,
            "turn_distribution": distribution.to_dict(),
            "remote_dataset_download": False,
            "paid_compute": False,
            "selected_row_manifest": [example.manifest_entry() for example in train_examples],
            "sequence_efficiency": {
                "train": sequence_efficiency_report(
                    train_examples,
                    max_sequence_tokens=model.config.context_length,
                    micro_batch_size=1,
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
                "eval": sequence_efficiency_report(
                    eval_examples,
                    max_sequence_tokens=model.config.context_length,
                    micro_batch_size=1,
                    sequence_packing=bool(sequence_config["sequence_packing"]),
                    pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
                    no_packing_rationale=str(sequence_config["no_packing_rationale"]),
                ),
            },
        },
    )
    write_selected_row_manifest(evidence_dir / "selected-row-manifest.jsonl", train_examples)
    write_selected_row_manifest(evidence_dir / "eval-smol-smoltalk-manifest.jsonl", eval_examples)
    write_selected_row_manifest(evidence_dir / "eval-tulu-3-personas-manifest.jsonl", eval_examples)
    write_selected_row_manifest(evidence_dir / "eval-no_robots-manifest.jsonl", eval_examples)

    result = run_sft_training(
        model,
        tokenizer,
        train_examples,
        eval_examples,
        SFTTrainerConfig(
            max_steps=int(config.payload["optimizer"]["smoke_max_steps"]),
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=0.05,
            scheduler=str(config.payload["optimizer"]["scheduler"]),
            warmup_steps=min(2, int(config.payload["optimizer"]["warmup_steps"])),
            weight_decay=float(config.payload["optimizer"]["weight_decay"]),
            precision="fp32",
            pad_to_multiple_of=sequence_config["pad_to_multiple_of"],
            seed=int(config.payload["optimizer"]["seed"]),
            output_dir=evidence_dir,
            artifact_name=config.artifact_name,
            eval_interval=10,
            checkpoint_interval=20,
            log_interval=5,
            sample_new_tokens=6,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(config.payload["monitoring"]["wandb_project"]),
                run_name=f"{config.run_id}-modal-smoke" if wandb_enabled else None,
                tags=("Esme-214M-Instruct", "sft", "multi-turn", "smoke", "fixture"),
                group=config.run_id,
                job_type="smoke",
                notes="Bounded Modal smoke uses the tiny multi-turn fixture, not the full dataset.",
                extra_config={
                    "model": config.artifact_name,
                    "stage": "sft",
                    "run_type": "smoke",
                    "dataset_mix": "fixture",
                    "eval_holdout": "fixture",
                    "gpu": config.runtime["selected_gpu"] if wandb_enabled else "cpu",
                    "tuning_mode": "full",
                    "resume": False,
                },
            ),
        ),
        eval_splits=_fixture_eval_splits(eval_examples),
        base_bundle_manifest={"mode": "local_cpu_fixture_multi_turn_tiny_base"},
    )

    write_multi_turn_samples(
        evidence_dir / "multi-turn-samples.md",
        model,
        tokenizer,
        eval_examples,
        sample_new_tokens=6,
        selected_step=result.selected_step,
    )
    write_json(evidence_dir / "eval-report.json", result.to_dict())
    write_json(
        evidence_dir / "cost.json",
        {"paid_compute": False, "estimated_cost_usd": 0.0, "runtime_spend_stop_usd": 0.0},
    )
    write_environment(evidence_dir / "environment.txt", device=torch.device("cpu"))
    refresh_manifest_files(evidence_dir, EXPECTED_ARTIFACTS)
    return {
        "status": "local_cpu_fixture_multi_turn_complete",
        "paid_compute": False,
        "wandb_enabled": wandb_enabled,
        "output_dir": str(evidence_dir),
        "assistant_only_multi_turn_masking_asserted": multi_turn_masking,
        "turn_distribution": distribution.to_dict(),
        "result": result.to_dict(),
        "required_artifacts_present": {
            name: (evidence_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


def _fixture_eval_splits(eval_examples: tuple[TokenizedExample, ...]) -> tuple[EvalSplit, ...]:
    return (
        EvalSplit("smol-smoltalk", eval_examples, selector_weight=0.85),
        EvalSplit("tulu-3-personas", eval_examples, selector_weight=0.15),
        EvalSplit("no_robots", eval_examples, guardrail=True),
    )


def _assert_multi_turn_masking(examples: tuple[TokenizedExample, ...]) -> bool:
    saw_multi_turn = False
    for example in examples:
        # Prompt span (system/first user) must be fully masked.
        if not all(label == -100 for label in example.labels[: example.prompt_tokens]):
            raise ValueError(f"{example.row_id}: prompt span leaked into the loss")
        # Supervised labels must equal their input ids exactly.
        if not all(
            label == token
            for token, label in zip(example.input_ids, example.labels, strict=True)
            if label != -100
        ):
            raise ValueError(f"{example.row_id}: supervised labels do not match input ids")
        if example.supervised_tokens <= 0:
            raise ValueError(f"{example.row_id}: no supervised assistant tokens")
        if example.assistant_turns > 1:
            saw_multi_turn = True
    if not saw_multi_turn:
        raise ValueError(
            "multi-turn fixture must include at least one example with >1 assistant turn"
        )
    return True


def _prepare_evidence_dir(config: MultiTurnLaunchConfig, output_dir: Path | None) -> Path:
    if output_dir is None:
        evidence_dir = (Path.cwd() / config.output_dir.parent / "local-cpu-fixture").resolve()
        if evidence_dir.exists():
            shutil.rmtree(evidence_dir)
        evidence_dir.mkdir(parents=True)
        return evidence_dir
    evidence_dir = output_dir.expanduser().resolve()
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise ValueError(f"custom output_dir must be empty or absent: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


def tiny_backbone_config() -> BackboneConfig:
    return BackboneConfig(
        name="tiny-multi-turn-sft-fixture",
        vocab_size=32,
        context_length=48,
        embedding_dim=16,
        layers=1,
        heads=2,
        kv_heads=1,
        feedforward_dim=32,
        z_loss_weight=0.0,
    )


def tiny_chat_tokenizer() -> Tokenizer:
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "user": 2,
        "assistant": 3,
        "system": 4,
        "say": 5,
        "red": 6,
        "blue": 7,
        "green": 8,
        "repeat": 9,
        "one": 10,
        "two": 11,
        "again": 12,
        "helpful": 13,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _tiny_train_examples() -> tuple[MultiTurnExample, ...]:
    return (
        MultiTurnExample(
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
            row_id="train-mt-1",
        ),
        MultiTurnExample(
            turns=(
                ChatTurn("user", "repeat one"),
                ChatTurn("assistant", "one"),
                ChatTurn("user", "repeat two"),
                ChatTurn("assistant", "two"),
            ),
            source="fixture",
            row_id="train-mt-2",
        ),
        MultiTurnExample(
            turns=(ChatTurn("user", "say green"), ChatTurn("assistant", "green")),
            source="fixture",
            row_id="train-st-1",
        ),
        MultiTurnExample(
            turns=(
                ChatTurn("user", "say blue"),
                ChatTurn("assistant", "blue"),
                ChatTurn("user", "again"),
                ChatTurn("assistant", "blue"),
            ),
            source="fixture",
            row_id="train-mt-3",
        ),
    )


def _tiny_eval_examples() -> tuple[MultiTurnExample, ...]:
    return (
        MultiTurnExample(
            turns=(
                ChatTurn("user", "say red"),
                ChatTurn("assistant", "red"),
                ChatTurn("user", "again"),
                ChatTurn("assistant", "red"),
            ),
            source="fixture",
            row_id="eval-mt-1",
        ),
        MultiTurnExample(
            turns=(ChatTurn("user", "repeat two"), ChatTurn("assistant", "two")),
            source="fixture",
            row_id="eval-st-1",
        ),
    )
