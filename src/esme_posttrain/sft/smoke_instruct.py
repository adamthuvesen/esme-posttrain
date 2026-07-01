from __future__ import annotations

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
    SingleTurnExample,
    sequence_efficiency_report,
    tokenize_single_turn,
)
from esme_posttrain.sft.launch_instruct import EXPECTED_ARTIFACTS, SFTLaunchConfig
from esme_posttrain.sft.launch_shared import prepare_evidence_dir as _prepare_evidence_dir
from esme_posttrain.sft.trainer import SFTTrainerConfig, WandbConfig, run_sft_training


def run_cpu_fixture_sft(
    config: SFTLaunchConfig,
    *,
    output_dir: Path | None = None,
    wandb_enabled: bool = False,
) -> dict[str, Any]:
    evidence_dir = _prepare_evidence_dir(config.output_dir, output_dir)

    tokenizer = tiny_tokenizer()
    model = DenseBackbone(tiny_backbone_config())
    train_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=model.config.context_length)
        for example in _tiny_train_examples()
    )
    eval_examples = tuple(
        tokenize_single_turn(tokenizer, example, max_sequence_tokens=model.config.context_length)
        for example in _tiny_eval_examples()
    )
    sequence_config = config.payload["sequence"]
    write_json(
        evidence_dir / "config.json",
        {"mode": "local_cpu_fixture", "source_config": config.run_id},
    )
    write_json(
        evidence_dir / "data-report.json",
        {
            "mode": "local_cpu_fixture",
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "prompt_masking_asserted": all(
                all(label == -100 for label in example.labels[: example.prompt_tokens])
                for example in train_examples
            ),
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
            eval_interval=10,
            checkpoint_interval=20,
            log_interval=5,
            sample_new_tokens=4,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(config.payload["monitoring"]["wandb_project"]),
                run_name=f"{config.run_id}-modal-smoke" if wandb_enabled else None,
                tags=("Esme-214M-Instruct", "sft", "smoke", "fixture"),
                group=config.run_id,
                job_type="smoke",
                notes="Bounded Modal smoke uses the tiny fixture path, not the full dataset.",
                extra_config={
                    "model": config.artifact_name,
                    "stage": "sft_cold_start",
                    "run_type": "smoke",
                    "dataset_mix": "fixture",
                    "eval_holdout": "fixture",
                    "gpu": config.runtime["selected_gpu"] if wandb_enabled else "cpu",
                    "tuning_mode": "full",
                    "resume": False,
                },
            ),
        ),
        base_bundle_manifest={"mode": "local_cpu_fixture_tiny_base"},
    )
    write_json(evidence_dir / "eval-report.json", result.to_dict())
    write_json(
        evidence_dir / "cost.json",
        {"paid_compute": False, "estimated_cost_usd": 0.0, "runtime_spend_stop_usd": 0.0},
    )
    write_environment(evidence_dir / "environment.txt", device=torch.device("cpu"))
    refresh_manifest_files(evidence_dir, EXPECTED_ARTIFACTS)
    return {
        "status": "local_cpu_fixture_complete",
        "paid_compute": False,
        "wandb_enabled": wandb_enabled,
        "output_dir": str(evidence_dir),
        "result": result.to_dict(),
        "required_artifacts_present": {
            name: (evidence_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


def tiny_backbone_config() -> BackboneConfig:
    return BackboneConfig(
        name="tiny-sft-fixture",
        vocab_size=32,
        context_length=24,
        embedding_dim=16,
        layers=1,
        heads=2,
        kv_heads=1,
        feedforward_dim=32,
        z_loss_weight=0.0,
    )


def tiny_tokenizer() -> Tokenizer:
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "user": 2,
        "assistant": 3,
        "say": 4,
        "red": 5,
        "blue": 6,
        "green": 7,
        "repeat": 8,
        "one": 9,
        "two": 10,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _tiny_train_examples() -> tuple[SingleTurnExample, ...]:
    return (
        SingleTurnExample("say red", "red", "fixture", "train-1"),
        SingleTurnExample("say blue", "blue", "fixture", "train-2"),
        SingleTurnExample("repeat one", "one", "fixture", "train-3"),
        SingleTurnExample("repeat two", "two", "fixture", "train-4"),
    )


def _tiny_eval_examples() -> tuple[SingleTurnExample, ...]:
    return (
        SingleTurnExample("say red", "red", "fixture", "eval-1"),
        SingleTurnExample("repeat two", "two", "fixture", "eval-2"),
    )
