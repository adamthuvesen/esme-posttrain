"""No-spend CPU fixture for the DPO chat-polish path.

Mirrors ``sft_multiturn_smoke`` but trains a tiny vanilla DPO pass on tiny
preference pairs so the fixture exercises: preference-pair prompt masking, the
frozen reference, margin increase, chosen/rejected logp tracking (likelihood-
displacement watch), checkpoint round-trip, a decoding pre-check on the fixture
(clearly marked as a harness demo, not the real SFT checkpoint), chat samples,
and the evaluation report.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.dpo.data import PreferencePair, tokenize_preference_pair
from esme_posttrain.dpo.decoding_precheck import run_decoding_precheck
from esme_posttrain.dpo.launch import EXPECTED_ARTIFACTS, DPOLaunchConfig
from esme_posttrain.dpo.sample_artifacts import write_chat_samples
from esme_posttrain.dpo.trainer import (
    DPOTrainerConfig,
    run_dpo_training,
)
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.run_artifacts import (
    refresh_manifest_files,
    write_environment,
    write_json,
    write_selected_row_manifest,
)
from esme_posttrain.sft.data import ChatTurn
from esme_posttrain.sft.smoke_multiturn import tiny_backbone_config, tiny_chat_tokenizer
from esme_posttrain.training.checkpointing import (
    latest_checkpoint_path,
    load_training_checkpoint,
)
from esme_posttrain.training.wandb_init import WandbConfig

FIXTURE_MAX_LENGTH = 48
FIXTURE_MAX_PROMPT_LENGTH = 24


class _FixtureInterrupted(RuntimeError):
    pass


def run_dpo_cpu_fixture(
    config: DPOLaunchConfig,
    *,
    output_dir: Path | None = None,
    wandb_enabled: bool = False,
    reference_checkpoint_path: Path | None = None,
    max_steps: int | None = None,
    interrupt_after_step: int | None = None,
) -> dict[str, Any]:
    evidence_dir = _prepare_evidence_dir(config, output_dir)
    fixture_steps = max(20, int(config.optimizer["smoke_max_steps"]))
    composable_fixture = max_steps is not None
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        fixture_steps = max_steps
    if interrupt_after_step is not None and not 0 < interrupt_after_step < fixture_steps:
        raise ValueError("interrupt_after_step must be positive and less than max_steps")

    policy, reference, tokenizer, reference_manifest = _fixture_models(reference_checkpoint_path)

    train_pairs = tuple(
        tokenize_preference_pair(
            tokenizer,
            pair,
            max_length=FIXTURE_MAX_LENGTH,
            max_prompt_length=FIXTURE_MAX_PROMPT_LENGTH,
        )
        for pair in _tiny_train_pairs()
    )
    eval_pairs = tuple(
        tokenize_preference_pair(
            tokenizer,
            pair,
            max_length=FIXTURE_MAX_LENGTH,
            max_prompt_length=FIXTURE_MAX_PROMPT_LENGTH,
        )
        for pair in _tiny_eval_pairs()
    )
    prompt_masking_asserted = _assert_preference_masking(train_pairs)

    write_json(
        evidence_dir / "config.json",
        {"mode": "local_cpu_fixture_dpo", "source_config": config.run_id},
    )
    write_json(
        evidence_dir / "data-report.json",
        {
            "mode": "local_cpu_fixture_dpo",
            "train_pairs": len(train_pairs),
            "eval_pairs": len(eval_pairs),
            "prompt_masking_asserted": prompt_masking_asserted,
            "remote_dataset_download": False,
            "paid_compute": False,
            "selected_pair_manifest": [pair.manifest_entry() for pair in train_pairs],
        },
    )
    write_selected_row_manifest(evidence_dir / "selected-pair-manifest.jsonl", train_pairs)
    write_selected_row_manifest(evidence_dir / "eval-pair-manifest.jsonl", eval_pairs)

    # Decoding pre-check on the fixture reference model. This is a HARNESS DEMO of
    # the pre-check; the real SFT checkpoint is not loadable in a CPU-only env.
    precheck = run_decoding_precheck(
        reference,
        tokenizer,
        _fixture_decoding_prompts(),
        is_real_checkpoint=False,
        note=(
            "HARNESS DEMO on the tiny fixture reference model, not the real Esme SFT "
            "checkpoint (which lives on the Modal Volume and is not loadable on CPU). "
            "The real pre-DPO decoding baseline must be produced inside the Modal job."
        ),
    )
    write_json(evidence_dir / "decoding-precheck.json", precheck.to_dict())

    trainer_config = DPOTrainerConfig(
        # A real batch per step (micro_batch 2, no accumulation) keeps the tiny
        # fixture's per-pair gradients from fighting each other; lr 0.02 is far
        # above the real 1e-6 so the tiny model actually moves in few steps.
        max_steps=fixture_steps,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=0.02,
        beta=float(config.payload["dpo"]["beta"]),
        length_normalized=bool(config.payload["dpo"]["length_normalized"]),
        scheduler=str(config.optimizer["scheduler"]),
        warmup_steps=min(
            fixture_steps,
            3,
            int(round(float(config.optimizer["warmup_ratio"]) * 10)),
        ),
        weight_decay=float(config.optimizer["weight_decay"]),
        precision="fp32",
        pad_to_multiple_of=config.payload["sequence"]["pad_to_multiple_of"],
        seed=int(config.optimizer["seed"]),
        output_dir=evidence_dir,
        artifact_name=config.artifact_name,
        eval_interval=1 if composable_fixture else 5,
        checkpoint_interval=1 if composable_fixture else 10,
        log_interval=1 if composable_fixture else 5,
        wandb=WandbConfig(
            enabled=wandb_enabled,
            project=str(config.payload["monitoring"]["wandb_project"]),
            run_name=f"{config.run_id}-local-cpu-fixture" if wandb_enabled else None,
            tags=tuple(config.payload["monitoring"]["wandb_tags"]) + ("smoke", "fixture"),
            group=config.run_id,
            job_type="smoke",
            notes=(
                "No-spend local CPU fixture on tiny preference pairs; the Modal "
                "smoke runs the real DPO path on UltraFeedback instead."
            ),
            extra_config={
                "model": config.artifact_name,
                "stage": "dpo",
                "run_type": "smoke",
                "beta": float(config.payload["dpo"]["beta"]),
                "length_normalized": bool(config.payload["dpo"]["length_normalized"]),
            },
        ),
    )
    resume_checkpoint: Path | None = None
    try:
        result = run_dpo_training(
            policy,
            reference,
            tokenizer,
            train_pairs,
            eval_pairs,
            trainer_config,
            reference_bundle_manifest=reference_manifest,
            step_callback=(
                _interrupt_during_next_step(interrupt_after_step)
                if interrupt_after_step is not None
                else None
            ),
        )
    except _FixtureInterrupted:
        resume_checkpoint = latest_checkpoint_path(evidence_dir)
        if resume_checkpoint is None:
            raise RuntimeError("DPO fixture interruption left no restartable checkpoint") from None
        loaded = load_training_checkpoint(resume_checkpoint)
        if loaded.step != interrupt_after_step:
            raise RuntimeError(
                "DPO fixture interruption checkpoint step mismatch: "
                f"expected {interrupt_after_step}, got {loaded.step}"
            ) from None
        policy, reference, tokenizer, reference_manifest = _fixture_models(
            reference_checkpoint_path
        )
        result = run_dpo_training(
            policy,
            reference,
            tokenizer,
            train_pairs,
            eval_pairs,
            replace(trainer_config, resume_from_latest=True),
            reference_bundle_manifest=reference_manifest,
        )

    write_chat_samples(
        evidence_dir / "chat-samples.md",
        policy,
        tokenizer,
        eval_pairs,
        selected_step=result.selected_step,
    )
    eval_payload = result.to_dict()
    eval_payload["decoding_precheck"] = precheck.to_dict()
    write_json(evidence_dir / "eval-report.json", eval_payload)
    write_json(
        evidence_dir / "cost.json",
        {"paid_compute": False, "estimated_cost_usd": 0.0, "runtime_spend_stop_usd": 0.0},
    )
    write_environment(evidence_dir / "environment.txt", device=torch.device("cpu"))
    refresh_manifest_files(evidence_dir, EXPECTED_ARTIFACTS)
    return {
        "status": "local_cpu_fixture_dpo_complete",
        "paid_compute": False,
        "wandb_enabled": wandb_enabled,
        "output_dir": str(evidence_dir),
        "prompt_masking_asserted": prompt_masking_asserted,
        "margin_increased": result.margin_increased,
        "chosen_logp_collapsed": result.chosen_logp_collapsed,
        "interrupted_and_resumed": interrupt_after_step is not None,
        "interrupted_after_step": interrupt_after_step,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "result": result.to_dict(),
        "decoding_precheck": precheck.to_dict(),
        "required_artifacts_present": {
            name: (evidence_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


def _fixture_models(
    reference_checkpoint_path: Path | None,
) -> tuple[DenseBackbone, DenseBackbone, Tokenizer, dict[str, Any]]:
    if reference_checkpoint_path is None:
        tokenizer = tiny_chat_tokenizer()
        reference = DenseBackbone(tiny_backbone_config())
        reference_manifest = {"mode": "local_cpu_fixture_dpo_tiny_reference"}
    else:
        checkpoint_path = reference_checkpoint_path.expanduser().resolve()
        loaded = load_training_checkpoint(checkpoint_path, map_location="cpu")
        reference = loaded.model
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
        manifest_path = checkpoint_path.parent / "manifest.json"
        if not tokenizer_path.is_file():
            raise ValueError(f"SFT reference tokenizer does not exist: {tokenizer_path}")
        if not manifest_path.is_file():
            raise ValueError(f"SFT reference manifest does not exist: {manifest_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        try:
            reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"SFT reference manifest is malformed: {manifest_path}") from error
        if not isinstance(reference_manifest, dict):
            raise ValueError(f"SFT reference manifest must be an object: {manifest_path}")
    policy = DenseBackbone(reference.config)
    policy.load_state_dict(reference.state_dict())
    return policy, reference, tokenizer, reference_manifest


def _interrupt_during_next_step(interrupt_after_step: int) -> Callable[[int], None]:
    def interrupt(step: int) -> None:
        # The trainer callback runs before the current step is checkpointed. Let
        # step N persist, then simulate preemption while step N+1 is in flight.
        if step == interrupt_after_step + 1:
            raise _FixtureInterrupted

    return interrupt


def _assert_preference_masking(pairs: tuple[Any, ...]) -> bool:
    for pair in pairs:
        for completion in (pair.chosen, pair.rejected):
            if not all(label == -100 for label in completion.labels[: completion.prompt_tokens]):
                raise ValueError(f"{pair.row_id}: prompt span leaked into a completion loss")
            supervised = [
                (token, label)
                for token, label in zip(completion.input_ids, completion.labels, strict=True)
                if label != -100
            ]
            if not supervised:
                raise ValueError(f"{pair.row_id}: completion has no supervised response tokens")
            if any(token != label for token, label in supervised):
                raise ValueError(f"{pair.row_id}: supervised labels do not match input ids")
    return True


def _tiny_train_pairs() -> tuple[PreferencePair, ...]:
    return (
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red",
            rejected="blue",
            source="fixture",
            row_id="train-1",
        ),
        PreferencePair(
            prompt_turns=(ChatTurn("system", "helpful"), ChatTurn("user", "say blue")),
            chosen="blue",
            rejected="red",
            source="fixture",
            row_id="train-2",
        ),
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say green"),),
            chosen="green",
            rejected="red",
            source="fixture",
            row_id="train-3",
        ),
        PreferencePair(
            prompt_turns=(ChatTurn("user", "repeat one"),),
            chosen="one",
            rejected="two",
            source="fixture",
            row_id="train-4",
        ),
    )


def _tiny_eval_pairs() -> tuple[PreferencePair, ...]:
    # Eval prompts reuse the train prompt contexts so the warm-started policy can
    # actually learn to prefer the chosen token; this is a fixture, not a held-out
    # generalization test.
    return (
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red",
            rejected="blue",
            source="fixture",
            row_id="eval-1",
        ),
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say green"),),
            chosen="green",
            rejected="two",
            source="fixture",
            row_id="eval-2",
        ),
    )


def _fixture_decoding_prompts() -> tuple[tuple[str, str], ...]:
    return (
        ("say_red", "user\nsay red\nassistant\n"),
        ("say_blue", "user\nsay blue\nassistant\n"),
    )


def _prepare_evidence_dir(config: DPOLaunchConfig, output_dir: Path | None) -> Path:
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
