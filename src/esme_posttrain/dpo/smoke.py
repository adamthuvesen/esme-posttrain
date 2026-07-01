"""No-spend CPU fixture for the DPO chat-polish path.

Mirrors ``sft_multiturn_smoke`` but trains a tiny vanilla DPO pass on tiny
preference pairs so the fixture exercises: preference-pair prompt masking, the
frozen reference, margin increase, chosen/rejected logp tracking (likelihood-
displacement watch), checkpoint round-trip, a decoding pre-check on the fixture
(clearly marked as a harness demo, not the real SFT checkpoint), chat samples,
and the K>=5 LLM-judge report.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from esme_posttrain.dpo.data import PreferencePair, tokenize_preference_pair
from esme_posttrain.dpo.decoding_precheck import run_decoding_precheck
from esme_posttrain.dpo.launch import EXPECTED_ARTIFACTS, DPOLaunchConfig
from esme_posttrain.dpo.sample_artifacts import write_chat_samples
from esme_posttrain.dpo.trainer import (
    DPOTrainerConfig,
    run_dpo_training,
)
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.run_artifacts import refresh_manifest_files, write_environment, write_json
from esme_posttrain.sft.data import ChatTurn
from esme_posttrain.sft.multiturn_judge import run_multi_turn_judge
from esme_posttrain.sft.smoke_multiturn import tiny_backbone_config, tiny_chat_tokenizer
from esme_posttrain.training.wandb_init import WandbConfig

FIXTURE_MAX_LENGTH = 48
FIXTURE_MAX_PROMPT_LENGTH = 24


def run_dpo_cpu_fixture(
    config: DPOLaunchConfig,
    *,
    output_dir: Path | None = None,
    wandb_enabled: bool = False,
) -> dict[str, Any]:
    evidence_dir = _prepare_evidence_dir(config, output_dir)

    tokenizer = tiny_chat_tokenizer()
    backbone_config = tiny_backbone_config()
    # Reference = a frozen SFT-like model; policy starts from the same weights
    # (warm-start), so the initial margin is 0 and DPO must push it up.
    reference = DenseBackbone(backbone_config)
    policy = DenseBackbone(backbone_config)
    policy.load_state_dict(reference.state_dict())

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
    _write_pair_manifest(evidence_dir / "selected-pair-manifest.jsonl", train_pairs)
    _write_pair_manifest(evidence_dir / "eval-pair-manifest.jsonl", eval_pairs)

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

    result = run_dpo_training(
        policy,
        reference,
        tokenizer,
        train_pairs,
        eval_pairs,
        DPOTrainerConfig(
            # A real batch per step (micro_batch 2, no accumulation) keeps the tiny
            # fixture's per-pair gradients from fighting each other; lr 0.02 is far
            # above the real 1e-6 so the tiny model actually moves in few steps.
            max_steps=max(20, int(config.optimizer["smoke_max_steps"])),
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=0.02,
            beta=float(config.payload["dpo"]["beta"]),
            length_normalized=bool(config.payload["dpo"]["length_normalized"]),
            scheduler=str(config.optimizer["scheduler"]),
            warmup_steps=min(3, int(round(float(config.optimizer["warmup_ratio"]) * 10))),
            weight_decay=float(config.optimizer["weight_decay"]),
            precision="fp32",
            pad_to_multiple_of=config.payload["sequence"]["pad_to_multiple_of"],
            seed=int(config.optimizer["seed"]),
            output_dir=evidence_dir,
            artifact_name=config.artifact_name,
            eval_interval=5,
            checkpoint_interval=10,
            log_interval=5,
            sample_new_tokens=6,
            wandb=WandbConfig(
                enabled=wandb_enabled,
                project=str(config.payload["monitoring"]["wandb_project"]),
                run_name=f"{config.run_id}-modal-smoke" if wandb_enabled else None,
                tags=tuple(config.payload["monitoring"]["wandb_tags"]) + ("smoke", "fixture"),
                group=config.run_id,
                job_type="smoke",
                notes="Bounded Modal smoke uses the tiny DPO fixture, not full UltraFeedback.",
                extra_config={
                    "model": config.artifact_name,
                    "stage": "dpo",
                    "run_type": "smoke",
                    "beta": float(config.payload["dpo"]["beta"]),
                    "length_normalized": bool(config.payload["dpo"]["length_normalized"]),
                },
            ),
        ),
        reference_bundle_manifest={"mode": "local_cpu_fixture_dpo_tiny_reference"},
    )

    write_chat_samples(
        evidence_dir / "chat-samples.md",
        policy,
        tokenizer,
        eval_pairs,
        selected_step=result.selected_step,
    )
    judge_report = run_multi_turn_judge(
        lambda prompt: "blue",  # fixture stand-in generator; no real judge configured
        judge=None,
        passes=int(config.payload["monitoring"]["judge_repeat_passes"]),
    )
    eval_payload = result.to_dict()
    eval_payload["dpo_judge"] = judge_report.to_dict()
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
        "result": result.to_dict(),
        "decoding_precheck": precheck.to_dict(),
        "required_artifacts_present": {
            name: (evidence_dir / name).is_file() for name in EXPECTED_ARTIFACTS
        },
    }


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


def _write_pair_manifest(path: Path, pairs: tuple[Any, ...]) -> None:
    rows = (json.dumps(pair.manifest_entry(), sort_keys=True) + "\n" for pair in pairs)
    path.write_text("".join(rows), encoding="utf-8")
