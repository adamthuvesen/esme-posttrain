from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from esme_posttrain.bundle import file_sha256
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.data import (
    DataError,
    DatasetSource,
    LossSemantics,
    SingleTurnExample,
    build_eval_set,
    build_matched_eval_sets,
    build_training_mix,
    tokenize_single_turn,
)
from esme_posttrain.sft.launch_instruct import (
    SFTLaunchConfig,
)
from esme_posttrain.sft.smoke_instruct import tiny_backbone_config, tiny_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "esme-214m-instruct.json"
WEIGHTS_FIELD = "key_format"


def test_dataset_filtering_mixing_and_token_caps(tmp_path: Path) -> None:
    smol_path = tmp_path / "smol.jsonl"
    tulu_path = tmp_path / "tulu.jsonl"
    _write_jsonl(
        smol_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                ]
            }
            for _ in range(9)
        ]
        + [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "red"},
                ]
            }
        ],
    )
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
            mix_ratio=0.8,
            path=smol_path,
        ),
        DatasetSource(
            name="tulu-3-personas",
            source="local-tulu",
            revision="1" * 40,
            license="odc-by",
            split="train",
            role="train",
            mix_ratio=0.2,
            path=tulu_path,
        ),
    )

    result = build_training_mix(
        sources,
        tiny_tokenizer(),
        max_samples=10,
        max_tokens=500,
        max_sequence_tokens=24,
    )

    assert len(result.examples) == 10
    assert result.counts_by_source["smol-smoltalk"].selected == 8
    assert result.counts_by_source["tulu-3-personas"].selected == 2
    assert result.counts_by_source["smol-smoltalk"].rejected_non_single_turn == 0
    assert [example.source for example in result.examples[:5]] == [
        "smol-smoltalk",
        "smol-smoltalk",
        "smol-smoltalk",
        "smol-smoltalk",
        "tulu-3-personas",
    ]

    capped = build_training_mix(
        sources,
        tiny_tokenizer(),
        max_samples=10,
        max_tokens=60,
        max_sequence_tokens=24,
    )
    assert capped.selected_tokens <= 60
    assert capped.budget_stop == "token_cap"
    first_manifest = result.examples[0].manifest_entry()
    assert first_manifest["source_dataset"] == "local-smol"
    assert first_manifest["revision"] == "0" * 40
    assert first_manifest["prompt_tokens"] > 0
    assert first_manifest["supervised_tokens"] == result.examples[0].response_tokens


def test_matched_eval_splits_skip_train_rows_and_count_target_truncation(
    tmp_path: Path,
) -> None:
    smol_path = tmp_path / "smol.jsonl"
    tulu_path = tmp_path / "tulu.jsonl"
    _write_jsonl(
        smol_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                ]
            }
            for _ in range(4)
        ]
        + [
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red red red red red red"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "say red"},
                    {"role": "assistant", "content": "red"},
                ]
            },
        ],
    )
    _write_jsonl(
        tulu_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "say blue"},
                    {"role": "assistant", "content": "blue"},
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
            mix_ratio=0.8,
            path=smol_path,
        ),
        DatasetSource(
            name="tulu-3-personas",
            source="local-tulu",
            revision="1" * 40,
            license="odc-by",
            split="train",
            role="train",
            mix_ratio=0.2,
            path=tulu_path,
        ),
    )
    tokenizer = tiny_tokenizer()
    train = build_training_mix(
        sources,
        tokenizer,
        max_samples=5,
        max_tokens=500,
        max_sequence_tokens=6,
    )

    matched = build_matched_eval_sets(
        sources,
        tokenizer,
        skip_selected_by_source={
            name: counts.selected for name, counts in train.counts_by_source.items()
        },
        max_samples_per_source=1,
        max_tokens_per_source=100,
        max_sequence_tokens=6,
    )

    train_keys = {(example.source, example.row_id) for example in train.examples}
    heldout_keys = {
        (example.source, example.row_id)
        for report in matched.values()
        for example in report.examples
    }
    assert heldout_keys.isdisjoint(train_keys)
    assert matched["smol-smoltalk"].counts_by_source["smol-smoltalk"].selected == 1
    assert (
        matched["smol-smoltalk"].counts_by_source["smol-smoltalk"].assistant_target_truncation_count
        == 1
    )
    manifest = matched["smol-smoltalk"].examples[0].manifest_entry()
    assert manifest["prompt_tokens"] > 0
    assert manifest["response_tokens"] > 0


def test_no_robots_adapter_is_eval_only(tmp_path: Path) -> None:
    path = tmp_path / "no_robots.jsonl"
    _write_jsonl(path, [{"instruction": "say red", "response": "red"}])
    source = DatasetSource(
        name="no_robots",
        source="HuggingFaceH4/no_robots",
        revision="2" * 40,
        license="cc-by-nc-4.0",
        split="test",
        role="eval",
        train_allowed=False,
        path=path,
    )

    result = build_eval_set(
        source,
        tiny_tokenizer(),
        max_samples=1,
        max_tokens=50,
        max_sequence_tokens=24,
    )

    assert len(result.examples) == 1
    assert result.examples[0].source == "no_robots"
    assert result.examples[0].manifest_entry()["source_dataset"] == "HuggingFaceH4/no_robots"

    bad_source = DatasetSource(
        name="no_robots",
        source="HuggingFaceH4/no_robots",
        revision="2" * 40,
        license="cc-by-nc-4.0",
        split="train",
        role="train",
        mix_ratio=1.0,
        train_allowed=True,
        path=path,
    )
    with pytest.raises(DataError, match="exactly two train sources"):
        build_training_mix(
            (bad_source,),
            tiny_tokenizer(),
            max_samples=1,
            max_tokens=50,
            max_sequence_tokens=24,
        )


def test_prompt_masking_is_directly_asserted() -> None:
    tokenized = tokenize_single_turn(
        tiny_tokenizer(),
        SingleTurnExample("say red", "red", "fixture", "1"),
        max_sequence_tokens=24,
    )

    assert all(label == -100 for label in tokenized.labels[: tokenized.prompt_tokens])
    assert tokenized.labels[tokenized.prompt_tokens - 1] == -100
    assert tokenized.labels[tokenized.prompt_tokens] == tokenized.input_ids[tokenized.prompt_tokens]
    assert (
        tokenized.labels[tokenized.prompt_tokens :]
        == tokenized.input_ids[tokenized.prompt_tokens :]
    )
    assert tokenized.response_tokens > 0

    with pytest.raises(DataError, match="assistant_only_loss"):
        tokenize_single_turn(
            tiny_tokenizer(),
            SingleTurnExample("say red", "red", "fixture", "1"),
            max_sequence_tokens=24,
            loss_semantics=LossSemantics(assistant_only_loss=False),
        )


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
