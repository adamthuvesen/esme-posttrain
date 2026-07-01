from __future__ import annotations

import json
from pathlib import Path

import pytest

from esme_posttrain.dpo.data import (
    MAX_VALIDATION_DROP_FRACTION,
    PreferencePair,
    build_preference_set,
    measure_preference_lengths,
    parse_ultrafeedback_row,
    tokenize_preference_pair,
)
from esme_posttrain.sft.data import IGNORE_INDEX, ChatTurn, DataError, DatasetSource
from esme_posttrain.sft.smoke_multiturn import tiny_chat_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "esme-214m-chat-dpo.json"


# --- preference-pair templating + prompt masking ------------------------------


def test_preference_pair_tokenization_masks_prompt_in_both_completions() -> None:
    tokenizer = tiny_chat_tokenizer()
    pair = PreferencePair(
        prompt_turns=(ChatTurn("system", "helpful"), ChatTurn("user", "say red")),
        chosen="red",
        rejected="blue",
        source="fixture",
        row_id="1",
    )
    tokenized = tokenize_preference_pair(tokenizer, pair, max_length=48, max_prompt_length=24)

    # Both completions share the exact same prompt prefix.
    assert tokenized.chosen.input_ids[: tokenized.chosen.prompt_tokens] == tokenized.prompt_ids
    assert tokenized.rejected.input_ids[: tokenized.rejected.prompt_tokens] == tokenized.prompt_ids

    eos_id = tokenizer.token_to_id("<eos>")
    for completion in (tokenized.chosen, tokenized.rejected):
        # The whole prompt span is masked; only response tokens score.
        assert all(label == IGNORE_INDEX for label in completion.labels[: completion.prompt_tokens])
        supervised = [
            (token, label)
            for token, label in zip(completion.input_ids, completion.labels, strict=True)
            if label != IGNORE_INDEX
        ]
        assert supervised, "completion must have supervised response tokens"
        # Supervised labels equal their input ids.
        assert all(token == label for token, label in supervised)
        # Exactly one trailing <eos> per response.
        assert completion.input_ids[-1] == eos_id
        assert completion.response_supervised_tokens == completion.response_tokens


def test_preference_pair_rejects_assistant_ending_prompt() -> None:
    with pytest.raises(DataError, match="must end on a user/system turn"):
        PreferencePair(
            prompt_turns=(ChatTurn("user", "hi"), ChatTurn("assistant", "hello")),
            chosen="a",
            rejected="b",
            source="f",
            row_id="1",
        )


def test_preference_pair_rejects_empty_responses() -> None:
    with pytest.raises(DataError, match="non-empty chosen and rejected"):
        PreferencePair(
            prompt_turns=(ChatTurn("user", "hi"),), chosen=" ", rejected="b", source="f", row_id="1"
        )


def test_preference_tokenization_enforces_prompt_length_cap() -> None:
    tokenizer = tiny_chat_tokenizer()
    pair = PreferencePair(
        prompt_turns=(ChatTurn("user", "say red green blue one two again"),),
        chosen="red",
        rejected="blue",
        source="f",
        row_id="1",
    )
    with pytest.raises(DataError, match="max_prompt_length"):
        tokenize_preference_pair(tokenizer, pair, max_length=48, max_prompt_length=2)


def test_preference_length_measurement_flags_truncation() -> None:
    tokenizer = tiny_chat_tokenizer()
    pair = PreferencePair(
        prompt_turns=(ChatTurn("user", "say red"),),
        chosen="red blue green one two",
        rejected="blue",
        source="f",
        row_id="1",
    )
    lengths = measure_preference_lengths(tokenizer, pair, max_length=6, max_prompt_length=24)
    assert lengths["prompt_truncated"] is False
    assert lengths["response_truncated"] is True


# --- UltraFeedback parsing + filtering ----------------------------------------


def _ultrafeedback_source(path: Path | None = None) -> DatasetSource:
    return DatasetSource(
        name="ultrafeedback-binarized",
        source="HuggingFaceH4/ultrafeedback_binarized",
        revision="0" * 40,
        license="mit",
        split="train_prefs",
        role="train",
        path=path,
        max_prompt_chars=4000,
        max_response_chars=4000,
    )


def test_parse_ultrafeedback_shared_prompt_pair() -> None:
    source = _ultrafeedback_source()
    row = {
        "prompt": "say red",
        "chosen": [
            {"role": "user", "content": "say red"},
            {"role": "assistant", "content": "red"},
        ],
        "rejected": [
            {"role": "user", "content": "say red"},
            {"role": "assistant", "content": "blue"},
        ],
    }
    pair = parse_ultrafeedback_row(source, "0", row)
    assert pair is not None
    assert pair.prompt_turns == (ChatTurn("user", "say red"),)
    assert pair.chosen == "red"
    assert pair.rejected == "blue"


def test_parse_ultrafeedback_rejects_identical_responses() -> None:
    source = _ultrafeedback_source()
    row = {
        "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "same"}],
        "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "same"}],
    }
    assert parse_ultrafeedback_row(source, "0", row) is None


def test_parse_ultrafeedback_rejects_mismatched_prompts() -> None:
    source = _ultrafeedback_source()
    row = {
        "chosen": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"}],
        "rejected": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "b"}],
    }
    assert parse_ultrafeedback_row(source, "0", row) is None


def test_build_preference_set_filters_and_holds_out(tmp_path: Path) -> None:
    path = tmp_path / "uf.jsonl"
    clean = [
        {
            "chosen": [{"role": "user", "content": w}, {"role": "assistant", "content": "red"}],
            "rejected": [{"role": "user", "content": w}, {"role": "assistant", "content": "blue"}],
        }
        for w in ("say red", "say blue", "say green", "repeat one")
    ]
    identical = {
        "chosen": [
            {"role": "user", "content": "say green"},
            {"role": "assistant", "content": "green"},
        ],
        "rejected": [
            {"role": "user", "content": "say green"},
            {"role": "assistant", "content": "green"},
        ],
    }
    rows = [*clean, identical]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tokenizer = tiny_chat_tokenizer()
    train = build_preference_set(
        _ultrafeedback_source(path),
        tokenizer,
        max_pairs=2,
        max_tokens=10000,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(train.pairs) == 2  # two clean pairs kept

    # A full pass over all rows must drop the identical-response row.
    full = build_preference_set(
        _ultrafeedback_source(path),
        tokenizer,
        max_pairs=10,
        max_tokens=10000,
        max_length=48,
        max_prompt_length=24,
    )
    assert full.counts.rejected_unparsable == 1

    # A held-out build that skips the rows the train build consumed is disjoint.
    heldout = build_preference_set(
        _ultrafeedback_source(path),
        tokenizer,
        max_pairs=2,
        max_tokens=10000,
        max_length=48,
        max_prompt_length=24,
        skip_selected=train.counts.selected,
    )
    train_keys = {(p.source, p.row_id) for p in train.pairs}
    heldout_keys = {(p.source, p.row_id) for p in heldout.pairs}
    assert heldout_keys.isdisjoint(train_keys)


def _uf_row(prompt: str, chosen: str, rejected: str) -> dict[str, object]:
    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }


def test_build_preference_set_drops_and_counts_empty_response_rows(tmp_path: Path) -> None:
    # One whitespace-only chosen among many clean rows must be dropped and counted,
    # not abort the whole build (the full UltraFeedback budget hits such rows). Keep
    # the bad fraction (1/61 ~ 1.6%) under the 2% sanity floor.
    path = tmp_path / "uf.jsonl"
    rows = [_uf_row("say red", "red", "blue") for _ in range(60)]
    rows.append(_uf_row("say green", "   ", "blue"))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = build_preference_set(
        _ultrafeedback_source(path),
        tiny_chat_tokenizer(),
        max_pairs=100,
        max_tokens=10**9,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(result.pairs) == 60  # clean rows survive
    assert result.counts.dropped_validation == 1
    assert result.counts.drop_reasons == {"empty_response": 1}
    # The drop is surfaced in the data report.
    report = result.to_dict()
    assert report["counts"]["dropped_validation"] == 1
    assert report["counts"]["drop_reasons"] == {"empty_response": 1}


def test_build_preference_set_keeps_clean_rows_with_no_drops(tmp_path: Path) -> None:
    path = tmp_path / "uf.jsonl"
    rows = [_uf_row("say red", "red", "blue"), _uf_row("say blue", "blue", "red")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = build_preference_set(
        _ultrafeedback_source(path),
        tiny_chat_tokenizer(),
        max_pairs=100,
        max_tokens=10**9,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(result.pairs) == 2
    assert result.counts.dropped_validation == 0
    assert result.counts.drop_reasons == {}


def test_build_preference_set_raises_above_drop_ceiling(tmp_path: Path) -> None:
    # 10% empty-response rows far exceeds the 2% sanity ceiling -> raise (bug signal).
    path = tmp_path / "uf.jsonl"
    rows = [_uf_row("say red", "red", "blue") for _ in range(18)]
    rows += [_uf_row("say green", "", "blue") for _ in range(2)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert MAX_VALIDATION_DROP_FRACTION == 0.02
    with pytest.raises(DataError, match="above the 2% ceiling"):
        build_preference_set(
            _ultrafeedback_source(path),
            tiny_chat_tokenizer(),
            max_pairs=100,
            max_tokens=10**9,
            max_length=48,
            max_prompt_length=24,
        )


def test_survivors_count_only_rows_that_tokenize(tmp_path: Path) -> None:
    # A row that parses but fails tokenization (here: over max_length) must not
    # count as a survivor, so it is never double-counted in the drop-rate
    # denominator (survivors + dropped_validation).
    path = tmp_path / "uf.jsonl"
    rows = [_uf_row("say red", "red", "blue") for _ in range(3)]
    rows.append(_uf_row("say green", "red blue green one two " * 20, "blue"))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = build_preference_set(
        _ultrafeedback_source(path),
        tiny_chat_tokenizer(),
        max_pairs=100,
        max_tokens=10**9,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(result.pairs) == 3
    assert result.counts.survivors == 3
    assert result.counts.rejected_too_long_tokens == 1
    assert result.counts.dropped_validation == 0


def _clean_preference_jsonl(path: Path, count: int) -> None:
    rows = [_uf_row(f"say red {i}", "red", "blue") for i in range(count)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_max_pairs_is_a_cap_not_a_required_minimum(tmp_path: Path) -> None:
    # 30 clean pairs available; cap 62000, floor 8 -> selecting 30 (below the cap
    # because of filtering, above the floor) must NOT report a shortfall.
    path = tmp_path / "uf.jsonl"
    _clean_preference_jsonl(path, 30)
    result = build_preference_set(
        _ultrafeedback_source(path),
        tiny_chat_tokenizer(),
        max_pairs=62000,
        min_pairs=8,
        max_tokens=10**9,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(result.pairs) == 30
    assert result.shortfalls == ()


def test_selecting_below_floor_reports_shortfall(tmp_path: Path) -> None:
    path = tmp_path / "uf.jsonl"
    _clean_preference_jsonl(path, 30)
    result = build_preference_set(
        _ultrafeedback_source(path),
        tiny_chat_tokenizer(),
        max_pairs=62000,
        min_pairs=40,  # floor above what survives
        max_tokens=10**9,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(result.pairs) == 30
    assert result.shortfalls
    assert "selected 30/40 (floor)" in result.shortfalls[0]


def test_max_pairs_still_caps(tmp_path: Path) -> None:
    path = tmp_path / "uf.jsonl"
    _clean_preference_jsonl(path, 30)
    result = build_preference_set(
        _ultrafeedback_source(path),
        tiny_chat_tokenizer(),
        max_pairs=10,
        min_pairs=5,
        max_tokens=10**9,
        max_length=48,
        max_prompt_length=24,
    )
    assert len(result.pairs) == 10  # never exceeds the cap
    assert result.shortfalls == ()


def test_min_pairs_must_not_exceed_cap(tmp_path: Path) -> None:
    path = tmp_path / "uf.jsonl"
    _clean_preference_jsonl(path, 5)
    with pytest.raises(DataError, match="min_pairs .floor. must be <= max_pairs"):
        build_preference_set(
            _ultrafeedback_source(path),
            tiny_chat_tokenizer(),
            max_pairs=10,
            min_pairs=20,
            max_tokens=10**9,
            max_length=48,
            max_prompt_length=24,
        )
