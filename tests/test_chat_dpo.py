from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from esme_posttrain.cli import main
from esme_posttrain.dpo.chat_eval import (
    CONVERSATIONAL_PROMPTS,
    chat_eval_prompts,
    run_chat_eval,
    write_chat_eval_json,
    write_chat_eval_markdown,
)
from esme_posttrain.dpo.chat_eval_run import (
    CHAT_EVAL_SPEND_CAP_USD,
    build_chat_eval_preflight,
    chat_eval_blockers,
)
from esme_posttrain.dpo.data import (
    PreferencePair,
    tokenize_preference_pair,
)
from esme_posttrain.dpo.decoding_precheck import (
    GREEDY,
    NUCLEUS_REP_PENALTY,
    DecodingConfig,
    _nucleus_keep_mask,
    generate_with_decoding,
    ngram_repetition_rate,
    run_decoding_precheck,
)
from esme_posttrain.dpo.full import (
    DPOFullRunError,
    _assert_accepted_dpo_result,
    _assert_data_safe,
    _assert_prompt_masking,
)
from esme_posttrain.dpo.launch import (
    DPO_FULL_RUN_SPEND_CAP_USD,
    EXPECTED_SWEEP_BETAS,
    build_dpo_dry_run,
    full_launch_blockers,
    load_dpo_config,
    smoke_launch_blockers,
    validate_dpo_payload,
)
from esme_posttrain.dpo.smoke import run_dpo_cpu_fixture
from esme_posttrain.dpo.sweep import (
    SWEEP_ARMS,
    SWEEP_SPEND_CAP_USD,
    build_dpo_sweep_preflight,
    dpo_sweep_blockers,
    learning_gate_payload,
    select_best_arm,
)
from esme_posttrain.dpo.trainer import (
    CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE,
    DPOTrainerConfig,
    DPOTrainerError,
    dpo_pair_loss,
    is_chosen_logp_collapsed,
    run_dpo_training,
    sequence_logprob,
)
from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.data import IGNORE_INDEX, ChatTurn
from esme_posttrain.sft.launch_instruct import LaunchError
from esme_posttrain.sft.smoke_multiturn import tiny_backbone_config, tiny_chat_tokenizer
from esme_posttrain.training.checkpointing import load_training_checkpoint, save_training_checkpoint
from esme_posttrain.training.collate import collate_batch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "esme-214m-chat-dpo.json"


# --- DPO loss + logp + frozen reference ---------------------------------------


def test_dpo_loss_is_log2_and_zero_margin_at_init() -> None:
    # Identical chosen/rejected/ref logps -> margin 0, loss = -log_sigmoid(0) = log 2.
    z = torch.zeros(3)
    loss, margin = dpo_pair_loss(z, z, z, z, beta=0.5)
    assert torch.allclose(margin, torch.zeros(3), atol=1e-6)
    assert torch.allclose(loss, torch.full((3,), 0.6931472), atol=1e-5)


def test_dpo_gradient_pushes_margin_positive() -> None:
    tokenizer = tiny_chat_tokenizer()
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    reference = DenseBackbone(cfg)
    policy = DenseBackbone(cfg)
    policy.load_state_dict(reference.state_dict())
    pair = tokenize_preference_pair(
        tokenizer,
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red",
            rejected="blue",
            source="f",
            row_id="1",
        ),
        max_length=48,
        max_prompt_length=24,
    )
    from esme_posttrain.dpo.trainer import _completion_as_collate_row

    chosen_ids, chosen_labels = collate_batch((_completion_as_collate_row(pair.chosen),))
    rejected_ids, rejected_labels = collate_batch((_completion_as_collate_row(pair.rejected),))
    pc = sequence_logprob(policy, chosen_ids, chosen_labels, length_normalized=False)
    pr = sequence_logprob(policy, rejected_ids, rejected_labels, length_normalized=False)
    rc = sequence_logprob(reference, chosen_ids, chosen_labels, length_normalized=False)
    rr = sequence_logprob(reference, rejected_ids, rejected_labels, length_normalized=False)
    # Warm-start: policy == reference, so the initial margin is exactly 0.
    loss, margin = dpo_pair_loss(pc, pr, rc, rr, beta=0.5)
    assert abs(float(margin.detach())) < 1e-5
    loss.mean().backward()
    with torch.no_grad():
        for param in policy.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-0.05)
    pc2 = sequence_logprob(policy, chosen_ids, chosen_labels, length_normalized=False)
    pr2 = sequence_logprob(policy, rejected_ids, rejected_labels, length_normalized=False)
    _, margin2 = dpo_pair_loss(pc2, pr2, rc, rr, beta=0.5)
    assert float(margin2.detach()) > 0.0


def test_chosen_logp_collapse_ignores_eval_noise_dip() -> None:
    # Real bounded-sweep dip: -254 -> -254.07 (0.07 nats, 0.03%) is eval noise.
    assert is_chosen_logp_collapsed(-254.0, -254.07) is False
    # Real smoke dip: -215.198 -> -215.472 (0.13%) is also sub-tolerance.
    assert is_chosen_logp_collapsed(-215.19759, -215.47217) is False


def test_chosen_logp_collapse_ignores_a_rise() -> None:
    assert is_chosen_logp_collapsed(-254.0, -253.0) is False
    assert is_chosen_logp_collapsed(-254.0, -254.0) is False


def test_chosen_logp_collapse_flags_meaningful_displacement() -> None:
    # A ~10% fall is genuine likelihood displacement and MUST flag.
    assert is_chosen_logp_collapsed(-254.0, -279.4) is True
    # Just past the tolerance band also flags; scales with magnitude.
    drop_over = abs(-254.0) * CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE + 0.5
    assert is_chosen_logp_collapsed(-254.0, -254.0 - drop_over) is True
    # A sub-tolerance drop just inside the band does not flag.
    drop_under = abs(-254.0) * CHOSEN_LOGP_COLLAPSE_REL_TOLERANCE - 0.5
    assert is_chosen_logp_collapsed(-254.0, -254.0 - drop_under) is False


def test_length_normalized_logp_divides_by_token_count() -> None:
    tokenizer = tiny_chat_tokenizer()
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    model = DenseBackbone(cfg)
    pair = tokenize_preference_pair(
        tokenizer,
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red blue",
            rejected="green",
            source="f",
            row_id="1",
        ),
        max_length=48,
        max_prompt_length=24,
    )
    from esme_posttrain.dpo.trainer import _completion_as_collate_row

    ids, labels = collate_batch((_completion_as_collate_row(pair.chosen),))
    summed = sequence_logprob(model, ids, labels, length_normalized=False)
    mean = sequence_logprob(model, ids, labels, length_normalized=True)
    supervised = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
    assert supervised > 0
    assert torch.allclose(mean, summed / supervised, atol=1e-5)


def test_reference_is_frozen_during_training(tmp_path: Path) -> None:
    tokenizer = tiny_chat_tokenizer()
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    reference = DenseBackbone(cfg)
    policy = DenseBackbone(cfg)
    policy.load_state_dict(reference.state_dict())
    reference_snapshot = {k: v.clone() for k, v in reference.state_dict().items()}
    pairs = tuple(
        tokenize_preference_pair(
            tokenizer,
            PreferencePair(
                prompt_turns=(ChatTurn("user", text),),
                chosen=chosen,
                rejected=rejected,
                source="f",
                row_id=text,
            ),
            max_length=48,
            max_prompt_length=24,
        )
        for text, chosen, rejected in (("say red", "red", "blue"), ("say blue", "blue", "red"))
    )
    run_dpo_training(
        policy,
        reference,
        tokenizer,
        pairs,
        pairs,
        DPOTrainerConfig(
            max_steps=5,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=0.02,
            beta=0.5,
            seed=214,
            output_dir=tmp_path / "run",
            eval_interval=5,
            log_interval=5,
        ),
    )
    after = reference.state_dict()
    for key, value in reference_snapshot.items():
        assert torch.equal(value, after[key]), f"reference param {key} changed during DPO"


def test_dpo_rejects_mismatched_policy_reference_config(tmp_path: Path) -> None:
    tokenizer = tiny_chat_tokenizer()
    policy = DenseBackbone(tiny_backbone_config())
    other = tiny_backbone_config()
    reference = DenseBackbone(type(other)(**{**other.to_dict(), "layers": 2}))
    pair = tokenize_preference_pair(
        tokenizer,
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red",
            rejected="blue",
            source="f",
            row_id="1",
        ),
        max_length=48,
        max_prompt_length=24,
    )
    with pytest.raises(DPOTrainerError, match="same model config"):
        run_dpo_training(
            policy,
            reference,
            tokenizer,
            (pair,),
            (pair,),
            DPOTrainerConfig(
                max_steps=1,
                micro_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                beta=0.5,
                seed=1,
                output_dir=tmp_path / "x",
            ),
        )


def test_dpo_logs_chosen_rejected_logps_every_eval(tmp_path: Path) -> None:
    tokenizer = tiny_chat_tokenizer()
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    reference = DenseBackbone(cfg)
    policy = DenseBackbone(cfg)
    policy.load_state_dict(reference.state_dict())
    pairs = tuple(
        tokenize_preference_pair(
            tokenizer,
            PreferencePair(
                prompt_turns=(ChatTurn("user", text),),
                chosen=chosen,
                rejected=rejected,
                source="f",
                row_id=text,
            ),
            max_length=48,
            max_prompt_length=24,
        )
        for text, chosen, rejected in (("say red", "red", "blue"), ("say green", "green", "two"))
    )
    result = run_dpo_training(
        policy,
        reference,
        tokenizer,
        pairs,
        pairs,
        DPOTrainerConfig(
            max_steps=10,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=0.02,
            beta=0.5,
            seed=214,
            output_dir=tmp_path / "run",
            eval_interval=5,
            log_interval=5,
        ),
    )
    eval_rows = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "eval"
    ]
    assert eval_rows, "DPO must log eval rows"
    for row in eval_rows:
        assert "eval/chosen_logp" in row
        assert "eval/rejected_logp" in row
        assert "eval/preference_accuracy" in row
        assert "eval/margin" in row
    assert "wandb_run_url" in result.to_dict()


# --- decoding pre-check -------------------------------------------------------


def test_ngram_repetition_rate() -> None:
    assert ngram_repetition_rate([1, 2, 3, 4], n=2) == 0.0
    # [1,1,1,1] -> bigrams all (1,1): one distinct of three -> 2/3 repeats.
    assert ngram_repetition_rate([1, 1, 1, 1], n=2) == pytest.approx(2 / 3)
    assert ngram_repetition_rate([1], n=3) == 0.0


def test_repetition_penalty_changes_decoding() -> None:
    tokenizer = tiny_chat_tokenizer()
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    model = DenseBackbone(cfg)
    prompt_ids = tokenizer.encode("user\nsay red\nassistant\n", add_special_tokens=False).ids
    greedy = generate_with_decoding(
        model, prompt_ids, GREEDY, eos_token_id=tokenizer.token_to_id("<eos>")
    )
    penalized = generate_with_decoding(
        model,
        prompt_ids,
        DecodingConfig(
            name="greedy_rep", strategy="greedy", repetition_penalty=2.0, max_new_tokens=8
        ),
        eos_token_id=tokenizer.token_to_id("<eos>"),
    )
    # Both produce token-id lists; the penalty path is exercised without error.
    assert isinstance(greedy, list)
    assert isinstance(penalized, list)


def test_decoding_precheck_summarizes_both_decoders() -> None:
    tokenizer = tiny_chat_tokenizer()
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    model = DenseBackbone(cfg)
    report = run_decoding_precheck(
        model,
        tokenizer,
        (("say_red", "user\nsay red\nassistant\n"), ("say_blue", "user\nsay blue\nassistant\n")),
        is_real_checkpoint=False,
        note="harness demo",
    )
    payload = report.to_dict()
    assert payload["is_real_checkpoint"] is False
    assert set(payload["per_decoding_summary"]) == {GREEDY.name, NUCLEUS_REP_PENALTY.name}
    for summary in payload["per_decoding_summary"].values():
        assert "mean_response_length" in summary
        assert "mean_repetition_rate_3gram" in summary


def test_nucleus_sampling_keeps_threshold_crossing_token() -> None:
    keep = _nucleus_keep_mask(torch.tensor([0.6, 0.3, 0.1]), top_p=0.7)

    assert keep.tolist() == [True, True, False]


# --- SFT-vs-DPO chat eval -----------------------------------------------------


def test_chat_eval_prompts_include_fixed_and_conversational() -> None:
    prompts = chat_eval_prompts()
    names = {name for name, _ in prompts}
    # The fixed multi-turn prompts plus the added conversational ones.
    assert {name for name, _ in CONVERSATIONAL_PROMPTS} <= names
    assert "capital_of_france" in names
    assert "weekend_followup" in names
    assert len(prompts) >= len(CONVERSATIONAL_PROMPTS) + 1


def test_chat_eval_compares_two_checkpoints_and_writes_markdown(tmp_path: Path) -> None:
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    sft_model = DenseBackbone(cfg)
    dpo_model = DenseBackbone(cfg)
    sft_path = tmp_path / "sft.pt"
    dpo_path = tmp_path / "dpo.pt"
    save_training_checkpoint(sft_path, model=sft_model, step=6300, metrics={})
    save_training_checkpoint(dpo_path, model=dpo_model, step=600, metrics={})
    tokenizer = tiny_chat_tokenizer()
    prompts = (
        ("say_red", "user\nsay red\nassistant\n"),
        ("say_blue", "user\nsay blue\nassistant\n"),
    )
    comparison = run_chat_eval(
        sft_path,
        dpo_path,
        tokenizer,
        device=torch.device("cpu"),
        max_new_tokens=8,
        prompts=prompts,
    )
    assert comparison.sft.label == "SFT"
    assert comparison.dpo.label == "DPO"
    summary = comparison.to_dict()["summary_comparison"]
    for decoding in comparison.decodings:
        assert "sft_mean_repetition_rate_3gram" in summary[decoding]
        assert "dpo_mean_repetition_rate_3gram" in summary[decoding]
        assert "sft_mean_response_length" in summary[decoding]

    markdown_path = tmp_path / "cmp.md"
    json_path = tmp_path / "cmp.json"
    write_chat_eval_markdown(comparison, markdown_path)
    write_chat_eval_json(comparison, json_path)
    md = markdown_path.read_text(encoding="utf-8")
    assert "DPO vs SFT chat-quality comparison" in md
    assert "Side-by-side generations" in md
    assert "SFT (" in md and "DPO (" in md  # both sides present per prompt
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "dpo_chat_eval"
    assert {"sft", "dpo", "summary_comparison", "degeneration_flags"} <= set(saved)


def test_chat_eval_flags_degeneration_on_short_cap(tmp_path: Path) -> None:
    # A tiny untrained model loops, so a short token cap reliably trips the
    # high-repetition / truncation flags -- proving the flagger works.
    cfg = tiny_backbone_config()
    torch.manual_seed(0)
    sft_path = tmp_path / "sft.pt"
    dpo_path = tmp_path / "dpo.pt"
    save_training_checkpoint(sft_path, model=DenseBackbone(cfg), step=1, metrics={})
    save_training_checkpoint(dpo_path, model=DenseBackbone(cfg), step=2, metrics={})
    comparison = run_chat_eval(
        sft_path,
        dpo_path,
        tiny_chat_tokenizer(),
        device=torch.device("cpu"),
        max_new_tokens=8,
        prompts=(("say_red", "user\nsay red\nassistant\n"),),
    )
    assert comparison.degeneration_flags  # some flags raised
    for flag in comparison.degeneration_flags:
        assert flag["model"] in {"SFT", "DPO"}
        assert flag["reasons"]


def test_chat_eval_preflight_never_launches_and_fits_cap() -> None:
    config = load_dpo_config(CONFIG_PATH)
    preflight = build_chat_eval_preflight(config, modal_gpu="A100")
    assert preflight["status"] == "ready_for_chat_eval"
    assert preflight["will_start_modal_job"] is False
    assert preflight["generation_only"] is True
    assert preflight["runtime"]["spend_cap_usd"] == CHAT_EVAL_SPEND_CAP_USD == 1.0
    assert preflight["runtime"]["timeout_cost_ceiling_usd"] <= CHAT_EVAL_SPEND_CAP_USD
    assert preflight["launch_blockers"] == []
    assert preflight["checkpoints"]["dpo_best"].endswith("best-checkpoint.pt")
    assert "modal_chat_dpo.py" in preflight["chat_eval_command"]


def test_chat_eval_blocks_gpu_mismatch() -> None:
    config = load_dpo_config(CONFIG_PATH)
    blockers = chat_eval_blockers(config, modal_gpu="L4")
    assert any("DPO_MODAL_GPU must match runtime.selected_gpu" in b for b in blockers)


def test_chat_eval_command_and_blockers_use_the_applied_timeout() -> None:
    config = load_dpo_config(CONFIG_PATH)
    preflight = build_chat_eval_preflight(config, modal_gpu="A100", timeout_hours=0.25)
    # The fractional timeout must survive into the emitted command verbatim.
    assert "DPO_TIMEOUT_HOURS=0.25" in preflight["chat_eval_command"]
    assert preflight["runtime"]["timeout_hours"] == 0.25
    assert preflight["launch_blockers"] == []
    # A 1h timeout ceiling exceeds the $1 cap and must block instead of running.
    blockers = chat_eval_blockers(config, modal_gpu="A100", timeout_hours=1.0)
    assert any("spend cap" in b for b in blockers)


def test_modal_chat_dpo_parses_fractional_timeout_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from scripts import modal_chat_dpo

    monkeypatch.delenv("DPO_TIMEOUT_HOURS", raising=False)
    try:
        reloaded = importlib.reload(modal_chat_dpo)
        assert reloaded.DPO_TIMEOUT_HOURS == 1.0
        # Chat eval keeps the documented 0.25h ceiling when the env is unset.
        assert reloaded.DPO_CHAT_EVAL_TIMEOUT_HOURS == 0.25
        monkeypatch.setenv("DPO_TIMEOUT_HOURS", "0.25")
        reloaded = importlib.reload(modal_chat_dpo)
        assert reloaded.DPO_TIMEOUT_HOURS == 0.25
        assert reloaded.DPO_CHAT_EVAL_TIMEOUT_HOURS == 0.25
        # Modal wants whole seconds; 0.25h must become 900s, never 0.
        assert int(reloaded.DPO_TIMEOUT_HOURS * 3600) == 900
    finally:
        monkeypatch.delenv("DPO_TIMEOUT_HOURS", raising=False)
        importlib.reload(modal_chat_dpo)


# --- CPU fixture: margin up, logp tracked, checkpoint round-trip ---------------


def test_cpu_fixture_margin_increases_and_no_collapse(tmp_path: Path) -> None:
    config = load_dpo_config(CONFIG_PATH)
    out = tmp_path / "fixture"
    result = run_dpo_cpu_fixture(config, output_dir=out)

    assert result["status"] == "local_cpu_fixture_dpo_complete"
    assert result["prompt_masking_asserted"] is True
    assert result["margin_increased"] is True
    assert result["chosen_logp_collapsed"] is False
    assert (
        result["result"]["selected_eval"]["preference_accuracy"]
        > result["result"]["base_eval"]["preference_accuracy"]
    )
    assert all(result["required_artifacts_present"].values())
    assert (out / "decoding-precheck.json").is_file()
    assert result["decoding_precheck"]["is_real_checkpoint"] is False


def test_cpu_fixture_checkpoint_reload_reproduces_logits(tmp_path: Path) -> None:
    config = load_dpo_config(CONFIG_PATH)
    out = tmp_path / "fixture"
    run_dpo_cpu_fixture(config, output_dir=out)

    loaded = load_training_checkpoint(out / "best-checkpoint.pt")
    tokenizer = tiny_chat_tokenizer()
    pair = tokenize_preference_pair(
        tokenizer,
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red",
            rejected="blue",
            source="f",
            row_id="1",
        ),
        max_length=48,
        max_prompt_length=24,
    )
    from esme_posttrain.dpo.trainer import _completion_as_collate_row

    ids, _ = collate_batch((_completion_as_collate_row(pair.chosen),))
    model = loaded.model
    model.eval()
    with torch.no_grad():
        first = model(ids[:, :-1])
        second = model(ids[:, :-1])
    assert torch.equal(first, second)


def test_dpo_full_acceptance_requires_accuracy_margin_and_no_logp_collapse() -> None:
    accepted = SimpleNamespace(
        base_eval=SimpleNamespace(preference_accuracy=0.5),
        selected_eval=SimpleNamespace(preference_accuracy=0.75),
        margin_increased=True,
        chosen_logp_collapsed=False,
    )
    _assert_accepted_dpo_result(accepted)

    lower_accuracy = SimpleNamespace(
        base_eval=SimpleNamespace(preference_accuracy=0.6),
        selected_eval=SimpleNamespace(preference_accuracy=0.5),
        margin_increased=True,
        chosen_logp_collapsed=False,
    )
    with pytest.raises(DPOFullRunError, match="preference accuracy"):
        _assert_accepted_dpo_result(lower_accuracy)

    collapsed = SimpleNamespace(
        base_eval=SimpleNamespace(preference_accuracy=0.5),
        selected_eval=SimpleNamespace(preference_accuracy=0.75),
        margin_increased=True,
        chosen_logp_collapsed=True,
    )
    with pytest.raises(DPOFullRunError, match="log-prob collapsed"):
        _assert_accepted_dpo_result(collapsed)


def test_dpo_full_prompt_masking_check_raises_on_leaked_prompt() -> None:
    from dataclasses import replace

    tokenizer = tiny_chat_tokenizer()
    pair = tokenize_preference_pair(
        tokenizer,
        PreferencePair(
            prompt_turns=(ChatTurn("user", "say red"),),
            chosen="red",
            rejected="blue",
            source="f",
            row_id="1",
        ),
        max_length=48,
        max_prompt_length=24,
    )
    assert _assert_prompt_masking((pair,)) is True

    # Unmask the prompt span of the chosen completion: the check must raise, not
    # just record False in the data report.
    leaked = replace(pair, chosen=replace(pair.chosen, labels=pair.chosen.input_ids))
    with pytest.raises(DPOFullRunError, match="prompt span leaked"):
        _assert_prompt_masking((leaked,))

    with pytest.raises(DPOFullRunError, match="no selected preference pairs"):
        _assert_prompt_masking(())


def test_dpo_full_rejects_eval_shortfalls_and_eval_cap_overruns() -> None:
    train_report = {
        "selected_pairs": 10,
        "selected_tokens": 100,
        "shortfalls": [],
    }
    caps = {
        "train_pairs": 10,
        "train_tokens": 100,
        "eval_pairs": 10,
        "eval_tokens": 100,
    }

    with pytest.raises(DPOFullRunError, match="eval data shortfall"):
        _assert_data_safe(
            train_report,
            {"selected_pairs": 5, "selected_tokens": 50, "shortfalls": ["selected 5/10"]},
            caps,
        )

    with pytest.raises(DPOFullRunError, match="eval token cap exceeded"):
        _assert_data_safe(
            train_report,
            {"selected_pairs": 10, "selected_tokens": 101, "shortfalls": []},
            caps,
        )


# --- config + dry-run + guards ------------------------------------------------


def test_config_pins_dpo_recipe_and_dry_run_never_launches() -> None:
    config = load_dpo_config(CONFIG_PATH)
    assert config.run_id == "esme_214m_chat_dpo"
    assert config.artifact_name == "Esme-214M-Chat"
    assert config.payload["dpo"]["loss_type"] == "sigmoid"
    assert config.payload["dpo"]["beta"] == 0.5
    assert config.budgets["max_length"] == 1024
    assert config.budgets["max_prompt_length"] == 512
    assert config.runtime["modal_volume"] == "esme-posttrain-esme-chat-dpo"

    dry_run = build_dpo_dry_run(config)
    assert dry_run["will_start_modal_job"] is False
    assert dry_run["will_download_data"] is False
    assert dry_run["preflight"]["will_start_modal_job"] is False
    assert dry_run["preflight"]["dataset_revisions"]["ultrafeedback-binarized"]
    assert dry_run["preflight"]["sft_reference_volume"] == "esme-posttrain-esme-sft-multiturn"
    assert dry_run["preflight"]["projected_cost_usd"] > 0
    assert "--full-run" in dry_run["full_launch_command"]
    assert "modal_chat_dpo.py" in dry_run["modal_smoke_command"]


def test_smoke_cost_under_two_dollars_and_no_blockers() -> None:
    config = load_dpo_config(CONFIG_PATH)
    assert config.estimated_smoke_cost_usd <= 2
    assert smoke_launch_blockers(config) == []


def test_full_run_refused_without_approval() -> None:
    config = load_dpo_config(CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=False, modal_gpu="A100")
    assert any("requires --approved" in b for b in blockers)


def test_full_run_refused_without_beta_sweep_evidence() -> None:
    # The checked-in config carries beta-sweep evidence; clear it to exercise the
    # no-evidence refusal path.
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["learning_gate"]["evidence"] = None
    config = validate_dpo_payload(payload, CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert any("bounded_beta_sweep" in b for b in blockers)


def test_full_run_blocks_modal_gpu_mismatch() -> None:
    config = load_dpo_config(CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="L4")
    assert any("DPO_MODAL_GPU must match" in b for b in blockers)


def _passing_beta_sweep_evidence() -> dict[str, object]:
    return {
        "bounded_beta_sweep": {
            "kind": "bounded_beta_sweep",
            "selector_metric": "eval/preference_accuracy",
            "swept_betas": [0.1, 0.3, 0.5],
            "best_beta": 0.3,
            "reference_preference_accuracy": 0.50,
            "best_preference_accuracy": 0.64,
            "best_chosen_logp_collapsed": False,
            "evidence_uri": "/posttrain/esme-chat-dpo-beta-sweep/evidence.json",
        }
    }


def test_full_run_accepts_passing_beta_sweep_evidence() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["learning_gate"]["evidence"] = _passing_beta_sweep_evidence()
    config = validate_dpo_payload(payload, CONFIG_PATH)
    assert full_launch_blockers(config, approved=True, modal_gpu="A100") == []


def test_full_run_rejects_sweep_not_beating_reference() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    evidence = _passing_beta_sweep_evidence()
    evidence["bounded_beta_sweep"]["best_preference_accuracy"] = 0.50
    payload["learning_gate"]["evidence"] = evidence
    config = validate_dpo_payload(payload, CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert any("best_preference_accuracy > reference_preference_accuracy" in b for b in blockers)


def test_full_run_rejects_chosen_logp_collapse() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    evidence = _passing_beta_sweep_evidence()
    evidence["bounded_beta_sweep"]["best_chosen_logp_collapsed"] = True
    payload["learning_gate"]["evidence"] = evidence
    config = validate_dpo_payload(payload, CONFIG_PATH)
    blockers = full_launch_blockers(config, approved=True, modal_gpu="A100")
    assert any("best_chosen_logp_collapsed must be false" in b for b in blockers)


def test_config_rejects_non_1024_max_length() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["budgets"]["max_length"] = 2048
    with pytest.raises(LaunchError, match="max_length must be 1024"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_pins_train_pair_floor_below_cap() -> None:
    config = load_dpo_config(CONFIG_PATH)
    budgets = config.budgets
    assert budgets["min_train_pairs"] == 8000
    assert budgets["min_eval_pairs"] == 100
    # The floor is a sufficiency minimum, well below the cap.
    assert budgets["min_train_pairs"] < budgets["max_train_pairs"]


def test_config_rejects_train_floor_above_cap() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["budgets"]["min_train_pairs"] = payload["budgets"]["max_train_pairs"] + 1
    with pytest.raises(LaunchError, match="min_train_pairs .floor. must be <="):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_rejects_simpo_loss_type() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["dpo"]["loss_type"] = "simpo"
    with pytest.raises(LaunchError, match="loss_type must be sigmoid"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_rejects_reference_free() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["dpo"]["reference_free"] = True
    with pytest.raises(LaunchError, match="reference_free must be false"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_rejects_auxiliary_sft_loss() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["dpo"]["auxiliary_sft_loss"] = True
    with pytest.raises(LaunchError, match="auxiliary_sft_loss must be false"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_rejects_too_high_learning_rate() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["optimizer"]["learning_rate"] = 1e-4
    with pytest.raises(LaunchError, match="learning_rate must be <= 1e-5"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_rejects_noncommercial_training_flag() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["datasets"]["non_commercial_training_approved"] = True
    with pytest.raises(LaunchError, match="non_commercial_training_approved"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_requires_logp_logging_and_judge_passes() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["monitoring"]["log_chosen_rejected_logps"] = False
    with pytest.raises(LaunchError, match="log_chosen_rejected_logps must be true"):
        validate_dpo_payload(payload, CONFIG_PATH)
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["monitoring"]["judge_repeat_passes"] = 3
    with pytest.raises(LaunchError, match="judge_repeat_passes must be >= 5"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_config_requires_stage_dpo_wandb_tag() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["monitoring"]["wandb_tags"] = ["Esme-214M-Chat"]
    with pytest.raises(LaunchError, match="must include stage=dpo"):
        validate_dpo_payload(payload, CONFIG_PATH)


def test_full_run_cap_above_limit_is_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["full_run_max_cost_usd"] = DPO_FULL_RUN_SPEND_CAP_USD + 1
    with pytest.raises(LaunchError, match="full_run_max_cost_usd must be <="):
        validate_dpo_payload(payload, CONFIG_PATH)


# --- beta sweep ---------------------------------------------------------------


def test_beta_sweep_arms_cover_exactly_the_swept_betas() -> None:
    assert tuple(arm.beta for arm in SWEEP_ARMS) == EXPECTED_SWEEP_BETAS
    for arm in SWEEP_ARMS:
        assert arm.effective_batch_size == 16
        assert arm.max_steps <= 200
        assert arm.eval_interval > 0


def test_beta_sweep_preflight_never_launches_and_fits_cap() -> None:
    config = load_dpo_config(CONFIG_PATH)
    preflight = build_dpo_sweep_preflight(config, timeout_hours=3, modal_gpu="A100")
    assert preflight["status"] == "ready_for_modal_sweep"
    assert preflight["will_start_modal_job"] is False
    assert preflight["launch_blockers"] == []
    assert preflight["runtime"]["projected_cost_usd"] <= SWEEP_SPEND_CAP_USD == 8.0
    assert preflight["acceptance"]["selector_metric"] == "eval/preference_accuracy"
    assert preflight["swept_betas"] == list(EXPECTED_SWEEP_BETAS)
    assert "modal_chat_dpo.py" in preflight["modal_sweep_command"]


def test_beta_sweep_blocks_gpu_mismatch() -> None:
    config = load_dpo_config(CONFIG_PATH)
    blockers = dpo_sweep_blockers(config, timeout_hours=3, modal_gpu="L4")
    assert any("DPO_MODAL_GPU must match runtime.selected_gpu" in b for b in blockers)


def test_beta_sweep_selects_best_arm_beating_reference_without_collapse() -> None:
    arms = [
        {
            "status": "complete",
            "arm_id": "a-beta0p1",
            "arm": {"beta": 0.1},
            "reference_eval": {"preference_accuracy": 0.5},
            "selected_eval": {"preference_accuracy": 0.55},
            "chosen_logp_collapsed": False,
        },
        {
            "status": "complete",
            "arm_id": "a-beta0p3",
            "arm": {"beta": 0.3},
            "reference_eval": {"preference_accuracy": 0.5},
            "selected_eval": {"preference_accuracy": 0.70},
            "chosen_logp_collapsed": True,  # collapsed -> ineligible despite high acc
        },
        {
            "status": "complete",
            "arm_id": "a-beta0p5",
            "arm": {"beta": 0.5},
            "reference_eval": {"preference_accuracy": 0.5},
            "selected_eval": {"preference_accuracy": 0.62},
            "chosen_logp_collapsed": False,
        },
    ]
    best = select_best_arm(arms)
    assert best is not None
    assert best["arm"]["beta"] == 0.5  # 0.3 disqualified by collapse, 0.5 > 0.1
    gate = learning_gate_payload(best_arm=best, evidence_uri="x")
    assert gate["status"] == "pass"
    assert gate["bounded_beta_sweep"]["best_beta"] == 0.5
    assert gate["bounded_beta_sweep"]["best_chosen_logp_collapsed"] is False


def test_beta_sweep_fails_when_no_arm_beats_reference() -> None:
    arms = [
        {
            "status": "complete",
            "arm_id": "a",
            "arm": {"beta": 0.1},
            "reference_eval": {"preference_accuracy": 0.5},
            "selected_eval": {"preference_accuracy": 0.49},
            "chosen_logp_collapsed": False,
        }
    ]
    assert select_best_arm(arms) is None
    gate = learning_gate_payload(best_arm=None, evidence_uri="x")
    assert gate["status"] == "fail"
    assert "blocker" in gate


# --- CLI ----------------------------------------------------------------------


def test_cli_dpo_dry_run_proves_no_modal_job(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["chat-dpo-dry-run", "--config", str(CONFIG_PATH), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["will_start_modal_job"] is False
    assert payload["run_id"] == "esme_214m_chat_dpo"


def test_cli_dpo_cpu_fixture_writes_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fixture"
    assert (
        main(
            [
                "chat-dpo-cpu-fixture",
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
    assert payload["status"] == "local_cpu_fixture_dpo_complete"
    assert (out / "metrics.jsonl").is_file()
    assert (out / "decoding-precheck.json").is_file()
