"""CPU-fixture proof for the GRPO-gain decomposition pipeline (esme-posttrain side).

Proves the two esme-posttrain halves of the decomposition without Modal: the random-reward
placebo GRPO mode and the grpo-decomp CompletionSet emitter. The grpo-decomp `report` that
consumes these artifacts is proven by the sibling test in that repo; here we assert the
artifacts are well formed and honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from esme_posttrain.bundle import load_dense_backbone_bundle
from esme_posttrain.rl import full as full_module
from esme_posttrain.rl.countdown_lite import load_countdown_lite_rows
from esme_posttrain.rl.decomp_emitter import (
    ESME_COUNTDOWN_SOURCE,
    EmitRequest,
    emit_completion_set,
    format_countdown_key,
)
from esme_posttrain.rl.full import run_countdown_lite_grpo_job
from esme_posttrain.rl.grpo import (
    CountdownGRPOTrainerConfig,
    _random_reward,
    run_countdown_lite_grpo,
)
from esme_posttrain.rl.launch import load_rlvr_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_BUNDLE = REPO_ROOT / "fixtures" / "tiny_bundle"
TINY_MANIFEST = REPO_ROOT / "fixtures" / "manifests" / "rl_tasks_tiny.json"
HELDOUT_MANIFEST = REPO_ROOT / "data" / "manifests" / "esme-214m-rl-heldout.tasks.json"
RL_FIXTURE_CONFIG = REPO_ROOT / "fixtures" / "configs" / "esme-214m-rl.fixture.json"


def _tiny_config(
    output_dir: Path, *, reward_mode: str, random_reward_seed: int = 214
) -> CountdownGRPOTrainerConfig:
    return CountdownGRPOTrainerConfig(
        max_steps=2,
        prompts_per_step=1,
        group_size=4,
        max_new_tokens=1,
        temperature=0.8,
        kl_beta=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        scheduler="constant",
        grad_clip=1.0,
        seed=214,
        output_dir=output_dir,
        max_rollout_tokens=100_000,
        exact_solve_reward=1.0,
        valid_expression_reward=0.3,
        invalid_reward=0.0,
        reward_mode=reward_mode,
        random_reward_seed=random_reward_seed,
        write_final_bundle=True,
    )


def _run_tiny_grpo(output_dir: Path, *, reward_mode: str) -> Path:
    loaded = load_dense_backbone_bundle(TINY_BUNDLE)
    reference = load_dense_backbone_bundle(TINY_BUNDLE).model
    train_rows = load_countdown_lite_rows(TINY_MANIFEST, split="train")
    result = run_countdown_lite_grpo(
        loaded.model,
        reference,
        loaded.tokenizer,
        train_rows,
        _tiny_config(output_dir, reward_mode=reward_mode),
    )
    return result.bundle_dir


def test_random_reward_draw_is_signal_free_and_over_support() -> None:
    import random

    config = _tiny_config(Path("unused"), reward_mode="random")
    rng = random.Random(config.random_reward_seed)
    support = {config.invalid_reward, config.valid_expression_reward, config.exact_solve_reward}
    draws = [_random_reward(rng, config) for _ in range(2_000)]
    # Every placebo reward is one of the three real support levels, and the draw touches
    # every level — it is not silently pinned to a single value.
    assert set(draws) == support
    assert all(value in support for value in draws)


def test_random_reward_requires_rng() -> None:
    config = _tiny_config(Path("unused"), reward_mode="random")
    with pytest.raises(Exception, match="placebo RNG"):
        _random_reward(None, config)


def test_random_reward_mode_is_deterministic(tmp_path: Path) -> None:
    first = _run_tiny_grpo(tmp_path / "random-a", reward_mode="random")
    second = _run_tiny_grpo(tmp_path / "random-b", reward_mode="random")
    first_rollouts = (first.parent / "rollouts.jsonl").read_text(encoding="utf-8")
    second_rollouts = (second.parent / "rollouts.jsonl").read_text(encoding="utf-8")
    assert first_rollouts == second_rollouts

    config_payload = json.loads((first.parent / "config.json").read_text(encoding="utf-8"))
    assert config_payload["reward_mode"] == "random"
    assert config_payload["random_reward_seed"] == 214


def test_random_reward_config_records_mode_and_placebo_is_not_verifier(tmp_path: Path) -> None:
    random_bundle = _run_tiny_grpo(tmp_path / "random", reward_mode="random")
    data_report = json.loads(
        (random_bundle.parent / "data-report.json").read_text(encoding="utf-8")
    )
    assert data_report["reward_policy"]["reward_mode"] == "random"
    # The placebo carries no task signal, so its reward policy is not marked verifiable-only.
    assert data_report["reward_policy"]["verifiable_only"] is False


def test_emitter_writes_valid_completion_set_schema(tmp_path: Path) -> None:
    out = tmp_path / "base__esme-countdown"
    summary = emit_completion_set(
        EmitRequest(
            bundle_path=TINY_BUNDLE,
            heldout_manifest_path=HELDOUT_MANIFEST,
            output_dir=out,
            set_name="heldout_fresh",
            n=2,
            temperature=0.8,
            max_new_tokens=4,
            seed=0,
        )
    )
    assert summary["dataset_name"] == ESME_COUNTDOWN_SOURCE
    assert summary["n_problems"] == 30
    assert summary["n_samples"] == 2

    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["dataset"]["name"] == ESME_COUNTDOWN_SOURCE
    assert provenance["dataset"]["config"] == "heldout_fresh"
    assert provenance["n_problems"] == 30
    assert provenance["sampling"]["n"] == 2

    lines = (out / "completions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    ids = set()
    for line in lines:
        item = json.loads(line)
        assert set(item["problem"]) == {"id", "question", "gold_answer"}
        ids.add(item["problem"]["id"])
        assert item["problem"]["gold_answer"].startswith("target=")
        # Uniform n, and every sample is a boxed answer the harness extractor can read.
        assert len(item["samples"]) == 2
        assert all(sample.startswith("\\boxed{") for sample in item["samples"])
    assert len(ids) == 30


def test_emitter_gold_answer_matches_countdown_key() -> None:
    assert format_countdown_key([9, 1], 10) == "target=10;numbers=1,9"
    assert format_countdown_key((3, 2, 1), 6) == "target=6;numbers=1,2,3"


def test_heldout_fresh_revision_is_pinned() -> None:
    """Guard the cross-repo contract: grpo-decomp commits this exact problem set + revision.

    grpo-decomp's ``esme-countdown`` eval set is a committed fixture keyed on this content
    hash. If Esme's held-out generator changes the 30 problem records, this pin trips here —
    in the repo that owns the generator — before the two sides silently diverge. Regenerate
    the grpo-decomp fixture and update this pin together when that is intentional.
    """
    from esme_posttrain.rl.decomp_emitter import _content_revision, _load_heldout_problems

    problems = _load_heldout_problems(HELDOUT_MANIFEST, "heldout_fresh")
    assert _content_revision("heldout_fresh", problems) == "heldout_fresh+e6e671c24ca56d27"


def test_three_arm_emit_produces_report_ready_layout(tmp_path: Path) -> None:
    """Train a placebo arm, then emit base/correct/random greedy sets in the report layout.

    This is the esme-posttrain side of the end-to-end proof: the three `<arm>__<set>`
    CompletionSets are written with identical problem records and dataset refs, greedy n=1,
    exactly what `grpo-decomp report --task-set esme-countdown` consumes.
    """
    random_bundle = _run_tiny_grpo(tmp_path / "random-run", reward_mode="random")
    correct_bundle = _run_tiny_grpo(tmp_path / "correct-run", reward_mode="verifier")
    completions_dir = tmp_path / "completions"

    arms = {
        "base": TINY_BUNDLE,
        "correct": correct_bundle,
        "random": random_bundle,
    }
    refs = {}
    problem_sets = {}
    for arm, bundle in arms.items():
        out = completions_dir / f"{arm}__esme-countdown"
        emit_completion_set(
            EmitRequest(
                bundle_path=bundle,
                heldout_manifest_path=HELDOUT_MANIFEST,
                output_dir=out,
                set_name="heldout_fresh",
                n=1,
                temperature=0.0,
                max_new_tokens=4,
                seed=0,
                model_label=arm,
            )
        )
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        refs[arm] = provenance["dataset"]
        assert provenance["sampling"]["n"] == 1
        assert provenance["sampling"]["temperature"] == 0.0
        problem_sets[arm] = [
            json.loads(line)["problem"]
            for line in (out / "completions.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    # All three arms must agree on the dataset ref and the exact problem records, or the
    # decomposition would be comparing arms over different problems.
    assert refs["base"] == refs["correct"] == refs["random"]
    assert problem_sets["base"] == problem_sets["correct"] == problem_sets["random"]


def test_skip_acceptance_eval_trains_and_exports_without_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skip-eval run trains + exports the bundle and never invokes the acceptance eval.

    This is the placebo relaunch shape: load base -> train -> export best-by-reward bundle,
    with no before/after eval (the ~4h evals that made the Modal run time out). We spy on the
    baseline eval entrypoint and assert it is never called, and that the bundle exists.
    """
    config = load_rlvr_config(RL_FIXTURE_CONFIG)

    eval_calls: list[object] = []

    def _no_eval(*args: object, **kwargs: object) -> dict[str, object]:
        eval_calls.append((args, kwargs))
        raise AssertionError("acceptance eval must not run when skip_acceptance_eval is set")

    monkeypatch.setattr(full_module, "run_countdown_lite_baseline", _no_eval)

    output_dir = tmp_path / "skip-eval"
    payload = run_countdown_lite_grpo_job(
        config,
        output_dir=output_dir,
        require_cuda=False,
        commit="fixture",
        dirty=False,
        skip_acceptance_eval=True,
    )

    # The acceptance eval never ran, so there is no before/after to compare.
    assert eval_calls == []
    assert payload["skip_acceptance_eval"] is True
    assert payload["grpo_result"] == "training-only"
    assert payload["before"] is None
    assert payload["after"] is None
    assert payload["before_report"] is None
    assert payload["after_report"] is None

    # Training still ran and the DenseBackbone bundle was exported by train/reward_mean.
    assert payload["trainer"]["steps_completed"] >= 1
    bundle_dir = Path(payload["bundle_dir"])
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "weights.pt").is_file()
    assert (output_dir / "best-checkpoint.pt").is_file()

    # The eval-derived artifacts are neither required nor written in skip mode.
    for eval_artifact in ("eval-before.json", "eval-after.json", "eval-after-final.json"):
        assert not (output_dir / eval_artifact).exists()
    assert eval_artifact not in payload["required_artifacts_present"]

    # Loading the exported bundle back proves it is a usable checkpoint for the emitter.
    load_dense_backbone_bundle(bundle_dir)


def test_skip_acceptance_eval_default_false_runs_eval(tmp_path: Path) -> None:
    """Default (no skip) still runs the before/after eval and writes the eval artifacts.

    Guards that the flag is opt-in: the accepted real run is unchanged.
    """
    config = load_rlvr_config(RL_FIXTURE_CONFIG)
    output_dir = tmp_path / "with-eval"
    payload = run_countdown_lite_grpo_job(
        config,
        output_dir=output_dir,
        require_cuda=False,
        commit="fixture",
        dirty=False,
    )
    assert payload.get("skip_acceptance_eval") is False
    assert payload["before"] is not None
    assert payload["after"] is not None
    assert (output_dir / "eval-before.json").is_file()
    assert (output_dir / "eval-after.json").is_file()
