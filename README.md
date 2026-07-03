# esme-posttrain

Esme is a 214M-parameter language model trained from scratch. `esme-posttrain` adapts an `Esme-214M-Base` checkpoint from `esme-pretrain` into instruction-following, preference-tuned, and verifier-trained model artifacts.

The standard post-training path has three stages:

```text
Esme-214M-Base
  -> SFT:  Esme-214M-Instruct
  -> DPO:  Esme-214M-Chat
  -> RLVR: Esme-214M-RL
```

RLVR uses GRPO on the current Countdown-Lite verifier task.

## Stage Summary

| Stage | Starts from | Produces | Accepted signal |
| --- | --- | --- | --- |
| SFT | `Esme-214M-Base` | `Esme-214M-Instruct` | Held-out response loss bottoms at `1.36` on step `6300`. |
| DPO | `Esme-214M-Instruct` | `Esme-214M-Chat` | Held-out preference accuracy peaks at `67.4%` on step `600`. |
| RLVR | `Esme-214M-Chat` | `Esme-214M-RL` | Best GRPO reward is `0.71` on step `234`; acceptance exact-solve rate is `16.35%`. |

SFT teaches chat format, turn-taking, and basic instruction following. DPO
trains the model to prefer chosen answers over rejected answers, without
on-policy RL. RLVR then trains against verifier-backed rewards on one task.

The current RLVR target is Countdown-Lite: generate a short arithmetic expression that uses each supplied number exactly once and reaches the target. The reward is verifier-backed. Style rewards are intentionally out of scope.

## Why 214M?

Esme-214M is intentionally small for learning purposes. That makes the full LLM lifecycle easier to build, keeps iteration fast and costs low, and makes failures easier to diagnose, while still going through real training, evaluation, export, post-training, and inference.

## What Is Here

- Stage code for SFT, DPO, RLVR, launch validation, dense-bundle export, and shared artifact writing.
- Configs for the current Esme-214M post-training path, validated in code by the stage launch modules.
- Evidence docs for SFT, DPO, and completed Countdown-Lite GRPO.
- Export tooling for `Esme-214M-Chat` bundles.

## Quickstart

```bash
uv sync --extra dev
uv run esme-posttrain --version
make check
```

Use `uv run ...` for Python commands. Default local commands do not download remote datasets, start Modal jobs, or spend money.

Modal SFT training launchers need a local `Esme-214M-Base` export bundle before
they can start a job. Set `ESME_BASE_BUNDLE_LOCAL` to the bundle directory, or
place the sibling `esme-pretrain` checkout next to this repo so the fallback
`../esme-pretrain/exports/esme-214m-base` exists.

```bash
export ESME_BASE_BUNDLE_LOCAL=/path/to/esme-214m-base
uv run python scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --dry-run --json
```

## Current Artifacts

- `docs/rlvr-countdown-lite.md` describes the RLVR task, baseline, and result.
- `docs/rlvr-countdown-lite-grpo.md` summarizes the completed Countdown-Lite GRPO run.
- `docs/rlvr-countdown-heldout-transfer.md` scores the RL and pre-RL checkpoints on held-out Countdown sets.
- Generated export bundles are written under gitignored `exports/`.
- Generated run reports are written under gitignored `artifacts/`.

## Training Telemetry

Static cards rendered from the accepted runs' Modal artifacts. The README keeps
one visual per post-training stage: SFT and DPO show checkpoint selection, and
RLVR shows verifier-scored behavior before and after RL. The GRPO reward curve
is still generated for run-level debugging, but the README leaves that detail
to the RL docs.

**Reading note:** These charts show training stability and checkpoint selection,
not broad capability. `Esme-214M` is intentionally small, so absolute gains are
capped by the base model; the key signal is improvement without collapse.

<p>
  <img src="assets/fig-sft-training-dynamics.svg" width="100%" alt="Multi-turn SFT training" />
</p>
<p>
  <img src="assets/fig-dpo-training-dynamics.svg" width="100%" alt="DPO preference training" />
</p>
<p>
  <img src="assets/fig-grpo-countdown-evidence.svg" width="100%" alt="Countdown-Lite evidence: acceptance and unseen-task transfer" />
</p>

The cards are rendered by `scripts/plot_run_telemetry.py` from the runs'
`rollouts.jsonl`, `metrics.jsonl`, and `best-checkpoint.json`, cross-checking
the derived curves against every logged metric record before rendering. The
script also exports `assets/fig-grpo-training-dynamics.svg` for the RL docs.
The RL evidence card uses sample-level valid-expression and exact-solve rates
transcribed from `docs/rlvr-countdown-lite-grpo.md` and
`docs/rlvr-countdown-heldout-transfer.md`; task-level pass@k metrics stay in
those detailed reports.

## Repository Layout

```text
src/esme_posttrain/
  cli/                command-line entry point (parser + one module per command group)
  bundle.py           dense backbone bundle loading and hashing
  modeling.py         shared dense model primitives
  run_artifacts.py    shared JSON/environment/manifest artifact writers
  sft/                supervised fine-tuning stage
  dpo/                preference-tuning stage
  rl/                 verifier-reward RL stage
  launch/             shared launch validation
  training/           shared training runtime (collate, metrics, checkpointing)
  export/             dense bundle export
```

Stage-specific code belongs in the stage package. Keep the package root small.

## Related Repositories

These repositories exchange artifacts, not imports:

- [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain): trains
  `Esme-214M-Base` from scratch.
- [`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain): adapts
  the base checkpoint with SFT, DPO, and verifier-backed RLVR.
- [`llm-infer`](https://github.com/adamthuvesen/llm-infer): loads, serves, and
  benchmarks exported Esme checkpoints.
- [`llm-rlvr`](https://github.com/adamthuvesen/llm-rlvr): provides a reusable
  RLVR harness with text-to-SQL as the reference task.
- [`grpo-decomp`](https://github.com/adamthuvesen/grpo-decomp): measures where
  GRPO gains come from, separating reliability from new capability.

## References

- Lambert et al., [_Tulu 3: Pushing Frontiers in Open Language Model Post-Training_](https://arxiv.org/abs/2411.15124), 2025.
- Chung et al., [_Scaling Instruction-Finetuned Language Models_](https://arxiv.org/abs/2210.11416), 2022.
- Rafailov et al., [_Direct Preference Optimization_](https://arxiv.org/abs/2305.18290), 2023.
- Shao et al., [_DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models_](https://arxiv.org/abs/2402.03300), 2024.
- Guo et al., [_DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning_](https://arxiv.org/abs/2501.12948), 2025.
- Wen et al., [_Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs_](https://arxiv.org/abs/2506.14245), 2025.
