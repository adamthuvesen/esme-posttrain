# Run Card: Esme-214M-RL

This is the verifier-reward stage of the standard post-training chain:

```text
Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL
```

`Esme-214M-RL` is the Countdown-Lite GRPO variant of `Esme-214M-Chat`.

## Method

- Objective: group-normalized REINFORCE-with-baseline plus a KL penalty
  against the frozen `Esme-214M-Chat` reference.
- The trainer takes one gradient step per rollout batch, so a PPO-style
  importance ratio would be identically 1 and no clipping term exists.
- Stability bundle: Dr. GRPO mean-only advantage (no std division), a graded
  reward (invalid 0.0 < format-only 0.05 < valid 0.3 + bounded closeness <
  exact 1.0, with exact kept >= 0.4 above the closeness ceiling), DAPO
  zero-variance group resampling (one resample per group), an ARPO
  success-replay buffer for all-failed groups, stratified difficulty sampling
  per batch, and per-step stability telemetry (token entropy, zero-variance
  fraction, replay injections, reward components).
- Hyperparameters: lr `5e-7`, `kl_beta 0.001`, temperature `1.0`, `8x16`
  rollouts per step, 240 steps, cosine decay, checkpoints every 20 steps,
  `max_new_tokens 12` (holds a 3-number expression).

## Artifact

- Produces: `Esme-214M-RL`
- Starts from: `Esme-214M-Chat`
- Config: `configs/esme-214m-rl.json`
- Config and dataset-manifest validation: `src/esme_posttrain/rl/launch.py`
- Launcher: `scripts/modal_rlvr_grpo.py`
- Public result summary: `docs/rlvr-countdown-lite-grpo.md`
- Operator provenance, training-shape verdict, incident record:
  `docs/internal/rlvr-countdown-lite-grpo-run.md`

## Task

- Dataset manifest: `data/manifests/esme-214m-rl.tasks.json`
- Local splits: `data/rl/countdown_lite/{train,dev,eval}.jsonl`
- Shape: 300 train tasks, 30 dev tasks, 30 eval tasks.
- Rule: generate an arithmetic expression using each supplied number exactly
  once to reach the target.
- Operators: `+`, `-`, `*`, and parentheses.
- Reward: exact verifier-backed execution check.
- Excluded rewards: style qualities such as friendliness, sharpness, or
  naturalness.
- Secondary transfer eval: GSM8K-lite, separate from the reward and acceptance
  target.

## Evaluation Profile

- Primary eval: Countdown-Lite eval split.
- Acceptance profile: `full_acceptance_30x32` (`30` tasks x `32` samples).
- Eval token budget: `eval_max_new_tokens 12`.
- Seed: `214`.
- Training rollout token budget: 5,300,000 (worst-case incl. the resample
  factor).
- Runtime hard stop: `$18.00`.
- Cost cap: `$20.00`, under the `$25.00` Countdown-Lite mission cap.

## Results

30 tasks x 32 samples, seed 214, evaluated on the best (step 234) and final
(step 240) checkpoints, which score identically.

| Report | pass@1 | pass@8 | pass@32 | Valid expressions | Exact solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Esme-214M-Chat` before-eval | 3.33% | 6.67% | 13.33% | 5.83% | 0.73% |
| `Esme-214M-RL` GRPO | 16.67% | 16.67% | 16.67% | 99.38% | 16.35% |

Training reward rose from 0.03 to a last-20-step mean of 0.51 (best 0.71 at
step 234) with no collapse; zero of 240 steps logged `reward_mean == 0`. GRPO
takes the valid-expression rate from 5.83% to 99.38% and lifts exact solves on
the easy band; all 5 solved tasks are easy-band, and medium/hard stay at 0%
pass — 214M does not master exact-solve.

## Acceptance

- Countdown-Lite verifier metrics improve over the chat baseline.
- The run writes reproducible config, metrics, report, checkpoint, tokenizer,
  manifest, cost, and environment artifacts.
- Eval records include phase, eval profile, config hash, model id, task/sample
  range, split, sample budget, and completion counts.
- Re-running eval with matching metadata resumes from completed task records;
  metadata mismatches fail loudly.
- W&B is disabled by default for local commands.

## Safe Local Commands

```bash
uv run esme-posttrain rlvr-dry-run --config configs/esme-214m-rl.json
uv run esme-posttrain rlvr-dry-run --config fixtures/configs/esme-214m-rl.fixture.json
uv run esme-posttrain rlvr-pipeline-smoke --config configs/esme-214m-rl.json --json
uv run esme-posttrain rlvr-countdown-lite-build-data --repo-root . --json
```
