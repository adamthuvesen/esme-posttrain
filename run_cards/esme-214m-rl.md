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
- Launcher: private operator module.
- Public result summary: `docs/rlvr-countdown-lite-grpo.md`

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
pass. The five solved tasks are the five acceptance tasks whose target is the sum of
the numbers. The model learned to write that sum (`docs/rlvr-research-report.md`).

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

## Placebo control

A random-reward placebo separates the reward signal from the training process. It runs
the accepted recipe and budget with `grpo.reward_mode = "random"`: each reward is drawn
(seeded by `random_reward_seed`) from the same three-level support {invalid, valid,
exact}, independent of the completion.

- **Configs:** real reward `configs/esme-214m-rl.json` (seed 214) and
  `configs/esme-214m-rl-seed{215..219}.json`; placebo
  `configs/esme-214m-rl-placebo.json` (seed 214) and
  `configs/esme-214m-rl-placebo-seed{215..219}.json`.
- **Runs:** training only (`skip_acceptance_eval`), 240 steps, 1x A100, about $0.74 to
  $1.12 per run.
- **Evaluation:** each run's step-240 training checkpoint is rebuilt into a bundle with
  `rlvr-checkpoint-bundle`, then sampled on `heldout_large` (200 tasks, 16 samples,
  temperature 1.0, 12 new tokens, seed 0) with `rlvr-emit-decomp-completions`.
- **Study:** `studies/rlvr-placebo-step240-large.json` pins every CompletionSet and
  bundle manifest by hash.

| Measure | Real reward | Placebo | Difference, 95% CI |
| --- | ---: | ---: | --- |
| Valid expressions | 97.9% | 1.3% | +96.6 pp [+95.9, +97.3] |
| Exact solves, per sample | 6.3% | 0.07% | +6.2 pp [+3.2, +9.6] |
| Problems solved at least once | 6.6% | 1.2% | +5.4 pp [+2.2, +9.0] |

Every comparison has a sign-flip p of 0.031, the floor for six seeds. The real-reward
gain is the sum template: every seed solves the 13 sum-target tasks and, between them,
one other task.

## Ablations

Each ablation changes one part of the accepted recipe and keeps everything else equal
to the seed 215-217 real-reward runs.

| Arm | Change from the accepted recipe | Configs |
| --- | --- | --- |
| `binary` | Reward is 1.0 for an exact solve and 0.0 otherwise. | `configs/esme-214m-rl-abl-binary-seed{215,216,217}.json` |
| `closeness` | Reward is 1.0 for an exact solve, `0.3 * exp(-abs(value - target) / target)` for any other valid expression, and 0.0 otherwise. No flat valid or format credit. | `configs/esme-214m-rl-abl-closeness-seed{215,216,217}.json` |
| `nostab` | No zero-variance resampling, no replay buffer, no stratified sampling. | `configs/esme-214m-rl-abl-nostab-seed{215,216,217}.json` |
| `stdnorm` | Advantages divided by the group standard deviation, as in the original GRPO. | `configs/esme-214m-rl-abl-stdnorm-seed{215,216,217}.json` |

- **Dataset:** the committed Countdown-Lite train split (300 tasks); no download.
- **Budget per run:** 240 steps, 8x16 rollouts, `max_new_tokens 12`, at most 5,300,000
  rollout tokens, training only. 1x A100, 1-hour timeout, hard stop $2.10.
- **Evaluation:** the same rebuild, sampling, and `heldout_large` protocol as the
  placebo control. Sampling runs on 1x A10G per job, at most 3 jobs at once
  (`scripts/modal_emit_completions.py`).
- **Studies:** `studies/rlvr-ablation-{binary,nostab,stdnorm}.json`, each against the
  matched real-reward seeds.

Step-240 results on `heldout_large`, mean of seeds 215-217:

| Arm | Valid | Samples that add all numbers | Problems solved (of 200) | Report |
| --- | ---: | ---: | ---: | --- |
| Full recipe | 98.0% | 97.8% | 13.0 | |
| `binary` | 59.2% | 50.0% | 28.3 | `studies/rlvr-ablation-binary.report.md` |
| `nostab` | 97.3% | 97.0% | 13.3 | `studies/rlvr-ablation-nostab.report.md` |
| `stdnorm` | 97.9% | 97.7% | 13.0 | `studies/rlvr-ablation-stdnorm.report.md` |

The reward shape decides the outcome. With the graded reward every run collapses to the
sum template and solves only the 13 sum-target tasks. With the binary reward the model
writes the sum on half of its samples and also solves 15-18 other tasks. The stability
techniques change validity by under 1 pp, and the advantage normalization changes
nothing. The `closeness` arm is in training.

## Spend

The placebo control cost about $9.80 for eleven training runs. The `binary`, `nostab`,
and `stdnorm` ablations cost $8.84 for nine runs on A100s. Evaluation sampling on A10Gs
cost about $6, estimated from container hours. All other evaluation ran
on a laptop CPU.
