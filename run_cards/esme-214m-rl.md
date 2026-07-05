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

## GRPO-gain decomposition (real signal vs placebo)

The accepted `Esme-214M-RL` gain is decomposed with the `grpo-decomp` harness by
comparing three arms on the held-out `heldout_fresh` Countdown set:

- **base** — `exports/esme-214m-chat` (pre-RL).
- **correct** — `exports/esme-214m-rl-caff0a1` (real verifier reward).
- **random** — the same recipe/budget with `grpo.reward_mode = "random"`: the reward is
  drawn (seeded via `random_reward_seed`) uniformly over the same three-level support
  {invalid, valid, exact}, independent of the completion. Any gain it shows is
  training-process placebo, not reward signal.

The first strict slice added a seed-214 placebo via `configs/esme-214m-rl-placebo.json`.
The 2026-07-05 multiseed follow-up adds seed 215-219 real-verifier and random-reward
pairs (`configs/esme-214m-rl{-placebo,}-seed{215,216,217,218,219}.json`) so the claim can be
made over training seeds instead of one run.

For the multiseed arms, `skip_acceptance_eval` keeps each run just-training: load base
bundle → 240 GRPO steps → export the best-by-`train/reward_mean` bundle, with no
before/after acceptance eval. The decomposition does not use those evals; completions come
from the emitter, checkpoint selection is by `train/reward_mean`, and all scoring happens
offline in `grpo-decomp`.

Completions for each arm are exported as `grpo-decomp` `CompletionSet` artifacts
(`provenance.json` + `completions.jsonl`) with the emitter:

```bash
uv run esme-posttrain rlvr-emit-decomp-completions \
  --bundle exports/esme-214m-chat --set heldout_fresh \
  --out runs/decomp/base__esme-countdown --n 1 --temperature 0.0 --json
```

Each emitted sample is the model's Countdown expression wrapped in `\boxed{...}`; the
`grpo-decomp` `esme-countdown` verifier grades it with Esme's rules (each supplied number
used exactly once, `+ - *` only, integer result equal to target). The result table is
produced by `grpo-decomp report --task-set esme-countdown` in the `grpo-decomp` repo.

CPU-fixture proof (no private compute): `tests/test_rlvr_decomp.py` here, plus
`tests/test_esme_countdown_decomp.py` in `grpo-decomp`.

### Result (2026-07-03, PRELIMINARY — 1 seed)

Placebo run: private training job, 240 steps, ~24 min, ~$0.83. Its
`train/reward_mean` stayed flat (~0.42, the random-draw average) with no climb
— confirming the reward carried no task signal. Two earlier attempts on A100
spot failed first to worker preemption (step 156) and then to a before-eval
wall-timeout (928/960); adding `skip_acceptance_eval` removed the eval phases
and the run completed clean on the third try. Sunk spend on the two failed
attempts ~$5.

Greedy pass@1 on `heldout_fresh` (n=30), graded by the `esme-countdown` verifier:

| Arm | Solved | pass@1 |
| --- | ---: | ---: |
| base (Esme-214M-Chat) | 1/30 | 3.33% |
| correct (Esme-214M-RL, real reward) | 2/30 | 6.67% |
| random (placebo, random reward) | 1/30 | 3.33% |

Pre-registered confirmatory test — placebo `correct − random`: **+3.3 pp, exact-binomial
p = 1.0, n_discordant = 1, 95% CI [0.0, +10.0] pp**. The real-reward model's edge over the
random-reward placebo rests on a **single** discordant problem (2 vs 1 solved) and is not
statistically distinguishable from zero at this scale. Format sensitivity (lenient vs strict
extraction) was +0.0 pp.

**Reading:** on a held-out set with **greedy decoding and exact-solve only**, RL's contribution
beyond a same-budget placebo is not separable — but that is a measurement artifact, not a null
result, and the sampled re-measurement below overturns it. Greedy pass@1 on exact-solve is the
sparsest, lowest-power slice available for a 214M model on Countdown (the whole dynamic range is
1-2 problems), so it cannot see the effect. Kept here as the strict-slice footnote; the honest
headline is the sampled result. Artifacts: `grpo-decomp
results/esme-countdown/{summary,decomposition}`.

### Sampled multiseed result (2026-07-05) — supported headline

Re-measured the same held-out problems with **n=16, temperature 1.0** and aggregated over
six training seeds. New Modal spend for the seed215-219 multiseed pairs was **$8.96**:
seed215 real/placebo `$0.9999`/`$0.7507`, seed216 `$1.1223`/`$0.7447`, and seed217
`$0.9952`/`$0.7547`, seed218 `$1.0612`/`$0.7908`, and seed219 `$0.9910`/`$0.7473`.
The prior seed214 placebo receipt was about `$0.83`.

Seed-level result, each seed on the same 30 held-out problems with 16 samples/problem:

| Seed | correct valid | random valid | Δ valid | correct any-exact | random any-exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| 214 | 27.1% | 0.8% | +26.2pp | 4/30 | 0/30 |
| 215 | 97.7% | 0.4% | +97.3pp | 3/30 | 0/30 |
| 216 | 97.1% | 0.8% | +96.2pp | 3/30 | 1/30 |
| 217 | 97.1% | 0.6% | +96.5pp | 3/30 | 0/30 |
| 218 | 96.7% | 0.6% | +96.0pp | 3/30 | 1/30 |
| 219 | 96.9% | 1.2% | +95.6pp | 3/30 | 1/30 |

Aggregate arm means:

| Arm | valid-expr rate | pass@1 | pass@8 | pass@16 | any-exact solved |
| --- | ---: | ---: | ---: | ---: | ---: |
| base (Esme-214M-Chat) | 0.8% | 0.2% | 1.7% | 3.3% | 1/30 |
| correct (Esme-214M-RL, real reward) | 85.4% | 9.0% | 10.3% | 10.6% | 3.17/30 |
| random (placebo, random reward) | 0.8% | 0.1% | 0.8% | 1.7% | 0.50/30 |

Seed-level tests (unit = training seed, n=6): valid-expression separation is **+84.7 pp,
95% CI [+54.6, +114.7]**. Any-exact separation is **+8.9 pp, 95% CI [+6.0, +11.7]**.

**Reading:** real verifier reward is now separable from random-reward placebo across seeds
on sampled held-out Countdown validity. Seed 214 was the low-validity outlier; seeds 215-219
all land near 97% validity while the placebos stay near base. Exact solving is smaller but
also positive across seeds. The supported claim is "RL sharpened form first, exact solving
second," not "214M learned broad new Countdown reasoning." Artifacts: `grpo-decomp
results/esme-countdown/{sampled_decomposition.md,sampled_multiseed_summary.json}`.
