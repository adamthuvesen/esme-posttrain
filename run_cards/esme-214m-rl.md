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
pass. 214M does not master exact-solve. The five solved tasks are the five acceptance tasks whose target is the sum of the numbers. The model learned to write that sum (`docs/rlvr-research-report.md`).

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

- **base**: `exports/esme-214m-chat` (pre-RL).
- **correct**: `exports/esme-214m-rl-caff0a1` (real verifier reward). This export held
  the step-22 bundle of run `caff0a1`, not the accepted step-234 model; see the fixed-step
  re-score below.
- **random**: the same recipe/budget with `grpo.reward_mode = "random"`: the reward is
  drawn (seeded via `random_reward_seed`) uniformly over the same three-level support
  {invalid, valid, exact}, independent of the completion. Any gain it shows is
  training-process placebo, not reward signal.

The first strict slice added a seed-214 placebo via `configs/esme-214m-rl-placebo.json`.
The 2026-07-05 multiseed follow-up adds seed 215-219 real-verifier and random-reward
pairs (`configs/esme-214m-rl{-placebo,}-seed{215,216,217,218,219}.json`) so the claim can be
made over training seeds instead of one run.

For the multiseed arms, `skip_acceptance_eval` keeps each run training-only: load base
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
used exactly once, `+ - *` only, integer result equal to target). The sampled multi-seed
result is produced offline by `scripts/esme_sampled_decomp.py` in the `grpo-decomp` repo.

CPU-fixture proof (no private compute): `tests/test_rlvr_decomp.py` here, plus
`tests/test_esme_countdown_decomp.py` in `grpo-decomp`.

### Result (2026-07-03, preliminary 1-seed read)

Placebo run: private training job, 240 steps, ~24 min, ~$0.83. Its
`train/reward_mean` stayed flat (~0.42, the random-draw average) with no climb
and confirmed the reward carried no task signal. Two earlier attempts on A100
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

Pre-registered confirmatory test, placebo `correct − random`: **+3.3 pp, exact-binomial
p = 1.0, n_discordant = 1, 95% CI [0.0, +10.0] pp**. The real-reward model's edge over the
random-reward placebo rests on a **single** discordant problem (2 vs 1 solved) and is not
statistically distinguishable from zero at this scale. Format sensitivity (lenient vs strict
extraction) was +0.0 pp.

**Reading:** on a held-out set with **greedy decoding and exact-solve only**, RL's contribution
beyond a same-budget placebo is not separable. That is a measurement artifact, not a null
result, and the sampled re-measurement below overturns it. Greedy pass@1 on exact-solve is the
sparsest, lowest-power slice available for a 214M model on Countdown (the whole dynamic range is
1-2 problems), so it cannot see the effect. Kept here as the strict-slice footnote; the honest
headline is the sampled result. Artifacts: `grpo-decomp
results/esme-countdown/{summary,decomposition}`.

### Sampled multiseed result (2026-07-05)

Superseded by the [fixed-step re-score](#fixed-step-re-score-2026-10-24) below, which
fixes a checkpoint-selection defect and a seed-214 provenance error in this section.

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
95% CI [+59.9, +97.5]**. Any-exact separation is **+8.9 pp, 95% CI [+0.6, +20.0]**. Both
have an exact sign-flip permutation p of 0.031, the smallest two-sided value six seeds
allow. The intervals come from a bootstrap that resamples seeds and, independently, one
shared set of problems (`studies/rlvr-placebo.report.md`). The paired Student t interval
first reported here,
[+54.6, +114.7], went above the 100 pp ceiling a rate difference can reach, so it was
replaced.

**Reading:** real verifier reward is now separable from random-reward placebo across seeds
on sampled held-out Countdown validity. Seed 214 was the low-validity outlier; seeds 215-219
all land near 97% validity while the placebos stay near base. Exact solving is smaller but
also positive across seeds. The supported claim is "RL sharpened form first, exact solving
second," not "214M learned broad new Countdown reasoning." Artifacts: `grpo-decomp
results/esme-countdown/{sampled_decomposition.md,sampled_multiseed_summary.json}`.

### Fixed-step re-score (2026-10-24)

The 2026-07-05 result above had two defects. This re-score fixes both. It used no new
training: every arm was rebuilt from the training checkpoint each run kept at step 240.

**Defect 1: checkpoint selection.** Each run exported only its bundle with the best
`train/reward_mean`. Under a random reward that choice is noise. The evaluated placebo
bundles came from steps 8, 49, 112, 162, and 219, while the real-reward bundles came from
steps 159 to 234. The arms were not compared at the same training budget.

**Defect 2: seed-214 provenance.** The study pinned the training manifest of run
`v2-ccb6287-1` (best step 234) for the seed-214 real-reward arm. Its completions came from
the bundle of run `caff0a1`, which selected step 22. Evidence: a re-emit from the `caff0a1`
bundle with the study's decoding settings reproduces 457 of 480 samples exactly; a re-emit
from the `v2-ccb6287-1` bundle reproduces 123. This is the "low-validity outlier" above.
The emitter wrote no commit or weights hash at the time, so the error was not visible in
the artifacts. The emitter now records both.

**Method.**

1. `rlvr-checkpoint-bundle` rebuilt a dense bundle from
   `checkpoints/step-000240/checkpoint.pt` for all 12 arms. For seed 214, the rebuilt
   bundle equals the trainer's own step-240 `bundle-final` in all 333 tensors.
2. `rlvr-emit-decomp-completions` sampled each bundle on local CPU with the study's
   decoding settings: `heldout_fresh`, n=16, temperature 1.0, 12 new tokens, seed 0.
3. `studies/rlvr-placebo-step240.json` pins every CompletionSet and rebuilt bundle
   manifest by hash. The baseline arm reuses the original Chat completions.

Spend: $0 compute. Downloads were about 29 GiB of checkpoints from the
`esme-posttrain-esme-rlvr-countdown` volume. The provenance of all 12 CompletionSets says
`dirty: true`. The emit and convert code was committed at `297d94e`; the uncommitted
changes during the run touched only the study report code and docs.

**Result** (`studies/rlvr-placebo-step240.report.md`):

| Seed | Real valid | Placebo valid | Real any-exact | Placebo any-exact |
| --- | ---: | ---: | ---: | ---: |
| 214 | 97.1% | 0.6% | 3/30 | 0/30 |
| 215 | 97.7% | 0.6% | 3/30 | 0/30 |
| 216 | 97.1% | 0.8% | 3/30 | 1/30 |
| 217 | 97.1% | 0.6% | 3/30 | 0/30 |
| 218 | 97.1% | 1.7% | 3/30 | 2/30 |
| 219 | 96.9% | 1.2% | 3/30 | 2/30 |

- Validity: **+96.2 pp, 95% CI [+94.5, +97.8]**, sign-flip p = 0.031.
- Any-exact: **+7.2 pp, 95% CI [+0.0, +17.2]**, sign-flip p = 0.031.

**Reading:** At a matched training budget, real verifier reward moves held-out validity
from near base to about 97% on every seed; the random-reward placebo does not move it.
The exact-solve gain is narrow. Every real seed solves the same three problems:
`easy_0000` (1, 9 → 10), `easy_0002` (1, 3 → 4), and `medium_0004`. Placebo seeds
sometimes solve the two easy ones, and the base model solves `easy_0002`. The exact-solve
effect is consistent across seeds (p at its floor), but it rests on those three problems,
so its interval over a fresh draw of problems reaches zero. The supported claim is "RL
sharpened form." An exact-solve gain needs a larger held-out set before it is claimed.

**Legacy bundles:** the exported `caff0a1` and `v2-ccb6287-1` bundles fail their own
manifest weights hash, although both files are complete and load. No current result
depends on them: the studies and the held-out docs use bundles rebuilt from training
checkpoints and verified by hash.

## Stability ablation (2026-10-24)

**Question:** which parts of the GRPO stability bundle produce the held-out validity gain?
The accepted recipe combines a graded reward, Dr. GRPO mean-only advantages, zero-variance
resampling, a success-replay buffer, and stratified difficulty sampling. Each ablation
removes one part and keeps everything else equal to the seed 215-217 real-reward runs.

| Arm | Change from the accepted recipe | Configs |
| --- | --- | --- |
| `binary` | Reward is 1.0 for an exact solve and 0.0 otherwise (no valid, format, or closeness credit). | `configs/esme-214m-rl-abl-binary-seed{215,216,217}.json` |
| `nostab` | No zero-variance resampling, no replay buffer, no stratified sampling. | `configs/esme-214m-rl-abl-nostab-seed{215,216,217}.json` |
| `stdnorm` | Advantages divided by the group standard deviation, as in the original GRPO. | `configs/esme-214m-rl-abl-stdnorm-seed{215,216,217}.json` |

- **Control:** the existing seed 215-217 real-reward runs, scored at step 240.
- **Dataset:** the committed Countdown-Lite train split (300 tasks); no download.
- **Budget per run:** 240 steps, 8x16 rollouts, `max_new_tokens 12`, at most 5,300,000
  rollout tokens. Training only (`skip_acceptance_eval`).
- **Hardware:** 1x A100 per run on the private training account; 9 runs.
- **Expected duration and cost:** about 30 minutes and $0.75-$1.15 per run, from the
  multiseed receipts; about $7-$10 total. Hard stop $2.10 per run with a 1-hour timeout ($18.90 worst case),
  inside the $25 approved for this campaign on 2026-10-24.
- **Evaluation sampling:** `scripts/modal_emit_completions.py` rebuilds and samples
  checkpoints on 1x A10G per job, at most 3 containers at once. Expected cost: a few
  dollars for the 200-task set, learning curves, and pass@k. Same $25 approval.
- **Artifacts:** each run's step-240 training checkpoint is rebuilt with
  `rlvr-checkpoint-bundle`, sampled with the placebo study's decoding settings, and
  scored with the study report against the matched real-reward seeds.

### Ablation result (2026-10-24)

All nine runs completed 240 steps for $8.84 in total. Each step-240 checkpoint was
rebuilt and sampled on `heldout_large` (200 tasks, 16 samples, temperature 1.0):

| Arm | Valid | Problems solved (of 200) | Report |
| --- | ---: | ---: | --- |
| Full recipe (seeds 215-217) | 98.0% | 13.0 | |
| `binary` | 59.2% | 28.3 | `studies/rlvr-ablation-binary.report.md` |
| `nostab` | 97.3% | 13.3 | `studies/rlvr-ablation-nostab.report.md` |
| `stdnorm` | 97.9% | 13.0 | `studies/rlvr-ablation-stdnorm.report.md` |

The graded reward is the component that matters. With it, every run collapses to one
template, the sum of all the numbers, which solves only the 13 tasks whose target is
that sum. With the binary reward the model writes the sum on half of its samples and
also solves 15-18 other tasks. The stability techniques change validity by under 1 pp
and leave the solved set unchanged. The advantage normalization changes nothing.
Evaluation sampling on A10Gs cost about $6, estimated from container hours. Analysis:
`docs/rlvr-research-report.md`.
