# What GRPO Teaches a 214M Model on Countdown

GRPO with a verifier reward makes `Esme-214M-Chat` write a valid Countdown expression
98% of the time on held-out tasks, up from 1.4%. It also solves more problems exactly,
but the solved set is small and it shrinks as training continues. The shape of the
reward controls this tradeoff. The usual GRPO stability techniques do not change it.

All numbers come from committed study reports and `studies/rlvr-research-summary.json`.
Figures are in `assets/`.

## Question

Countdown gives a set of numbers and a target. The model must write an arithmetic
expression that uses each number exactly once and reaches the target. A verifier checks
the answer, so the reward needs no human or model judge.

This report asks three questions:

1. Does the gain come from the reward signal, or from the training process itself?
2. Does the model learn to solve problems, or only to write valid expressions?
3. Which parts of the training recipe produce the gain?

## Setup

**Model.** `Esme-214M-Chat`: a 214M-parameter dense transformer, pretrained from scratch
in [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain), then tuned with SFT
and DPO in this repo.

**Training.** GRPO for 240 steps on 300 Countdown-Lite training tasks: 8 prompts x 16
samples per step, learning rate `5e-7`, KL penalty `0.001` toward the Chat model,
temperature 1.0, at most 12 new tokens. The accepted recipe uses:

- a graded reward: 0 invalid, 0.05 well-formed, 0.3 valid plus up to 0.3 for closeness,
  1.0 exact,
- Dr. GRPO advantages: reward minus the group mean, with no division by the group
  standard deviation,
- three stability techniques: resampling of zero-variance groups, a replay buffer of
  successful samples, and stratified difficulty sampling.

Configs: `configs/esme-214m-rl*.json`. Run card: `run_cards/esme-214m-rl.md`.

**Control.** A random-reward placebo runs the same recipe and budget, but draws each
reward from the same three-level support, independent of the answer. Any gain in the
placebo comes from the training process, not from the reward signal.

**Evaluation set.** `heldout_large`: 200 tasks (120 medium and 80 hard, 199 of them with
three numbers), selected from the same generator and verified disjoint from every
training and earlier held-out task (`src/esme_posttrain/rl/countdown_heldout.py`).

**Decoding.** 16 samples per task at temperature 1.0, 12 new tokens, seed 0.

**Checkpoints.** Every arm is evaluated at training step 240. Each bundle is rebuilt
from the run's step-240 training checkpoint with `rlvr-checkpoint-bundle`.

**Statistics.** The unit of analysis is the training seed. Each comparison reports a
two-way bootstrap interval that resamples seeds and, independently, one shared set of
problems. It also reports an exact sign-flip permutation p-value over seed effects.
With six seeds the smallest possible p is 0.031. With three seeds it is 0.25. See
`docs/study-reports.md`.

## Results

### 1. The gain comes from the reward signal

Six real-reward seeds against six placebo seeds
(`studies/rlvr-placebo-step240-large.report.md`):

| Measure | Real reward | Placebo | Difference, 95% CI |
| --- | ---: | ---: | --- |
| Valid expressions | 97.9% | 1.3% | +96.6 pp [+95.9, +97.3] |
| Exact solves, per sample | 6.3% | 0.07% | +6.2 pp [+3.2, +9.6] |
| Problems solved at least once | 6.6% | 1.2% | +5.4 pp [+2.2, +9.0] |

Every comparison has p = 0.031, the floor for six seeds. The placebo stays at the level
of the Chat model on every measure, at every checkpoint measured. The reward signal produces the
whole gain.

### 2. The model learns form first, and its solved set contracts

Most of the gain is validity. The model learns the answer format almost completely. It
solves about 13 of 200 problems, and all six seeds solve the same 13: their intersection
is 13 problems and their union is 14. For seed 215, the model is right on 88-100% of
samples for those problems. It almost never solves anything else.

The learning curves show how this happens (mean of seeds 215-217,
`assets/fig-learning-curves.svg`):

| Step | Valid | Problems solved (of 200) | Exact per sample |
| ---: | ---: | ---: | ---: |
| 0 (Chat) | 1.4% | 3.0 | 0.09% |
| 20 | 6.8% | 7.7 | 0.39% |
| 40 | 31.4% | 19.3 | 2.20% |
| 60 | 64.8% | 18.7 | 4.05% |
| 120 | 94.4% | 14.0 | 5.79% |
| 240 | 98.0% | 13.0 | 6.25% |

The number of distinct problems solved peaks at step 40 and then falls by a third.
Per-sample accuracy keeps rising. After step 40, training makes the model more reliable
on a smaller set of problems. A checkpoint chosen by training reward picks the narrowest
model.

### 3. The reward shape controls the tradeoff, and the stability techniques do not

Each ablation removes one part of the recipe, over seeds 215-217
(`studies/rlvr-ablation-*.report.md`, `assets/fig-reward-shape-tradeoff.svg`):

| Arm | Valid | Problems solved (of 200) | Full recipe minus arm, problems solved |
| --- | ---: | ---: | --- |
| Full recipe | 98.0% | 13.0 | |
| Binary exact-only reward | 59.2% | 28.3 | -7.7 pp [-11.7, -4.0] |
| No stability techniques | 97.3% | 13.3 | -0.2 pp [-1.0, 0.0] |
| Std-normalized advantages | 97.9% | 13.0 | 0.0 pp [0.0, 0.0] |

- **Binary reward.** Without credit for valid but wrong answers, the model writes valid
  expressions 39 points less often. It solves more than twice as many distinct problems:
  25 problems are solved by every binary seed, and 18 are solved by no full-recipe seed.
  Its success is spread across problems instead of concentrated on a few.
- **Stability techniques.** Removing resampling, the replay buffer, and stratified
  sampling changes validity by 0.7 pp and the solved set by less than one problem.
- **Advantage normalization.** Dividing by the group standard deviation gives the same
  results as the Dr. GRPO default.

Three seeds cannot give a permutation p below 0.25, so these intervals rest on problem
resampling. The binary result is large and consistent across all three seeds. The two
null results agree with each other and with the learning curves.

The graded reward pays 0.3 or more for any valid expression. Validity is easy to reach,
so the policy gradient pushes the model toward valid answers on every problem and away
from exploring answers that are invalid but sometimes correct. This is a consistent
reading of the results. It is not a tested mechanism.

### 4. RL sharpens what the base model can reach

Unbiased pass@k from 256 samples per task, for the Chat model and the seed-214 RL model
at step 240 (`assets/fig-pass-at-k.svg`):

| k | Esme-214M-Chat | Esme-214M-RL |
| ---: | ---: | ---: |
| 1 | 0.1% | 6.3% |
| 16 | 1.7% | 6.7% |
| 64 | 5.8% | 7.2% |
| 256 | 14.0% | 9.5% |

RL leads up to k = 64 because it solves its few problems on almost every sample. The
curves cross between k = 64 and k = 128. With 256 samples the Chat model solves 28
distinct problems and the RL model solves 19. At this scale GRPO makes the model
reliable on part of what the base model can already reach, and it loses access to the
rest. This matches the pattern Yue et al. (2025) report for larger models.

## Corrections made during this audit

The headline result was re-audited before this report. These errors were found and
fixed. Each fix is in the repository history.

1. **An impossible interval.** The first study used a paired t interval, which gave
   `[+54.6, +114.7]` pp for a rate difference that cannot exceed 100 pp. It was replaced
   with the bootstrap and permutation test above. A later review found that the first
   bootstrap resampled problems separately for each seed. That narrowed the intervals,
   because every seed sees the same problems. The bootstrap now draws one problem set
   per resample.
2. **Noise as checkpoint selection.** Each run exported the bundle with the best
   training reward. Under a random reward that choice is noise, so placebo arms were
   evaluated at steps 8 to 219. All arms are now evaluated at step 240.
3. **A mislabeled model.** The first study's seed-214 real-reward arm came from a
   step-22 bundle of an earlier run, not the step-234 run it named. A re-emit
   reproduces 457 of 480 samples from the step-22 bundle. This was the low "outlier"
   seed. The same mislabeled export made an earlier token-budget report claim that
   validity collapses beyond 12 tokens. That claim is withdrawn
   (`docs/rlvr-countdown-3number-tokenbudget.md`).
4. **A silent CPU fallback.** The bundle loader placed the model on the CPU whatever
   device was requested, so "GPU" evaluations ran on the container CPU. This is the
   likely cause of earlier remote evaluations that timed out after hours.

To prevent repeats, CompletionSet provenance now records the code commit and a weights
hash. A study report refuses a bundle whose weights did not produce its samples.

## Limits

- One task, one model size. The findings describe Countdown at 214M parameters.
- The held-out tasks come from the training generator's distribution. A target shift
  (`heldout_shift`, targets 65-128) is solved 0% of the time by every model.
- Ablations use three seeds. They test one change each. KL strength, learning rate,
  and longer generation budgets are not ablated.
- `heldout_large` samples were drawn on a GPU. Earlier studies sampled on a CPU. Each
  comparison uses one sampling path for all of its arms.
- The 12-token budget truncates most answers. With 48 tokens, validity falls to 75-80%
  and three-number exact solving does not rise
  (`docs/rlvr-countdown-3number-tokenbudget.md`).

## Cost

Nine ablation runs cost $8.84 on A100s. Sampling for the 200-task evaluations, learning
curves, and pass@k cost about $6 on A10Gs, estimated from container hours. All other evaluation ran on a laptop
CPU.

## Reproduce

```bash
uv run esme-posttrain study-report --study studies/rlvr-placebo-step240-large.json
uv run esme-posttrain study-report --study studies/rlvr-ablation-binary.json
uv run --with plotly --with kaleido python scripts/plot_rlvr_research.py \
  --completions runs/gpu-emit/batch1
```

The study reports check every input by hash. The CompletionSets are under gitignored
`runs/`, so rebuilding a report needs those files. `scripts/modal_emit_completions.py`
regenerates them from the training checkpoints.

## References

- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO), 2024.
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) (Dr. GRPO), 2025.
- Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837), 2025.
- Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (unbiased pass@k), 2021.
