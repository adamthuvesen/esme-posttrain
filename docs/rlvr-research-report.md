# What GRPO Teaches a 214M Model on Countdown

GRPO with the accepted graded reward teaches `Esme-214M-Chat` one move: add all the
numbers. At step 240, 98% of its answers on held-out tasks are the sum of the supplied
numbers, whatever the target. That template is always a valid expression and often
close to the target, so the graded reward pays well for it. The model solves exactly
the held-out tasks whose target is the sum, and almost nothing else.

A random-reward placebo, a larger held-out set, learning curves, and reward ablations
found this. Earlier docs on the same runs described the result as a "transferable
answer format plus real 2-operand skill". They described the template without noticing it.

All numbers come from committed study reports and `studies/rlvr-research-summary.json`.
Figures are in `assets/`.

## Question

Countdown gives a set of numbers and a target. The model must write an arithmetic
expression that uses each number exactly once and reaches the target. A verifier checks
the answer, so the reward needs no human or model judge.

This report asks three questions:

1. Does the gain come from the reward signal, or from the training process itself?
2. What does the model learn: arithmetic, or something that scores well?
3. Which parts of the training recipe produce it?

## Setup

**Model.** `Esme-214M-Chat`: a 214M-parameter dense transformer, pretrained from scratch
in [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain), then tuned with SFT
and DPO in this repo.

**Training.** GRPO for 240 steps on 300 Countdown-Lite training tasks: 8 prompts x 16
samples per step, learning rate `5e-7`, KL penalty `0.001` toward the Chat model,
temperature 1.0, at most 12 new tokens. The accepted recipe uses:

- a graded reward: 0 invalid, 0.05 well-formed, 0.3 for a valid expression plus up to
  0.3 for closeness to the target, and 1.0 for an exact solve,
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
Exactly 13 of the 200 tasks have a target equal to the sum of their numbers.

**Decoding.** 16 samples per task at temperature 1.0, 12 new tokens, seed 0.

**Checkpoints.** Every arm is evaluated at training step 240. Each bundle is rebuilt
from the run's step-240 training checkpoint with `rlvr-checkpoint-bundle`.

**Statistics.** The unit of analysis is the training seed. Each comparison reports a
two-way bootstrap interval that resamples seeds and, independently, one shared set of
problems. It also reports an exact sign-flip permutation p-value over seed effects.
With six seeds the smallest possible p is 0.031. With three seeds it is 0.25. See
`docs/study-reports.md`.

## Results

### 1. The reward signal causes the change

Six real-reward seeds against six placebo seeds
(`studies/rlvr-placebo-step240-large.report.md`):

| Measure | Real reward | Placebo | Difference, 95% CI |
| --- | ---: | ---: | --- |
| Valid expressions | 97.9% | 1.3% | +96.6 pp [+95.9, +97.3] |
| Exact solves, per sample | 6.3% | 0.07% | +6.2 pp [+3.2, +9.6] |
| Problems solved at least once | 6.6% | 1.2% | +5.4 pp [+2.2, +9.0] |

Every comparison has p = 0.031, the floor for six seeds. The placebo stays at the level
of the Chat model on every measure, at every checkpoint measured. The reward signal
produces the whole change. The next section shows what that change is.

### 2. The model learns to add all the numbers

At step 240, 97.7% of the full-recipe samples are valid expressions that use only `+`.
A valid expression with only `+` uses every number once, so it is the sum of the
numbers. The validity gain in section 1 is almost entirely this template.

The template is exact only when the target is the sum. All six seeds solve all 13 sum
tasks in `heldout_large`. Between them, they solve one other task, in one seed. The
reward explains why: on the 300 training tasks, a policy that always writes the sum
earns a mean reward of 0.54. The logged training reward plateaued near that value. In
the training set, 17% of targets are the sum (36% of the 2-number tasks).

The learning curves show the policy converge on the template (mean of seeds 215-217,
`assets/fig-learning-curves.svg`):

| Step | Samples that add all numbers | Sum tasks solved | Other tasks solved | Exact per sample |
| ---: | ---: | ---: | ---: | ---: |
| 0 (Chat) | 0.8% | 2 | 1 | 0.09% |
| 20 | 5.1% | 5.7 | 2.0 | 0.39% |
| 40 | 28.5% | 12.0 | 7.3 | 2.20% |
| 60 | 63.1% | 12.0 | 6.7 | 4.05% |
| 120 | 94.0% | 13.0 | 1.0 | 5.79% |
| 240 | 97.8% | 13.0 | 0.0 | 6.25% |

At step 40 the model solves 7 tasks that are not sums. As the template takes over the
samples, those solves disappear. Per-sample exact rate keeps rising only because the
template answers the 13 sum tasks on nearly every sample.

### 3. The reward shape controls the collapse, and the stability techniques do not

Each ablation removes one part of the recipe, over seeds 215-217
(`studies/rlvr-ablation-*.report.md`, `assets/fig-reward-shape-tradeoff.svg`):

| Arm | Valid | Samples that add all numbers | Sum tasks solved | Other tasks solved |
| --- | ---: | ---: | ---: | ---: |
| Full recipe | 98.0% | 97.8% | 13 | 0 |
| Binary exact-only reward | 59.2% | 50.0% | 12 | 15-18 |
| No stability techniques | 97.3% | 97.0% | 13 | 0-1 |
| Std-normalized advantages | 97.9% | 97.7% | 13 | 0 |

- **Binary reward.** Without credit for valid but wrong answers, the sum template stops
  paying unless it is exact. The model still adds the numbers on half of its samples,
  but it also solves 15-18 tasks that are not sums. Across its three seeds it solves 31
  distinct tasks, 18 of which no full-recipe seed solves. The graded reward trains this
  arithmetic away.
- **Stability techniques.** Removing resampling, the replay buffer, and stratified
  sampling lowers validity by 0.7 pp (CI [+0.3, +1.2]) and does not change the solved
  set. The run still collapses to the template.
- **Advantage normalization.** Dividing by the group standard deviation gives the same
  results as the Dr. GRPO default.

Three seeds cannot give a permutation p below 0.25, so these intervals rest on problem
resampling. The binary result is large and holds on all three seeds.

### 4. RL trades breadth for reliability

Unbiased pass@k from 256 samples per task, for the Chat model and the seed-214 RL model
at step 240 (`assets/fig-pass-at-k.svg`):

| k | Esme-214M-Chat | Esme-214M-RL |
| ---: | ---: | ---: |
| 1 | 0.1% | 6.3% |
| 16 | 1.7% | 6.7% |
| 64 | 5.8% | 7.2% |
| 256 | 14.0% | 9.5% |

RL leads up to k = 64 because it answers the 13 sum tasks on almost every sample. The
curves cross between k = 64 and k = 128. With 256 samples:

- the Chat model solves 28 tasks, 9 sums and 19 others,
- the RL model solves 19 tasks, all 13 sums and 6 others,
- 11 tasks are solved by both, 17 only by Chat, and 8 only by RL.

RL makes the model reliable on the sum tasks, including four that the Chat model never
solves in 256 samples. Its four other new solves are single hits in 256 samples. It
loses most of the base model's reach on everything else. This is close to the pattern
Yue et al. (2025) describe, where RL sharpens rather than extends what the base model
can do. The exception is the four sum tasks, which the template reaches.

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
5. **A template read as skill.** Earlier docs credited RL with "real 2-operand skill"
   that transfers to held-out tasks. The held-out tasks it solved were the sum tasks.
   This report and the held-out docs now say so.

To prevent repeats, CompletionSet provenance now records the code commit and a weights
hash. A study report refuses a bundle whose weights did not produce its samples.

## Limits

- One task, one model size. The findings describe Countdown at 214M parameters.
- The held-out tasks come from the training generator's distribution. A target shift
  (`heldout_shift`, targets 65-128, above any sum of three numbers from 1-9) is solved
  0% of the time by every model.
- Ablations use three seeds. They test one change each. KL strength, learning rate,
  the closeness weight, and longer generation budgets are not ablated.
- The analysis identifies the sum template from output form. It does not inspect what
  the model represents internally.
- `heldout_large` samples were drawn on a GPU. Earlier studies sampled on a CPU. Each
  comparison uses one sampling path for all of its arms.

## Cost

Nine ablation runs cost $8.84 on A100s. Sampling for the 200-task evaluations, learning
curves, and pass@k cost about $6 on A10Gs, estimated from container hours. All other
evaluation ran on a laptop CPU.

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
