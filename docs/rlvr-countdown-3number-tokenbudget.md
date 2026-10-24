# RLVR Countdown 3-Number Token-Budget Diagnostic

Is `Esme-214M-RL`'s near-zero 3-operand exact-solve on the held-out transfer eval a
12-token truncation artifact, or a real capability wall? This diagnostic reruns the
held-out cells at larger rollout budgets (24 and 48 tokens) and compares them with the
12-token numbers in `docs/rlvr-countdown-heldout-transfer.md`. Eval only, no training.
Same acceptance protocol: seed 214, 32 samples per task, greedy sample 0 plus 31
stochastic samples at temperature 0.8.

## Correction (2026-10-24)

The first version of this report did not evaluate the accepted model. Its commands
named the `exports/esme-214m-rl-caff0a1` bundle, which holds the step-22 checkpoint of
an earlier run, and its 24- and 48-token reports were not kept. It reported that
3-number validity collapses to 13-19% once the budget grows. That result does not
reproduce on the accepted model. Every number below comes from a re-run on
hash-verified bundles:

- RL: the step-240 bundle rebuilt from run `v2-ccb6287-1` (weights sha256
  `b35cdfb5…`), tensor-identical to the trainer's own step-240 export.
- Chat: the warm-start bundle (weights sha256 `3e7c1a45…`).

The verdict on 3-operand solving still holds. The "validity collapse" finding is
withdrawn.

## Verdict

**Real 3-operand wall, not truncation.** With 2x or 4x the rollout budget, 3-number
exact-solve does not rise: 4.00% → 3.62% → 3.25% on the fresh set at 12, 24, and 48
tokens, and 0.00% at every budget on the target-shifted set. Validity falls gradually
as the budget grows (99% → 89% → 75% on fresh 3-number tasks): with room to keep
writing, the model sometimes continues past a valid expression and corrupts it. The
2-number control keeps its skill at every budget (pass@32 40%).

## 3-number cells

| Model | Split | Budget | Tasks | pass@1 | pass@8 | pass@32 | Valid expr | **Exact solve** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Esme-214M-RL | fresh | 12 | 25 | 4.00% | 4.00% | 4.00% | 99.12% | **4.00%** |
| Esme-214M-RL | fresh | 24 | 25 | 4.00% | 4.00% | 4.00% | 88.50% | **3.62%** |
| Esme-214M-RL | fresh | 48 | 25 | 0.00% | 4.00% | 4.00% | 74.75% | **3.25%** |
| Esme-214M-RL | shift | 12 | 30 | 0.00% | 0.00% | 0.00% | 99.38% | **0.00%** |
| Esme-214M-RL | shift | 24 | 30 | 0.00% | 0.00% | 0.00% | 87.40% | **0.00%** |
| Esme-214M-RL | shift | 48 | 30 | 0.00% | 0.00% | 0.00% | 79.79% | **0.00%** |
| Esme-214M-Chat | fresh | 12 | 25 | 0.00% | 0.00% | 0.00% | 2.25% | **0.00%** |
| Esme-214M-Chat | fresh | 24 | 25 | 0.00% | 0.00% | 0.00% | 3.62% | **0.00%** |
| Esme-214M-Chat | fresh | 48 | 25 | 0.00% | 0.00% | 0.00% | 4.12% | **0.00%** |
| Esme-214M-Chat | shift | 12 | 30 | 0.00% | 0.00% | 0.00% | 1.46% | **0.00%** |
| Esme-214M-Chat | shift | 24 | 30 | 0.00% | 0.00% | 0.00% | 2.40% | **0.00%** |
| Esme-214M-Chat | shift | 48 | 30 | 0.00% | 0.00% | 0.00% | 4.38% | **0.00%** |

## 2-number control (fresh split, 5 tasks)

| Model | Budget | pass@32 | Valid expr | Exact solve |
| --- | ---: | ---: | ---: | ---: |
| Esme-214M-RL | 12 | 40.00% | 98.75% | 38.75% |
| Esme-214M-RL | 24 | 40.00% | 97.50% | 38.12% |
| Esme-214M-RL | 48 | 40.00% | 90.00% | 34.38% |
| Esme-214M-Chat | 12 | 20.00% | 10.00% | 3.12% |
| Esme-214M-Chat | 24 | 40.00% | 10.62% | 4.38% |
| Esme-214M-Chat | 48 | 40.00% | 15.00% | 5.62% |

## Early stop and EOS

Generated-token counts come from re-tokenizing each stored completion (text before
`<eos>`) with the bundle tokenizer. A sample whose length equals the budget ran to the
cap; a shorter one stopped on `<eos>`.

| Model | Split | Budget | Median gen tokens | Ran to cap |
| --- | --- | ---: | ---: | ---: |
| Esme-214M-RL | fresh (3-num) | 12 | 12 | 98.88% |
| Esme-214M-RL | fresh (3-num) | 24 | 24 | 90.88% |
| Esme-214M-RL | fresh (3-num) | 48 | 48 | 77.25% |
| Esme-214M-RL | shift (3-num) | 12 | 12 | 98.23% |
| Esme-214M-RL | shift (3-num) | 24 | 24 | 89.17% |
| Esme-214M-RL | shift (3-num) | 48 | 48 | 75.21% |

At 12 tokens the RL model runs to the cap on about 99% of 3-number samples, so the
12-token eval does truncate it. With more budget it stops on `<eos>` more often, yet
3-number exact-solve does not rise. The budget is not what limits 3-operand solving.

## What this settles

- **Settles:** for `Esme-214M-RL`, 3-operand Countdown exact-solve is a capability
  wall on both held-out sets, not an artifact of the 12-token budget. GRPO taught a
  transferable answer format plus real 2-operand skill.
- **Withdrawn:** the claim that validity collapses to 13-19% beyond 12 tokens. On the
  accepted model it declines to 75-80% at 48 tokens.

## Execution and spend

Local CPU, $0. Each cell:

```bash
uv run esme-posttrain rlvr-countdown-lite-baseline \
  --manifest data/manifests/esme-214m-rl-heldout.tasks.json \
  --bundle <verified-bundle-dir> --output-dir <out> \
  --split <heldout_fresh|heldout_shift> \
  --samples-per-task 32 --max-new-tokens <12|24|48> --seed 214 --json
```

Reports are under gitignored `artifacts/verified-heldout/`.
