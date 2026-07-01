# RLVR Countdown-Lite

Countdown-Lite is the first verifier-reward target for `Esme-214M-RL`. The model starts from `Esme-214M-Chat` and is trained with GRPO on a small arithmetic-expression task.

## Task

- Inputs: 2-3 supplied integers.
- Target: small integer.
- Operators: `+`, `-`, and `*`.
- Answer format: arithmetic expression only.
- Rule: use each supplied number exactly once and evaluate exactly to the target.
- Reward: exact verifier-backed execution check.

Countdown-Lite is the only RLVR training target in this run. GSM8K-lite remains a secondary transfer eval, not a reward source.

## Data And Outputs

- Manifest: `data/manifests/esme-214m-rl.tasks.json`
- Data: `data/rl/countdown_lite/{train,dev,eval}.jsonl`
- Result summary: `docs/rlvr-countdown-lite-grpo.md`

## Metrics

- Bundle: `Esme-214M-Chat`
- Split: eval
- Tasks: 30
- Samples per task: 32

| Report | pass@1 | pass@8 | pass@32 | Valid expressions | Exact solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline report | 3.33% | 3.33% | 6.67% | 2.71% | 0.42% |
| GRPO result | 16.67% | 16.67% | 20.00% | 35.73% | 15.83% |

`Esme-214M-RL` is the current Countdown-Lite GRPO artifact. Secondary transfer evals are separate from the verifier reward.
