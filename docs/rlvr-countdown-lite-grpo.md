# RLVR Countdown-Lite GRPO

`Esme-214M-RL` is the Countdown-Lite GRPO variant of `Esme-214M-Chat`.

## Result

- Artifact: `Esme-214M-RL`
- Reference artifact: `Esme-214M-Chat`
- Task variant: Countdown-Lite GRPO
- Objective: group-normalized REINFORCE-with-baseline plus a KL penalty
  against the chat reference (one gradient step per rollout batch, so
  no PPO-style ratio clipping)
- Result: completed
- Eval profile: `full_acceptance_30x32`
- Reward source: Countdown-Lite verifier only
- Secondary transfer eval: GSM8K-lite is outside this result

## Acceptance Metrics

| Metric | `Esme-214M-Chat` | `Esme-214M-RL` |
| --- | ---: | ---: |
| pass@1 | 3.33% | 16.67% |
| pass@8 | 3.33% | 16.67% |
| pass@32 | 6.67% | 20.00% |
| valid-expression rate | 2.71% | 35.73% |
| exact-solve rate | 0.42% | 15.83% |

## Inputs

- Dataset: `data/manifests/esme-214m-rl.tasks.json`
- Task overview: `docs/rlvr-countdown-lite.md`
