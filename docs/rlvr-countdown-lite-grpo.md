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
- Eval profile: `full_acceptance_30x32` (`eval_max_new_tokens 12`)
- Reward source: Countdown-Lite verifier only
- Secondary transfer eval: GSM8K-lite is outside this result

## Acceptance Metrics

30 tasks x 32 samples, seed 214, evaluated on the best and final checkpoints
(both score identically).

| Metric | `Esme-214M-Chat` | `Esme-214M-RL` |
| --- | ---: | ---: |
| pass@1 | 3.33% | 16.67% |
| pass@8 | 6.67% | 16.67% |
| pass@32 | 13.33% | 16.67% |
| valid-expression rate | 5.83% | 99.38% |
| exact-solve rate | 0.73% | 16.35% |

GRPO nearly always produces a well-formed expression after training
(valid-expression rate 5.83% -> 99.38%) and lifts exact solves on the easy
band; medium and hard tasks stay unsolved at this model size.

## Inputs

- Dataset: `data/manifests/esme-214m-rl.tasks.json`
- Task overview: `docs/rlvr-countdown-lite.md`
- Held-out transfer: `docs/rlvr-countdown-heldout-transfer.md`
- Operator provenance and training-shape verdict:
  `docs/internal/rlvr-countdown-lite-grpo-run.md`
