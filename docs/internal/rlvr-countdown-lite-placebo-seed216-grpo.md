# RLVR Countdown-Lite GRPO — seed216 random reward

- Result: `complete`
- Role: multiseed `random` placebo arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-L8Lc2NGzasHMdBf2Jj95R3`
- Modal call: `fc-01KWSH4V0X1ZGCKZ7Y86BZK57D`
- W&B run: `q3e470sb`
- Output stem: `esme-214m-rlvr-placebo-seed216-grpo`
- Config: `configs/esme-214m-rl-placebo-seed216.json`
- Commit at launch: `650f275`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$0.7447`
- Runtime: `1277.4s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `219`, `train/reward_mean = 0.5218750238`

## Local Evidence

- Receipt: `runs/multiseed/seed216-placebo/cost.json`
- Manifest: `runs/multiseed/seed216-placebo/manifest.json`
- Bundle: `runs/multiseed/seed216-placebo/bundle`
- CompletionSet: `runs/multiseed-completions/seed216/random__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 0.8% |
| pass@1 | 0.2% |
| pass@8 | 1.7% |
| pass@16 | 3.3% |
| Any-exact solved | 1/30 |

