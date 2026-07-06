# RLVR Countdown-Lite GRPO: seed217 real reward

- Result: `complete`
- Role: multiseed `correct` arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-cROQ4hUW1qG9IEb201q8Gr`
- Modal call: `fc-01KWSKDKJXV6MH2GMTTM228MV4`
- W&B run: `wma5ic39`
- Output stem: `esme-214m-rlvr-real-seed217-grpo`
- Config: `configs/esme-214m-rl-seed217.json`
- Commit at launch: `276eebf`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$0.9952`
- Runtime: `1707.0s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `234`, `train/reward_mean = 0.7249559760`

## Local Evidence

- Receipt: `runs/multiseed/seed217-real/cost.json`
- Manifest: `runs/multiseed/seed217-real/manifest.json`
- Bundle: `runs/multiseed/seed217-real/bundle`
- CompletionSet: `runs/multiseed-completions/seed217/correct__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 97.1% |
| pass@1 | 9.6% |
| pass@8 | 10.0% |
| pass@16 | 10.0% |
| Any-exact solved | 3/30 |
