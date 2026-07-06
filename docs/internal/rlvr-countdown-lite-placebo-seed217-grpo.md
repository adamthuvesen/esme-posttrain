# RLVR Countdown-Lite GRPO: seed217 random reward

- Result: `complete`
- Role: multiseed `random` placebo arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-e56WpbEYj4lgXfiSOCUYTR`
- Modal call: `fc-01KWSKDKPMMPFRBXSK5MMW7CWT`
- W&B run: `k9ktl1a0`
- Output stem: `esme-214m-rlvr-placebo-seed217-grpo`
- Config: `configs/esme-214m-rl-placebo-seed217.json`
- Commit at launch: `276eebf`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$0.7547`
- Runtime: `1294.6s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `112`, `train/reward_mean = 0.5398437381`

## Local Evidence

- Receipt: `runs/multiseed/seed217-placebo/cost.json`
- Manifest: `runs/multiseed/seed217-placebo/manifest.json`
- Bundle: `runs/multiseed/seed217-placebo/bundle`
- CompletionSet: `runs/multiseed-completions/seed217/random__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 0.6% |
| pass@1 | 0.0% |
| pass@8 | 0.0% |
| pass@16 | 0.0% |
| Any-exact solved | 0/30 |
