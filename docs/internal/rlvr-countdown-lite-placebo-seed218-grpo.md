# RLVR Countdown-Lite GRPO: seed218 random reward

- Result: `complete`
- Role: multiseed `random` placebo arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-FDDOjSl9PDyx70VE02w6sr`
- Modal call: `fc-01KWSPJN7P2HG1ZZVEHPA019CC`
- W&B run: `vdqwpmqp`
- Output stem: `esme-214m-rlvr-placebo-seed218-grpo`
- Config: `configs/esme-214m-rl-placebo-seed218.json`
- Commit at launch: `f066485`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$0.7908`
- Runtime: `1356.4s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `8`, `train/reward_mean = 0.5296875238`

## Local Evidence

- Receipt: `runs/multiseed/seed218-placebo/cost.json`
- Manifest: `runs/multiseed/seed218-placebo/manifest.json`
- Bundle: `runs/multiseed/seed218-placebo/bundle`
- CompletionSet: `runs/multiseed-completions/seed218/random__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 0.6% |
| pass@1 | 0.2% |
| pass@8 | 1.7% |
| pass@16 | 3.3% |
| Any-exact solved | 1/30 |
