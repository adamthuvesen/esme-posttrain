# RLVR Countdown-Lite GRPO — seed219 random reward

- Result: `complete`
- Role: multiseed `random` placebo arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-Ril8Uxn79a7W69IYxD6Zh9`
- Modal call: `fc-01KWSPKHW1P09H2XTGVPFB9XJH`
- W&B run: `7h8t83k0`
- Output stem: `esme-214m-rlvr-placebo-seed219-grpo`
- Config: `configs/esme-214m-rl-placebo-seed219.json`
- Commit at launch: `f066485`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$0.7473`
- Runtime: `1281.9s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `162`, `train/reward_mean = 0.5375000238`

## Local Evidence

- Receipt: `runs/multiseed/seed219-placebo/cost.json`
- Manifest: `runs/multiseed/seed219-placebo/manifest.json`
- Bundle: `runs/multiseed/seed219-placebo/bundle`
- CompletionSet: `runs/multiseed-completions/seed219/random__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 1.2% |
| pass@1 | 0.2% |
| pass@8 | 1.7% |
| pass@16 | 3.3% |
| Any-exact solved | 1/30 |
