# RLVR Countdown-Lite GRPO — seed218 real reward

- Result: `complete`
- Role: multiseed `correct` arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-c4tj6fRsyQeU3uCUC0m2ZR`
- Modal call: `fc-01KWSPJ6PZBG95C4KTMPGD502V`
- W&B run: `xfp6dxry`
- Output stem: `esme-214m-rlvr-real-seed218-grpo`
- Config: `configs/esme-214m-rl-seed218.json`
- Commit at launch: `f066485`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$1.0612`
- Runtime: `1820.3s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `159`, `train/reward_mean = 0.7109227777`

## Local Evidence

- Receipt: `runs/multiseed/seed218-real/cost.json`
- Manifest: `runs/multiseed/seed218-real/manifest.json`
- Bundle: `runs/multiseed/seed218-real/bundle`
- CompletionSet: `runs/multiseed-completions/seed218/correct__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 96.7% |
| pass@1 | 9.6% |
| pass@8 | 10.0% |
| pass@16 | 10.0% |
| Any-exact solved | 3/30 |

