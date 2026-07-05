# RLVR Countdown-Lite GRPO — seed216 real reward

- Result: `complete`
- Role: multiseed `correct` arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-5UbugwN3wY6cCWGqMAcvaC`
- Modal call: `fc-01KWSH2T5DWES0M07B7RT5N6HT`
- W&B run: `4gsq6ioa`
- Output stem: `esme-214m-rlvr-real-seed216-grpo`
- Config: `configs/esme-214m-rl-seed216.json`
- Commit at launch: `650f275`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$1.1223`
- Runtime: `1925.1s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `234`, `train/reward_mean = 0.7166057825`

## Local Evidence

- Receipt: `runs/multiseed/seed216-real/cost.json`
- Manifest: `runs/multiseed/seed216-real/manifest.json`
- Bundle: `runs/multiseed/seed216-real/bundle`
- CompletionSet: `runs/multiseed-completions/seed216/correct__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 97.1% |
| pass@1 | 9.8% |
| pass@8 | 10.0% |
| pass@16 | 10.0% |
| Any-exact solved | 3/30 |

