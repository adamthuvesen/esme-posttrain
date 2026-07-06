# RLVR Countdown-Lite GRPO: seed219 real reward

- Result: `complete`
- Role: multiseed `correct` arm for `grpo-decomp` Esme held-out sampled decomposition.
- Modal app: `ap-1rRYGD9JBT9bEnngiV2lsF`
- Modal call: `fc-01KWSPK3PHA0KSQVGS55521JVN`
- W&B run: `mjghkkie`
- Output stem: `esme-214m-rlvr-real-seed219-grpo`
- Config: `configs/esme-214m-rl-seed219.json`
- Commit at launch: `f066485`
- Hardware: `A100`
- Runtime hard stop: `$4.50`
- Spend: `$0.9910`
- Runtime: `1699.9s`
- Training status: `complete`, step `240`
- Selected checkpoint: step `234`, `train/reward_mean = 0.7059140801`

## Local Evidence

- Receipt: `runs/multiseed/seed219-real/cost.json`
- Manifest: `runs/multiseed/seed219-real/manifest.json`
- Bundle: `runs/multiseed/seed219-real/bundle`
- CompletionSet: `runs/multiseed-completions/seed219/correct__esme-countdown`

## Held-Out Decomposition Slice

On `heldout_fresh` with `n=16`, `temperature=1.0`, `max_new_tokens=12`:

| Metric | Value |
| --- | ---: |
| Valid-expression rate | 96.9% |
| pass@1 | 9.8% |
| pass@8 | 10.0% |
| pass@16 | 10.0% |
| Any-exact solved | 3/30 |
