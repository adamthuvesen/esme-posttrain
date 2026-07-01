# RLVR Countdown-Lite GRPO Run Details

This internal note keeps operator-grade provenance for the completed Countdown-Lite GRPO run. The public summary lives in `docs/rlvr-countdown-lite-grpo.md`.

## Checkpoint

- Preserved output: `esme-posttrain-esme-rlvr-countdown:/esme-214m-rlvr-countdown-grpo-caff0a1`
- Best checkpoint: `best-checkpoint.pt` (`816.3 MiB`)
- Full checkpoint: `checkpoint.pt` (`2.4 GiB`)
- Best checkpoint selector: `train/reward_mean = 0.7718750238418579` at step `22`
- Local evidence commit: `810856e`

## Bounds

- Sample budget: `360`
- Token budget: `512000`
- Estimated train tokens: `303360`
- Hardware: A100
- Spend: `$1.8300` against a `$25.00` cap
- Runtime hard stop: `$8.00`

Detailed machine-readable evidence is generated under gitignored `artifacts/` when the run is reproduced.
