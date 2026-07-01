# Run Card: Esme-214M-Chat (DPO polish)

## Artifact

- Produces: `Esme-214M-Chat`
- Starts from: `Esme-214M-Instruct` (the accepted multi-turn SFT foundation)
- Ladder: `Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL`
- Config: `configs/esme-214m-chat-dpo.json`
- Launcher: `scripts/modal_chat_dpo.py`

## Method

One offline, vanilla DPO pass (`loss_type=sigmoid`, the same objective TRL's
`DPOTrainer` ships) implemented on this repo's native `DenseBackbone` stack — TRL
is not used because it requires a HuggingFace `PreTrainedModel`/`AutoTokenizer`,
which the from-scratch Esme model is not. The policy is warm-started from the SFT
foundation checkpoint; the frozen reference is the same checkpoint.

- Preference data: `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs` /
  `test_prefs`), revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`, license MIT.
  Chat-templated to this repo's multi-turn template + `<eos>`, prompt tokens masked
  so only response tokens score. Light filtering: char caps + drop identical pairs.
- Config (SmolLM2-360M anchor): `beta 0.5`, `lr 1.0e-6`, cosine + `warmup_ratio 0.1`,
  2 epochs, `max_length 1024` / `max_prompt_length 512`, effective batch 128
  (micro 4 x grad-accum 32), bf16, adamw_torch.
- Length-normalization is a config toggle (`dpo.length_normalized`), the dominant
  mitigation for verbosity reward-hacking; default off, flip on if length hacking shows.
- Chosen/rejected logps are logged every eval to catch likelihood displacement.
- No auxiliary SFT/NLL term, no SimPO, no on-policy generation, no RL (out of scope).

## Warm-start reference

The accepted multi-turn SFT foundation (W&B run `hhmk7uk2`, best checkpoint step
6300, matched held-out 2.117 -> 1.356) lives on the SFT Modal Volume
`esme-posttrain-esme-sft-multiturn` at output stem `esme-214m-sft-multiturn-full`.
The DPO Modal job mounts that Volume read-only and loads `best-checkpoint.pt` +
`tokenizer.json`; DPO never writes to it. DPO writes to its own Volume
`esme-posttrain-esme-chat-dpo`.

## Decoding pre-check

Before DPO, the Modal job evaluates the SFT checkpoint under greedy vs
nucleus (p=0.95) + repetition penalty (1.3) on fixed multi-turn prompts and records
n-gram repetition rate + response-length distribution as the pre-DPO baseline. Some
"rambling" may be a decoding artifact fixable for free. (The CPU fixture runs this
harness on the tiny fixture model and marks it `is_real_checkpoint=false`.)

## Acceptance (honest small-model bar)

The DPO checkpoint must:
- beat the SFT reference on held-out preference accuracy (chosen margin > 0),
- not collapse chosen-logp below the reference (likelihood-displacement guard),
- reduce n-gram repetition and/or stabilize response length vs the pre-DPO baseline,
  without degrading coherence.

This is chat polish measured by repetition/length/coherence proxies — not a 7B-style
AlpacaEval win, which is not a credible target at 214M. The K>=5 LLM-judge chat score
is reported with spread but is never the selector.

## Spend & approval gates

- Smoke: capped at **$2** (`runtime.smoke_max_cost_usd`), no env bypass.
- Full run: capped at **$15** runaway; refuses without `--approved` AND bounded
  beta-sweep learning-gate evidence (`learning_gate.evidence.bounded_beta_sweep`).
- Beta sweep `{0.1, 0.3, 0.5}`: bounded, $8 cap; selects the beta that beats the SFT
  reference on held-out preference accuracy without chosen-logp collapse, and emits
  the learning-gate evidence the full run requires.

## Commands

```bash
# Dry-run (never starts Modal): proves will_start_modal_job:false + cost + blockers
uv run esme-posttrain chat-dpo-dry-run --config configs/esme-214m-chat-dpo.json --json

# No-spend local CPU fixture (margin up, chosen-logp tracked, checkpoint round-trip)
uv run esme-posttrain chat-dpo-cpu-fixture --config configs/esme-214m-chat-dpo.json --json

# Capped Modal smoke (requires Modal + W&B creds + chat approval)
DPO_MODAL_GPU='A100' DPO_TIMEOUT_HOURS=10 uv run --with modal==1.5.1 \
  modal run --detach scripts/modal_chat_dpo.py \
  --config configs/esme-214m-chat-dpo.json --modal-smoke --approved --json

# Bounded beta sweep {0.1,0.3,0.5} (learning gate; $8 cap)
DPO_MODAL_GPU='A100' DPO_SWEEP_TIMEOUT_HOURS=3 uv run --with modal==1.5.1 \
  modal run scripts/modal_chat_dpo.py \
  --config configs/esme-214m-chat-dpo.json --beta-sweep --approved --json

# Full DPO launch — NOT APPROVED in this mission; refuses without --approved AND
# bounded beta-sweep evidence written back into learning_gate.evidence.
DPO_MODAL_GPU='A100' DPO_TIMEOUT_HOURS=10 uv run --with modal==1.5.1 \
  modal run --detach scripts/modal_chat_dpo.py \
  --config configs/esme-214m-chat-dpo.json --full-run --approved --json
```

## Projected full-run cost

~100M scored tokens at the conservative A100 projection (4200 tok/s, $2.0988/h) ≈
$13.9, under the $15 cap. The projection is unmeasured for the DPO triple-forward
(policy chosen + policy rejected + no-grad reference); it must be re-measured by a
bounded probe or read off the beta sweep's measured rate before the full launch.
