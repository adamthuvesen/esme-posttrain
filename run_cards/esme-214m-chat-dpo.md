# Run Card: Esme-214M-Chat DPO

This is the preference-tuning stage of the standard post-training chain:

```text
Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL
```

## Artifact

- Produces: `Esme-214M-Chat`
- Starts from: `Esme-214M-Instruct`
- Config: `configs/esme-214m-chat-dpo.json`
- Launcher: private operator module.

## Method

The stage runs one offline DPO pass (`loss_type=sigmoid`) on the repo's native
`DenseBackbone` stack. TRL is not used because Esme is a from-scratch model, not
a Hugging Face `PreTrainedModel`.

- Preference data: `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs` /
  `test_prefs`), revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`, MIT.
- Prompts and responses use the repo's multi-turn chat template plus `<eos>`.
- Prompt tokens are masked so only response tokens score.
- Light filtering drops overlong rows and identical chosen/rejected pairs.
- Config anchor: `beta 0.5`, `lr 1.0e-6`, cosine schedule,
  `warmup_ratio 0.1`, 2 epochs, `max_length 1024`,
  `max_prompt_length 512`, effective batch `128`, bf16.
- Chosen/rejected log probabilities are logged on eval to catch likelihood
  displacement.

Length normalization is available through `dpo.length_normalized` and is the
first fix to try if response length starts driving the preference margin.

## Acceptance

The DPO checkpoint should:

- improve held-out preference accuracy over the SFT reference,
- keep chosen-logp from collapsing relative to the reference,
- reduce repetition and/or stabilize response length without degrading
  coherence.

This is chat polish for a 214M model. Repetition, length, likelihood, and
coherence proxies matter more here than benchmark claims meant for much larger
models.

## Safe Local Commands

```bash
uv run esme-posttrain chat-dpo-dry-run --config configs/esme-214m-chat-dpo.json --json
uv run esme-posttrain chat-dpo-cpu-fixture --config configs/esme-214m-chat-dpo.json --json
```
