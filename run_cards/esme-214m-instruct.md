# Run Card: Esme-214M-Instruct SFT

## Artifact

- Produces: `Esme-214M-Instruct`
- Starts from: `Esme-214M-Base`
- Bundle format: `llm_pretrain_dense_v1`
- Config: `configs/esme-214m-instruct.json`
- Config schema: `schemas/instruct-sft-config.schema.json`

This single-turn SFT recipe is retained for reproducibility. The active public
post-training path now uses the multi-turn SFT foundation in
`run_cards/esme-214m-sft-multiturn.md`.

## Data

- 80% `HuggingFaceTB/smol-smoltalk`
  (`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0), filtered to concise
  single-turn rows.
- 20% `allenai/tulu-3-sft-personas-instruction-following`
  (`fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`, ODC-BY), with persona
  constraints folded into the user prompt.
- Eval only: `HuggingFaceH4/no_robots`
  (`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, CC-BY-NC-4.0). This
  non-commercial dataset is a guardrail metric, not training data.
- Sample cap: 50,000 train examples.
- Target train tokens: 30,000,000.
- Hard token cap: 50,000,000 selected train tokens.
- Max sequence length: 1,024 tokens.

## Method

The trainer renders user/assistant rows, masks prompt tokens with `-100`, and
optimizes only assistant response tokens. It writes selected-row manifests,
namespaced train/eval metrics, restartable checkpoints, deterministic samples,
and a dense bundle manifest for downstream stages.

The selected recipe uses learning rate `3e-5`, microbatch size `2`, gradient
accumulation `8`, effective batch size `16`, cosine decay, and `700` warmup
steps.

## Acceptance

- Base bundle manifest hashes validate before weights load.
- Prompt tokens are masked; loss applies only to assistant response tokens.
- Matched held-out SmolTalk/Tulu response loss improves over Base.
- `no_robots` remains an out-of-distribution guardrail metric only.
- Checkpoint reload reproduces logits.
- Required outputs include config, data report, selected-row manifest, eval
  manifests, metrics, checkpoint files, samples, tokenizer, manifest, eval
  report, cost report, and environment record.

## Safe Local Commands

```bash
uv run esme-posttrain instruct-sft-dry-run --config configs/esme-214m-instruct.json --json
uv run esme-posttrain instruct-sft-cpu-fixture --config configs/esme-214m-instruct.json --json
```
