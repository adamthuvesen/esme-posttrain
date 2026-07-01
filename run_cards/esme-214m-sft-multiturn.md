# Run Card: Esme-214M Multi-Turn SFT Foundation

This is the SFT stage of the standard post-training chain:

```text
Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL
```

It trains the instruction foundation from Base on multi-turn conversational and
instruction data. This is the foundation used by the DPO chat model.

## Artifact

- Produces: `Esme-214M-Instruct`
- Starts from: `Esme-214M-Base`
- Bundle format: `llm_pretrain_dense_v1`
- Config: `configs/esme-214m-sft-multiturn.json`

## Data

- 85% `HuggingFaceTB/smol-smoltalk`
  (`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0), including
  multi-turn rows. Function-calling and hardest reasoning subsets are dropped for
  this 214M-capacity recipe.
- 15% `allenai/tulu-3-sft-personas-instruction-following`
  (`fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`, ODC-BY), single-turn
  instruction-following with constraints folded into the user turn.
- Eval only: `HuggingFaceH4/no_robots`
  (`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, CC-BY-NC-4.0), used as a
  non-commercial out-of-distribution guardrail.
- Sample cap: 50,000 train examples.
- Target / hard token cap: 60M / 80M selected train tokens.
- Max sequence length: 1,024 tokens.

## Chat Template And Masking

Conversations render with the repo's role markers:

```text
system
{system}
user
{user}
assistant
{assistant}<eos>
```

Every assistant turn's content and trailing `<eos>` are supervised. System/user
turns and role markers are masked with `-100`.

## Recipe

- AdamW, cosine decay, about 4% warmup, weight decay `0.1`, grad clip `1.0`,
  seed `214`.
- Effective batch `16` = microbatch `2` x gradient accumulation `8`.
- `bf16` on CUDA.
- `sequence_packing=false`; padding metrics are emitted.
- Learning rate is selected by matched held-out response loss before a full
  recipe is accepted.

## Acceptance

- Matched held-out response loss improves over Base across multi-turn chat and
  single-turn instruction rows.
- `no_robots` is reported as a guardrail, never as the selector.
- Multi-turn samples show coherent turn-taking.
- Instruction following remains present after multi-turn training.
- LLM-judge chat scores, when run, are reported with repeated-judge spread and
  are never the selector.

## Safe Local Commands

```bash
uv run esme-posttrain sft-multiturn-dry-run --config configs/esme-214m-sft-multiturn.json --json
uv run esme-posttrain sft-multiturn-cpu-fixture --config configs/esme-214m-sft-multiturn.json --json
```
