# Run Card: Esme-214M Multi-Turn SFT Foundation

This is the SFT step of the standard post-training chain
`Esme-214M-Base -> SFT (Esme-214M-Instruct, multi-turn) -> DPO -> RL`. It is one
real SFT on multi-turn conversational + instruction data, trained fresh from
Base. This is the foundation used by the DPO chat model; the single-turn
Instruct run remains retained evidence and the replay-SFT chat path is not part
of the active recipe.

## Artifact

- Produces: `Esme-214M-Instruct` (the multi-turn SFT foundation)
- Starts from: read-only `Esme-214M-Base` bundle at
  `/Users/adamthuvesen/dev/menti/esme-pretrain/exports/esme-214m-base`
- Bundle format: `llm_pretrain_dense_v1`
- Config: `configs/esme-214m-sft-multiturn.json`
- Planned output: `runs/esme-214m-sft-multiturn/esme_214m_sft_multiturn`
- Modal Volume: `esme-posttrain-esme-sft-multiturn` (separate from the Instruct Volume)
- Full-run status: NOT approved. The full launch needs explicit approval, the
  `--approved` flag, and bounded-matched-sweep learning-gate evidence.

## Data

- Train mix: 85% `HuggingFaceTB/smol-smoltalk`
  (`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0), **multi-turn included**.
  Capacity-filtered: rows tagged with the `apigen-80k`, `xlam-function-calling-60k`,
  or `self-oss-instruct-sc2-exec-filter-50k` subsets (function calling, hardest
  reasoning) are dropped, following SmolLM2's small-model SFT mix.
- Train mix: 15% `allenai/tulu-3-sft-personas-instruction-following`
  (`fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`, ODC-BY), single-turn
  instruction-following with constraints folded into the user turn.
- Eval only: `HuggingFaceH4/no_robots`
  (`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, CC-BY-NC-4.0), non-commercial OOD
  guardrail. Never used for training unless a separate non-commercial-training
  approval is recorded.
- Sample cap: 50,000 train examples.
- Target / hard token cap: 60M / 80M selected train tokens (~2 effective epochs).
- Max sequence length: **1,024 tokens**, matching the `Esme-214M-Base` context
  (`context_length=1024`). Multi-turn conversations that fit within 1,024 tokens
  are kept whole; longer conversations are dropped (never truncated down to a
  single turn) so multi-turn coverage stays real. Padding-efficiency and
  truncation/drop counts are recorded in the data report.

## Chat Template & Masking

Conversations render as the repo's role markers extended to N turns:

```
system\n{system}\n
user\n{user}\nassistant\n{assistant}<eos>
user\n{user}\nassistant\n{assistant}<eos>
...
```

Every assistant turn's content plus its trailing `<eos>` is supervised
(`assistant_only_loss`); all system/user turns and the role markers themselves
are masked with `-100`. Single-turn rows are the one-assistant-turn special case.

## Recipe

- AdamW, cosine decay, ~4% warmup, weight decay 0.1, grad clip 1.0, seed 214.
- Effective batch 16 = microbatch 2 x grad-accum 8.
- `bf16` on CUDA; `sequence_packing=false` initially (padding metrics emitted).
- ~2 effective epochs over the curated subset.
- Learning rate: the config carries `1e-4` as the small-model starting point
  (SmolLM2 used a higher LR than the 2e-5 large-model default). The full launch is
  gated on a bounded matched sweep that picks the LR by matched held-out response
  loss; the config LR is replaced by the sweep winner before any full run.

## Eval & Acceptance

- Selector / early-stop metric: weighted matched response loss over a held-out
  covering **both** multi-turn chat (smol-smoltalk, weight 0.85) and single-turn
  instruction (tulu-3-personas, weight 0.15). Held-out rows are skipped past the
  selected train rows so train/eval are disjoint.
- `no_robots` is an OOD guardrail with a catastrophic-regression multiplier; never
  the selector.
- Fixed multi-turn sample prompts are generated into `multi-turn-samples.md`.
- An MT-Bench-style LLM-judge chat score is reported as the **mean of K>=5
  re-judge passes with its spread**. It is reporting-only and never the selector.
- Acceptance: the foundation beats Base on matched held-out response loss;
  multi-turn samples show coherent turn-taking; instruction-following present.

## Commands

No-spend dry-run (never starts Modal):

```bash
uv run esme-posttrain sft-multiturn-dry-run --config configs/esme-214m-sft-multiturn.json --json
```

No-spend multi-turn CPU fixture evidence:

```bash
uv run esme-posttrain sft-multiturn-cpu-fixture --config configs/esme-214m-sft-multiturn.json --json
```

Detached Modal smoke, capped at `$2`, no `SFT_MODAL_GPU` spend bypass:

```bash
SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=12 uv run --with modal==1.5.1 modal run --detach scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --approved --json
```

Bounded 1024-len throughput probe (`<=$3`, no W&B):

```bash
SFT_MODAL_GPU='A100' SFT_PROBE_TIMEOUT_HOURS=1 uv run --with modal==1.5.1 modal run scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --throughput-probe --approved --json
```

Bounded matched-eval LR sweep (`<=$8`, W&B `stage=sft`); must beat step 0 to
populate the learning gate:

```bash
SFT_MODAL_GPU='A100' SFT_SWEEP_TIMEOUT_HOURS=3 uv run --with modal==1.5.1 modal run scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --modal-sweep --approved --json
```

Detached full run, capped at `$40`, refused without `--approved` AND
learning-gate evidence:

```bash
SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=12 uv run --with modal==1.5.1 modal run --detach scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --full-run --approved --json
```

## Approval Gates

- Local/CPU evidence and a small Modal smoke are capped at `$2`.
- The full multi-turn SFT launch requires `--approved`, bounded-matched-sweep
  learning-gate evidence (`eval/matched/response_loss` lower than step 0), and a
  blocker-free no-spend dry-run.
- `$40` is the full-run runaway cap and runtime stop (raised from `$25` with
  explicit approval for this multi-turn foundation; the single-turn Instruct path
  keeps its `$25` cap).
- A 1024-len throughput probe re-confirms A100 tokens/s before the full launch;
  the recipe trains at the same 1024 length as the carried-over Instruct rate.
- Any dataset revision/license change, use of `no_robots` for training, higher
  cap, retry behavior, or full launch requires a new written approval.
```
