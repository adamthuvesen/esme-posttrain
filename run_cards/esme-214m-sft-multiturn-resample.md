# Run Card: Esme-214M Multi-Turn SFT Evidence Resample

Bounded, generation-only follow-up to the accepted multi-turn SFT foundation
(`run_cards/esme-214m-sft-multiturn.md`). It does not train anything.

## What And Why

The full run's `multi-turn-samples.md` evidence was generated with a truncation
bug: prompts were cut at `prompt_tokens`, i.e. before the FIRST assistant turn,
so the samples showed the model continuing from the opening user message
instead of continuing the final assistant turn of a multi-turn conversation.
The sampling code is fixed at commit `78f2094`
(`final_assistant_cut` + `write_multi_turn_samples` in
`src/esme_posttrain/sft/sample_artifacts.py`). The trained checkpoint exists
only in the Modal Volume, so this job regenerates the evidence from that
checkpoint with the fixed truncation.

## Model / Config

- Checkpoint: `best-checkpoint.pt` from the completed full run, in Modal Volume
  `esme-posttrain-esme-sft-multiturn` at `/posttrain/esme-214m-sft-multiturn-full/`
  (tokenizer: `tokenizer.json` in the same directory). Read in place; the run
  directory gains only the new artifact.
- Config: `configs/esme-214m-sft-multiturn.json` (same validated config as the
  full run; the launcher refuses anything else).

## Data

- Same matched held-out eval source as the full run, streamed INSIDE the Modal
  container only (the local CLI path never downloads):
  - `HuggingFaceTB/smol-smoltalk` (`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0)
  - `allenai/tulu-3-sft-personas-instruction-following`
    (`fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`, ODC-BY)
- Held-out selection is reproduced exactly like the full run: the per-source
  skip counts are read from the full run's persisted `data-report.json`
  (`train.counts_by_source[*].selected`), then the matched eval sets are rebuilt
  with the config budgets (256 samples / 524,288 tokens per source, 1,024 max
  sequence tokens). The sample pool is the first 3 multi-turn smol-smoltalk
  held-out rows, identical to the full run's pool.

## Sample Budget

- 3 samples, 256 new tokens each (`monitoring.sample_new_tokens` in the config).

## Hardware

- 1x A100 on Modal (`runtime.selected_gpu`), $2.0988/h. `SFT_MODAL_GPU` must
  match or the launch is blocked.

## Expected Duration

- Minutes, not hours: container start + checkpoint load (~1-2 min), streamed
  matched-eval rebuild dominates (skip ~42.5k selected smol rows and ~7.5k tulu
  rows, then tokenize 256 held-out rows per source; a few minutes), generation
  of 3 x 256 tokens on A100 is seconds. Realistic total: ~5-10 minutes.
- Hard Modal timeout: 0.25 h (15 min). If the streamed rebuild ever exceeds it,
  the job dies inside the ceiling below; `SFT_RESAMPLE_TIMEOUT_HOURS` may be
  raised only while `timeout x $2.0988/h` stays under the $1 blocker cap.

## Expected Cost

- Timeout cost ceiling: 0.25 h x $2.0988/h = **$0.5247**, validated by a
  launch blocker against the hard **$1** resample spend cap (mirrors the DPO
  chat-eval cap). Realistic cost at 5-10 min: ~$0.17-0.35.

## Artifact Produced

- `multi-turn-samples-v2.md` written next to the original in the Volume:
  `/posttrain/esme-214m-sft-multiturn-full/multi-turn-samples-v2.md`.
- The original `multi-turn-samples.md` is never modified, so the buggy evidence
  stays inspectable beside the corrected artifact.
- Local mirror after the job completes:

```bash
uv run --with modal==1.5.1 modal volume get esme-posttrain-esme-sft-multiturn \
  esme-214m-sft-multiturn-full/multi-turn-samples-v2.md \
  runs/esme-214m-sft-multiturn/esme_214m_sft_multiturn/multi-turn-samples-v2.md
```

## Launch

Dry-run preflight (no Modal, no download, no spend):

```bash
uv run python scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --resample-evidence --dry-run --json
```

Approved launch (spawns the bounded Modal function and prints a receipt without
waiting; results land in the Volume):

```bash
SFT_MODAL_GPU='A100' SFT_RESAMPLE_TIMEOUT_HOURS=0.25 uv run --with modal==1.5.1 modal run --detach scripts/modal_chat_sft.py --config configs/esme-214m-sft-multiturn.json --resample-evidence --approved --json
```

## Abort Rules

- The exact command above has not been approved in chat.
- `SFT_MODAL_GPU` differs from `runtime.selected_gpu`.
- The timeout cost ceiling (`SFT_RESAMPLE_TIMEOUT_HOURS` x GPU $/h) exceeds the
  $1 resample spend cap, at launch or re-checked inside Modal.
- `best-checkpoint.pt`, `tokenizer.json`, or `data-report.json` is missing from
  the full-run output directory (the job fails loudly instead of sampling from
  the wrong weights or the wrong held-out rows).
