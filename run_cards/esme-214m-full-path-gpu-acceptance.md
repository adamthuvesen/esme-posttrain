# Run Card: Esme-214M Full-Path GPU Acceptance

One tiny paid run proving the full SFT -> DPO -> RLVR -> export handoff chain
on real CUDA. This is the GPU tier of the acceptance path; the CPU tier is the
committed `full-path-cpu-smoke` CLI.

**Status: written ahead of approval, not run.** Trigger: run right before the
next release or paid training campaign, and no earlier. Any spend under this
card still needs Adam's chat approval first.

## Coverage Delta vs the CPU Smoke

`full-path-cpu-smoke` already proves, at zero spend: stage chaining, DPO
interrupt/resume state equality against an uninterrupted control, bundle
contract validation at every handoff, and downstream verifier scoring -- on
fixture-scale models. This run re-proves none of that. It covers only what
CPU fixtures cannot:

- **CUDA precision.** Every stage runs `bf16` on CUDA. Losses, log-probs, and
  rewards must stay finite; the exported bundle must reload on CPU afterward
  and produce a bounded generation, so the bf16 weights round-trip.
- **Remote storage.** The base bundle is read from a Modal volume, each stage
  writes its artifacts to its stage volume, and everything retained is
  downloaded back off the volumes. The download is part of the run, not an
  afterthought (esme-pretrain retrospective, section 9).
- **Real memory behavior.** The real 214M `DenseBackbone` at the full-run
  micro-batch and `max_length 1024` shapes (SFT micro-batch 2, DPO micro-batch
  4 with a frozen reference model resident). Peak CUDA memory
  (`torch.cuda.max_memory_allocated`) is recorded per stage.

## Method

Chain the three existing per-stage Modal smoke modes, then export:

1. **SFT smoke** (`configs/esme-214m-sft-multiturn.json` smoke budgets:
   10 train samples, 16,384 train tokens, 4 eval samples) from the base
   bundle.
2. **DPO smoke** (`configs/esme-214m-chat-dpo.json` smoke budgets: 12 pairs,
   24,576 train tokens, 6 eval pairs) with `sft_reference` overridden to the
   step-1 smoke checkpoint, not the accepted full-run checkpoint -- the point
   is the handoff.
3. **RLVR pipeline smoke** (`configs/esme-214m-rl.json` `pipeline_smoke`
   profile: 1 step, max 4,096 rollout tokens) on the exported step-2 bundle.
   Countdown-Lite data is committed in-repo; no dataset download.
4. **Export + retrieve.** Export the final bundle, download all retained
   artifacts off the volumes, revalidate the bundle contract locally, and run
   one bounded CPU generation from it.

Wiring gap to close before launch (small, no spend): the per-stage smokes do
not yet chain their volume paths, and peak-CUDA-memory recording is not wired
into the stage environment records. Both land as normal reviewed changes
before approval is requested.

## Budgets

| Axis | Value |
| --- | --- |
| GPU | 1x A100 (Modal `A100`, $2.0988/h -- same profile as every accepted full run, so memory behavior is comparable) |
| Hard cost cap | **$4.00 total**; per-stage runtime spend stop $2.00 (the existing `smoke_max_cost_usd`) |
| Timeout | 0.5 h per stage job; three jobs worst-case ceiling $3.15, under the cap |
| Expected duration | 25-45 GPU-minutes total (SFT smoke alone projects 8 min on A100 per its config) |
| Expected cost | ~$0.90-$1.60 |
| Token budget | <= 16,384 SFT train + 24,576 DPO train + 4,096 RLVR rollout tokens: <= 45,056 total, hard stop 60,000 |
| Dataset download | Bounded smoke-budget slices of the two pinned HF datasets (smol-smoltalk / tulu-3-personas mix and ultrafeedback_binarized) only; approval of this card covers exactly that |
| W&B | Disabled |
| Retries | None (`allow_retries false`). A failed attempt is recorded here with its sunk spend; retrying needs fresh approval |

## Retained Artifacts

Per stage: `config.json`, `metrics.jsonl`, `best-checkpoint.json`,
`cost.json`, `environment.txt` (including peak CUDA memory), `manifest.json`.
Plus the final exported bundle (`config.json`, `tokenizer.json`, `weights.pt`,
`manifest.json`) and the chain-level acceptance report. All of it downloaded
to gitignored `runs/full-path-gpu-acceptance/` and hashed.

Checkpoints (`.pt`) other than the final bundle weights are not retained
past acceptance; this run proves the path, not a model.

## Abort Rules

Abort the whole chain, keep partial artifacts for diagnosis, and record the
attempt and sunk spend in this card, when any of these fires:

- A stage preflight reports launch blockers (never override them).
- Elapsed cost passes the per-stage $2.00 stop or the $4.00 total cap.
- A stage hits its 0.5 h timeout or is preempted.
- Any loss, log-prob, or reward goes NaN/inf, or a stage OOMs.
- Bundle contract validation fails at any handoff.

## Definition of Done

The run is accepted only when **all** of the following hold. A green Modal
job with artifacts still on the volume is not done -- off-machine artifacts
are unaccepted (esme-pretrain retrospective, section 9: evidence left on
volumes and laptops was lost).

1. All three stages and the export complete with finite metrics, bf16
   confirmed on CUDA, within the caps.
2. Every retained artifact is downloaded off the Modal volumes to local
   `runs/full-path-gpu-acceptance/`.
3. The exported bundle revalidates locally (`validate_bundle_contract`) and
   produces a bounded CPU generation.
4. An **Accepted Result** section is added to this card -- numbers, peak
   CUDA memory per stage, per-stage cost, and the sha256 of each retained
   artifact -- and committed as part of accepting the run.

## Safe Local Commands

```bash
uv run esme-posttrain full-path-cpu-smoke --json
uv run esme-posttrain sft-multiturn-dry-run --config configs/esme-214m-sft-multiturn.json --json
uv run esme-posttrain chat-dpo-dry-run --config configs/esme-214m-chat-dpo.json --json
uv run esme-posttrain rlvr-dry-run --config configs/esme-214m-rl.json
```
