# Run Card: Esme-214M Full-Path GPU Acceptance

The full SFT -> DPO -> RLVR -> export handoff chain is proven on real CUDA by
one tiny paid run. This is the GPU tier of the acceptance path; the CPU tier
is the committed `full-path-cpu-smoke` CLI.

**Status: written ahead of approval, not run.** Trigger: to be run right
before the next release or paid training campaign, and no earlier. Chat
approval from Adam is still required before any spend under this card.

## Coverage Delta vs the CPU Smoke

Stage chaining, DPO interrupt/resume state equality against an uninterrupted
control, bundle contract validation at every handoff, and downstream verifier
scoring are already proven at zero spend by `full-path-cpu-smoke` -- on
fixture-scale models. None of that is re-proven here. Only what cannot be
reached by CPU fixtures is covered:

- **CUDA precision.** Every stage is run in `bf16` on CUDA. Losses,
  log-probs, and rewards must stay finite; the exported bundle must be
  reloaded on CPU afterward and a bounded generation produced, so the bf16
  weights are shown to round-trip.
- **Remote storage.** The base bundle is read from a Modal volume, each
  stage's artifacts are written to its stage volume, and everything retained
  is downloaded back off the volumes. The download is part of the run, not an
  afterthought (esme-pretrain retrospective, section 9).
- **Real memory behavior.** The real 214M `DenseBackbone` is run at the
  full-run micro-batch and `max_length 1024` shapes (SFT micro-batch 2, DPO
  micro-batch 4 with a frozen reference model resident). Peak CUDA memory
  (`torch.cuda.max_memory_allocated`) is recorded per stage.

## Method

The three existing per-stage Modal smoke modes are chained, then the result
is exported:

1. **SFT smoke.** Run from the base bundle with the smoke budgets in
   `configs/esme-214m-sft-multiturn.json`: 10 train samples, 16,384 train
   tokens, 4 eval samples.
2. **DPO smoke.** Run with the smoke budgets in
   `configs/esme-214m-chat-dpo.json` (12 pairs, 24,576 train tokens, 6 eval
   pairs), with `sft_reference` overridden to the step-1 smoke checkpoint,
   not the accepted full-run checkpoint -- the handoff is the point.
3. **RLVR pipeline smoke.** Run on the exported step-2 bundle with the
   `pipeline_smoke` profile in `configs/esme-214m-rl.json`: 1 step, max
   4,096 rollout tokens. Countdown-Lite data is committed in-repo; no
   dataset download is needed.
4. **Export + retrieve.** The final bundle is exported, all retained
   artifacts are downloaded off the volumes, the bundle contract is
   revalidated locally, and one bounded CPU generation is produced from it.

Wiring gap to be closed before launch (small, no spend): the per-stage
smokes do not yet chain their volume paths, and peak-CUDA-memory recording
is not wired into the stage environment records. Both must be landed as
normal reviewed changes before approval is requested.

## Budgets

| Axis | Value |
| --- | --- |
| GPU | 1x A100 (Modal `A100`, $2.0988/h -- the same profile as every accepted full run, so memory behavior is comparable) |
| Hard cost cap | **$4.00 total**; per-stage runtime spend stop $2.00 (the existing `smoke_max_cost_usd`) |
| Timeout | 0.5 h per stage job; a three-job worst case is capped at a $3.15 ceiling, under the cap |
| Expected duration | 25-45 GPU-minutes total (8 min on A100 is projected for the SFT smoke alone, per its config) |
| Expected cost | ~$0.90-$1.60 |
| Token budget | <= 16,384 SFT train + 24,576 DPO train + 4,096 RLVR rollout tokens: <= 45,056 total, hard stop 60,000 |
| Dataset download | Bounded smoke-budget slices of the two pinned HF datasets (the smol-smoltalk / tulu-3-personas mix and ultrafeedback_binarized) only; exactly that download, and nothing more, is covered by approval of this card |
| W&B | Disabled |
| Retries | None (`allow_retries false`). A failed attempt is recorded here with its sunk spend; fresh approval is required for any retry |

## Retained Artifacts

Per stage: `config.json`, `metrics.jsonl`, `best-checkpoint.json`,
`cost.json`, `environment.txt` (including peak CUDA memory), `manifest.json`.
Plus the final exported bundle (`config.json`, `tokenizer.json`, `weights.pt`,
`manifest.json`) and the chain-level acceptance report. All of it is
downloaded to gitignored `runs/full-path-gpu-acceptance/` and hashed.

Checkpoints (`.pt`) other than the final bundle weights are not retained
past acceptance; the path is being proven here, not a model.

## Abort Rules

The whole chain is aborted, partial artifacts are kept for diagnosis, and
the attempt and its sunk spend are recorded in this card when any of these
fires:

- Launch blockers are reported by a stage preflight (blockers are never
  overridden).
- The per-stage $2.00 stop or the $4.00 total cap is reached.
- A stage's 0.5 h timeout is hit, or the worker is preempted.
- A NaN/inf loss, log-prob, or reward is seen, or CUDA memory is exhausted.
- The bundle contract check is failed at any handoff.

## Definition of Done

The run is accepted only when **all** of the following hold. A green Modal
job with artifacts still on the volume is not done -- off-machine artifacts
are treated as unaccepted (esme-pretrain retrospective, section 9: evidence
left on volumes and laptops was lost).

1. All three stages and the export are completed with finite metrics, bf16
   confirmed on CUDA, within the caps.
2. Every retained artifact is downloaded off the Modal volumes to local
   `runs/full-path-gpu-acceptance/`.
3. The exported bundle is revalidated locally (`validate_bundle_contract`)
   and a bounded CPU generation is produced from it.
4. An **Accepted Result** section -- the numbers, peak CUDA memory per
   stage, per-stage cost, and the sha256 of each retained artifact -- is
   added to this card and committed as part of accepting the run.

## Safe Local Commands

```bash
uv run esme-posttrain full-path-cpu-smoke --json
uv run esme-posttrain sft-multiturn-dry-run --config configs/esme-214m-sft-multiturn.json --json
uv run esme-posttrain chat-dpo-dry-run --config configs/esme-214m-chat-dpo.json --json
uv run esme-posttrain rlvr-dry-run --config configs/esme-214m-rl.json
```
