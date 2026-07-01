# Run Card: Esme-214M-Instruct SFT

## Artifact

- Produces: `Esme-214M-Instruct`
- Starts from: read-only `Esme-214M-Base` bundle at
  `/Users/adamthuvesen/dev/menti/esme-pretrain/exports/esme-214m-base`
- Bundle format: `llm_pretrain_dense_v1`
- Config: `configs/esme-214m-instruct.json`
- Config schema: `schemas/instruct-sft-config.schema.json`
- Planned output: `runs/esme-214m-instruct-sft-pilot/esme_214m_instruct_sft_pilot`
- Bounded sweep approval: Adam approved the real-data interval-eval Modal sweep
  on 2026-06-27. This is not approval for a new full-data SFT launch.
- Full-run status: the learning gate records stopped-run reconciliation and
  bounded matched interval-eval sweep evidence. Adam approved the full A100
  launch after a blocker-free no-spend dry-run.

## Data

- Train mix: 80% `HuggingFaceTB/smol-smoltalk`
  (`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`, Apache-2.0), filtered to concise
  single-turn rows.
- Train mix: 20% `allenai/tulu-3-sft-personas-instruction-following`
  (`fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e`, ODC-BY), with constraints folded
  into the user prompt.
- Eval only: `HuggingFaceH4/no_robots`
  (`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, CC-BY-NC-4.0). This dataset is
  non-commercial and is not allowed in training unless a separate approval records
  that change.
- Sample cap: 50,000 train examples.
- Target train tokens: 30,000,000 trained tokens, roughly two effective epochs
  over the observed selected train tokens.
- Hard token cap: 50,000,000 selected train tokens.
- Max sequence length: 1,024 tokens.

## Evidence Contract

The pilot is not launch-ready unless all of these are true:

### Stopped-Run Reconciliation

The stopped A100 full-data run and the older diagnostic full run are separate
evidence:

- Stopped A100 path:
  `/posttrain/esme-instruct-sft-showcase-full/metrics.jsonl` in Modal Volume
  `esme-posttrain-esme-instruct-sft-pilot`. Local evidence mirror:
  `runs/esme-214m-instruct-sft-pilot/stopped-run-evidence/showcase-full.metrics.jsonl`.
  It has 98 interval eval rows for `no_robots`: step 0, then every 200 steps
  through step 19,400. The best `no_robots` response loss was step 600
  (`2.617136564552844`); latest interval eval was step 19,400
  (`2.6772786582507764`) while train loss kept improving.
- Older diagnostic path:
  `/posttrain/esme-instruct-sft-full/metrics.jsonl`. Local evidence mirror:
  `runs/esme-214m-instruct-sft-pilot/stopped-run-evidence/older-full.metrics.jsonl`.
  It has only step-0 and final-style eval rows; final step 48,828 had
  `no_robots` response loss `3.4976762224550835` versus Base step-0
  `2.747025366096404`.
- These paths must not be conflated. The stopped A100 run shows why
  `no_robots` is useful as an OOD guardrail but insufficient as the checkpoint
  selector for an SFT recipe trained on SmolTalk/Tulu.

### Bounded Matched Interval-Eval Sweep

The bounded real-data A100 sweep completed on 2026-06-27 under isolated Modal
Volume paths rooted at `/posttrain/esme-instruct-sft-interval-sweep/`.

- Evidence path:
  `/posttrain/esme-instruct-sft-interval-sweep/sweep-20260627T143203Z-evidence/interval-eval-sweep.json`.
  Local mirror:
  `runs/esme-214m-instruct-sft-pilot/interval-sweep-evidence/sweep-20260627T143203Z/interval-eval-sweep.json`.
- Base step 0 matched response loss was `2.187998926732106`, from the weighted
  selector `0.8 * smol-smoltalk + 0.2 * tulu-3-personas` with components
  `2.0866214573707644` and `2.593508804177473`.
- The selected arm is `sweep-20260627T143203Z-lr3e-5-mb2-ga8-eb16`: best
  matched response loss `1.7372412149933547` at step 60. This is the basis for
  saying the corrected recipe beats Base on the matched selector.
- The `lr1e-5` arm also beat Base (`1.7755725221338137` at step 160), but it was
  not the best arm.
- `no_robots` remains an OOD guardrail/reporting metric, not the selector: Base
  step 0 was `2.752275518437169`; the selected arm's best-checkpoint
  `no_robots` loss was `2.7090167036172934`.
- The selected arm's final step drifted back up (`1.8420357651088226` matched,
  `2.8486045192173606` `no_robots`), so the full report must use the selected
  best checkpoint, not the final training step.
- Generation evidence for the selected arm is mirrored at
  `runs/esme-214m-instruct-sft-pilot/interval-sweep-evidence/sweep-20260627T143203Z/lr3e-5.samples.md`.

- Base bundle manifest SHA256s validate before weights are loaded.
- Prompt tokens are masked with `-100`; loss applies only to assistant response tokens.
- Survivor counts, selected/unused examples, selected tokens, supervised tokens, eval
  examples, and effective epochs are reported.
- `selected-row-manifest.jsonl` records source, revision, row id, token counts, and
  prompt/response lengths for selected training rows.
- Train/eval metrics use namespaced W&B-ready keys and interval eval.
- Periodic checkpoints include model, optimizer, scheduler, completed step, and token
  totals; full-run resume loads the latest checkpoint from the stable Modal Volume
  output directory and records whether training was fresh or resumed.
- `sequence_packing=false` is explicit until a tested packer exists; data reports
  emit padding efficiency and max-sequence-slot efficiency.
- `checkpoint.pt` reload reproduces logits.
- Matched heldout SmolTalk/Tulu response loss and perplexity are written for
  Base and Instruct, with component metrics and weighted selector loss.
- `no_robots` response loss and perplexity are written for Base and Instruct as
  an OOD guardrail/reporting metric only.
- Instruct matched heldout response loss is lower than Base response loss.
- Required artifacts exist: `config.json`, `data-report.json`,
  `selected-row-manifest.jsonl`, `eval-smol-smoltalk-manifest.jsonl`,
  `eval-tulu-3-personas-manifest.jsonl`, `eval-no_robots-manifest.jsonl`,
  `metrics.jsonl`, `checkpoint.pt`, `best-checkpoint.pt`,
  `best-checkpoint.json`, `samples.md`, `tokenizer.json`, `manifest.json`,
  `eval-report.json`, `cost.json`, and `environment.txt`.

## Commands

No-spend dry-run:

```bash
uv run esme-posttrain instruct-sft-dry-run --config configs/esme-214m-instruct.json --json
```

No-spend CPU fixture evidence:

```bash
uv run esme-posttrain instruct-sft-cpu-fixture --config configs/esme-214m-instruct.json --json
```

Detached Modal smoke, capped at `$2`:

```bash
SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=10 uv run --with modal==1.5.1 modal run --detach scripts/modal_instruct_sft.py --config configs/esme-214m-instruct.json --approved --json
```

Completed bounded real-data interval-eval Modal sweep evidence, capped
separately and isolated from the full-run path. The active config selects A100
for the accepted full SFT recipe; any new sweep with a different GPU/profile
needs a separate run card and approval.

```bash
SFT_MODAL_GPU='A100' SFT_SWEEP_TIMEOUT_HOURS=2 uv run --with modal==1.5.1 modal run scripts/modal_instruct_sft.py --config configs/esme-214m-instruct.json --modal-sweep --approved --json
```

This command writes per-arm artifacts under
`/posttrain/esme-instruct-sft-interval-sweep/<arm-id>`, records
`interval-eval-sweep.json` and `learning-gate.json`, and must use the approved
train datasets plus `HuggingFaceH4/no_robots` eval-only holdout.

Detached full run, capped at `$25` and refused without `--approved`:

```bash
SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=10 uv run --with modal==1.5.1 modal run --detach scripts/modal_instruct_sft.py --config configs/esme-214m-instruct.json --full-run --approved --json
```

Set `SFT_MODAL_FULL_OUTPUT_STEM` to a fresh single directory stem when a stopped
diagnostic run already wrote the default full-run path. For the approved
2026-06-27 recovery launch, use
`esme-instruct-sft-showcase-full-20260627-a100-matched`. The corrected recipe is
learning rate `3e-5`, microbatch size 2, gradient accumulation 8, effective
batch size 16 on A100, cosine decay, 700 warmup steps, and a 30M trained-token
target for roughly two effective epochs over the observed 14.5M selected train
tokens.

The accepted local Esme-214M-Instruct SFT run selects step 6400 by matched eval.
Matched response loss improves from `2.1186577799740736` to
`1.1952465851220688`, and cost is `$3.9872` under cap.

Resume the detached full run from the latest checkpoint in the stable Volume
output directory:

```bash
SFT_MODAL_GPU='A100' SFT_TIMEOUT_HOURS=10 uv run --with modal==1.5.1 modal run --detach scripts/modal_instruct_sft.py --config configs/esme-214m-instruct.json --full-run --approved --resume --json
```

## Approval Gates

- Local/CPU evidence and a small Modal smoke remain capped at `$2`.
- The bounded real-data interval-eval sweep approval covered only that sweep,
  not a new full-data SFT launch.
- Full Esme-214M-Instruct SFT reruns require `--approved`, the learning gate,
  and a separate explicit full-data launch approval.
- Adam explicitly approved the A100 full SFT launch after the matched
  interval-eval evidence and dry-run gate were recorded.
- `$25` is the full-run runaway cap and runtime stop.
- GPU profile selection is explicit in config. `A100` is selected from the
  bounded throughput probe; `L4` and `A10G` remain documented profiles.
- Any dataset revision/license change, use of `no_robots` for training, higher cap,
  retry behavior, or full SFT launch requires a new written approval.
