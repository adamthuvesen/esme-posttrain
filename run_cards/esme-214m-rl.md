# Run Card: Esme-214M-RL

## Artifact

- Produces: `Esme-214M-RL`
- Starts from: `Esme-214M-Chat`
- Ladder: `Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL`
- Config: `configs/esme-214m-rl.json`
- Dataset manifest schema: `schemas/rl-task-manifest.schema.json`
- Launch config schema: `schemas/rlvr-grpo-config.schema.json`
- Launcher: `scripts/modal_rlvr_grpo.py`

## Plan

- Method: Countdown-Lite RLVR from Chat using verifiable rewards only.
- Dataset: local RL task manifest at `data/manifests/esme-214m-rl.tasks.json`.
- Dataset shape: deterministic local train/dev/eval JSONL splits with 300/30/30 tasks.
- Task rules: 2-3 supplied integers, small integer targets, `+`, `-`, `*`, parentheses allowed,
  and each supplied number must be used exactly once.
- Reward policy: every reward must be measurable and verifier-backed.
- Excluded reward terms: style qualities such as sharpness, friendliness, and naturalness.
- Primary eval: Countdown-Lite eval split.
- Full acceptance eval profile: `full_acceptance_30x32` (`30` tasks x `32` samples = `960`
  rollouts).
- No-spend lifecycle gate: `pipeline_smoke` (`1` train task, `1` GRPO step, before/after
  eval at `1 x 1`) on fixtures only.
- Secondary transfer eval: GSM8K-lite only; it is not a training target or reward source.
- Sample budget: 360 local Countdown-Lite tasks.
- Token budget: 512,000 tokens.
- Seed: `214`.
- Hardware for the first bounded GRPO tranche: Modal `A100`.
- Expected dry-run duration: under 1 minute.
- Expected dry-run cost: `$0`.
- Expected GRPO duration: about 23 minutes of projected train-token work, plus Modal startup
  and before/after eval overhead.
- Expected GRPO cost: `$0.8039` from the checked-in dry-run.
- Runtime hard stop: `$8.00`.
- Timeout cost ceiling: `$6.2964` from `3h * $2.0988/hr`, below the runtime hard stop.
- Mission cap: under `$25.00`.
- Planned run output: `runs/esme-214m-rlvr-countdown-grpo` (ignored by git).
- Checked-in report: `artifacts/rlvr-countdown-lite/grpo-report.json`.
- Checked-in doc: `docs/rlvr-countdown-lite-grpo.md`.

## Countdown-Lite Baseline

Baseline command:

```bash
uv run esme-posttrain rlvr-countdown-lite-baseline \
  --manifest data/manifests/esme-214m-rl.tasks.json \
  --bundle exports/esme-214m-chat \
  --output-dir artifacts/rlvr-countdown-lite \
  --split eval \
  --samples-per-task 32 \
  --max-new-tokens 4 \
  --seed 214 \
  --json
```

Evidence:

- Report: `artifacts/rlvr-countdown-lite/baseline-report.json`
- pass@1: 3.33%
- pass@8: 3.33%
- pass@32: 6.67%
- valid-expression rate: 2.71%
- exact-solve rate: 0.42%
- easy pass@32: 20.00%
- medium pass@32: 0.00%
- hard pass@32: 0.00%

Decision: `GRPO-ready`. The local chat bundle has a small but nonzero easy-band foothold, so the
first bounded attempt is GRPO, not a tiny SFT/hint cold-start.

## Approval Gates

Adam approved Modal/GPU spend for this mission after this run card/config has explicit bounds.
Hard cap: under `$25` total expected spend for Countdown-Lite only; stop and report before
exceeding the cap or running any task outside this card.

Current status: bounded GRPO trainer/launcher exists and the dry-run is inside cap. The first
Modal attempt hydrated app `ap-0NVYhugDtfOpDsdHqRJlhX` and showed one active task, but returned no
function-call payload and emitted no logs before it was stopped to avoid leaving work in flight.
Read-only post-stop evidence from `uv run python -m modal app list --json` reports
`state=stopped` and `tasks=0` for `ap-0NVYhugDtfOpDsdHqRJlhX`. No after-eval artifacts were
produced. The second approved probe launched cleanly as app `ap-ISGdjVzIZQWy1c92PHzsoh` /
function call `fc-01KWE1VZ2HK5JS1S3JAP5FD5YB`, then stalled at `before_eval_start` for the full
`30` eval tasks x `32` samples. HQ stopped it at `2026-07-01 07:48:13+02:00` with `0` tasks.

Launcher fix status: implemented as a no-spend patch. The approved full-run path now uses the
shared pinned detached Modal command, spawns the GRPO function, writes an in-flight report, prints a
launch receipt with the Modal app/function-call ids and log commands, and returns without waiting
for the remote result. Direct unhydrated Python execution fails loudly instead of entering an
`app.run()` fallback. The remote function now prints flushed lifecycle milestones for remote entry,
config validation, output selection, CUDA selection, bundle/data load, before eval, trainer, after
eval, and return serialization; it also mirrors those milestones to a small Volume status JSONL.
No Modal/GPU rerun has been performed for this fix.

Before/after eval observability status: implemented as a no-spend patch after the stopped
`before_eval_start` probe. RLVR eval generation now emits start, progress, complete, and timeout
milestones with task/sample counters, total sample budget, sample batch size, and elapsed seconds.
Milestones print to Modal logs, mirror to the Volume status JSONL, and write local
`eval-progress.jsonl`. Full eval remains `30 x 32`, generated in bounded sample batches of `4`,
with fail-loud guards checked between batches. The effective eval wall/no-progress timeouts are now
derived from the selected eval sample budget: `max(900s, samples * 2.5s)` and
`max(300s, samples * 0.75s)`. For the full `30 x 32` acceptance profile this yields `2400s`
wall-time and `720s` no-progress guards, so the old fixed `900s` wall guard cannot kill the run
before training at the observed `569/960` sample mark. The reduced debug probe uses `2` eval tasks
x `4` samples and exits before trainer spend.

Pipeline smoke gate status: implemented as a no-spend local path. It uses the explicit
`pipeline_smoke` config profile, runs on the checked-in tiny fixture bundle/data, disables online
W&B, does not start Modal/GPU work, and must reach `before_eval`, `trainer_start`, `after_eval`,
and report generation before any renewed Modal spend.

Eval resume/provenance status: every before/after Countdown-Lite eval writes
`baseline-partial.jsonl` beside its report. Each completed task record includes the phase,
eval profile, config hash, model id, task/sample range, completed/total counts, split, and
sample budget. Re-running the eval with matching metadata resumes from completed task records
instead of duplicating samples; metadata mismatches fail loudly.

W&B status: RLVR uses the shared SFT/DPO W&B initializer and metric logger. Local job calls default
to W&B disabled. Tests exercise W&B with `wandb_mode="offline"` and a fake module, so no network or
API key is required. Future approved Modal RLVR calls enable W&B online and mount
`modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])`; `monitoring.wandb_project` is
`esme-posttrain`, `monitoring.wandb_required_for_modal=true`, and tags include `stage=rlvr`.

Actual/estimated spend so far: `$6.2964` conservative timeout ceiling for the stopped
`ap-ISGdjVzIZQWy1c92PHzsoh` call; no remote `cost.json` artifact returned.

Dry-run only:

```bash
uv run esme-posttrain rlvr-dry-run --config configs/esme-214m-rl.json
```

Fixture smoke test:

```bash
uv run esme-posttrain rlvr-dry-run --config fixtures/configs/esme-214m-rl.fixture.json
```

No-spend `pipeline_smoke` lifecycle gate:

```bash
uv run esme-posttrain rlvr-pipeline-smoke --config fixtures/configs/esme-214m-rl.fixture.json --output-dir runs/esme-214m-rlvr-pipeline-smoke-fixture --report-path artifacts/rlvr-countdown-lite/pipeline-smoke-report.json --doc-path docs/internal/rlvr-countdown-lite-pipeline-check.md --json
```

Approved Modal `pipeline_smoke` lifecycle smoke, left unrun:

```bash
RLVR_MODAL_OUTPUT_STEM='esme-214m-rlvr-pipeline-smoke' RLVR_MODAL_GPU='A100' RLVR_TIMEOUT_HOURS=3 uv run --with modal==1.5.1 modal run --detach scripts/modal_rlvr_grpo.py \
  --config configs/esme-214m-rl.json --modal-pipeline-smoke --approved --json
```

Approved bounded full `30 x 32` GRPO launch, left unrun:

```bash
RLVR_MODAL_GPU='A100' RLVR_TIMEOUT_HOURS=3 uv run --with modal==1.5.1 modal run --detach scripts/modal_rlvr_grpo.py \
  --config configs/esme-214m-rl.json --full-run --approved --json
```

Approved before-eval debug probe, left unrun:

```bash
RLVR_MODAL_OUTPUT_STEM='esme-214m-rlvr-before-eval-debug' RLVR_MODAL_GPU='A100' RLVR_TIMEOUT_HOURS=3 uv run --with modal==1.5.1 modal run --detach scripts/modal_rlvr_grpo.py \
  --config configs/esme-214m-rl.json --debug-before-eval --approved --json
```

The launcher refuses any `RLVR_TIMEOUT_HOURS` value that does not exactly match
`runtime.timeout_hours` in `configs/esme-214m-rl.json`, including dry-runs, so the Modal decorator
timeout cannot exceed the checked-in hard-stop evidence.

The launch receipt is not a completion payload. It should include `status`,
`will_start_modal_job`, `debug_before_eval`, `modal_result_awaited=false`, `modal_app`,
`modal_app_id`, `modal_call_id`, `modal_logs_command`, `modal_call_logs_command`,
`modal_stop_command`, `modal_status_command`, `remote_status_path`, `full_launch_command`,
`resume_command`, `volume_output_dir`, `projected_cost_usd`, `runtime_spend_stop_usd`,
`timeout_cost_ceiling_usd`, `wandb_project`, `wandb_required_for_modal`, `wandb_mode`,
`report_path`, and `doc_path`.

Use the receipt's log commands immediately after an approved live launch:

```bash
modal app logs <app-id> --timestamps --show-function-call-id --show-container-id
modal app logs <app-id> --timestamps --show-function-call-id --show-container-id --function-call <fc-id>
```

The launch must return the JSON receipt immediately and write an in-flight
`artifacts/rlvr-countdown-lite/grpo-report.json`. Final before/after Countdown-Lite metrics are a
separate completion artifact from the remote run, not launch stdout. If the Modal app is active
without lifecycle logs beyond the projected startup window, stop it and record the blocker:

```bash
uv run python -m modal app stop <app-id> --yes
```

Countdown-Lite data generation:

```bash
uv run esme-posttrain rlvr-countdown-lite-build-data --repo-root . --json
```
