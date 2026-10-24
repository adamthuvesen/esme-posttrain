# RLVR Countdown Held-Out Transfer

Did Countdown-Lite GRPO teach `Esme-214M-RL` the task family, or did it memorize
the training distribution? This report scores the RL step-240 checkpoint and its
pre-RL warm start (`Esme-214M-Chat`, the DPO artifact GRPO started from) on two
held-out task sets that are dedup-verified disjoint from all 360 committed
Countdown-Lite tasks, using the acceptance protocol (`eval_max_new_tokens 12`,
32 samples/task, seed 214).

## Verdict

**A transferable template, not transferable arithmetic.**
After GRPO the RL model emits a valid expression ~99% of the time on fresh,
never-seen tasks (both the fresh and target-shifted sets), versus 1-4% for the
warm start. The expression is almost always the sum of the supplied numbers,
whatever the target (`docs/rlvr-research-report.md`). Every exact solve below is a
task whose target is that sum: `easy_0000` (1 + 9 = 10), `easy_0002` (1 + 3 = 4),
and `medium_0004`. The target-shifted set (targets 65-128) has no sum targets, and
the model solves none of it.

## Metrics (32 samples/task, max_new_tokens=12, seed 214)

Both models were scored locally with
the same code path (`rlvr-countdown-lite-baseline`), so the comparison is
device- and RNG-consistent. In-distribution acceptance is shown for reference.

| Model | Task set | pass@1 | pass@8 | pass@32 | Valid expressions | Exact solves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Esme-214M-Chat | acceptance (in-distribution) | 3.33% | 6.67% | 13.33% | 5.83% | 0.73% |
| Esme-214M-RL | acceptance (in-distribution) | 16.67% | 16.67% | 16.67% | 99.38% | 16.35% |
| Esme-214M-Chat | held-out fresh | 3.33% | 3.33% | 3.33% | 3.54% | 0.52% |
| Esme-214M-RL | held-out fresh | 10.00% | 10.00% | 10.00% | 99.06% | 9.79% |
| Esme-214M-Chat | held-out shift | 0.00% | 0.00% | 0.00% | 1.46% | 0.00% |
| Esme-214M-RL | held-out shift | 0.00% | 0.00% | 0.00% | 99.38% | 0.00% |

### Width-stratified view (the load-bearing comparison)

The overall fresh-set numbers mix operand counts (the fresh set has only 5
two-number tasks against 25 three-number; see set construction). Stratified by
operand count:

| Stratum | Set | Chat valid | RL valid | Chat exact | RL exact | RL pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2-number | held-out fresh (5 tasks) | 10.00% | 98.75% | 3.12% | 38.75% | 40.00% |
| 3-number | held-out fresh (25 tasks) | 2.25% | 99.12% | 0.00% | 4.00% | 4.00% |
| 3-number | held-out shift (30 tasks) | 1.46% | 99.38% | 0.00% | 0.00% | 0.00% |

The RL model writes a valid 3-number expression ~99% of the time on numbers it
has never seen, because it adds them. It solves the one fresh 3-number task whose
target is the sum (4%) and nothing on the shifted set. The two solved 2-number
tasks are also sum targets.

## Held-out set construction

Both sets contain 30 tasks (matching the acceptance eval size), live in
`data/rl/countdown_heldout/`, and are declared by
`data/manifests/esme-214m-rl-heldout.tasks.json`. Generator:
`uv run esme-posttrain rlvr-countdown-heldout-build-data` (deterministic;
builder in `src/esme_posttrain/rl/countdown_heldout.py`, selection seed 4126).

- **Dedup key:** canonical `(sorted numbers tuple, target)`, the same key the
  Countdown-Lite generator dedups on. Disjointness against all 360 committed
  train/dev/eval tasks is enforced by a loud check inside the builder and by
  unit tests (`tests/test_countdown_heldout.py`).
- **Set A `heldout_fresh` (fresh-unseen):** same generator and distribution
  (numbers 1-9, 2-3 operands, targets 0-64), restricted to tasks the committed
  dataset never selected and re-ranked with the new seed. Mix: 5 easy / 15
  medium / 10 hard; by width, 5 two-number and 25 three-number.
- **Set B `heldout_shift` (shifted-variant):** targets 65-128 instead of 0-64,
  with numbers (1-9), operand counts, and operators (`+ - *`) unchanged. Every
  selected task has 3 operands; disjoint from the committed tasks by
  construction and verified with the same key.

## Checkpoint provenance

- `Esme-214M-RL`: the step-240 bundle rebuilt from the accepted run's training
  checkpoint with `rlvr-checkpoint-bundle` (weights sha256 `b35cdfb5…`). It is
  tensor-identical to the trainer's own step-240 export, and it scores the same as the
  step-234 best checkpoint on the acceptance eval.
- `Esme-214M-Chat`: the exported DPO best checkpoint (weights sha256 `3e7c1a45…`), the
  exact artifact every GRPO run warm-started from.

## Execution and spend

- All evals ran locally on CPU for $0. Reports are under gitignored
  `artifacts/verified-heldout/`.
- Eval command per cell (bundle and split vary):
  `uv run esme-posttrain rlvr-countdown-lite-baseline --manifest
  data/manifests/esme-214m-rl-heldout.tasks.json --bundle <bundle-dir>
  --output-dir <out> --split <heldout_fresh|heldout_shift> --samples-per-task 32
  --max-new-tokens 12 --seed 214 --json`
