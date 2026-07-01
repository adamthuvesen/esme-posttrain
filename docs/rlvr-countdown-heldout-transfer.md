# RLVR Countdown Held-Out Transfer

Did Countdown-Lite GRPO teach `Esme-214M-RL` the task family, or did it
memorize the training distribution? This report scores the RL best checkpoint
and its pre-RL warm start (`Esme-214M-Chat`, the DPO artifact GRPO started
from) on two held-out task sets that are dedup-verified disjoint from all 360
committed Countdown-Lite tasks, using the exact acceptance protocol.

## Verdict

**Transfer, not memorization — on the stratum the acceptance protocol can
measure.** On fresh, never-seen 2-number tasks the RL model keeps nearly its
entire advantage over the warm start (valid-expression delta 87.5pp held-out
vs 89.8pp in-distribution; exact-solve delta 32.5pp vs 41.8pp). It emits
well-formed `a op b` expressions for number pairs it never trained on, so GRPO
taught a reusable answer format plus real 2-operand arithmetic skill, not a
lookup table of training tasks.

**Caveat: the acceptance protocol cannot measure 3-number tasks at all.** The
eval caps generation at `max_new_tokens=4`, and the shortest tokenization of
any 3-number expression (e.g. `2+3*4`) is 5 tokens under the bundle tokenizer.
Every 3-number task is therefore structurally unsolvable for both models —
which also means the accepted in-training numbers (35.73% valid after RL) were
always driven entirely by the 11 two-number tasks in the 30-task acceptance
set. On the shifted-target set (all 3-number) both models score exactly 0
under the frozen protocol, so the target-range-shift question is unanswerable
at 4 tokens; a supplementary 8-token diagnostic (below) shows what changes
when the cap is lifted.

## Metrics — acceptance protocol (32 samples/task, max_new_tokens=4, seed 214)

All six runs below were executed locally on CPU in this session with the same
code path (`rlvr-countdown-lite-baseline`), so the RL-vs-Chat comparison is
device- and RNG-consistent. Full per-task JSON lives under gitignored
`artifacts/heldout-transfer/`.

| Model | Task set | pass@1 | pass@8 | pass@32 | Valid expressions | Exact solves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Esme-214M-Chat | acceptance eval (local rerun) | 3.33% | 3.33% | 6.67% | 2.71% | 0.42% |
| Esme-214M-Chat | held-out fresh | 3.33% | 3.33% | 3.33% | 1.15% | 0.62% |
| Esme-214M-Chat | held-out shift | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Esme-214M-RL | acceptance eval (local rerun) | 16.67% | 20.00% | 20.00% | 35.62% | 15.73% |
| Esme-214M-RL | held-out fresh | 6.67% | 6.67% | 6.67% | 15.73% | 6.04% |
| Esme-214M-RL | held-out shift | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

Reference in-training numbers from the accepted GRPO run (Modal A100, same
protocol, `caff0a1/eval-{before,after}.json`): Chat 3.33/3.33/16.67 pass@1/8/32,
valid 3.23%, exact 1.25%; RL 16.67/16.67/20.00, valid 35.73%, exact 15.83%.
The local reruns reproduce these within sampling noise — the Chat rerun matches
the committed local baseline (`docs/rlvr-countdown-lite.md`) exactly, and the
RL rerun matches Modal `eval-after` to within one task on pass@8 (stochastic
samples differ across devices at fixed seed).

### Width-stratified view (the load-bearing comparison)

The overall fresh-set drop (35.62% -> 15.73% valid) is a difficulty-mix
artifact: the acceptance set has 11 two-number tasks, the fresh set only 5
(see set construction). Stratified by operand count:

| Stratum | Set | Chat valid | RL valid | Chat exact | RL exact | RL pass@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2-number | acceptance (11 tasks) | 7.39% | 97.16% | 1.14% | 42.90% | 54.55% |
| 2-number | held-out fresh (5 tasks) | 6.88% | 94.37% | 3.75% | 36.25% | 40.00% |
| 3-number | any set, either model | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

On 3-number tasks the RL model almost always emits a well-formed 2-number
expression using a subset of the prompt numbers (783/800 invalid fresh-set
width-3 samples fail only the "use each number exactly once" rule) — the
4-token cap truncates before a third operand can appear.

## Supplementary diagnostic — max_new_tokens=8 (protocol deviation, labeled)

To separate "protocol ceiling" from "capability ceiling" on 3-number tasks,
the held-out sets were re-run once with `--max-new-tokens 8` (everything else
identical). This is NOT the acceptance protocol and is reported only as a
diagnostic:

| Model | Task set | pass@1 | pass@8 | pass@32 | Valid expressions | Exact solves |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Esme-214M-Chat | held-out fresh | 3.33% | 3.33% | 3.33% | 2.71% | 0.52% |
| Esme-214M-Chat | held-out shift | 0.00% | 0.00% | 0.00% | 0.94% | 0.00% |
| Esme-214M-RL | held-out fresh | 6.67% | 6.67% | 6.67% | 28.02% | 5.94% |
| Esme-214M-RL | held-out shift | 0.00% | 0.00% | 0.00% | 22.29% | 0.00% |

With room to express three operands, the RL model's learned format transfers
but its arithmetic does not: on 3-number tasks its valid-expression rate rises
from 0% to 14.88% (fresh) and 22.29% (shift) vs the warm start's ~1%, yet its
3-number exact-solve rate stays exactly 0 on both sets, and pass@k does not
move (all solves remain 2-number). GRPO taught a transferable expression
format and 2-operand arithmetic; it did not create 3-operand solving ability,
in or out of distribution.

## Held-out set construction

Both sets contain 30 tasks (matching the acceptance eval size), live in
`data/rl/countdown_heldout/`, and are declared by
`data/manifests/esme-214m-rl-heldout.tasks.json`. Generator:
`uv run esme-posttrain rlvr-countdown-heldout-build-data` (deterministic;
builder in `src/esme_posttrain/rl/countdown_heldout.py`, selection seed 4126).

- **Dedup key:** canonical `(sorted numbers tuple, target)` — the same key the
  Countdown-Lite generator dedups on. Disjointness against all 360 committed
  train/dev/eval tasks is enforced by a loud check inside the builder and by
  unit tests (`tests/test_countdown_heldout.py`).
- **Set A `heldout_fresh` (fresh-unseen):** same generator and distribution
  (numbers 1-9, 2-3 operands, targets 0-64), restricted to tasks the committed
  dataset never selected and re-ranked with the new seed. Mix: 5 easy /
  15 medium / 10 hard. The acceptance mix (10/12/8) is impossible for an
  unseen same-distribution set: the committed dataset already consumed 110 of
  the 115 unique easy tasks the generator can produce, so the fresh set takes
  the entire remaining easy population (5) and keeps the acceptance
  medium:hard ratio (3:2) for the rest.
- **Set B `heldout_shift` (shifted-variant):** one modest shift on the target
  axis the generator exposes — targets 65-128 instead of 0-64, with numbers
  (1-9), operand counts (2-3), and operators (`+ - *`) unchanged. In practice
  every selected task has 3 operands (2-operand targets above 64 are rare) and
  targets span 66-126. Disjoint from the committed tasks by construction
  (their targets cap at 64) and still verified with the same key.

## Checkpoint provenance

- `Esme-214M-RL`: Modal Volume `esme-posttrain-esme-rlvr-countdown`,
  `esme-214m-rlvr-countdown-grpo-caff0a1/bundle/` — the dense bundle the GRPO
  job exported after restoring the best checkpoint
  (`best-checkpoint.pt`, selected step 22, `train/reward_mean` 0.7719), i.e.
  the same weights as `best-checkpoint.pt` in loadable bundle form. Local copy
  sha256-verified against the run manifest
  (`weights.pt` = `8557333a3383e1f4...`).
- `Esme-214M-Chat`: Modal Volume `llm-infer-esme-bundles`, `esme-214m-chat/` —
  the exported DPO best checkpoint (step 600), the exact artifact GRPO
  warm-started from. Local copy sha256-verified
  (`weights.pt` = `3e7c1a45fb398614...`); `config.json` and `tokenizer.json`
  hashes are identical across both bundles.

## Execution and spend

- All evals ran locally on CPU: **local, $0**. No Modal function was launched;
  Modal was used only for read-only `modal volume ls` / `modal volume get`
  (bundle download, ~1.7 GiB total). No W&B run.
- Eval command per cell (split and bundle vary):
  `uv run esme-posttrain rlvr-countdown-lite-baseline --manifest
  data/manifests/esme-214m-rl-heldout.tasks.json --bundle <bundle-dir>
  --output-dir <out> --split <split> --samples-per-task 32 --max-new-tokens 4
  --seed 214 --json` (the supplementary diagnostic reruns the two held-out
  splits with `--max-new-tokens 8`, all else identical)
- Protocol anchors: the local Chat acceptance rerun reproduces the committed
  local baseline exactly; the local RL acceptance rerun matches the accepted
  Modal `eval-after` within sampling noise (see metrics table).
