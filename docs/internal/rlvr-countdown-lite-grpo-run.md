# RLVR Countdown-Lite GRPO Run Details

Operator-grade provenance, training-shape verdict, and incident record for the
completed Countdown-Lite GRPO run. The public summary lives in
`docs/rlvr-countdown-lite-grpo.md`; the held-out transfer analysis in
`docs/rlvr-countdown-heldout-transfer.md`.

The run uses the stability bundle described on the run card
(`run_cards/esme-214m-rl.md`): Dr. GRPO mean-only advantage, graded reward,
DAPO zero-variance resampling, ARPO success-replay buffer, stratified
difficulty sampling, and `max_new_tokens 12` (a shorter rollout budget cannot
hold any 3-number expression).

## Result

- Artifact: `Esme-214M-RL` (output stem `esme-214m-rlvr-countdown-grpo-v2-ccb6287-1`)
- Reference artifact: `Esme-214M-Chat`
- Config: `configs/esme-214m-rl.json`
- Completed: 240/240 steps, before-eval plus after-eval on both the best
  (step 234) and final (step 240) checkpoints
- Eval profile: `full_acceptance_30x32`, `eval_max_new_tokens 12`, seed 214
- Training reward rose from 0.03 to a last-20 mean of 0.51 (best step 0.71)
  with no collapse; zero of 240 steps logged `reward_mean == 0`

## Before/After Eval

The "before" row is the same `Esme-214M-Chat` bundle the run re-baselined under
the 12-token protocol.

| Metric | Chat baseline (before) | Best ckpt (step 234) | Final ckpt (step 240) |
| --- | ---: | ---: | ---: |
| pass@1 | 3.33% | 16.67% | 16.67% |
| pass@8 | 6.67% | 16.67% | 16.67% |
| pass@32 | 13.33% | 16.67% | 16.67% |
| valid-expression rate | 5.83% | 99.38% | 99.38% |
| exact-solve rate | 0.73% | 16.35% | 16.35% |

The best and final checkpoints are behaviorally identical on this eval: all 960
generated samples match byte for byte between the two after-evals (both eval
phases genuinely ran, ~78 min each). By step 234 the cosine schedule had
decayed the learning rate to ~1e-13, so the last six steps barely moved the
weights and the seeded eval reproduces the same generations.

Solved tasks are 5/30, all in the easy band (50% of easy tasks at every
pass@k); medium and hard stay at 0% pass despite ~99% valid expressions, which
matches the expectation that 214M does not master exact-solve.

## Training-Shape Verdict

Success criteria checked against
`runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/metrics.jsonl` (240 train
records, steps 1–240):

| Criterion | Verdict | Evidence |
| --- | --- | --- |
| Reward EMA non-decreasing over >= 70% of steps | **Borderline** | EMA(alpha 0.05): 70.7% of transitions non-decreasing (pass); EMA(alpha 0.1): 64.0% (miss). EMA(0.1) went 0.03 → peak 0.55 at step 234 → 0.51 at end; a sustained climb, but per-step noise keeps the strict transition count near the bar. |
| `frac_zero_variance` < 0.5 throughout | **Fail on the letter, pass on the intent** | `frac_zero_variance_groups` reaches >= 0.5 on 71/240 steps (first at step 101, max 0.875); the post-resample `..._sampled` variant on 125/240 (max 1.0). But every one of the 502 zero-variance groups in rollouts.jsonl is an all-valid tie (reward >= 0.3) or all-exact tie (reward 1.0); zero groups tied at reward 0. The criterion was written to catch an all-fail death mode, which never appeared; late-run ties are the model succeeding uniformly, which starves the gradient but does not kill the run. |
| No unrecovered > 0.3 single-step reward drop | **Pass** | Largest single-step drop is 0.236 (step 85); no drop exceeds 0.3. |
| Final checkpoint within 20% of best | **Pass (on eval); train-metric caveat** | The final checkpoint's 30x32 eval is identical to the best checkpoint's (all 960 samples byte-identical). On the noisy per-step train metric, the final logged step's reward_mean (0.418) is 58.6% of the best step's (0.714); the last-20-step mean is 0.511 (71.6% of best). The checkpoint that ships is indistinguishable from best where it counts. |

Supporting facts (verified from artifacts, not assumed):

- `reward_mean == 0` on 0 of 240 steps.
- `valid_expression_rate` at step 240: 0.969 (mean of last 10 steps 0.967);
  `invalid_rate` 0.0. Eval-side valid rate 99.38%.
- Token entropy declined smoothly 2.02 → 0.35 with only 5 single-step
  increases > 0.1 — controlled convergence, also the cause of the late-run
  zero-variance ties.
- Replay buffer filled to 295 cached successes with **zero** injections.
  All-failed groups did occur — 113, all in steps 1–42 — but none of those
  tasks had a fresh (<= 40-step-old) prior success the buffer could inject, so
  the ARPO path correctly stayed silent; after step 42 no group was ever
  all-failed.
- Zero-variance resampling was active: 776 resamples, 502 cap hits
  (`zero_variance_max_resamples: 1`), concentrated late-run where success ties
  dominate.
- `exact_solve_rate` in training rose from ~0.01 (first 20 steps) to ~0.16
  (last 20), peaking at 0.48.

## Preemption Incident

The detached Modal run (`ap-6kXrTZLCQm3jexY4MPkkPk`, spawned call
`fc-01KWGWE0PTCBSVK3REEDYMJPEJ`) executed twice:

- **First attempt** (`esme-214m-rlvr-countdown-grpo-v2-ccb6287`, W&B `39k48a1x`):
  completed the before-eval and all 240 training steps, wrote
  `best-checkpoint.json`, then died mid-after-eval — last progress record is
  sample 352/960, task `countdown_lite_eval_medium_0000`, at monotonic
  ~6,507 s. Forensics verdict: Modal infrastructure preemption (SIGTERM; W&B
  closed the run as "finished"; the launcher configures no Modal retries, so a
  code exception could not have restarted it; zero GPU memory errors).
- **Second attempt** (`...-ccb6287-1`, W&B `qfkygv2y`): Modal's infra requeue of
  the same call. Full clean lifecycle: before-eval, 240/240 steps, after-eval
  on the best checkpoint, after-eval on the final checkpoint,
  `return_serialization` at monotonic ~15,790 s. **This is the canonical run.**

The first attempt's `metrics.jsonl` is byte-identical to the second's (same seed
214, deterministic trainer — the requeue replayed the same trajectory), its
`eval-before.json` is byte-identical too, and its volume dir has no
`cost.json`/`eval-after.json` (the process died before writing them). The app
is stopped with 0 running tasks. Spend impact: the preemption cost roughly one
duplicated training pass, because the Modal function restarts from zero instead
of resuming from the volume artifacts it had already written.

## Spend

| Item | Basis | Amount |
| --- | ---: | ---: |
| Canonical attempt | `cost.json`: 15,780.8 s @ $2.0988/h, status `complete` | $9.20 |
| Preempted attempt | estimated: ~6,510 s @ $2.0988/h (no cost.json written) | ~$3.79 |
| **Total** | | **~$12.99** |

Against the $20 hard cap (runtime stop $18, launch projection $13.84): under cap
even after paying for the duplicated pass. No new compute was spent on the
close-out; all artifacts were fetched with read-only volume gets.

## Identifiers

- Modal app: `ap-6kXrTZLCQm3jexY4MPkkPk` (stopped)
- Modal function call: `fc-01KWGWE0PTCBSVK3REEDYMJPEJ`
- W&B project: `esme-posttrain`
  - Preempted attempt: `39k48a1x` — https://wandb.ai/adam-thuvesen-mentimeter/esme-posttrain/runs/39k48a1x
  - Canonical attempt: `qfkygv2y` — https://wandb.ai/adam-thuvesen-mentimeter/esme-posttrain/runs/qfkygv2y
- Volume dirs (`esme-posttrain-esme-rlvr-countdown`):
  `esme-214m-rlvr-countdown-grpo-v2-ccb6287` (preempted),
  `esme-214m-rlvr-countdown-grpo-v2-ccb6287-1` (canonical)
- Best checkpoint: step 234, `train/reward_mean` 0.7142
  (`best-checkpoint.pt`, sha256 `58a48b58…df4bf808` per `manifest.json`)

## Follow-Up Recommendation

Make the Modal GRPO function resume from existing volume artifacts instead of
restarting from zero. The run dir already carries everything a requeue needs:
`eval-before/baseline-partial.jsonl` and `eval-progress.jsonl` (skip or resume
completed eval phases), `checkpoints/` and `best-checkpoint.pt` (resume training
at the last checkpoint), and `metrics.jsonl` (append, not truncate). The
preempted attempt died 352 samples into an after-eval it never got credit for;
a resume path would have turned the requeue into ~75 minutes of remaining eval
instead of a full 4.4-hour rerun, roughly the $3.79 the preemption cost.
Recommendation only — not implemented.
