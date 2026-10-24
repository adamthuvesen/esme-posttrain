# Study reports

Study reports turn immutable completion artifacts into checked JSON and generated
Markdown. A study specification fixes the hypothesis, arms, seeds, task-manifest
identity, sample and decoding budgets, planned comparisons, confidence interval,
acceptance rule, and the claims an accepted result may support.

Each artifact reference includes its SHA-256 digest. Report generation stops on a
missing or changed artifact. It also compares task and decoding provenance across
arms. Missing treatment or control seeds, mismatched task manifests, different sample
counts, or different decoding settings force a rejected verdict with no supported
claims.

The training seed is the unit of analysis. Samples generated within one training run
are not independent model replications, so each comparison pairs the arms by seed and
reports two numbers:

- `seed_problem_bootstrap_95`: a two-way percentile bootstrap. Seeds and problems are
  crossed, because every seed is scored on the same problems. Each resample draws
  training seeds with replacement and, independently, one set of problems with
  replacement. That problem set is shared by every drawn seed and both arms. The
  interval is the 2.5th and 97.5th percentile of 10,000 resampled mean effects, with
  a fixed resampling seed. Unlike a paired Student t interval, it can never leave the
  `[-1, 1]` range of a rate difference.
- An exact two-sided sign-flip permutation p-value over the seed-level paired effects.
  With `n` seeds whose effects are nonzero, the smallest possible p is `2 / 2**n`,
  which is `0.031` for six seeds. The Markdown report prints that floor beside the
  p-value.

The two numbers answer different questions. The permutation p asks whether the effect
holds across training seeds on these exact problems. The bootstrap interval also asks
whether it would hold on a fresh draw of problems. When an effect rests on a few
problems, the p can be at its floor while the interval reaches zero. A percentile
bootstrap over six seeds also tends to give an interval that is too narrow. The
acceptance rule gates on the interval's lower bound.

A run may also pin `evaluated_bundle_manifest`: the manifest of the bundle its
completions were sampled from. The report then checks that the bundle's weights hash
matches the `model_revision` in the CompletionSet provenance, and shows the evaluated
training step. Every committed study pins it, because every arm is evaluated at one
fixed step. GRPO exports the bundle with the best `train/reward_mean`, and under a
random reward that choice is noise. `rlvr-checkpoint-bundle` rebuilds a bundle from the
periodic training checkpoint at any step:

```bash
uv run esme-posttrain rlvr-checkpoint-bundle \
  --checkpoint <run>/checkpoints/step-000240/checkpoint.pt \
  --template-bundle <run>/bundle --out <bundle-dir> --expected-step 240
```

The Countdown CompletionSet scorer matches the decomposition contract: it reads the
expression inside `\boxed{...}`, rejects decimal tokens with leading zeroes, and then
applies the in-repo Countdown number-use and arithmetic verifier. This detail is pinned
by a fixture because accepting `06` as the supplied number `6` changes the scores.

The library entry point is
`esme_posttrain.studies.report.generate_study_report`. The CLI exposes it as:

```bash
uv run esme-posttrain study-report --study studies/rlvr-placebo-step240-large.json
```

By default the two outputs sit beside the specification as `<name>.report.json` and
`<name>.report.md`. Markdown reports carry a generated-file notice and link to the JSON
report. Edit the specification or source artifacts, then rebuild; do not hand-edit a
generated report.

The study specifications and reports under `studies/` are checked in. Their source
CompletionSets and training receipts stay under gitignored `runs/`, so rebuilding a
real report needs those hash-matched local files. A unit test checks that each
committed report matches the hash of its specification. The compact
`fixtures/studies/` report is the clean-checkout contract test.

A specification may also pin `reference_summary`: a hashed projection of a result
computed elsewhere. Any seed, arm, or mean-effect drift from it rejects the report.
