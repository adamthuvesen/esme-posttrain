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

- `seed_problem_bootstrap_95`: a two-level percentile bootstrap. Each resample draws
  training seeds with replacement, then draws problems with replacement inside each
  drawn seed, using the same problems for both arms. The interval is the 2.5th and
  97.5th percentile of 10,000 resampled mean effects, with a fixed resampling seed.
  A rate difference can never leave `[-1, 1]`. A paired Student t interval did leave
  it: with six seeds and one outlier, it gave an upper bound of `+114.7%`.
- An exact two-sided sign-flip permutation p-value over the seed-level paired effects.
  With `n` seeds, the smallest possible p is `2 / 2**n`, which is `0.031` for six
  seeds. The Markdown report prints that floor beside the p-value.

A percentile bootstrap over six seeds tends to give an interval that is too narrow.
The permutation p-value makes no distributional assumption, so read the two together.
The acceptance rule still gates on the interval's lower bound.

A run may also pin `evaluated_bundle_manifest`: the manifest of the bundle its
completions were sampled from. Use this when that bundle is not the run's selected
bundle. The report shows both the step the training run selected and the step that was
evaluated. Without it, the evaluated step is the selected step.

`studies/rlvr-placebo-step240.json` uses this to evaluate every arm at training step
240. GRPO exports only the bundle with the best `train/reward_mean`. Under a random
reward that choice is noise, so the six-seed placebo arms were evaluated at steps 8 to
219, not at the end of the training budget. `rlvr-checkpoint-bundle` rebuilds a bundle
from the periodic training checkpoint at any step:

```bash
uv run esme-posttrain rlvr-checkpoint-bundle \
  --checkpoint <run>/checkpoints/step-000240/checkpoint.pt \
  --template-bundle <run>/bundle --out <bundle-dir> --expected-step 240
```

The Countdown CompletionSet scorer matches the decomposition contract: it reads the
expression inside `\boxed{...}`, rejects decimal tokens with leading zeroes, and then
applies the in-repo Countdown number-use and arithmetic verifier. This detail is pinned
by a fixture because accepting `06` as the supplied number `6` changes the checked
placebo table.

The library entry point is
`esme_posttrain.studies.report.generate_study_report`. The CLI exposes it as:

```bash
uv run esme-posttrain study-report --study studies/rlvr-placebo.json
```

By default the two outputs sit beside the specification as `<name>.report.json` and
`<name>.report.md`. Markdown reports carry a generated-file notice and link to the JSON
report. Edit the specification or source artifacts, then rebuild; do not hand-edit a
generated report.

The accepted six-seed report is checked in, while its source CompletionSets and
training receipts remain under gitignored `runs/`. Rebuilding that real report needs
those hash-matched local run artifacts; the compact `fixtures/studies/` report is the
clean-checkout contract test.

The real specification also hashes a committed projection of
`grpo-decomp/results/esme-countdown/sampled_multiseed_summary.json`. A seed, arm,
aggregate, or confidence-interval drift rejects the report instead of silently
replacing the retrospective's checked result.
