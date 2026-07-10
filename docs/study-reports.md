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

The current confidence interval method is `paired_student_t_95`: a two-sided 95%
Student t interval over seed-level paired effects. The training seed is the unit of
analysis. This matches the six-seed RLVR placebo study and avoids pretending that the
samples generated within one training run are independent model replications.

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
