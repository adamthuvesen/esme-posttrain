<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# GRPO stability ablation: no zero-variance resampling, replay buffer, or stratified sampling

Machine-readable report: [rlvr-ablation-nostab.report.json](rlvr-ablation-nostab.report.json)

**Verdict:** accepted

The accepted GRPO recipe outperforms the same recipe with no zero-variance resampling, replay buffer, or stratified sampling on 200 fresh held-out Countdown tasks at training step 240.

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `seed_problem_bootstrap_95`
- Sample budget per run: 3200
- Evidence warning: full_recipe seed 215 has no training manifest
- Evidence warning: full_recipe seed 216 has no training manifest
- Evidence warning: full_recipe seed 217 has no training manifest
- Evidence warning: ablation_nostab seed 215 has no training manifest
- Evidence warning: ablation_nostab seed 216 has no training manifest
- Evidence warning: ablation_nostab seed 217 has no training manifest

## Per-seed results

### full_recipe (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 98.0% | 6.3% | 13/200 | 0.9999 |
| 216 | 98.2% | 6.3% | 13/200 | 1.1223 |
| 217 | 97.8% | 6.2% | 13/200 | 0.9952 |

[Table data](rlvr-ablation-nostab.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-nostab.report.json)

### ablation_nostab (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 97.4% | 6.2% | 13/200 | 0.7331 |
| 216 | 97.2% | 6.2% | 14/200 | 0.8118 |
| 217 | 97.4% | 6.2% | 13/200 | 0.7464 |

[Table data](rlvr-ablation-nostab.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-nostab.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 1.4% | 0.1% | 3/200 | n/a |

[Table data](rlvr-ablation-nostab.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a | n/a |

[Table data](rlvr-ablation-nostab.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| full_recipe | treatment | 98.0% | 6.2% | 6.5% | 3.1174 |
| ablation_nostab | control | 97.3% | 6.2% | 6.7% | 2.2913 |
| chat_baseline | baseline | 1.4% | 0.1% | 1.5% | 0.0000 |

[Table data](rlvr-ablation-nostab.report.json)

## Planned comparisons

### full-vs-ablation-validity

- Metric: `valid_expression_rate`
- Mean effect: +0.7%
- 95% CI: [+0.3%, +1.2%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-nostab.report.json)

### full-vs-ablation-exact-sample

- Metric: `exact_solve_rate`
- Mean effect: +0.1%
- 95% CI: [-0.0%, +0.2%]
- Sign-flip permutation p (two-sided): 0.5000 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-nostab.report.json)

### full-vs-ablation-exact

- Metric: `any_exact_solve_rate`
- Mean effect: -0.2%
- 95% CI: [-1.0%, +0.0%]
- Sign-flip permutation p (two-sided): 1.0000 (smallest possible with these seeds: 1.0000)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-nostab.report.json)


## Supported claims

- Removing zero-variance resampling, the replay buffer, and stratified sampling changes held-out validity by less than 1.5 percentage points at training step 240.

## Excluded runs

None.

## Artifact provenance

- `runs/gpu-emit/batch1/heldout_large-real-seed215-step240/completions.jsonl` — `42ac3c982c72ab221869adaceea84662b3d5eb4399393e3af1e97c991b6e9b73`
- `runs/gpu-emit/batch1/heldout_large-real-seed215-step240/provenance.json` — `274c6f95237b08a8df7874eae9096f8af588e83ebaa225f90da908af5d7e71cc`
- `runs/gpu-emit/batch1/heldout_large-real-seed215-step240/bundle-manifest.json` — `a73aa795812b561c4ac1bffba7a403def30896d88918e10e94a8a12bc657c300`
- `runs/multiseed/seed215-real/cost.json` — `711b3145cdc93181c05ea0333f4d2b715faa83583e6753de5c5382a546f147b0`
- `runs/gpu-emit/batch1/heldout_large-real-seed216-step240/completions.jsonl` — `9d27b945d4c920e2cb954a67389dcc9eaad17b634e351294fab6226ef415ac81`
- `runs/gpu-emit/batch1/heldout_large-real-seed216-step240/provenance.json` — `b9cb746f2e2414bab097c41a4b7bbeb19065c8f3b1d806526b05a5f2e0247e8d`
- `runs/gpu-emit/batch1/heldout_large-real-seed216-step240/bundle-manifest.json` — `437f18283999bf4d8c9ff427b23187fe11c58cc04cc71b18597fb9c9df9988d7`
- `runs/multiseed/seed216-real/cost.json` — `c636fc7a3fc344ec5811d2861a42885b7f6b27d01cca6ecd42d32ce9bbd1fd15`
- `runs/gpu-emit/batch1/heldout_large-real-seed217-step240/completions.jsonl` — `7c9315b9b8d8265f37653ec1670fa8233364a2d9a105b0b6e6180b4732849643`
- `runs/gpu-emit/batch1/heldout_large-real-seed217-step240/provenance.json` — `ab1511f17ff72cf8956262952142fb23a2a761402414ba2e971020c9f2e28539`
- `runs/gpu-emit/batch1/heldout_large-real-seed217-step240/bundle-manifest.json` — `88ae6d58011bb77690b0090923f7a1ca52f1ff9c9833ad25f68a133968e0c87b`
- `runs/multiseed/seed217-real/cost.json` — `07762084cde98585028f86e0329b0a7c3dcb7421bf43bad2e8b05b8d31444889`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed215-step240/completions.jsonl` — `433af6ef681ea81df2249ca7ed907cf61f1f240c9aa5c3ecc2842b47f9b9298e`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed215-step240/provenance.json` — `2433e20d24403478894e355c6f5e10f2e882f4a87aa9ea24479981df50eabc8d`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed215-step240/bundle-manifest.json` — `8002e1aa05c6a95275be06dded4fe0b636fe9b7b9fe9eee75becbbdfbabacddb`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-nostab-seed215-grpo.json` — `b2b038c0aa547a63bcb0a071a5717322d18573632a6a27e1c31e4b4ab293a1bb`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed216-step240/completions.jsonl` — `8d76c9b383bdea7c9e783198e77cbe34947294b2263fd046233f9dde3da5e175`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed216-step240/provenance.json` — `a52dfe638496beb2714a00746c23eb0d12b1e9b38058cc07811ec6c2f0ee59ca`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed216-step240/bundle-manifest.json` — `b01df0f984ba97805b58c8159dc42fa7d2b84add4bc07b7b1554f9de3294fe6d`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-nostab-seed216-grpo.json` — `d9fe0865f9b4eb99030729d328fd4c9b0fa15d4db6695c914128f1c11a242337`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed217-step240/completions.jsonl` — `cba88d76b7fd31f12a90ffb7ca8c5a42a9bbf96c49bcf9f7cb2db464229316a7`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed217-step240/provenance.json` — `79422855035134cde6f258f2841bf2d39327dc232daa3ca51054e0aa551e2ccd`
- `runs/gpu-emit/batch1/heldout_large-nostab-seed217-step240/bundle-manifest.json` — `99babc6049453619ed3a21b23a6d1c393ee3a341b115eff576482899e1b0cf3f`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-nostab-seed217-grpo.json` — `4a25c977105f7a879c6eb8d38f0d7c1e60d7058dc7a9efcceb1c2f7c920fb729`
- `runs/gpu-emit/batch1/heldout_large-chat/completions.jsonl` — `d58c2a1aef5114e2f577ddeac59a9d4a598a1818af16a1941c488bd05e2f7529`
- `runs/gpu-emit/batch1/heldout_large-chat/provenance.json` — `3e2546d1e854e702a6284030cf2e062ee98527f1f286f1e6849873b45569e8d4`
