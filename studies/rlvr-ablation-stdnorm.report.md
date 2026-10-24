<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# GRPO stability ablation: std-normalized advantages

Machine-readable report: [rlvr-ablation-stdnorm.report.json](rlvr-ablation-stdnorm.report.json)

**Verdict:** rejected

The accepted GRPO recipe outperforms the same recipe with std-normalized advantages on 200 fresh held-out Countdown tasks at training step 240.

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `seed_problem_bootstrap_95`
- Sample budget per run: 3200
- Rejection reason: 95% confidence interval lower bound does not exceed 0.0
- Evidence warning: full_recipe seed 215 has no training manifest
- Evidence warning: full_recipe seed 216 has no training manifest
- Evidence warning: full_recipe seed 217 has no training manifest
- Evidence warning: ablation_stdnorm seed 215 has no training manifest
- Evidence warning: ablation_stdnorm seed 216 has no training manifest
- Evidence warning: ablation_stdnorm seed 217 has no training manifest

## Per-seed results

### full_recipe (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 98.0% | 6.3% | 13/200 | 0.9999 |
| 216 | 98.2% | 6.3% | 13/200 | 1.1223 |
| 217 | 97.8% | 6.2% | 13/200 | 0.9952 |

[Table data](rlvr-ablation-stdnorm.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-stdnorm.report.json)

### ablation_stdnorm (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 97.9% | 6.3% | 13/200 | 0.9712 |
| 216 | 98.0% | 6.2% | 13/200 | 1.0054 |
| 217 | 97.8% | 6.3% | 13/200 | 1.0122 |

[Table data](rlvr-ablation-stdnorm.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-stdnorm.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 1.4% | 0.1% | 3/200 | n/a |

[Table data](rlvr-ablation-stdnorm.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a | n/a |

[Table data](rlvr-ablation-stdnorm.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| full_recipe | treatment | 98.0% | 6.2% | 6.5% | 3.1174 |
| ablation_stdnorm | control | 97.9% | 6.3% | 6.5% | 2.9888 |
| chat_baseline | baseline | 1.4% | 0.1% | 1.5% | 0.0000 |

[Table data](rlvr-ablation-stdnorm.report.json)

## Planned comparisons

### full-vs-ablation-validity

- Metric: `valid_expression_rate`
- Mean effect: +0.1%
- 95% CI: [-0.2%, +0.4%]
- Sign-flip permutation p (two-sided): 0.5000 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-stdnorm.report.json)

### full-vs-ablation-exact-sample

- Metric: `exact_solve_rate`
- Mean effect: -0.0%
- 95% CI: [-0.1%, +0.1%]
- Sign-flip permutation p (two-sided): 1.0000 (smallest possible with these seeds: 0.5000)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-stdnorm.report.json)

### full-vs-ablation-exact

- Metric: `any_exact_solve_rate`
- Mean effect: +0.0%
- 95% CI: [+0.0%, +0.0%]
- Sign-flip permutation p (two-sided): 1.0000 (smallest possible with these seeds: 1.0000)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-stdnorm.report.json)


## Supported claims

No claims are accepted from this report.

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
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed215-step240/completions.jsonl` — `ba4a7529dd4b6927c7e817dd926e2513dcd7d872bbc1586b3246f2477fcb5bea`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed215-step240/provenance.json` — `9cb9d3bf5112ec28afa6ef55167d519fd6a4adadd6510372812f82af93759f32`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed215-step240/bundle-manifest.json` — `3211cf2cbcbe7cb7588d0fc264199007a13abbe6d032ee7f91aa4f562487142e`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-stdnorm-seed215-grpo.json` — `124f6e4d9b4a5a7f01c6a7ff25deff5038ab5a853902279eec6c182bb3f8e32e`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed216-step240/completions.jsonl` — `904900f3f6228f90aa72685fb782fbde60a4d9a67666e6aea76ac911e7f8510d`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed216-step240/provenance.json` — `60b4b4d765e26b482b7a0e258f14aaf65dcb119f9cc28ca210d1ee269c5ac746`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed216-step240/bundle-manifest.json` — `df560ed56dda8fa6e3493c0c54c3640dff5b19922b47a05cce2ea84eecd908d9`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-stdnorm-seed216-grpo.json` — `e8510c94c272501b6c032a134b093a765eff17a7afa9a645c6a9356b7ccc024e`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed217-step240/completions.jsonl` — `b0c03a65b8e420c9082a5ad5403171c8f9da683303dc242eb2c08f8883b3b01f`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed217-step240/provenance.json` — `26cc3e4bcf45372f0289efba164563733ecd831b91345128ef4c649a2376a72e`
- `runs/gpu-emit/batch1/heldout_large-stdnorm-seed217-step240/bundle-manifest.json` — `cfebe5ce5e3ba29d45d42ad07eb4a906ecff0730bd7226b061640e2fd6cccef0`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-stdnorm-seed217-grpo.json` — `6b1a0ddc752d19090753d463a4ac0d7163a52c227eea3dff9050e856830c43ad`
- `runs/gpu-emit/batch1/heldout_large-chat/completions.jsonl` — `d58c2a1aef5114e2f577ddeac59a9d4a598a1818af16a1941c488bd05e2f7529`
- `runs/gpu-emit/batch1/heldout_large-chat/provenance.json` — `3e2546d1e854e702a6284030cf2e062ee98527f1f286f1e6849873b45569e8d4`
