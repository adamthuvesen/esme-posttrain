<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# GRPO reward ablation: closeness-only partial credit

Machine-readable report: [rlvr-ablation-closeness.report.json](rlvr-ablation-closeness.report.json)

**Verdict:** accepted

The accepted GRPO recipe outperforms the same recipe with closeness-only partial credit (no flat valid or format credit) on 200 fresh held-out Countdown tasks at training step 240.

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `seed_problem_bootstrap_95`
- Sample budget per run: 3200
- Evidence warning: full_recipe seed 215 has no training manifest
- Evidence warning: full_recipe seed 216 has no training manifest
- Evidence warning: full_recipe seed 217 has no training manifest
- Evidence warning: ablation_closeness seed 215 has no training manifest
- Evidence warning: ablation_closeness seed 216 has no training manifest
- Evidence warning: ablation_closeness seed 217 has no training manifest

## Per-seed results

### full_recipe (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 98.0% | 6.3% | 13/200 | 0.9999 |
| 216 | 98.2% | 6.3% | 13/200 | 1.1223 |
| 217 | 97.8% | 6.2% | 13/200 | 0.9952 |

[Table data](rlvr-ablation-closeness.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-closeness.report.json)

### ablation_closeness (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 95.3% | 5.9% | 14/200 | 0.9635 |
| 216 | 95.4% | 5.9% | 14/200 | 0.9813 |
| 217 | 94.4% | 5.8% | 14/200 | 0.9700 |

[Table data](rlvr-ablation-closeness.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-closeness.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 1.4% | 0.1% | 3/200 | n/a |

[Table data](rlvr-ablation-closeness.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a | n/a |

[Table data](rlvr-ablation-closeness.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| full_recipe | treatment | 98.0% | 6.2% | 6.5% | 3.1174 |
| ablation_closeness | control | 95.1% | 5.9% | 7.0% | 2.9148 |
| chat_baseline | baseline | 1.4% | 0.1% | 1.5% | 0.0000 |

[Table data](rlvr-ablation-closeness.report.json)

## Planned comparisons

### full-vs-ablation-validity

- Metric: `valid_expression_rate`
- Mean effect: +2.9%
- 95% CI: [+2.1%, +3.8%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-closeness.report.json)

### full-vs-ablation-exact-sample

- Metric: `exact_solve_rate`
- Mean effect: +0.4%
- 95% CI: [+0.0%, +0.9%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-closeness.report.json)

### full-vs-ablation-exact

- Metric: `any_exact_solve_rate`
- Mean effect: -0.5%
- 95% CI: [-1.5%, +0.0%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-closeness.report.json)


## Supported claims

- Removing the flat valid-expression credit (closeness-only partial credit) lowers held-out validity at training step 240.

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
- `runs/gpu-emit/batch1/heldout_large-closeness-seed215-step240/completions.jsonl` — `f2c38ae0d4b4b4c75fd914362f946ae36973792e91c11cb0f5eba30a89ac2897`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed215-step240/provenance.json` — `6e894973e9289bac79a73b15cef5f1ff7961153e0d5995c872b959013d3f428d`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed215-step240/bundle-manifest.json` — `131cdd40013bab41f4b189664852c24d54edeec526361f4f61654495e011f8db`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-closeness-seed215-grpo.json` — `7e342eec880db9304d23f44dce6899ec9fc839ad0b065179596ed7da401d478b`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed216-step240/completions.jsonl` — `c2c94da976552f3441dd32842ab7219ea66631ec782c12e40e58c47d2a7760be`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed216-step240/provenance.json` — `ac1d665d72fcc6520a6706ab30c0468f175f9ad06ed2b94713480ba2b36b58a3`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed216-step240/bundle-manifest.json` — `ac57f9fc8d5232d816a7dd6684463f6a374b248da4201d78a555720d00f2972d`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-closeness-seed216-grpo.json` — `babdba09d890762c65fe3d26d0c593407b17aba1931729745037d389cfd31585`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed217-step240/completions.jsonl` — `c5db8d33b908b59b56e1bf3d125f967ae05a58134e5c4a237dee82fe400a211c`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed217-step240/provenance.json` — `0fd6c319d3d46efa70d017b4c1ec799699857fab371e54ee5233f19f10f16839`
- `runs/gpu-emit/batch1/heldout_large-closeness-seed217-step240/bundle-manifest.json` — `2d26cb55d120da61621dfa98bf84e608e4da91fbbc3ce9fab0f8ae6a3c5fa274`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-closeness-seed217-grpo.json` — `f94182d89e6a6b573a23bef430cff5a262e600026928bf09b2d9eb580061b117`
- `runs/gpu-emit/batch1/heldout_large-chat/completions.jsonl` — `d58c2a1aef5114e2f577ddeac59a9d4a598a1818af16a1941c488bd05e2f7529`
- `runs/gpu-emit/batch1/heldout_large-chat/provenance.json` — `3e2546d1e854e702a6284030cf2e062ee98527f1f286f1e6849873b45569e8d4`
