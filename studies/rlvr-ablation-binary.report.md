<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# GRPO stability ablation: a binary exact-only reward

Machine-readable report: [rlvr-ablation-binary.report.json](rlvr-ablation-binary.report.json)

**Verdict:** accepted

The accepted GRPO recipe outperforms the same recipe with a binary exact-only reward on 200 fresh held-out Countdown tasks at training step 240.

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `seed_problem_bootstrap_95`
- Sample budget per run: 3200
- Evidence warning: full_recipe seed 215 has no training manifest
- Evidence warning: full_recipe seed 216 has no training manifest
- Evidence warning: full_recipe seed 217 has no training manifest
- Evidence warning: ablation_binary seed 215 has no training manifest
- Evidence warning: ablation_binary seed 216 has no training manifest
- Evidence warning: ablation_binary seed 217 has no training manifest

## Per-seed results

### full_recipe (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 98.0% | 6.3% | 13/200 | 0.9999 |
| 216 | 98.2% | 6.3% | 13/200 | 1.1223 |
| 217 | 97.8% | 6.2% | 13/200 | 0.9952 |

[Table data](rlvr-ablation-binary.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-binary.report.json)

### ablation_binary (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 215 | 59.5% | 3.7% | 28/200 | 1.1788 |
| 216 | 57.1% | 3.8% | 30/200 | 1.1769 |
| 217 | 60.9% | 3.9% | 27/200 | 1.2004 |

[Table data](rlvr-ablation-binary.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-ablation-binary.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 1.4% | 0.1% | 3/200 | n/a |

[Table data](rlvr-ablation-binary.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a | n/a |

[Table data](rlvr-ablation-binary.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| full_recipe | treatment | 98.0% | 6.2% | 6.5% | 3.1174 |
| ablation_binary | control | 59.2% | 3.8% | 14.2% | 3.5561 |
| chat_baseline | baseline | 1.4% | 0.1% | 1.5% | 0.0000 |

[Table data](rlvr-ablation-binary.report.json)

## Planned comparisons

### full-vs-ablation-validity

- Metric: `valid_expression_rate`
- Mean effect: +38.8%
- 95% CI: [+35.2%, +42.7%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-binary.report.json)

### full-vs-ablation-exact-sample

- Metric: `exact_solve_rate`
- Mean effect: +2.4%
- 95% CI: [+0.6%, +4.6%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-binary.report.json)

### full-vs-ablation-exact

- Metric: `any_exact_solve_rate`
- Mean effect: -7.7%
- 95% CI: [-11.7%, -4.0%]
- Sign-flip permutation p (two-sided): 0.2500 (smallest possible with these seeds: 0.2500)
- Paired seeds: 215, 216, 217
- [Comparison data](rlvr-ablation-binary.report.json)


## Supported claims

- Removing the graded reward (binary exact-only reward) lowers held-out validity at training step 240.
- The binary-reward arm solves more distinct held-out problems than the full recipe at training step 240.

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
- `runs/gpu-emit/batch1/heldout_large-binary-seed215-step240/completions.jsonl` — `5d4d0f0084bc9739b0203c935385bddac84bee9a7e342add5dc13116fff75cfc`
- `runs/gpu-emit/batch1/heldout_large-binary-seed215-step240/provenance.json` — `5a51aedeecdedf6a4b2271cc2c62db53f2f54e42aece0ccc9a12fded7e61461c`
- `runs/gpu-emit/batch1/heldout_large-binary-seed215-step240/bundle-manifest.json` — `851b9b52e52d9edb9c2e4b6fc037f50b98110fe0a94765f09889b4e6c479e50a`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-binary-seed215-grpo.json` — `32965c62447043a245b2c28fdd0f7c6f28699bf43cc6267bfdf0c41cb25b5249`
- `runs/gpu-emit/batch1/heldout_large-binary-seed216-step240/completions.jsonl` — `8d65f52909c67b165b293d12e4dd082a9f6bf5a101772fc95099717e63cf0cc8`
- `runs/gpu-emit/batch1/heldout_large-binary-seed216-step240/provenance.json` — `71d299ab7272f8b178056541cc17027be64a81066da686f4cdeabb5bebaa052e`
- `runs/gpu-emit/batch1/heldout_large-binary-seed216-step240/bundle-manifest.json` — `a91392617c2542a4e8f7387c1ca23665a668cc39536059d71f36c889cc25f8e5`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-binary-seed216-grpo.json` — `847f18b07a3228a9cd21a4ccdd720b36fb751ea742757b2d8c3aca4da6ed6036`
- `runs/gpu-emit/batch1/heldout_large-binary-seed217-step240/completions.jsonl` — `9d0c52cd5fb6fbc12f12c173b86729ca7e8bad588644be206cbccac26a898ce2`
- `runs/gpu-emit/batch1/heldout_large-binary-seed217-step240/provenance.json` — `966ad063b3e88441ad0d6e392c1e7cea1964e6837fbdfd8071c1d9914b3e7c60`
- `runs/gpu-emit/batch1/heldout_large-binary-seed217-step240/bundle-manifest.json` — `518fb1340159b4aa3faa6c70e5ac4b38ce8270bd9010046941ba31b8e588eac9`
- `runs/ablation-launch/cost/esme-214m-rlvr-abl-binary-seed217-grpo.json` — `2d850e77fd316a036769f95774033e99d4038eb1780699c54846e8e231b193e4`
- `runs/gpu-emit/batch1/heldout_large-chat/completions.jsonl` — `d58c2a1aef5114e2f577ddeac59a9d4a598a1818af16a1941c488bd05e2f7529`
- `runs/gpu-emit/batch1/heldout_large-chat/provenance.json` — `3e2546d1e854e702a6284030cf2e062ee98527f1f286f1e6849873b45569e8d4`
