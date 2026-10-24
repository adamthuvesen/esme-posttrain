<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# RLVR Countdown-Lite placebo study at training step 240

Machine-readable report: [rlvr-placebo-step240.report.json](rlvr-placebo-step240.report.json)

**Verdict:** accepted

Verifier-reward GRPO improves held-out Countdown validity and exact solving beyond a random-reward placebo when every arm is evaluated at the same training step (240).

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `seed_problem_bootstrap_95`
- Sample budget per run: 480
- Evidence warning: real_reward seed 214 has no comparable cost artifact
- Evidence warning: random_reward seed 214 has no training manifest
- Evidence warning: random_reward seed 214 has no comparable cost artifact

## Per-seed results

### real_reward (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 97.1% | 9.6% | 3/30 | n/a |
| 215 | 97.7% | 9.8% | 3/30 | 0.9999 |
| 216 | 97.1% | 9.8% | 3/30 | 1.1223 |
| 217 | 97.1% | 9.6% | 3/30 | 0.9952 |
| 218 | 97.1% | 9.6% | 3/30 | 1.0612 |
| 219 | 96.9% | 9.8% | 3/30 | 0.9910 |

[Table data](rlvr-placebo-step240.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | 234 | train/reward_mean=0.7142 | 240 | 5300000 |
| 215 | complete | 234 | train/reward_mean=0.7211 | 240 | 5300000 |
| 216 | complete | 234 | train/reward_mean=0.7166 | 240 | 5300000 |
| 217 | complete | 234 | train/reward_mean=0.7250 | 240 | 5300000 |
| 218 | complete | 159 | train/reward_mean=0.7109 | 240 | 5300000 |
| 219 | complete | 234 | train/reward_mean=0.7059 | 240 | 5300000 |

[Table data](rlvr-placebo-step240.report.json)

### random_reward (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 0.6% | 0.0% | 0/30 | n/a |
| 215 | 0.6% | 0.0% | 0/30 | 0.7507 |
| 216 | 0.8% | 0.2% | 1/30 | 0.7447 |
| 217 | 0.6% | 0.0% | 0/30 | 0.7547 |
| 218 | 1.7% | 0.6% | 2/30 | 0.7908 |
| 219 | 1.2% | 0.4% | 2/30 | 0.7473 |

[Table data](rlvr-placebo-step240.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | 240 | 5300000 |
| 215 | complete | 49 | train/reward_mean=0.5242 | 240 | 5300000 |
| 216 | complete | 219 | train/reward_mean=0.5219 | 240 | 5300000 |
| 217 | complete | 112 | train/reward_mean=0.5398 | 240 | 5300000 |
| 218 | complete | 8 | train/reward_mean=0.5297 | 240 | 5300000 |
| 219 | complete | 162 | train/reward_mean=0.5375 | 240 | 5300000 |

[Table data](rlvr-placebo-step240.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 0.8% | 0.2% | 1/30 | n/a |

[Table data](rlvr-placebo-step240.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a | n/a |

[Table data](rlvr-placebo-step240.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| real_reward | treatment | 97.2% | 9.7% | 10.0% | 5.1697 |
| random_reward | control | 0.9% | 0.2% | 2.8% | 3.7883 |
| chat_baseline | baseline | 0.8% | 0.2% | 3.3% | 0.0000 |

[Table data](rlvr-placebo-step240.report.json)

## Planned comparisons

### real-vs-random-validity

- Metric: `valid_expression_rate`
- Mean effect: +96.2%
- 95% CI: [+95.3%, +97.0%]
- Sign-flip permutation p (two-sided): 0.0312 (smallest possible with these seeds: 0.0312)
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo-step240.report.json)

### real-vs-random-exact

- Metric: `any_exact_solve_rate`
- Mean effect: +7.2%
- 95% CI: [+3.3%, +11.7%]
- Sign-flip permutation p (two-sided): 0.0312 (smallest possible with these seeds: 0.0312)
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo-step240.report.json)

### real-vs-baseline-validity

- Metric: `valid_expression_rate`
- Mean effect: +96.2%
- 95% CI: n/a
- Sign-flip permutation p (two-sided): n/a
- Paired seeds: 214
- [Comparison data](rlvr-placebo-step240.report.json)


## Supported claims

- Across six training seeds, the verifier-reward arm has a higher held-out Countdown valid-expression rate than the random-reward arm.
- The exact-solve gain is positive but smaller than the valid-expression gain; this study does not support a claim of broad reasoning improvement.

## Excluded runs

- `esme-214m-rlvr-countdown-grpo-v2-ccb6287-attempt1`: Preempted before durable completion and excluded before the multiseed comparison.

## Artifact provenance

- `runs/fixed-step-240-completions/seed214/correct__esme-countdown/completions.jsonl` — `6fdc30d60eebf1b1d1476ffd18f77e0f66b3be89750b94a65c977b91f734c983`
- `runs/fixed-step-240-completions/seed214/correct__esme-countdown/provenance.json` — `450feed827c4643b6e9e4bed85c3d5d17d77b1827665e5892fc10a7f19c45807`
- `runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/manifest.json` — `e827777155fbf0bb70fca846b7c466258d243c9a4e6f50ba7187558035dd20c6`
- `runs/fixed-step-240/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/bundle/manifest.json` — `c25a576dcdcfd1e6af81a7d3a94d5d48759eb16e1117fde6813e80145bcbd332`
- `runs/fixed-step-240-completions/seed215/correct__esme-countdown/completions.jsonl` — `e65c47aafac954cfe559d7f10db693bdd2a5c3b8be09d970faca7f5c65a3bfa2`
- `runs/fixed-step-240-completions/seed215/correct__esme-countdown/provenance.json` — `7d014e015661665c8a8c6aa711fd3ac187d82417fbec200ddabdc5b8e3b2e029`
- `runs/multiseed/seed215-real/manifest.json` — `4083dd6cb61dbc47c86157329b1ebe207165084146003896c5f6919affc58a57`
- `runs/fixed-step-240/esme-214m-rlvr-real-seed215-grpo/bundle/manifest.json` — `b3a47cf2a41cedbbee74cfbf70f88ce6d2663b72fd54ff42e306bbeb2d3efb1d`
- `runs/multiseed/seed215-real/cost.json` — `711b3145cdc93181c05ea0333f4d2b715faa83583e6753de5c5382a546f147b0`
- `runs/fixed-step-240-completions/seed216/correct__esme-countdown/completions.jsonl` — `65606650c656e78a3f6b2268149eb8405206ecb6bbde6c1bbdd556098793dfa8`
- `runs/fixed-step-240-completions/seed216/correct__esme-countdown/provenance.json` — `e7abb25d0e2284c03b1c9c2e0ecc279edf787a699bafc0f4a03365ae5ab6df00`
- `runs/multiseed/seed216-real/manifest.json` — `dd3387bd4f50fd7320dded21d22666eb550d9dabc50c30101e514646432e4751`
- `runs/fixed-step-240/esme-214m-rlvr-real-seed216-grpo/bundle/manifest.json` — `8bdc59b78a766f1977a2ae4ba6f8578fbcd740cf6419e3b5b7be627c9a0f2c05`
- `runs/multiseed/seed216-real/cost.json` — `c636fc7a3fc344ec5811d2861a42885b7f6b27d01cca6ecd42d32ce9bbd1fd15`
- `runs/fixed-step-240-completions/seed217/correct__esme-countdown/completions.jsonl` — `9fe2cfd460c0b09f2499fe5faef3e96b479ceb64be38baf56875ddf4ab980db4`
- `runs/fixed-step-240-completions/seed217/correct__esme-countdown/provenance.json` — `7b66a3c92381f6d9328dcfaad5ed04bc3fc06c9bd74101045756d220f09dacc5`
- `runs/multiseed/seed217-real/manifest.json` — `5c18747f33386899505625a9637a69651d870b34b03e8569c03045fc529832ba`
- `runs/fixed-step-240/esme-214m-rlvr-real-seed217-grpo/bundle/manifest.json` — `93f5b9089d85a750a59449276c2ae9155e7d9d4337d91c36b420178cf310d4e9`
- `runs/multiseed/seed217-real/cost.json` — `07762084cde98585028f86e0329b0a7c3dcb7421bf43bad2e8b05b8d31444889`
- `runs/fixed-step-240-completions/seed218/correct__esme-countdown/completions.jsonl` — `6fdc30d60eebf1b1d1476ffd18f77e0f66b3be89750b94a65c977b91f734c983`
- `runs/fixed-step-240-completions/seed218/correct__esme-countdown/provenance.json` — `7dd71df96685196b0f982d27ee200114e8535daef52a4af9b4fc550475ad067a`
- `runs/multiseed/seed218-real/manifest.json` — `e7de8bc1780658372e24ec3a141f7654ab8aab5c60940bf648495d48cd808f91`
- `runs/fixed-step-240/esme-214m-rlvr-real-seed218-grpo/bundle/manifest.json` — `35154a593236aa306268f81d4f51109d1a0cad0b2ee8539c9621d8e58c0bc2bf`
- `runs/multiseed/seed218-real/cost.json` — `230a48cc98b7379db0b3b1ea97b558815a922149b31e38dd9345d167533fff72`
- `runs/fixed-step-240-completions/seed219/correct__esme-countdown/completions.jsonl` — `049ca981bc221f2fd4ce05ab6d843c56492e382f4bd7f565bf8a97d3e434b8d7`
- `runs/fixed-step-240-completions/seed219/correct__esme-countdown/provenance.json` — `afb7f364fd7e025bf3594ee1c4cf9f92480638434ecd75ba8c94a958b29ee0f6`
- `runs/multiseed/seed219-real/manifest.json` — `35e8b8104031f42ce6c4ac301ca87ba7b02225e16d8cc93af617c249a036a9a1`
- `runs/fixed-step-240/esme-214m-rlvr-real-seed219-grpo/bundle/manifest.json` — `b3d475f02b941023791b1cb94d241b425135e1fad48ffeba7a608accefeaacae`
- `runs/multiseed/seed219-real/cost.json` — `f7bf65e93f63163067a89644c596194fc77700b8f50ceed080a65f92c6021c31`
- `runs/fixed-step-240-completions/seed214/random__esme-countdown/completions.jsonl` — `26b2acc473cdd03d247606e3cf0d7f2f46a6083f871e0e800cf30939b099d922`
- `runs/fixed-step-240-completions/seed214/random__esme-countdown/provenance.json` — `b11205a199a52bed1dbeaab2e43e88759d2a3dff94ac97989fa1662f767b9a95`
- `runs/fixed-step-240/esme-214m-rlvr-placebo-grpo-2/bundle/manifest.json` — `db8494ca3d1b9128722eb9b2f889a3abf5e95a48786fb1702ce1cd5f2c0c1aa0`
- `runs/fixed-step-240-completions/seed215/random__esme-countdown/completions.jsonl` — `41a071a1e8e9c66d7fb3cfe0f959dfaa7af75a4097efd5a37dbbdb9d800a7093`
- `runs/fixed-step-240-completions/seed215/random__esme-countdown/provenance.json` — `498a3ac6f371885493d7ea464b04cdec01f4e67145ba17193200440d89d51015`
- `runs/multiseed/seed215-placebo/manifest.json` — `ca8376c3e27858eb81f321addfd7182d054b86be7345b567c56489e26baf98e5`
- `runs/fixed-step-240/esme-214m-rlvr-placebo-seed215-grpo/bundle/manifest.json` — `20fef466ffcc1df8cd414789cf23649cc83baa59a0974bd9c78c626ea157600c`
- `runs/multiseed/seed215-placebo/cost.json` — `3bd5180a6b9ee2ed4df68bffbf05e255dfe03ee333f5aa933db3c4153723d1f9`
- `runs/fixed-step-240-completions/seed216/random__esme-countdown/completions.jsonl` — `579c7f798a8e90fcc5273bc02c6fa06abb2eabb5547ba7665d464e51837ea4fd`
- `runs/fixed-step-240-completions/seed216/random__esme-countdown/provenance.json` — `d20384528e0b8dda62118b083e8344300ffdf2757863eb3662f014cbb3f6da7a`
- `runs/multiseed/seed216-placebo/manifest.json` — `972ee4ce77b6afa99fef76b6c92fc071184bfd0bb01cd9e177c5cc7ca62fccfa`
- `runs/fixed-step-240/esme-214m-rlvr-placebo-seed216-grpo/bundle/manifest.json` — `4736b428de964bf80eafa00f8842543479f87420a4bf3348d09909bff496b81c`
- `runs/multiseed/seed216-placebo/cost.json` — `84801cfa37bcbcfe98b27b38d8a063212d9cd1159a5efa044bd9a565178848ec`
- `runs/fixed-step-240-completions/seed217/random__esme-countdown/completions.jsonl` — `0a40ebccc926a5c47cc0b868086b44209081783f5fc8c14b1d37ce8d2fc15196`
- `runs/fixed-step-240-completions/seed217/random__esme-countdown/provenance.json` — `c2e63f5c0cd70747456d4133096352c85ad8e879a9ffe80d48a4a55a6129724a`
- `runs/multiseed/seed217-placebo/manifest.json` — `217883d85b2dda6b1bb345da1f7e361d8384c8d6690184b6288293dbee81ac1c`
- `runs/fixed-step-240/esme-214m-rlvr-placebo-seed217-grpo/bundle/manifest.json` — `46a02c1a503acf01810f31d0dbf8a1be4c2fce82628ce72a51f8928cd113d75a`
- `runs/multiseed/seed217-placebo/cost.json` — `ac7ea4cfeb0209f4622d45a6528bea88972e45f17dbaf17685955a6fe8880c9d`
- `runs/fixed-step-240-completions/seed218/random__esme-countdown/completions.jsonl` — `b0936e6df6143783aca3d7755f9330b934d01140f831ed28c35eee87b6e66fd4`
- `runs/fixed-step-240-completions/seed218/random__esme-countdown/provenance.json` — `e59833acabbffbe5d00c5de195f8c5c09435b5a8f7b7ea066db72e1881c9e3ed`
- `runs/multiseed/seed218-placebo/manifest.json` — `d3bb4724575fa087fb2539d76c4b19987644b3a83193c84c32035bba2b4c272f`
- `runs/fixed-step-240/esme-214m-rlvr-placebo-seed218-grpo/bundle/manifest.json` — `176d70583a7d8afd505588e57cf5d1457e4143ca3b1500f3ef1838caadde882a`
- `runs/multiseed/seed218-placebo/cost.json` — `8f5429c466b4efd672814b381e63bad1d0ac9c9798b705024e693e75f9a7f985`
- `runs/fixed-step-240-completions/seed219/random__esme-countdown/completions.jsonl` — `f2be28a312346ff4adb1dbf20403ebec9716a0bbae36f081bb1dafca516445d3`
- `runs/fixed-step-240-completions/seed219/random__esme-countdown/provenance.json` — `97f5ff890f3877fb7926a2235eb4a5208b41a7aa7305b21a0ca5b217aed3062b`
- `runs/multiseed/seed219-placebo/manifest.json` — `3f8f17333607a000d473520c110ddbcf1565a76658a9416ddbdbb33e59d32fe5`
- `runs/fixed-step-240/esme-214m-rlvr-placebo-seed219-grpo/bundle/manifest.json` — `869d92dea104f77cea3bdbcaf732def285ae96c173461b7adfe20e0447ae54c6`
- `runs/multiseed/seed219-placebo/cost.json` — `1059a74fa31bc7aed4da116e71e9fd3fcf2bf2e404340297bd21838bba639ceb`
- `runs/decomp-sampled/base__esme-countdown/completions.jsonl` — `04877330a37afd092993399751d232daff33740bdff49290be33783980282b72`
- `runs/decomp-sampled/base__esme-countdown/provenance.json` — `5f5cfe7d00a1e799ef02a5e720bcc0c9f8b3ec243bbbad860f61cda20916dac2`
