<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# RLVR Countdown-Lite placebo study

Machine-readable report: [rlvr-placebo.report.json](rlvr-placebo.report.json)

**Verdict:** accepted

Verifier-reward GRPO improves held-out Countdown validity and exact solving beyond a random-reward placebo.

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `paired_student_t_95`
- Sample budget per run: 480
- Evidence warning: real_reward seed 214 has no comparable cost artifact
- Evidence warning: random_reward seed 214 has no training manifest
- Evidence warning: random_reward seed 214 has no comparable cost artifact

## Per-seed results

### real_reward (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 27.1% | 5.6% | 4/30 | n/a |
| 215 | 97.7% | 9.8% | 3/30 | 0.9999 |
| 216 | 97.1% | 9.8% | 3/30 | 1.1223 |
| 217 | 97.1% | 9.6% | 3/30 | 0.9952 |
| 218 | 96.7% | 9.6% | 3/30 | 1.0612 |
| 219 | 96.9% | 9.8% | 3/30 | 0.9910 |

[Table data](rlvr-placebo.report.json)

| Seed | Train status | Selected step | Selected metric | Token budget |
| ---: | --- | ---: | ---: | ---: |
| 214 | n/a | 234 | train/reward_mean=0.7142 | 5300000 |
| 215 | complete | 234 | train/reward_mean=0.7211 | 5300000 |
| 216 | complete | 234 | train/reward_mean=0.7166 | 5300000 |
| 217 | complete | 234 | train/reward_mean=0.7250 | 5300000 |
| 218 | complete | 159 | train/reward_mean=0.7109 | 5300000 |
| 219 | complete | 234 | train/reward_mean=0.7059 | 5300000 |

[Table data](rlvr-placebo.report.json)

### random_reward (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 0.8% | 0.0% | 0/30 | n/a |
| 215 | 0.4% | 0.0% | 0/30 | 0.7507 |
| 216 | 0.8% | 0.2% | 1/30 | 0.7447 |
| 217 | 0.6% | 0.0% | 0/30 | 0.7547 |
| 218 | 0.6% | 0.2% | 1/30 | 0.7908 |
| 219 | 1.2% | 0.2% | 1/30 | 0.7473 |

[Table data](rlvr-placebo.report.json)

| Seed | Train status | Selected step | Selected metric | Token budget |
| ---: | --- | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | 5300000 |
| 215 | complete | 49 | train/reward_mean=0.5242 | 5300000 |
| 216 | complete | 219 | train/reward_mean=0.5219 | 5300000 |
| 217 | complete | 112 | train/reward_mean=0.5398 | 5300000 |
| 218 | complete | 8 | train/reward_mean=0.5297 | 5300000 |
| 219 | complete | 162 | train/reward_mean=0.5375 | 5300000 |

[Table data](rlvr-placebo.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 0.8% | 0.2% | 1/30 | n/a |

[Table data](rlvr-placebo.report.json)

| Seed | Train status | Selected step | Selected metric | Token budget |
| ---: | --- | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a |

[Table data](rlvr-placebo.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| real_reward | treatment | 85.4% | 9.0% | 10.6% | 5.1697 |
| random_reward | control | 0.8% | 0.1% | 1.7% | 3.7883 |
| chat_baseline | baseline | 0.8% | 0.2% | 3.3% | 0.0000 |

[Table data](rlvr-placebo.report.json)

## Planned comparisons

### real-vs-random-validity

- Metric: `valid_expression_rate`
- Mean effect: +84.7%
- 95% CI: [+54.6%, +114.7%]
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo.report.json)

### real-vs-random-exact

- Metric: `any_exact_solve_rate`
- Mean effect: +8.9%
- 95% CI: [+6.0%, +11.7%]
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo.report.json)

### real-vs-baseline-validity

- Metric: `valid_expression_rate`
- Mean effect: +26.2%
- 95% CI: n/a
- Paired seeds: 214
- [Comparison data](rlvr-placebo.report.json)


## Supported claims

- Across six training seeds, the verifier-reward arm has a higher held-out Countdown valid-expression rate than the random-reward arm.
- The exact-solve gain is positive but smaller than the valid-expression gain; this study does not support a claim of broad reasoning improvement.

## Excluded runs

- `esme-214m-rlvr-countdown-grpo-v2-ccb6287-attempt1`: Preempted before durable completion and excluded before the multiseed comparison.

## Artifact provenance

- `runs/decomp-sampled/correct__esme-countdown/completions.jsonl` — `573a42accdc4e3e608d3522818887a31419ad5d90ea744b63514c92ae0f5cf59`
- `runs/decomp-sampled/correct__esme-countdown/provenance.json` — `2432ea5f092f5f3dca407329bdc80ec32e51029b2d5b77b0627bf3d0e6ec4cc6`
- `runs/esme-214m-rlvr-countdown-grpo-v2-ccb6287-1/manifest.json` — `e827777155fbf0bb70fca846b7c466258d243c9a4e6f50ba7187558035dd20c6`
- `runs/multiseed-completions/seed215/correct__esme-countdown/completions.jsonl` — `e65c47aafac954cfe559d7f10db693bdd2a5c3b8be09d970faca7f5c65a3bfa2`
- `runs/multiseed-completions/seed215/correct__esme-countdown/provenance.json` — `d3a379bb2b1c4c2297c64c0ea469be2898909de3f08052a4853d98bd878f457c`
- `runs/multiseed/seed215-real/manifest.json` — `4083dd6cb61dbc47c86157329b1ebe207165084146003896c5f6919affc58a57`
- `runs/multiseed/seed215-real/cost.json` — `711b3145cdc93181c05ea0333f4d2b715faa83583e6753de5c5382a546f147b0`
- `runs/multiseed-completions/seed216/correct__esme-countdown/completions.jsonl` — `65606650c656e78a3f6b2268149eb8405206ecb6bbde6c1bbdd556098793dfa8`
- `runs/multiseed-completions/seed216/correct__esme-countdown/provenance.json` — `db28f9635e203bd57f47e1c52056dce39b2ed9c92130e30a2f9915b599266298`
- `runs/multiseed/seed216-real/manifest.json` — `dd3387bd4f50fd7320dded21d22666eb550d9dabc50c30101e514646432e4751`
- `runs/multiseed/seed216-real/cost.json` — `c636fc7a3fc344ec5811d2861a42885b7f6b27d01cca6ecd42d32ce9bbd1fd15`
- `runs/multiseed-completions/seed217/correct__esme-countdown/completions.jsonl` — `9fe2cfd460c0b09f2499fe5faef3e96b479ceb64be38baf56875ddf4ab980db4`
- `runs/multiseed-completions/seed217/correct__esme-countdown/provenance.json` — `102fc5e6e9c6323f444fc03b258e2a25dcfe2ea99522f6280d342940684d4cba`
- `runs/multiseed/seed217-real/manifest.json` — `5c18747f33386899505625a9637a69651d870b34b03e8569c03045fc529832ba`
- `runs/multiseed/seed217-real/cost.json` — `07762084cde98585028f86e0329b0a7c3dcb7421bf43bad2e8b05b8d31444889`
- `runs/multiseed-completions/seed218/correct__esme-countdown/completions.jsonl` — `4d5f26f7b4b9c5efe84bcdcdbf8a7c7d579ef0a78061af7fdc1801c2ce4307d6`
- `runs/multiseed-completions/seed218/correct__esme-countdown/provenance.json` — `5ede81118911be6672589a434f5abd5f79c76e99037a65fa8576d87af6ad8b7a`
- `runs/multiseed/seed218-real/manifest.json` — `e7de8bc1780658372e24ec3a141f7654ab8aab5c60940bf648495d48cd808f91`
- `runs/multiseed/seed218-real/cost.json` — `230a48cc98b7379db0b3b1ea97b558815a922149b31e38dd9345d167533fff72`
- `runs/multiseed-completions/seed219/correct__esme-countdown/completions.jsonl` — `049ca981bc221f2fd4ce05ab6d843c56492e382f4bd7f565bf8a97d3e434b8d7`
- `runs/multiseed-completions/seed219/correct__esme-countdown/provenance.json` — `f4910d69cb776c9cf34951bf239e1c4274135e08db7c9b831669adbccb22f05f`
- `runs/multiseed/seed219-real/manifest.json` — `35e8b8104031f42ce6c4ac301ca87ba7b02225e16d8cc93af617c249a036a9a1`
- `runs/multiseed/seed219-real/cost.json` — `f7bf65e93f63163067a89644c596194fc77700b8f50ceed080a65f92c6021c31`
- `runs/decomp-sampled/random__esme-countdown/completions.jsonl` — `41575550e01010bdfbc7e0efc1092ae962015ee9f91e5ec300db257c59c3356e`
- `runs/decomp-sampled/random__esme-countdown/provenance.json` — `c540f8d4e31d00f61b145cda16b9ca47b855340ee28ff3aa5f20232677881a53`
- `runs/multiseed-completions/seed215/random__esme-countdown/completions.jsonl` — `325b8663672077bcf8eeae69c4a495580d26ab5c6dc9def5488d42477b5cd094`
- `runs/multiseed-completions/seed215/random__esme-countdown/provenance.json` — `75f8906adc0748441ab72bf0d9f3586ff3743bc641bdbec6a48d3da2a69eb64c`
- `runs/multiseed/seed215-placebo/manifest.json` — `ca8376c3e27858eb81f321addfd7182d054b86be7345b567c56489e26baf98e5`
- `runs/multiseed/seed215-placebo/cost.json` — `3bd5180a6b9ee2ed4df68bffbf05e255dfe03ee333f5aa933db3c4153723d1f9`
- `runs/multiseed-completions/seed216/random__esme-countdown/completions.jsonl` — `579c7f798a8e90fcc5273bc02c6fa06abb2eabb5547ba7665d464e51837ea4fd`
- `runs/multiseed-completions/seed216/random__esme-countdown/provenance.json` — `69e4cf2313865dbe11c46f0c511070c40396fe81bbc17d9ed76e6d82419406e5`
- `runs/multiseed/seed216-placebo/manifest.json` — `972ee4ce77b6afa99fef76b6c92fc071184bfd0bb01cd9e177c5cc7ca62fccfa`
- `runs/multiseed/seed216-placebo/cost.json` — `84801cfa37bcbcfe98b27b38d8a063212d9cd1159a5efa044bd9a565178848ec`
- `runs/multiseed-completions/seed217/random__esme-countdown/completions.jsonl` — `66b9c926b78857b2c724fab43c9d2d9d5e556a39704397709f59f4300a01ffc6`
- `runs/multiseed-completions/seed217/random__esme-countdown/provenance.json` — `86750c7eb4d2dafc3b752c245edeb75b953d65fca16969d59e84294e1c3e1c80`
- `runs/multiseed/seed217-placebo/manifest.json` — `217883d85b2dda6b1bb345da1f7e361d8384c8d6690184b6288293dbee81ac1c`
- `runs/multiseed/seed217-placebo/cost.json` — `ac7ea4cfeb0209f4622d45a6528bea88972e45f17dbaf17685955a6fe8880c9d`
- `runs/multiseed-completions/seed218/random__esme-countdown/completions.jsonl` — `7f1015190542a65bb18d086a4e7c273fd08c53b8b69790cf34098c0ba327d6b5`
- `runs/multiseed-completions/seed218/random__esme-countdown/provenance.json` — `b306f02907b144e0ca64ed9ce25a355ed1a6ed6ebb4e4315b9d04eaf6accd86b`
- `runs/multiseed/seed218-placebo/manifest.json` — `d3bb4724575fa087fb2539d76c4b19987644b3a83193c84c32035bba2b4c272f`
- `runs/multiseed/seed218-placebo/cost.json` — `8f5429c466b4efd672814b381e63bad1d0ac9c9798b705024e693e75f9a7f985`
- `runs/multiseed-completions/seed219/random__esme-countdown/completions.jsonl` — `8b11136e39ff96cb92c6c4db02391745e1700822b3f2cab6d27314704e065a49`
- `runs/multiseed-completions/seed219/random__esme-countdown/provenance.json` — `5a6a9da22236edd7852c0b1870568ae7c8f2cae5c5c707897a99e69d0a79737d`
- `runs/multiseed/seed219-placebo/manifest.json` — `3f8f17333607a000d473520c110ddbcf1565a76658a9416ddbdbb33e59d32fe5`
- `runs/multiseed/seed219-placebo/cost.json` — `1059a74fa31bc7aed4da116e71e9fd3fcf2bf2e404340297bd21838bba639ceb`
- `runs/decomp-sampled/base__esme-countdown/completions.jsonl` — `04877330a37afd092993399751d232daff33740bdff49290be33783980282b72`
- `runs/decomp-sampled/base__esme-countdown/provenance.json` — `5f5cfe7d00a1e799ef02a5e720bcc0c9f8b3ec243bbbad860f61cda20916dac2`
- `studies/references/grpo-decomp-sampled-multiseed-summary.json` — `daa7ecea735d2ff13f1366f81d2a472abc9f89dbcce3be492d09b570a2fc4e90`
