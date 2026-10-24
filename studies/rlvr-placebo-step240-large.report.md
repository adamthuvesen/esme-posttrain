<!-- GENERATED FILE. Edit the study specification or source artifacts, then rebuild. -->
# RLVR Countdown-Lite placebo study at step 240 on 200 held-out tasks

Machine-readable report: [rlvr-placebo-step240-large.report.json](rlvr-placebo-step240-large.report.json)

**Verdict:** accepted

At a matched training step (240), verifier-reward GRPO improves validity and exact solving on 200 fresh, mostly 3-number held-out Countdown tasks beyond a random-reward placebo.

## Completeness and compatibility

- Complete: yes
- Compatible: yes
- Confidence interval: `seed_problem_bootstrap_95`
- Sample budget per run: 3200
- Evidence warning: real_reward seed 214 has no training manifest
- Evidence warning: real_reward seed 214 has no comparable cost artifact
- Evidence warning: real_reward seed 215 has no training manifest
- Evidence warning: real_reward seed 216 has no training manifest
- Evidence warning: real_reward seed 217 has no training manifest
- Evidence warning: real_reward seed 218 has no training manifest
- Evidence warning: real_reward seed 219 has no training manifest
- Evidence warning: random_reward seed 214 has no training manifest
- Evidence warning: random_reward seed 214 has no comparable cost artifact
- Evidence warning: random_reward seed 215 has no training manifest
- Evidence warning: random_reward seed 216 has no training manifest
- Evidence warning: random_reward seed 217 has no training manifest
- Evidence warning: random_reward seed 218 has no training manifest
- Evidence warning: random_reward seed 219 has no training manifest

## Per-seed results

### real_reward (treatment)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 98.1% | 6.3% | 13/200 | n/a |
| 215 | 98.0% | 6.3% | 13/200 | 0.9999 |
| 216 | 98.2% | 6.3% | 13/200 | 1.1223 |
| 217 | 97.8% | 6.2% | 13/200 | 0.9952 |
| 218 | 97.8% | 6.3% | 13/200 | 1.0612 |
| 219 | 97.5% | 6.2% | 14/200 | 0.9910 |

[Table data](rlvr-placebo-step240-large.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | 240 | 5300000 |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |
| 218 | complete | n/a | n/a | 240 | 5300000 |
| 219 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-placebo-step240-large.report.json)

### random_reward (control)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 1.2% | 0.1% | 2/200 | n/a |
| 215 | 0.8% | 0.1% | 2/200 | 0.7507 |
| 216 | 1.7% | 0.1% | 2/200 | 0.7447 |
| 217 | 1.3% | 0.1% | 3/200 | 0.7547 |
| 218 | 1.4% | 0.1% | 3/200 | 0.7908 |
| 219 | 1.4% | 0.1% | 2/200 | 0.7473 |

[Table data](rlvr-placebo-step240-large.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | 240 | 5300000 |
| 215 | complete | n/a | n/a | 240 | 5300000 |
| 216 | complete | n/a | n/a | 240 | 5300000 |
| 217 | complete | n/a | n/a | 240 | 5300000 |
| 218 | complete | n/a | n/a | 240 | 5300000 |
| 219 | complete | n/a | n/a | 240 | 5300000 |

[Table data](rlvr-placebo-step240-large.report.json)

### chat_baseline (baseline)

| Seed | Valid expression | Exact sample | Any-exact problems | Cost (USD) |
| ---: | ---: | ---: | ---: | ---: |
| 214 | 1.4% | 0.1% | 3/200 | n/a |

[Table data](rlvr-placebo-step240-large.report.json)

| Seed | Train status | Selected step | Selected metric | Evaluated step | Token budget |
| ---: | --- | ---: | ---: | ---: | ---: |
| 214 | n/a | n/a | n/a | n/a | n/a |

[Table data](rlvr-placebo-step240-large.report.json)


## Aggregate arms

| Arm | Role | Valid expression | Exact sample | Any-exact rate | Cost (USD) |
| --- | --- | ---: | ---: | ---: | ---: |
| real_reward | treatment | 97.9% | 6.3% | 6.6% | 5.1697 |
| random_reward | control | 1.3% | 0.1% | 1.2% | 3.7883 |
| chat_baseline | baseline | 1.4% | 0.1% | 1.5% | 0.0000 |

[Table data](rlvr-placebo-step240-large.report.json)

## Planned comparisons

### real-vs-random-validity

- Metric: `valid_expression_rate`
- Mean effect: +96.6%
- 95% CI: [+95.9%, +97.3%]
- Sign-flip permutation p (two-sided): 0.0312 (smallest possible with these seeds: 0.0312)
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo-step240-large.report.json)

### real-vs-random-exact-sample

- Metric: `exact_solve_rate`
- Mean effect: +6.2%
- 95% CI: [+3.2%, +9.6%]
- Sign-flip permutation p (two-sided): 0.0312 (smallest possible with these seeds: 0.0312)
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo-step240-large.report.json)

### real-vs-random-exact

- Metric: `any_exact_solve_rate`
- Mean effect: +5.4%
- 95% CI: [+2.2%, +9.0%]
- Sign-flip permutation p (two-sided): 0.0312 (smallest possible with these seeds: 0.0312)
- Paired seeds: 214, 215, 216, 217, 218, 219
- [Comparison data](rlvr-placebo-step240-large.report.json)


## Supported claims

- At a matched training step, the verifier-reward arm has a higher valid-expression rate than the random-reward arm on 200 fresh held-out Countdown tasks.
- At a matched training step, the verifier-reward arm solves more distinct held-out problems than the random-reward arm, but far fewer than it makes valid.

## Excluded runs

None.

## Artifact provenance

- `runs/gpu-emit/batch1/heldout_large-real-seed214-step240/completions.jsonl` — `b4ed25758cedbbd17f4e25ce43f074bd5e2ee6f6585bd8670b4b9c53bcb697b3`
- `runs/gpu-emit/batch1/heldout_large-real-seed214-step240/provenance.json` — `fffa8276ff24f484d7449e0f90939439f84ee2dd2852deaeaf95d5d5d3ee00b2`
- `runs/gpu-emit/batch1/heldout_large-real-seed214-step240/bundle-manifest.json` — `a3f6068faf1f9f65ce035ed20eeee0c563eb9267a44185dbfb8c48bfe5ac3f1d`
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
- `runs/gpu-emit/batch1/heldout_large-real-seed218-step240/completions.jsonl` — `3e101bf02605e7ce6f78457b79166e3284f2239fa3c4299c34172ad593102dbc`
- `runs/gpu-emit/batch1/heldout_large-real-seed218-step240/provenance.json` — `fa549651cd396740e138c4bf08404b216f037304d4e60440e02b148216eeb519`
- `runs/gpu-emit/batch1/heldout_large-real-seed218-step240/bundle-manifest.json` — `5ff684a238ecd59b89337c017b821b9e8fe0aafbe2833c27da841afa650d2d27`
- `runs/multiseed/seed218-real/cost.json` — `230a48cc98b7379db0b3b1ea97b558815a922149b31e38dd9345d167533fff72`
- `runs/gpu-emit/batch1/heldout_large-real-seed219-step240/completions.jsonl` — `5f6c45a7ab8101b5c2fc451d773eaea0cbfd3ec5e8f51f8c12af6283f686c0d1`
- `runs/gpu-emit/batch1/heldout_large-real-seed219-step240/provenance.json` — `1c43a932e0b10bf347a0edb0161afb5189628e1c029d96981ce5cd227b344432`
- `runs/gpu-emit/batch1/heldout_large-real-seed219-step240/bundle-manifest.json` — `2a41830b3c6074b5a1aaaee2c073201a0b4f4a625a94204b9f023115f3531e1e`
- `runs/multiseed/seed219-real/cost.json` — `f7bf65e93f63163067a89644c596194fc77700b8f50ceed080a65f92c6021c31`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed214-step240/completions.jsonl` — `e5b545ab78aa3b9011e516f144c25392551557a2914e56cb65ae24e313bdee48`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed214-step240/provenance.json` — `fe6cffd297885a0b787aeaa0770b5de00672c89b84715cca1f1fd8fa88c69b9a`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed214-step240/bundle-manifest.json` — `42f98ab6b49062fb12e852882474012d4df31393d626ccdcb7768a5abe293b2e`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed215-step240/completions.jsonl` — `951de6456546d6256f8ae2fbd9970dffeaeb85579435bb42403aa6a0e99cb015`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed215-step240/provenance.json` — `17caf1001b44448ec1cd8d36248f68c3c6dbacd091a6f9a97b76d503dfb11940`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed215-step240/bundle-manifest.json` — `f98c910822adbc2cd4ce35f1fd1a1cf06b2a6037c6b937784083e4a718efe183`
- `runs/multiseed/seed215-placebo/cost.json` — `3bd5180a6b9ee2ed4df68bffbf05e255dfe03ee333f5aa933db3c4153723d1f9`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed216-step240/completions.jsonl` — `7f60f9511ea5bb0e9707e509dec4d88549f8ee5ede421f86e179f51dd8e15755`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed216-step240/provenance.json` — `b7cc525084d3cf6dfe8931932aee248dfdd0673aefcdd589be68e3f7cf8f7d9c`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed216-step240/bundle-manifest.json` — `a8495274ef6f4f45ceec8eab56807cba666024c146e069e515466ad7a538d5c1`
- `runs/multiseed/seed216-placebo/cost.json` — `84801cfa37bcbcfe98b27b38d8a063212d9cd1159a5efa044bd9a565178848ec`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed217-step240/completions.jsonl` — `f30d13b66244b3d8b5c82079c7f9e2a71a09c7d22282b1b646dfbe0aee94a9ac`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed217-step240/provenance.json` — `ab0d52d1138c9975ed41a5a8b78b84a8561a1dd208be3f6e99a47ae30e3b435d`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed217-step240/bundle-manifest.json` — `9b193cca2e818ff8cee86e057a1eec3a9f4070e04e3dbc3c6a57a33e043c95fa`
- `runs/multiseed/seed217-placebo/cost.json` — `ac7ea4cfeb0209f4622d45a6528bea88972e45f17dbaf17685955a6fe8880c9d`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed218-step240/completions.jsonl` — `f54cb79416871ca1ac13894df6dcff6c91f04bfff48626b33b318ff1e0abc5a1`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed218-step240/provenance.json` — `854d372f9e5695a7b94b46476878ffdecbb95be90ab64fa4b852e23a2ffd6e0e`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed218-step240/bundle-manifest.json` — `3cc06ef5e98730e79b4735501ecffeaaef549a246144007f2d23561d2536f299`
- `runs/multiseed/seed218-placebo/cost.json` — `8f5429c466b4efd672814b381e63bad1d0ac9c9798b705024e693e75f9a7f985`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed219-step240/completions.jsonl` — `177974d43a8cebc327649e527783d2a9df0b4d9a2107027e45936f911fe9c73c`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed219-step240/provenance.json` — `ff9b97395c3af3fa56622be3fa7f28066a206635f265842ffc3ebefa041a0701`
- `runs/gpu-emit/batch1/heldout_large-placebo-seed219-step240/bundle-manifest.json` — `0966a9276a3b7f31cffd8648f9f7e88a29274ceed5e7d686e12dcf9048efaab3`
- `runs/multiseed/seed219-placebo/cost.json` — `1059a74fa31bc7aed4da116e71e9fd3fcf2bf2e404340297bd21838bba639ceb`
- `runs/gpu-emit/batch1/heldout_large-chat/completions.jsonl` — `d58c2a1aef5114e2f577ddeac59a9d4a598a1818af16a1941c488bd05e2f7529`
- `runs/gpu-emit/batch1/heldout_large-chat/provenance.json` — `3e2546d1e854e702a6284030cf2e062ee98527f1f286f1e6849873b45569e8d4`
