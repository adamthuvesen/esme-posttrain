# Run Card: Esme-214M-RL

This is the verifier-reward stage of the standard post-training chain:

```text
Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL
```

`Esme-214M-RL` is the Countdown-Lite GRPO variant of `Esme-214M-Chat`.

## Artifact

- Produces: `Esme-214M-RL`
- Starts from: `Esme-214M-Chat`
- Config: `configs/esme-214m-rl.json`
- Dataset manifest schema: `schemas/rl-task-manifest.schema.json`
- Launch config schema: `schemas/rlvr-grpo-config.schema.json`
- Launcher: `scripts/modal_rlvr_grpo.py`
- Public result summary: `docs/rlvr-countdown-lite-grpo.md`

## Task

- Dataset manifest: `data/manifests/esme-214m-rl.tasks.json`
- Local splits: `data/rl/countdown_lite/{train,dev,eval}.jsonl`
- Shape: 300 train tasks, 30 dev tasks, 30 eval tasks.
- Rule: generate an arithmetic expression using each supplied number exactly
  once to reach the target.
- Operators: `+`, `-`, `*`, and parentheses.
- Reward: exact verifier-backed execution check.
- Excluded rewards: style qualities such as friendliness, sharpness, or
  naturalness.
- Secondary transfer eval: GSM8K-lite, separate from the reward and acceptance
  target.

## Evaluation Profile

- Primary eval: Countdown-Lite eval split.
- Acceptance profile: `full_acceptance_30x32` (`30` tasks x `32` samples).
- Seed: `214`.
- Token budget: 512,000 tokens.
- Runtime hard stop: `$8.00`.
- Mission cap: under `$25.00`.

## Results

| Report | pass@1 | pass@8 | pass@32 | Valid expressions | Exact solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Esme-214M-Chat` baseline | 3.33% | 3.33% | 6.67% | 2.71% | 0.42% |
| `Esme-214M-RL` GRPO | 16.67% | 16.67% | 20.00% | 35.73% | 15.83% |

The baseline had a small but nonzero easy-band foothold, so the first bounded
RLVR target was GRPO rather than an additional supervised hinting stage.

## Acceptance

- Countdown-Lite verifier metrics improve over the chat baseline.
- The run writes reproducible config, metrics, report, checkpoint, tokenizer,
  manifest, cost, and environment artifacts.
- Eval records include phase, eval profile, config hash, model id, task/sample
  range, split, sample budget, and completion counts.
- Re-running eval with matching metadata resumes from completed task records;
  metadata mismatches fail loudly.
- W&B is disabled by default for local commands.

## Safe Local Commands

```bash
uv run esme-posttrain rlvr-dry-run --config configs/esme-214m-rl.json
uv run esme-posttrain rlvr-dry-run --config fixtures/configs/esme-214m-rl.fixture.json
uv run esme-posttrain rlvr-countdown-lite-build-data --repo-root . --json
```
