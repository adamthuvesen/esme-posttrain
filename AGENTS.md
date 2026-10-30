# AGENTS.md

Local instructions for `esme-posttrain`. The global agent rules still apply; this file only pins the workflow and red lines for this repo.

## Workflow

- Work only inside this repo unless Adam explicitly says otherwise.
- This repo has a public GitHub remote (`adamthuvesen/esme-posttrain`). Never push without Adam's explicit approval; work on a branch and let Adam decide when it lands.
- Nothing under `docs/internal/` may be committed or published; it is gitignored local working material.
- Commit accepted work with small conventional commits.
- Before committing, run the repo gates that match the change and leave `git status --short` clean.

## Commands

```bash
make check        # lint, format, package + Modal types, tests; green before any commit
make fmt          # auto-format + safe lint fixes
uv run pytest     # tests, including the full CPU stage chain
```

Everything runs through `uv` (`uv run …`, never bare `python`/`pip`).

## Doc Routing

- Start with `README.md` for the current stage of the Esme post-training stack, active CLI commands, and launch prep overview.
- Read `docs/architecture.md` for the stage pipeline, module seams, and artifact handoffs.
- Use the change table in `docs/package-layout.md` to find ownership and focused checks.
- Read the matching run card before changing a launch config, run budget, data source, acceptance gate, or artifact path:
  - `run_cards/esme-214m-sft-multiturn.md` for the multi-turn SFT foundation.
  - `run_cards/esme-214m-chat-dpo.md` for chat DPO.
  - `run_cards/esme-214m-rl.md` for RLVR prep.
- Each stage's `config.py` owns config shape. Its launch validator owns recipe constraints. Change both when a field changes behavior.
- Consume typed settings after validation. Use `to_dict()` only at JSON boundaries.

## Enforced Rules

| Rule | Enforced by | Added |
| --- | --- | --- |
| Type-check Modal entrypoints alongside package code | `pyproject.toml` mypy scope, run by local and CI `make check` | 2026-11-04 |
| Resolve launch-script imports in both direct and Makefile test runs | `pyproject.toml` pytest `pythonpath` | 2026-11-04 |

## Scope

- This is the post-training stage: take the from-scratch base model exported by `esme-pretrain`, run an SFT cold-start, then simple-task RLVR, and hand the adapted model to `llm-infer`.
- Sibling repos (`esme-pretrain`, `llm-infer`, `grpo-decomp`) are related but independent. Exchange artifacts (checkpoints, eval results), never code imports.
- Prefer CPU-first, deterministic, inspectable code until the tiny end-to-end path is proven.
- Keep changes small and evidence-backed: configs, fixtures, launch guards, and trainer code should move with their matching tests.
- Do not start or expand training infrastructure without the matching run card and approval.
- Standing rules from the retired staff-level roadmap: no task registry or plugin framework beyond what real tasks need; no paid judges as a required eval dependency; no shared base trainer; test count and coverage are not headline measures.
- Deferred until their triggers fire: eval breadth (a second symbolic verifier task plus one language-oriented eval on the shared contract) waits on wanting to claim posttraining capability beyond Countdown-Lite; module splits of `rl/grpo.py`, `rl/launch.py`, and `sft/trainer.py` wait until a new task forces edits inside them. The GPU acceptance run carries its own trigger in `run_cards/esme-214m-full-path-gpu-acceptance.md`.

## Spend And Data

- No remote dataset download without a written run card and Adam approval.
- No Modal, GPU, paid API, or other compute spend without a written run card and Adam approval.
- A run card must name the dataset, token/sample budget, model/config, hardware, expected duration, expected cost, and artifact produced.

## Code Style

- Minimal runtime dependencies until a milestone needs them.
- Use clear domain names: base model, cold-start, SFT, RLVR, reward, checkpoint, run card.
- Make failures loud when data is missing, malformed, skipped, or outside a declared budget.
