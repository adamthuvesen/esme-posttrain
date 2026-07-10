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
make check        # the gate: ruff lint + format-check + unit tests; green before any commit
make fmt          # auto-format + safe lint fixes
uv run pytest     # unit tests
```

Everything runs through `uv` (`uv run …`, never bare `python`/`pip`).

## Doc Routing

- Start with `README.md` for the current stage of the Esme post-training stack, active CLI commands, and launch prep overview.
- Read `docs/package-layout.md` before moving modules, adding stage code, or changing imports.
- If the local-only `docs/internal/instruct-sft-recipe.md` exists, read it before touching the single-turn Instruct SFT recipe, trainer loop, eval selection, checkpoint/resume behavior, Modal smoke, or full-run path.
- Read the matching run card before changing a launch config, run budget, data source, acceptance gate, or artifact path:
  - `run_cards/esme-214m-instruct.md` for the original single-turn Instruct SFT path.
  - `run_cards/esme-214m-sft-multiturn.md` for the multi-turn SFT foundation.
  - `run_cards/esme-214m-chat-dpo.md` for chat DPO.
  - `run_cards/esme-214m-rl.md` for RLVR prep.
- When editing config shape, fixtures, or CLI dry-run payloads, update the matching stage launch validator (`sft/launch_instruct.py`, `sft/launch_multiturn.py`, `dpo/launch.py`, `rl/launch.py`) and its tests together. The validators are the single source of truth for config shape.

## Scope

- This is the post-training stage: take the from-scratch base model exported by `esme-pretrain`, run an SFT cold-start, then simple-task RLVR, and hand the adapted model to `llm-infer`.
- Sibling repos (`esme-pretrain`, `llm-infer`, `grpo-decomp`) are related but independent. Exchange artifacts (checkpoints, eval results), never code imports.
- Prefer CPU-first, deterministic, inspectable code until the tiny end-to-end path is proven.
- Keep changes small and evidence-backed: configs, fixtures, launch guards, and trainer code should move with their matching tests.
- Do not start or expand training infrastructure without the matching run card and approval.

## Spend And Data

- No remote dataset download without a written run card and Adam approval.
- No Modal, GPU, paid API, or other compute spend without a written run card and Adam approval.
- A run card must name the dataset, token/sample budget, model/config, hardware, expected duration, expected cost, and artifact produced.

## Code Style

- Minimal runtime dependencies until a milestone needs them.
- Use clear domain names: base model, cold-start, SFT, RLVR, reward, checkpoint, run card.
- Make failures loud when data is missing, malformed, skipped, or outside a declared budget.
