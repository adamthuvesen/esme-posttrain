# AGENTS.md

Local rules for `esme-posttrain`. The global agent rules still apply.

## Commands

```bash
make install      # install the committed dependency versions
make check        # the gate: architecture, lint, types, tests; same command as CI
make architecture # dependency boundaries with repair guidance
make fmt          # auto-format + safe lint fixes
make test         # tests in parallel (pytest -n auto), including the full CPU stage chain
```

Everything runs through `uv` (`uv run …`, never bare `python`/`pip`). Make targets use `uv --locked`, so run `uv lock` after a dependency change. CI runs `make check`.

Prove changes with `make verify` (`verify-esme-posttrain` skill at `.agents/skills/verify-esme-posttrain/`).

## Red Lines

- The GitHub remote is public. Never push without Adam's explicit approval.
- No Modal, GPU, paid API, remote dataset download, or other spend without a written run card and Adam's approval. A run card names the dataset, token or sample budget, model and config, hardware, expected duration, expected cost, and artifact produced.
- Do not start or expand training infrastructure without the matching run card.

## Enforced Rules

`tests/test_enforced_rules.py` fails when this section is missing, a row cites nothing, or a cited name is not defined in its file.

| Rule | Enforced by | Added |
| --- | --- | --- |
| Stage boundaries and shared-code dependency direction | `pyproject.toml::dependency-direction`, `make architecture` | 2026-11-04 |
| DPO must not import the SFT trainer, even indirectly | `pyproject.toml::dpo-without-sft-trainer`, `make architecture` | 2026-11-04 |
| Type-check Modal entrypoints with package code | `pyproject.toml::files`, `make type` | 2026-11-04 |
| Import each name from its defining module, not a re-export | `pyproject.toml::no_implicit_reexport`, `make type` | 2026-11-06 |
| A parsed stage config is a valid recipe: run id, volume, data, caps | `src/esme_posttrain/sft/config.py::_check_recipe`, `src/esme_posttrain/dpo/config.py::_check_recipe`, `src/esme_posttrain/rl/config.py::_check_recipe`, `tests/test_launch_cap_enforcement.py::test_settings_model_rejects_recipe_drift` | 2026-11-06 |
| Smoke spend cap ($2), per-stage full-run cap, and Modal timeout bound | `src/esme_posttrain/launch/config_models.py::check_stage_runtime`, `tests/test_launch_cap_enforcement.py::test_settings_model_rejects_over_cap_runtime_without_launch_module` | 2026-11-06 |
| Every path to a paid Modal call passes an approval gate first | `tests/test_modal_launch_gates.py::test_modal_launches_pass_an_approval_gate` | 2026-11-06 |
| Full runs refuse without `--approved` | `src/esme_posttrain/launch/config_guards.py::full_launch_blockers`, `tests/test_sft_multiturn.py::test_full_run_refused_without_approval`, `tests/test_chat_dpo.py::test_full_run_refused_without_approval` | 2026-11-06 |
| Remote dataset reads need an explicit download flag | `tests/test_sft_multiturn.py::test_remote_dataset_rows_refuse_without_download_approval` | 2026-11-06 |
| Nothing under `docs/internal/` is committed | `tests/test_enforced_rules.py::test_docs_internal_is_never_tracked` | 2026-11-06 |
| The verify drives must observe FAIL against a wrong expectation | `.agents/skills/verify-esme-posttrain/verify::control`, `make verify` | 2026-11-06 |

## Where Things Live

- `README.md`: current stage of the stack, active CLI commands, launch prep.
- `docs/architecture.md`: stage pipeline, module seams, artifact handoffs.
- `docs/package-layout.md`: change table with owners and focused checks.
- Each stage's `config.py` owns config shape and recipe constraints. Its launch module checks files on disk, projects cost, and builds blockers.
- Read the matching run card before changing a launch config, budget, data source, acceptance gate, or artifact path: `run_cards/esme-214m-sft-multiturn.md`, `run_cards/esme-214m-chat-dpo.md`, `run_cards/esme-214m-rl.md`.

## Judgment Calls

- Scope: take the base model from `esme-pretrain`, run SFT, DPO, then simple-task RLVR, and hand the model to `llm-infer`. Sibling repos exchange artifacts, never code imports.
- Prefer CPU-first, deterministic, inspectable code. Configs, fixtures, launch guards, and trainer code move with their matching tests.
- No task registry or plugin framework beyond what real tasks need. No paid judges as a required eval dependency. No shared base trainer. Test count and coverage are not headline measures.
- Deferred until their triggers fire: eval breadth (a second symbolic verifier and one language eval) waits on a capability claim beyond Countdown-Lite. Splits of `rl/grpo.py`, `rl/launch.py`, and `sft/trainer.py` wait until a new task forces edits inside them. The GPU acceptance run's trigger is in `run_cards/esme-214m-full-path-gpu-acceptance.md`.
- Consume typed settings after parsing. Use `to_dict()` only at JSON boundaries.
- Make failures loud when data is missing, malformed, skipped, or outside a declared budget.
- Keep runtime dependencies minimal. Use clear domain names: base model, cold-start, SFT, RLVR, reward, checkpoint, run card.
