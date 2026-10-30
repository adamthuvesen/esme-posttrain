# Package Layout

This repo keeps stage-specific implementation code out of the package root. When adding or moving code, prefer the canonical package below instead of creating another root-level module.

## Find the Owner

| Change | Start here | Verify with |
| --- | --- | --- |
| Launch config field or recipe constraint | The stage's `config.py`, then typed consumers | `make type`, stage launch tests |
| Budget or launch condition | The matching run card, the stage's `config.py` for caps, and its launch module for blockers | `tests/test_launch_cap_enforcement.py` and stage launch tests |
| Dense-bundle field shared by stages | `export/bundle_writer.py` and `bundle.py` | `tests/test_artifact_contract.py` |
| DPO export provenance | `export/dense_bundle.py` | `tests/test_export_dense_bundle.py` |
| RL export provenance or checkpoint conversion | `rl/bundle.py` and `rl/checkpoint_bundle.py` | `tests/test_rlvr_decomp.py` |
| Training objective | The stage trainer | The stage's trainer tests |
| Stage handoff | `acceptance/full_path_smoke.py` | `tests/test_full_path_smoke.py` |
| Shared tiny model or chat tokenizer | `sft/fixtures.py` | SFT and DPO fixture tests |
| CLI command | Its group in `cli/` | The matching CLI or stage tests |

Run test files with `uv run --locked pytest <path>`. Run `make check` before committing.
It includes dependency boundaries, the full CPU stage chain, and type checking for Modal entrypoints.

Config shape errors identify the input file and nested field. Recipe errors
name the failed constraint. Basedpyright reports the
consumer that uses a missing field, assigns to frozen settings, or passes
the wrong stage config. Fix the owning model or consumer shown by the error.

## Root Package

`src/esme_posttrain/` should stay small:

- `cli/` owns the `esme-posttrain` command surface: `parser.py` assembles the
  subparsers, one module per command group (`rl.py`, `sft.py`, `dpo.py`,
  `export.py`, `artifacts.py`, `studies.py`, `acceptance.py`), and `output.py`
  holds payload emission and error exits.
  Handlers take an `argparse.Namespace` and return an int exit code.
- `bundle.py` owns canonical dense-bundle v1 validation, loading, and hashing.
- `modeling.py` owns the shared dense model primitives.
- `run_artifacts.py` owns shared JSON, environment, and manifest artifact writers.
- `__init__.py` and `__main__.py` expose package metadata and module execution.

Do not add root compatibility aliases.

## Stage Packages

- `src/esme_posttrain/sft/`: one conversation-based supervised fine-tuning path for
  single- and multi-turn rows, plus launch validation, evaluation, sampling, and
  stage artifacts.
- `src/esme_posttrain/dpo/`: preference data, DPO trainer, launch validation, checkpoint evaluation, and chat-quality comparison.
- `src/esme_posttrain/rl/`: verifier-backed RLVR data, Countdown-Lite baseline, GRPO training, launch validation, and result reports.
- `src/esme_posttrain/evals/`: typed records for the evaluation contract: task rows, per-sample scores, per-task results, resume lines, and aggregate summaries. Serialized shapes are pinned by the golden fixtures under `fixtures/outputs/`.
- `src/esme_posttrain/studies/`: strict study specifications and generated
  artifact-backed JSON/Markdown reports.
- `src/esme_posttrain/launch/`: shared frozen config models, spend/blocker helpers,
  command construction, and launch utilities.
- `src/esme_posttrain/acceptance/`: the full-path CPU smoke that coordinates stage handoffs.
- `src/esme_posttrain/training/`: shared training runtime used by the stage packages: collation, metrics, checkpointing, seeding/precision, and W&B setup.
- `src/esme_posttrain/export/`: one canonical dense-bundle writer plus DPO export and generation verification.

Each stage's `config.py` owns its typed settings and recipe checks, so a parsed
settings object is always a valid recipe. Its launch module checks files on disk,
resolves paths, and projects cost. Consumers use typed attributes, then serialize
settings only when writing JSON. `rl/bundle.py` owns RL export identity and calls
the shared writer. It serves both training and checkpoint conversion.

## Import Rules

`make architecture` enforces the dependency graph declared in `pyproject.toml`.
It reports the offending import chain and a repair direction.
Shared code cannot depend on stages. Stages cannot depend on CLI or acceptance code.
DPO can reuse `sft.data`, `sft.data_types`, `sft.sampling`, and `sft.fixtures`, but cannot reach the SFT trainer.
New modules inside these packages inherit their boundaries.
For a new top-level package, declare its position in the graph before running the check.

- Import each name from the module that defines it. `make type` rejects imports through a re-export (`reportPrivateLocalImportUsage`).
- Keep package `__init__.py` files boring. Avoid broad re-export barrels; they hide ownership and can create import cycles.
- If a module grows large, split by responsibility inside the stage package before adding a new top-level package.

## Related Docs

- `README.md` explains the current Esme post-training stack.
- `docs/architecture.md` maps the stage pipeline, module seams, and artifact handoffs.
- `docs/rlvr-countdown-lite.md` records the Countdown-Lite RLVR task, baseline, and result.
- `esme-pretrain/docs/bundle-format.md` is the canonical cross-repo bundle
  contract and compatibility policy; this repo does not maintain a parallel copy.
- `docs/study-reports.md` defines reproducible study reports.
