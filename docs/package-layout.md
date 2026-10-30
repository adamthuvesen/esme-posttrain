# Package Layout

This repo keeps stage-specific implementation code out of the package root. When adding or moving code, prefer the canonical package below instead of creating another root-level module.

## Find the Owner

| Change | Start here | Verify with |
| --- | --- | --- |
| Launch config field | The stage's `config.py`, then its launch validator and typed consumers | `make type`, stage launch tests |
| Budget or launch condition | The matching run card and stage launch validator | `tests/test_launch_cap_enforcement.py` and stage launch tests |
| Dense-bundle field shared by stages | `export/bundle_writer.py` and `bundle.py` | `tests/test_artifact_contract.py` |
| DPO export provenance | `export/dense_bundle.py` | `tests/test_export_dense_bundle.py` |
| RL export provenance or checkpoint conversion | `rl/bundle.py` and `rl/checkpoint_bundle.py` | `tests/test_rlvr_decomp.py` |
| Training objective | The stage trainer | The stage's trainer tests |
| Stage handoff | `launch/full_path_smoke.py` | `tests/test_full_path_smoke.py` |
| CLI command | Its group in `cli/` | The matching CLI or stage tests |

Run test files with `uv run pytest <path>`. Run `make check` before committing.
It includes the full CPU stage chain and type checking for Modal entrypoints.

Config shape errors identify the input file and nested field. Recipe errors
name the failed constraint. Mypy reports the
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
  command construction, launch utilities, and the full-path CPU smoke.
- `src/esme_posttrain/training/`: shared training runtime used by the stage packages: collation, metrics, checkpointing, seeding/precision, and W&B setup.
- `src/esme_posttrain/export/`: one canonical dense-bundle writer plus DPO export and generation verification.

Each stage's `config.py` owns its typed settings. Its launch validator owns
recipe checks and resolved paths. Consumers use typed attributes, then serialize
settings only when writing JSON. `rl/bundle.py` owns RL export identity and calls
the shared writer. It serves both training and checkpoint conversion.

## Import Rules

- Internal imports should use canonical package paths such as `esme_posttrain.sft.trainer`, `esme_posttrain.dpo.launch`, `esme_posttrain.rl.launch`, `esme_posttrain.launch.config_guards`, and `esme_posttrain.export.dense_bundle`.
- Keep package `__init__.py` files boring. Avoid broad re-export barrels; they hide ownership and can create import cycles.
- If a module grows large, split by responsibility inside the stage package before adding a new top-level package.

## Related Docs

- `README.md` explains the current Esme post-training stack.
- `docs/architecture.md` maps the stage pipeline, module seams, and artifact handoffs.
- `docs/rlvr-countdown-lite.md` records the Countdown-Lite RLVR task, baseline, and result.
- `esme-pretrain/docs/bundle-format.md` is the canonical cross-repo bundle
  contract and compatibility policy; this repo does not maintain a parallel copy.
- `docs/study-reports.md` defines reproducible study reports.
