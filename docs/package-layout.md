# Package Layout

This repo keeps stage-specific implementation code out of the package root. When adding or moving code, prefer the canonical package below instead of creating another root-level module.

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
- `src/esme_posttrain/launch/`: shared config validation, spend/blocker helpers,
  command construction, launch utilities, and the full-path CPU smoke.
- `src/esme_posttrain/training/`: shared training runtime used by the stage packages: collation, metrics, checkpointing, seeding/precision, and W&B setup.
- `src/esme_posttrain/export/`: export code for adapted dense bundles handed to downstream inference repos.

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
