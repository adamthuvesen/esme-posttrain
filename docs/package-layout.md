# Package Layout

This repo keeps stage-specific implementation code out of the package root. When adding or moving code, prefer the canonical package below instead of creating another root-level module.

## Root Package

`src/esme_posttrain/` should stay small:

- `cli.py` owns the `esme-posttrain` command surface.
- `bundle.py` owns dense backbone bundle loading and hashing.
- `modeling.py` owns the shared dense model primitives.
- `run_artifacts.py` owns shared JSON, environment, and manifest artifact writers.
- `__init__.py` and `__main__.py` expose package metadata and module execution.

Do not add root compatibility aliases.

## Stage Packages

- `src/esme_posttrain/sft/`: supervised fine-tuning data, trainers, launch validation, evaluation, sampling, and stage artifacts.
- `src/esme_posttrain/dpo/`: preference data, DPO trainer, launch validation, checkpoint evaluation, and chat-quality comparison.
- `src/esme_posttrain/rl/`: verifier-backed RLVR data, Countdown-Lite baseline, GRPO training, launch validation, and result reports.
- `src/esme_posttrain/launch/`: shared config validation, spend/blocker helpers, command construction, and launch utilities.
- `src/esme_posttrain/export/`: export code for adapted dense bundles handed to downstream inference repos.

## Import Rules

- Internal imports should use canonical package paths such as `esme_posttrain.sft.trainer`, `esme_posttrain.dpo.launch`, `esme_posttrain.rl.launch`, `esme_posttrain.launch.validate`, and `esme_posttrain.export.dense_bundle`.
- Keep package `__init__.py` files boring. Avoid broad re-export barrels; they hide ownership and can create import cycles.
- If a module grows large, split by responsibility inside the stage package before adding a new top-level package.

## Related Docs

- `README.md` explains the current Esme post-training stack.
- `docs/rlvr-countdown-lite.md` records the Countdown-Lite RLVR task, baseline, and result.
