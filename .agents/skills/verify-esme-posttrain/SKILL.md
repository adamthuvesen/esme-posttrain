---
name: verify-esme-posttrain
description: "Drives the real esme-posttrain SFT, DPO, and RL stage chain and the Modal launch entrypoints on CPU, offline, with no spend. Use to run, launch, drive, or verify esme-posttrain, or to prove a change to it works."
---

# verify-esme-posttrain

`make check` proves the code is consistent. This skill proves behavior. It trains SFT, DPO, and RL on CPU through `full-path-cpu-smoke`, rebuilds and checks the bundles, then resolves every paid Modal launch plan and checks that each one refuses without `--approved`.

## Run

```bash
make verify                   # drives that cover files changed against main; stage-chain when none match
make verify ARGS=all          # every drive, about 25 seconds
make verify ARGS=launch-plans
make verify ARGS=list         # each drive, what it proves, and the paths it covers
make verify ARGS=control      # re-runs stage-chain with an SFT loss threshold it cannot reach; exits 0 only when FAIL is observed
```

The helper is `.agents/skills/verify-esme-posttrain/verify`. It runs children with `.venv/bin/python`, so run `make install` first.

## Reading Results

- Each drive prints `PASS|FAIL <drive> <seconds>s <detail>`. A FAIL detail names what the drive expected and what it observed. A drive that crashes is a FAIL.
- The run fails when it changes the checkout: a tracked, untracked, or ignored file, the index, HEAD, or `.git/config`. Only `.verify/` and `__pycache__/` may change.
- Evidence is in `.verify/runs/<utc>-<sha7>[-dirty]/`. Each step has `.cmd`, `.stdout`, `.stderr`, and `.exit` files. `stage-chain/artifacts/` holds the full chain output (metrics, checkpoints, bundles). `summary.json` holds the overall `result`, `checkout_changed`, the drive results, and the git sha.
- `tripwire fired` means a child tried a non-loopback network connection or called `modal`, `wandb`, or `hf`. Treat it as a spend or data-leak risk, not a flake.

## Isolation

Each run gets a temp HOME and temp XDG, HF, W&B, and inductor dirs. It sets `HF_HUB_OFFLINE=1` and `WANDB_MODE=disabled`. It removes `MODAL_*`, `WANDB_*`, `HF_*`, the stage env overrides, and secret-named variables, and points `MODAL_SERVER_URL` at a closed loopback port. A `sitecustomize.py` refuses non-loopback sockets, and PATH stubs refuse `modal`, `wandb`, and `hf`.

## Limits

- The drives never reach a GPU, Modal, a remote dataset, or W&B. A real GPU run is proven only by an approved launch from its run card.
- The chain uses `fixtures/tiny_bundle`. It proves the stage lifecycle, resume, and bundle contracts, not model quality. The tiny RL run has zero-variance groups, so its loss is 0 and its weights do not move.
- The launch-plans drive also runs each dry-run launcher file (not emit-completions, which only runs under `modal run`) as a subprocess with `--dry-run`, offline, and checks for a clean plan, because the entrypoint shim skips the file's own `__main__` path. It calls each script's `local_entrypoint` body with the argv `modal run` would pass. It does not prove Modal image builds or volume layout.

## Gotchas

- An unapproved launch writes nothing. The RL drive runs the committed fixture config, so a refusal that writes a report changes the checkout and fails the run.
- The RL launcher refuses even a dry run unless `RLVR_TIMEOUT_HOURS` equals the config's `runtime.timeout_hours`. The helper sets it to the fixture's value, 1.
- `rlvr-checkpoint-bundle` needs an RL bundle as its template. A DPO bundle has no `source_manifest` and is rejected.
