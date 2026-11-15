# Esme Post-training

Terms for `esme-posttrain`: SFT, DPO, and RLVR on top of the `Esme-214M-Base` bundle. Bundle and chat terms come from `esme-pretrain`.

## Contract terms (owned by esme-pretrain)

Each term below: "Defined in esme-pretrain `GLOSSARY.md`."

- **Bundle**, **Bundle format**, **Manifest**, **Base bundle / chat bundle**
- **Chat template** and **Chat template version**. This repo writes the template: `src/esme_posttrain/export/chat_template.py`.
- **EOS (`<eos>`)** and **EOS ids**
- **Special tokens** and **add_special_tokens**
- **Extension fields**. This repo writes them: `src/esme_posttrain/export/bundle_writer.py::write_dense_bundle`.

## Post-training terms

**Stage**:
One of SFT, DPO, RLVR. Stages hand off artifacts (checkpoint or bundle), never code. Stages never import each other.

**SFT**:
Supervised fine-tuning on conversations with all-assistant masking. Output: `Esme-214M-Instruct`.

**DPO**:
Preference tuning from the SFT checkpoint. The SFT checkpoint is the frozen reference. Output: `Esme-214M-Chat`.

**RLVR**:
GRPO on Countdown-Lite with a verifier reward and a KL penalty to the frozen Chat model. Output: `Esme-214M-RL`.

**Frozen reference**:
The previous stage's artifact. It is the warm start and the regularization target. It never changes during the stage.

**Sweep**:
A bounded short-arm run that gates a full run. DPO sweeps beta over `EXPECTED_SWEEP_BETAS` (0.1, 0.3, 0.5). SFT sweeps learning rate on matched held-out loss.

**Learning gate**:
The `learning_gate` config block. A full run refuses until the sweep evidence passes. The refusal text is `LEARNING_GATE_BLOCKER` (`src/esme_posttrain/dpo/launch.py`). `_check_learning_gate` (`src/esme_posttrain/dpo/config.py`) checks only the config shape.

**Spend cap**:
The hard USD limit per launch. Smoke: $2 (`src/esme_posttrain/launch/config_models.py::SMOKE_SPEND_CAP_USD`). Full runs: SFT $40, DPO $15, RLVR $25. A full run refuses without `--approved`.
_Avoid_: budget (a projected cost, not the limit)

**Run card**:
The written spec Adam approves before spend: data, budget, config, hardware, cost, artifact. Files are in `run_cards/`.

**Countdown-Lite**:
The only RLVR reward target: a small arithmetic-expression task with a verifier. GSM8K-lite is a transfer eval, not a reward source.

## Name drift

- "Chat bundle" means the DPO bundle. The RLVR bundle has `model.stage: "rlvr"` and no `chat_template`.
- `decoding.default_add_special_tokens` is written here but `llm-infer` does not read it. Its runtime parameter of the same name is a different thing.
