# Esme-214M-Instruct SFT Recipe

This internal page records the single-turn SFT recipe and accepted evidence for `Esme-214M-Instruct`. The public docs describe the current multi-turn SFT -> DPO -> RLVR path.

## Scope

This is supervised full fine-tuning from `Esme-214M-Base` to `Esme-214M-Instruct`. It is not RL, not DPO, and not LoRA/QLoRA.

The loop formats user/assistant rows, masks prompt tokens with `-100`, trains only on assistant completions, records namespaced train/eval metrics, writes selected-row manifests, saves restartable checkpoints, and generates fixed samples with EOS-aware stopping.

## Data And Selection

- Train mix: concise single-turn rows from `HuggingFaceTB/smol-smoltalk` plus instruction-following rows from `allenai/tulu-3-sft-personas-instruction-following`.
- Eval-only guardrail: `HuggingFaceH4/no_robots`; it is not part of training.
- Selection target: bounded train-token budget with explicit survivor counts and manifests.
- Selector: matched held-out response loss over the approved train-distribution eval mix.

## Accepted Evidence

`learning_gate.evidence` records stopped-run reconciliation and bounded matched interval-eval sweep evidence. The matched Base step-0 response loss was `2.187998926732106`; the selected `lr3e-5-mb2-ga8-eb16` arm reached `1.7372412149933547` at step 60.

The accepted local Esme-214M-Instruct SFT run selects step 6400 by matched eval. Matched response loss improves from `2.1186577799740736` to `1.1952465851220688`, and cost is `$3.9872` under cap.

## Rerun Rule

Each full-data launch still needs explicit approval. Fresh full runs must write to a fresh output directory; resume runs must find a valid latest checkpoint in the target directory. Checkpoints written at format v3 carry RNG state and data position, and resume restores them so a resumed run continues the uninterrupted run's stream. Checkpoints without those fields still load and resume from the fresh seed.

The selected optimizer recipe is learning rate `3e-5`, microbatch size 2, gradient accumulation 8, effective batch size 16, cosine decay, and 700 warmup steps on A100.
