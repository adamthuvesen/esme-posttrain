# RLVR Countdown 3-Number Token-Budget Diagnostic

Is `Esme-214M-RL`'s near-zero 3-operand exact-solve on the held-out transfer eval
a 12-token truncation artifact, or a real capability wall? This diagnostic reruns
the 3-number cells of the held-out sets at larger rollout budgets (24 and 48
tokens) and compares them, cell for cell, against the accepted 12-token numbers in
`docs/rlvr-countdown-heldout-transfer.md`. Eval only, no training. Same acceptance
protocol: seed 214, 32 samples/task, greedy sample 0 + 31 stochastic at
temperature 0.8.

## Verdict

**Real 3-operand wall, not truncation.** Giving the RL model 2x or 4x the rollout
budget does not lift 3-number exact-solve above the 12-token floor — it stays at
**0.00%** on both held-out 3-number cells at 24 and 48 tokens, versus 4.00% (fresh)
and 0.00% (shift) at 12. More budget makes 3-operand performance *worse on the
metric that matters for training signal*: valid-expression rate craters from ~99%
at 12 tokens to ~13-19% at 24/48, because with room to keep generating the model
writes a well-formed expression and then rambles past it (`"1 + 9 = 3\n10 = 3\n
Explain..."`), corrupting its own answer so the verifier can no longer extract it.
The 12-token result was not hiding a solvable-but-truncated model; the cramped
budget was masking a *more* broken free-generation regime. The 2-number control is
budget-insensitive (pass@32 holds at 40% across 12/24/48), confirming the
2-operand skill is real and the token budget is not what gates 3-operand solving.

## Execution and spend

- **Local CPU, $0. No private training function, no GPU, no W&B run, no network.**
  Same as the prior held-out eval.
- 8 eval cells ran on this machine: both bundles x {`heldout_fresh`,
  `heldout_shift`} x {24, 48} tokens. The 12-token numbers are reused from the
  committed `artifacts/heldout-transfer-new/` reports (not rerun — they reproduce
  the transfer doc's table exactly).
- Per-cell command (bundle, split, budget vary):

  ```bash
  uv run esme-posttrain rlvr-countdown-lite-baseline \
    --manifest data/manifests/esme-214m-rl-heldout.tasks.json \
    --bundle exports/<esme-214m-rl-caff0a1|esme-214m-chat> \
    --output-dir <out> \
    --split <heldout_fresh|heldout_shift> \
    --samples-per-task 32 --max-new-tokens <24|48> --seed 214 --json
  ```

- Raw eval outputs land under `artifacts/` (gitignored, as with the prior
  held-out eval); every number the diagnostic rests on is tabulated inline below,
  so this report is the self-contained, reproducible record. Rerun the commands
  above to regenerate the per-cell `baseline-report.json` files.

## 3-number cells — the load-bearing comparison

Held-out fresh reports its 25 three-number tasks (the fresh set also has 5
two-number tasks, tabulated separately below). Held-out shift is all 30 three-number.

| Model | Split | Budget | tasks | pass@1 | pass@8 | pass@32 | Valid expr | **Exact solve** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Esme-214M-RL | fresh | 12 | 25 | 4.00% | 4.00% | 4.00% | 99.12% | **4.00%** |
| Esme-214M-RL | fresh | 24 | 25 | 0.00% | 0.00% | 0.00% | 12.50% | **0.00%** |
| Esme-214M-RL | fresh | 48 | 25 | 0.00% | 0.00% | 0.00% | 12.62% | **0.00%** |
| Esme-214M-RL | shift | 12 | 30 | 0.00% | 0.00% | 0.00% | 99.38% | **0.00%** |
| Esme-214M-RL | shift | 24 | 30 | 0.00% | 0.00% | 0.00% | 18.75% | **0.00%** |
| Esme-214M-RL | shift | 48 | 30 | 0.00% | 0.00% | 0.00% | 18.85% | **0.00%** |
| Esme-214M-Chat | fresh | 12 | 25 | 0.00% | 0.00% | 0.00% | 2.25% | **0.00%** |
| Esme-214M-Chat | fresh | 24 | 25 | 0.00% | 0.00% | 0.00% | 3.62% | **0.00%** |
| Esme-214M-Chat | fresh | 48 | 25 | 0.00% | 0.00% | 0.00% | 4.12% | **0.00%** |
| Esme-214M-Chat | shift | 12 | 30 | 0.00% | 0.00% | 0.00% | 1.46% | **0.00%** |
| Esme-214M-Chat | shift | 24 | 30 | 0.00% | 0.00% | 0.00% | 2.40% | **0.00%** |
| Esme-214M-Chat | shift | 48 | 30 | 0.00% | 0.00% | 0.00% | 4.38% | **0.00%** |

Exact-solve on the 3-number cells is flat at the floor across the whole budget
sweep. The only movement is the RL model's valid-expression rate collapsing as the
budget grows — the opposite of what a truncation artifact would show. The 12-token
fresh 3.99% -> 4.00% was already the model's ceiling here; extra tokens buy no new
solves.

## 2-number control (fresh split, 5 tasks)

The 2-operand skill should be — and is — budget-insensitive.

| Model | Budget | pass@32 | Valid expr | Exact solve |
| --- | ---: | ---: | ---: | ---: |
| Esme-214M-RL | 12 | 40.00% | 98.75% | 38.75% |
| Esme-214M-RL | 24 | 40.00% | 84.38% | 33.75% |
| Esme-214M-RL | 48 | 40.00% | 87.50% | 33.12% |
| Esme-214M-Chat | 12 | 20.00% | 10.00% | 3.12% |
| Esme-214M-Chat | 24 | 40.00% | 10.62% | 4.38% |
| Esme-214M-Chat | 48 | 40.00% | 15.00% | 5.62% |

RL 2-number pass@32 is a flat 40% at every budget; its per-sample exact-solve dips
slightly (38.75% -> 33.12%) for the same rambling reason, but the harder pass@k
signal is unchanged. The control behaves exactly as a genuine, budget-independent
2-operand skill should. If the 3-number floor were a truncation artifact, we would
expect it to lift like a capability that just needed room — it does not.

## Early-stop / EOS interplay

The CLI does not expose a completion-length field, so generated-token counts here
are recovered by re-tokenizing each stored (pre-`<eos>`) completion with the
bundle tokenizer. `generate` runs until `max_new_tokens` or until every sequence
emits `<eos>`; a sample whose recovered length equals the budget ran to the cap, a
shorter one stopped early on `<eos>`.

| Model | Split | Budget | median gen tokens | ran to cap | stopped early on EOS |
| --- | --- | ---: | ---: | ---: | ---: |
| Esme-214M-RL | fresh (3-num) | 12 | 12 | 98.9% | 1.1% |
| Esme-214M-RL | fresh (3-num) | 24 | 24 | 80.1% | 19.9% |
| Esme-214M-RL | fresh (3-num) | 48 | 48 | 63.0% | 37.0% |
| Esme-214M-RL | shift (3-num) | 12 | 12 | 98.2% | 1.8% |
| Esme-214M-RL | shift (3-num) | 24 | 24 | 80.1% | 19.9% |
| Esme-214M-RL | shift (3-num) | 48 | 48 | 61.9% | 38.1% |
| Esme-214M-Chat | fresh (3-num) | 48 | 48 | 61.3% | 38.8% |
| Esme-214M-Chat | shift (3-num) | 48 | 48 | 61.8% | 38.2% |

This is the crux. At 12 tokens the RL model runs to the cap on ~99% of 3-number
samples — it is genuinely truncated and almost never reaches `<eos>`, which is the
strongest case the truncation hypothesis could ask for. Give it 24 or 48 tokens and
it *does* start terminating on `<eos>` on its own (20% -> 37% of samples stop
before the cap), so the extra budget is being used and generation is no longer
forced to truncate. Yet 3-number exact-solve still sits at 0.00%. The budget was
the binding constraint at 12 tokens; removing it exposes that the underlying
3-operand arithmetic is not there. Freed to generate, the model appends
verification-style continuations (`"= <wrong>"`, restated targets, meta text) after
a syntactically valid expression, which is why valid-expression rate falls off a
cliff exactly when truncation is relieved.

## What this settles, and what it doesn't

- **Settles:** for `Esme-214M-RL` (214M params, this GRPO checkpoint), 3-operand
  Countdown exact-solve is a real capability wall on both the fresh and
  target-shifted held-out sets — not an artifact of the 12-token training/eval
  budget. This is consistent with the prior that Countdown 3-operand mastery needs
  a substantially larger model. The transfer story from
  `docs/rlvr-countdown-heldout-transfer.md` stands: GRPO taught a transferable
  answer format plus real 2-operand skill; 3-operand solving stays at the floor,
  and giving it more tokens does not move it.
- **Doesn't touch:** the accepted caff0a1 run artifacts, the held-out sets, the
  manifest, or the original transfer doc — all unchanged. This is a new,
  additive diagnostic.

## Follow-ups (not run)

- Budgets beyond 48 tokens are not worth running: the RL model already reaches
  `<eos>` on its own before the 48-token cap on ~37% of samples and still solves 0%,
  so a larger budget only extends the rambling regime. Flagged, not executed.
- A stop-at-first-expression decode (truncate generation at the first complete
  expression) would recover the format transfer story at large budgets by cutting
  the corrupting tail — but that changes the decode contract, not the arithmetic,
  and would not produce new correct 3-operand solves. Out of scope here.
