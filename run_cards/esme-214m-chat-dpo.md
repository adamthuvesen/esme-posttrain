# Run Card: Esme-214M-Chat DPO

This is the preference-tuning stage of the standard post-training chain:

```text
Esme-214M-Base -> Esme-214M-Instruct -> Esme-214M-Chat -> Esme-214M-RL
```

## Artifact

- Produces: `Esme-214M-Chat`
- Starts from: `Esme-214M-Instruct`
- Config: `configs/esme-214m-chat-dpo.json`
- Launcher: private operator module.

## Method

The stage runs one offline DPO pass (`loss_type=sigmoid`) on the repo's native
`DenseBackbone` stack. TRL is not used because Esme is a from-scratch model, not
a Hugging Face `PreTrainedModel`.

- Preference data: `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs` /
  `test_prefs`), revision `3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`, MIT.
- Prompts and responses use the repo's multi-turn chat template plus `<eos>`.
- Prompt tokens are masked so only response tokens score.
- Light filtering drops overlong rows and identical chosen/rejected pairs.
- Config anchor: `beta 0.5`, `lr 1.0e-6`, cosine schedule,
  `warmup_ratio 0.1`, 2 epochs, `max_length 1024`,
  `max_prompt_length 512`, effective batch `128`, bf16.
- Chosen/rejected log probabilities are logged on eval to catch likelihood
  displacement.

Length normalization is available through `dpo.length_normalized` and is the
first fix to try if response length starts driving the preference margin.

## Acceptance

The DPO checkpoint should:

- improve held-out preference accuracy over the SFT reference,
- keep chosen-logp from collapsing relative to the reference,
- reduce repetition and/or stabilize response length without degrading
  coherence.

This is chat polish for a 214M model. Repetition, length, likelihood, and
coherence proxies matter more here than benchmark claims meant for much larger
models.

## Accepted Result

The accepted run is `esme-214m-chat-dpo-full`, selected on
`eval/preference_accuracy`:

- Selected checkpoint: step `600`, preference accuracy `0.67362` over `959`
  held-out pairs.
- Eval margin `0.38544`; eval DPO loss `0.61030`.
- Chosen/rejected mean logp `-234.717` / `-238.552`; the
  likelihood-displacement watch held (chosen stays above rejected).

Source artifact: `runs/esme-214m-chat-dpo-full/best-checkpoint.json`
(sha256 `146cc3239f3c5bf92a39cd1a9c196ac91a737956ca058d490729bab87c1412eb`).
Run outputs are gitignored; the accepted numbers are quoted here so the
result is citable from a fresh clone.

## Chat Eval Result (SFT vs DPO)

The side-by-side chat-quality eval (`dpo/chat_eval.py`) was run once on Modal
(A100, CUDA, torch `2.12.1+cu130`, python `3.12.10`) on 2026-06-28 against the
accepted checkpoints: the DPO best checkpoint (step `600`) and the SFT
reference (`esme-214m-sft-multiturn-full`, step `6300`). Both models were
evaluated with the same tokenizer, the same 8 fixed conversational prompts,
and the same two decoders, at `96` max new tokens. The results were left on
the Modal volume until 2026-07-10, when they were downloaded and pinned here.

| Decoder | Metric | SFT | DPO |
| --- | --- | --- | --- |
| greedy | mean response length | 38.0 | 38.5 |
| greedy | mean 3-gram repetition | 0.038 | 0.027 |
| greedy | max 3-gram repetition | 0.263 | 0.150 |
| nucleus_p0.95_rep1.3 | mean response length | 40.0 | 46.0 |
| nucleus_p0.95_rep1.3 | mean 3-gram repetition | 0.007 | 0.005 |
| nucleus_p0.95_rep1.3 | max 3-gram repetition | 0.053 | 0.026 |

Mean and max 3-gram repetition are reduced by DPO under both decoders, with
no degenerate length shift. Five truncation-at-token-cap flags were raised
(three SFT, two DPO); no loops and no empty generations were seen. The
`context_carryover` prompt is still confabulated by both models under greedy
decoding; under nucleus decoding it is answered correctly by the DPO model,
while the SFT model rambles to the cap.

No LLM judge is configured, so no judge scores are carried in this record.
Per the acceptance policy, any future judge scores are reported with
repeated-judge spread (K>=5 passes over fixed generations) and are never the
checkpoint selector.

Source artifacts (gitignored under `runs/esme-214m-chat-dpo-full/`, also kept
on the `esme-posttrain-esme-chat-dpo` volume):

- `chat-eval-sft-vs-dpo.json`
  (sha256 `566ec6518a0bdef937dfc77aee9773672fd0e805adcdfd4bd1880cbd61f184d3`)
- `chat-eval-sft-vs-dpo.md`, the full side-by-side transcripts
  (sha256 `3dc42bee58bfa6da7b0ee86708b24fd489799a10848a33b6f110764bc0fe3a65`)
- `chat-eval-environment.txt`
  (sha256 `33b43d815c55cb869773ed7e21a906e8e9d6c1b2241ed4aa3fccd245f4a2ab4f`)

## Safe Local Commands

```bash
uv run esme-posttrain chat-dpo-dry-run --config configs/esme-214m-chat-dpo.json --json
uv run esme-posttrain chat-dpo-cpu-fixture --config configs/esme-214m-chat-dpo.json --json
```
