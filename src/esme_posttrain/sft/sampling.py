from __future__ import annotations

import re

import torch
from tokenizers import Tokenizer

from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.data import TokenizedExample
from esme_posttrain.sft.eval_suite import EvalSplit


def markdown_fenced_text(text: str) -> list[str]:
    max_backticks = max((len(match.group(0)) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, max_backticks + 1)
    return [f"{fence}text", text, fence]


def generate_samples(
    model: DenseBackbone,
    tokenizer: Tokenizer,
    eval_examples: tuple[TokenizedExample, ...],
    sample_new_tokens: int,
) -> tuple[str, ...]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    eos_token_id = tokenizer.token_to_id("<eos>")
    outputs: list[str] = []
    with torch.no_grad():
        for example in eval_examples[:3]:
            prompt_ids = example.input_ids[: example.prompt_tokens]
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(
                input_tensor,
                max_new_tokens=sample_new_tokens,
                eos_token_id=eos_token_id,
            )
            generated_ids = generated[0].detach().cpu().tolist()
            generated_ids = truncate_after_prompt_eos(
                generated_ids,
                prompt_tokens=len(prompt_ids),
                eos_token_id=eos_token_id,
            )
            outputs.append(
                tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                )
            )
    if was_training:
        model.train()
    return tuple(outputs)


def truncate_after_prompt_eos(
    generated_ids: list[int],
    *,
    prompt_tokens: int,
    eos_token_id: int | None,
) -> list[int]:
    if eos_token_id is None:
        return generated_ids
    try:
        eos_index = generated_ids.index(eos_token_id, prompt_tokens)
    except ValueError:
        return generated_ids
    return generated_ids[:eos_index]


def sample_prompt_examples(eval_splits: tuple[EvalSplit, ...]) -> tuple[TokenizedExample, ...]:
    examples: list[TokenizedExample] = []
    for split in eval_splits:
        if split.selector_weight <= 0:
            continue
        examples.extend(split.examples[: max(0, 3 - len(examples))])
        if len(examples) >= 3:
            break
    if not examples:
        examples.extend(eval_splits[0].examples[:3])
    return tuple(examples)
