from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.data import TokenizedExample
from esme_posttrain.sft.sampling import generate_samples, markdown_fenced_text


def write_multi_turn_samples(
    path: Path,
    model: DenseBackbone,
    tokenizer: Tokenizer,
    eval_examples: tuple[TokenizedExample, ...],
    *,
    sample_new_tokens: int,
    selected_step: int,
) -> None:
    multi_turn = tuple(example for example in eval_examples if example.assistant_turns > 1) or (
        eval_examples
    )
    generations = generate_samples(model, tokenizer, multi_turn, sample_new_tokens)
    lines = [
        "# Multi-Turn SFT Samples",
        "",
        f"Selected checkpoint step: {selected_step}",
        "",
        "Each prompt is a multi-turn conversation truncated before the final assistant turn; "
        "the model continues it.",
        "",
    ]
    for index, example in enumerate(multi_turn[:3], start=1):
        prompt_ids = example.input_ids[: example.prompt_tokens]
        prompt = tokenizer.decode(list(prompt_ids), skip_special_tokens=False)
        lines.extend(
            [
                f"## Multi-Turn Sample {index} (assistant_turns={example.assistant_turns})",
                "",
                "Prompt:",
                "",
                *markdown_fenced_text(prompt),
                "",
                "Generation:",
                "",
                *markdown_fenced_text(generations[index - 1]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
