from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer

from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.data import DataError, TokenizedExample
from esme_posttrain.sft.sampling import generate_samples, markdown_fenced_text
from esme_posttrain.training.collate import IGNORE_INDEX


def final_assistant_cut(example: TokenizedExample) -> int:
    """Index of the first supervised token of the last assistant turn.

    Supervised label spans mark assistant-turn content; each span is preceded by
    an unsupervised ``assistant\\n`` header segment, so slicing ``input_ids`` up
    to this index keeps the whole conversation through the final assistant
    header and drops only the final turn's content. For single-assistant-turn
    examples this equals ``prompt_tokens``.
    """
    cut: int | None = None
    previous_supervised = False
    for index, label in enumerate(example.labels):
        supervised = label != IGNORE_INDEX
        if supervised and not previous_supervised:
            cut = index
        previous_supervised = supervised
    if cut is None:
        raise DataError(f"{example.source}:{example.row_id} has no supervised assistant tokens")
    return cut


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
    # Re-point prompt_tokens at the final assistant turn so both the rendered
    # prompt and the generation continue the last turn, not the first.
    prompts = tuple(
        replace(example, prompt_tokens=final_assistant_cut(example)) for example in multi_turn[:3]
    )
    generations = generate_samples(model, tokenizer, prompts, sample_new_tokens)
    lines = [
        "# Multi-Turn SFT Samples",
        "",
        f"Selected checkpoint step: {selected_step}",
        "",
        "Each prompt is a multi-turn conversation truncated before the final assistant "
        "turn's content (earlier turns and the trailing assistant header are kept); "
        "the model continues that turn.",
        "",
    ]
    for index, example in enumerate(prompts, start=1):
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
