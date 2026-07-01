from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from esme_posttrain.modeling import DenseBackbone
from esme_posttrain.sft.sampling import markdown_fenced_text


def write_chat_samples(
    path: Path,
    model: DenseBackbone,
    tokenizer: Tokenizer,
    eval_pairs: tuple[Any, ...],
    *,
    selected_step: int,
) -> None:
    eos_id = tokenizer.token_to_id("<eos>")
    lines = [
        "# DPO Chat Samples",
        "",
        f"Selected checkpoint step: {selected_step}",
        "",
        "Each prompt is rendered with the chat template; the DPO policy continues it.",
        "",
    ]
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for index, pair in enumerate(eval_pairs[:3], start=1):
            prompt_ids = list(pair.prompt_ids)
            generated = model.generate(
                torch.tensor([prompt_ids], dtype=torch.long, device=device),
                max_new_tokens=8,
                eos_token_id=eos_id,
            )
            new_ids = generated[0].detach().cpu().tolist()[len(prompt_ids) :]
            if eos_id is not None and eos_id in new_ids:
                new_ids = new_ids[: new_ids.index(eos_id)]
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
            generation = tokenizer.decode(new_ids, skip_special_tokens=False)
            lines.extend(
                [
                    f"## Chat Sample {index}",
                    "",
                    "Prompt:",
                    "",
                    *markdown_fenced_text(prompt_text),
                    "",
                    "Generation:",
                    "",
                    *markdown_fenced_text(generation),
                    "",
                ]
            )
    if was_training:
        model.train()
    path.write_text("\n".join(lines), encoding="utf-8")
