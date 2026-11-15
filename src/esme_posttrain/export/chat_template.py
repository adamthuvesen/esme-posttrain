"""The one owner of the Esme newline chat template.

Training renders and the exported bundle manifest both come from this module.
`llm-infer` renders a served conversation from the manifest's `chat_template`
object: each message through `roles[role]`, then `generation_prompt`, and
tokenizes the text with `add_special_tokens=False`.

Training ends each assistant turn with the `<eos>` token id. The assistant
pattern ends with the literal text `<eos>`, so served history matches training
only when the tokenizer encodes that text as the single `<eos>` id.
`require_eos_text_is_eos_id` checks that before a bundle ships the template.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from tokenizers import Tokenizer

ChatRole = Literal["system", "user", "assistant"]

CHAT_TEMPLATE_ID = "esme_newline_v2"
EOS_TOKEN = "<eos>"
GENERATION_PROMPT = "assistant\n"
ROLE_PATTERNS: dict[ChatRole, str] = {
    "system": "system\n{content}\n",
    "user": "user\n{content}\n",
    "assistant": GENERATION_PROMPT + "{content}" + EOS_TOKEN,
}


class ChatTemplateError(ValueError):
    pass


def render_turn(role: ChatRole, content: str) -> str:
    return ROLE_PATTERNS[role].format(content=content)


def render_prompt(turns: Iterable[tuple[ChatRole, str]]) -> str:
    """Render turns, then open the assistant reply with the generation prompt."""
    return "".join(render_turn(role, content) for role, content in turns) + GENERATION_PROMPT


def chat_template_manifest() -> dict[str, Any]:
    """The `chat_template` object written into an exported chat bundle manifest."""
    return {
        "id": CHAT_TEMPLATE_ID,
        "format": "role_newline_content_newline_assistant_content_eos",
        "roles": dict(ROLE_PATTERNS),
        "generation_prompt": GENERATION_PROMPT,
        "example": render_prompt([("user", "...")]),
        "add_special_tokens": False,
    }


def require_eos_text_is_eos_id(tokenizer: Tokenizer) -> int:
    """Return the `<eos>` id once the text `<eos>` encodes to exactly that id."""
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ChatTemplateError(f"tokenizer is missing required {EOS_TOKEN} token")
    encoded = list(tokenizer.encode(EOS_TOKEN, add_special_tokens=False).ids)
    if encoded != [eos_id]:
        raise ChatTemplateError(
            f"tokenizer encodes the text {EOS_TOKEN!r} as {encoded}, not [{eos_id}]; "
            f"{CHAT_TEMPLATE_ID} ends assistant turns with that text, so served history "
            f"would not match training. Register {EOS_TOKEN} as a special token in "
            "tokenizer.json added_tokens (special: true, same id) and export again"
        )
    return int(eos_id)
