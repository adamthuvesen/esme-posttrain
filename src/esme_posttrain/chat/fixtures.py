"""Tiny model and tokenizer builders for local CPU fixtures."""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from esme_posttrain.modeling import BackboneConfig


def tiny_backbone_config() -> BackboneConfig:
    return BackboneConfig(
        name="tiny-multi-turn-sft-fixture",
        vocab_size=32,
        context_length=48,
        embedding_dim=16,
        layers=1,
        heads=2,
        kv_heads=1,
        feedforward_dim=32,
        z_loss_weight=0.0,
    )


def tiny_chat_tokenizer() -> Tokenizer:
    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "user": 2,
        "assistant": 3,
        "system": 4,
        "say": 5,
        "red": 6,
        "blue": 7,
        "green": 8,
        "repeat": 9,
        "one": 10,
        "two": 11,
        "again": 12,
        "helpful": 13,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<eos>"])
    return tokenizer
