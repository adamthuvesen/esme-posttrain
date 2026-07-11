from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, cast

import torch

F = torch.nn.functional


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    vocab_size: int
    context_length: int
    embedding_dim: int
    layers: int
    heads: int
    feedforward_dim: int
    kv_heads: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True
    logit_soft_cap: float = 0.0
    z_loss_weight: float = 1e-4
    attention_kind: str = "gqa"
    mtp_predict_tokens: int = 0

    def __post_init__(self) -> None:
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.effective_kv_heads < 1:
            raise ValueError("kv_heads must be at least 1")
        if self.heads % self.effective_kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        if self.attention_kind == "mha" and self.effective_kv_heads != self.heads:
            raise ValueError("attention_kind='mha' requires kv_heads to equal heads")
        if self.attention_kind not in {"mha", "gqa"}:
            raise ValueError("only mha and gqa attention bundles are loadable in posttrain")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.logit_soft_cap < 0:
            raise ValueError("logit_soft_cap must be non-negative")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight must be non-negative")
        if self.mtp_predict_tokens != 0:
            raise ValueError("mtp_predict_tokens must be 0 for the SFT pilot")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.heads

    @property
    def effective_kv_heads(self) -> int:
        return self.heads if self.kv_heads is None else self.kv_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackboneConfig:
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - field_names)
        if unknown:
            raise ValueError(f"unknown backbone config keys: {unknown}")
        return cls(**payload)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden_fp32 = hidden.float()
        normed = hidden_fp32 * torch.rsqrt(hidden_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


def build_rope_cache(
    context_length: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(context_length, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def apply_rope(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(query.dtype)[None, None, :, :]
    sin = sin.to(query.dtype)[None, None, :, :]
    return (query * cos) + (_rotate_half(query) * sin), (key * cos) + (_rotate_half(key) * sin)


class SwiGLU(torch.nn.Module):
    def __init__(self, dim: int, feedforward_dim: int) -> None:
        super().__init__()
        self.w_gate = torch.nn.Linear(dim, feedforward_dim, bias=False)
        self.w_up = torch.nn.Linear(dim, feedforward_dim, bias=False)
        self.w_down = torch.nn.Linear(feedforward_dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(hidden)) * self.w_up(hidden))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        if config.effective_kv_heads != config.heads:
            raise ValueError("MultiHeadAttention requires kv_heads to equal heads")
        self.heads = config.heads
        self.head_dim = config.head_dim
        dim = config.embedding_dim
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        query = self.wq(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        key = self.wk(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        value = self.wv(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query, key = apply_rope(query, key, cos, sin)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.wo(attention)


class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.effective_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeat = self.heads // self.kv_heads
        dim = config.embedding_dim
        kv_dim = self.kv_heads * self.head_dim
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, kv_dim, bias=False)
        self.wv = torch.nn.Linear(dim, kv_dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        query = self.wq(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        key = self.wk(hidden).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.wv(hidden).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query, key = apply_rope(query, key, cos, sin)
        if self.kv_repeat != 1:
            key = key.repeat_interleave(self.kv_repeat, dim=1)
            value = value.repeat_interleave(self.kv_repeat, dim=1)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.wo(attention)


def build_attention(config: BackboneConfig) -> MultiHeadAttention | GroupedQueryAttention:
    if config.attention_kind == "mha":
        return MultiHeadAttention(config)
    if config.attention_kind == "gqa":
        return GroupedQueryAttention(config)
    raise ValueError(f"unsupported attention_kind: {config.attention_kind}")


class DecoderBlock(torch.nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.attention = build_attention(config)
        self.feedforward_norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.feedforward = SwiGLU(config.embedding_dim, config.feedforward_dim)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), cos, sin)
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class DenseBackbone(torch.nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = torch.nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.lm_head = torch.nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        cos, sin = build_rope_cache(
            config.context_length, config.head_dim, config.rope_theta, torch.device("cpu")
        )
        self.rope_cos: torch.Tensor
        self.rope_sin: torch.Tensor
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = (2.0 * self.config.layers) ** -0.5
        with torch.no_grad():
            for module in self.blocks:
                block = cast(DecoderBlock, module)
                block.attention.wo.weight.mul_(scale)
                block.feedforward.w_down.weight.mul_(scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input ids must have shape [batch, sequence]")
        seq = input_ids.shape[1]
        if seq > self.config.context_length:
            raise ValueError(f"sequence length {seq} exceeds context {self.config.context_length}")
        cos = self.rope_cos[:seq].to(input_ids.device)
        sin = self.rope_sin[:seq].to(input_ids.device)
        hidden = self.token_embedding(input_ids)
        for module in self.blocks:
            block = cast(DecoderBlock, module)
            hidden = block(hidden, cos, sin)
        return self.lm_head(self.final_norm(hidden))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id must be inside the vocabulary")
        was_training = self.training
        self.eval()
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            conditioned = generated[:, -self.config.context_length :]
            next_logits = soft_cap_logits(self(conditioned)[:, -1, :], self.config.logit_soft_cap)
            if temperature <= 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            if eos_token_id is not None:
                eos_ids = torch.full_like(next_id, eos_token_id)
                next_id = torch.where(finished[:, None], eos_ids, next_id)
                finished |= next_id.squeeze(-1) == eos_token_id
            generated = torch.cat((generated, next_id), dim=1)
            if bool(finished.all()):
                break
        if was_training:
            self.train()
        return generated


def soft_cap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    if cap <= 0.0:
        return logits
    return cap * torch.tanh(logits / cap)


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    z_loss_weight: float = 0.0,
    logit_soft_cap: float = 0.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_logits = logits.reshape(-1, logits.shape[-1]).float()
    capped_logits = soft_cap_logits(raw_logits, logit_soft_cap)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if valid.sum() == 0:
        total = raw_logits.sum() * 0.0
        return total, {"ce_loss": 0.0, "total_loss": float(total.detach())}

    log_z = torch.logsumexp(capped_logits, dim=-1)
    target_logits = capped_logits.gather(-1, flat_targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    cross_entropy = (log_z - target_logits)[valid].mean()
    total = cross_entropy
    components = {"ce_loss": float(cross_entropy.detach())}
    if z_loss_weight > 0.0:
        raw_log_z = torch.logsumexp(raw_logits, dim=-1)
        z_loss = z_loss_weight * (raw_log_z[valid] ** 2).mean()
        total = total + z_loss
        components["z_loss"] = float(z_loss.detach())
    components["total_loss"] = float(total.detach())
    return total, components


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))
