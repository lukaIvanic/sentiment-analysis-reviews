"""Tiny bidirectional transformer encoder implemented in PyTorch."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig


def sinusoidal_position_encoding(max_length: int, d_model: int) -> torch.Tensor:
    """Create non-trainable sinusoidal position encodings."""

    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    dims = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    angle_rates = torch.exp(-(math.log(10_000.0) / d_model) * (2 * torch.div(dims, 2, rounding_mode="floor")))
    angles = positions * angle_rates
    encoding = torch.zeros((max_length, d_model), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles[:, 0::2])
    encoding[:, 1::2] = torch.cos(angles[:, 1::2])
    return encoding


class SwiGLUFeedForward(nn.Module):
    """Small gated feed-forward block."""

    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.up_gate = nn.Linear(d_model, ff_dim * 2, bias=False)
        self.down = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, values = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * values)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention through PyTorch SDPA so CUDA can choose fused kernels."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        allowed_mask = attention_mask.to(torch.bool)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out(attended)


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional transformer encoder block."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.ff_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attn_norm(x), attention_mask))
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x


class TinyTransformerClassifier(nn.Module):
    """Tiny BPE-token transformer encoder for binary sentiment classification."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        if config.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.RMSNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(config.max_length, config.d_model),
            persistent=False,
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode token ids with the bidirectional transformer stack."""

        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids)
        x = x + self.position_encoding[:seq_len].to(dtype=x.dtype)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.final_norm(x)

    def pool(self, encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token states for classification."""

        if self.config.pooling == "cls":
            return encoded[:, 0, :]

        mask = attention_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        summed = (encoded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(input_ids, attention_mask)
        pooled = self.pool(encoded, attention_mask)
        return self.classifier(pooled)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
