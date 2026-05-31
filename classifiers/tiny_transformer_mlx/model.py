"""Tiny bidirectional transformer encoder implemented in MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig


def sinusoidal_position_encoding(max_length: int, d_model: int) -> mx.array:
    """Create non-trainable sinusoidal position encodings."""

    positions = np.arange(max_length, dtype=np.float32)[:, None]
    dims = np.arange(d_model, dtype=np.float32)[None, :]
    angle_rates = np.exp(-(np.log(10_000.0) / d_model) * (2 * (dims // 2)))
    angles = positions * angle_rates
    encoding = np.zeros((max_length, d_model), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return mx.array(encoding)


def attention_bias(attention_mask: mx.array) -> mx.array:
    """Convert a 1/0 padding mask into an additive attention bias."""

    return mx.where(
        attention_mask[:, None, None, :] > 0,
        mx.array(0.0, dtype=mx.float32),
        mx.array(-1e9, dtype=mx.float32),
    )


def causal_attention_bias(attention_mask: mx.array) -> mx.array:
    """Combine a padding mask with a causal no-lookahead mask."""

    seq_len = attention_mask.shape[1]
    padding = attention_bias(attention_mask)
    future = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
    causal = mx.where(
        future[None, None, :, :],
        mx.array(-1e9, dtype=mx.float32),
        mx.array(0.0, dtype=mx.float32),
    )
    return padding + causal


class SwiGLUFeedForward(nn.Module):
    """Small gated feed-forward block."""

    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.up_gate = nn.Linear(d_model, ff_dim * 2, bias=False)
        self.down = nn.Linear(ff_dim, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, values = mx.split(self.up_gate(x), 2, axis=-1)
        return self.down(nn.silu(gate) * values)


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional transformer encoder block."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.attention = nn.MultiHeadAttention(
            dims=config.d_model,
            num_heads=config.num_heads,
            bias=False,
        )
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.ff_dim)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        normalized = self.attn_norm(x)
        attended = self.attention(normalized, normalized, normalized, mask=mask.astype(normalized.dtype))
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x


class TinyTransformerClassifier(nn.Module):
    """Tiny BPE-token transformer encoder for binary sentiment classification."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if config.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = [EncoderBlock(config) for _ in range(config.num_layers)]
        self.final_norm = nn.RMSNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self._position_encoding = sinusoidal_position_encoding(config.max_length, config.d_model)

    def encode(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """Encode token ids with the bidirectional transformer stack."""

        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids)
        x = x + self._position_encoding[:seq_len].astype(x.dtype)[None, :, :]
        x = self.dropout(x)

        mask = attention_bias(attention_mask)
        for block in self.blocks:
            x = block(x, mask)

        return self.final_norm(x)

    def pool(self, encoded: mx.array, attention_mask: mx.array) -> mx.array:
        """Pool token states for classification."""

        if self.config.pooling == "cls":
            return encoded[:, 0, :]

        mask = attention_mask.astype(mx.float32)[:, :, None]
        summed = mx.sum(encoded * mask, axis=1)
        counts = mx.maximum(mx.sum(mask, axis=1), mx.array(1.0, dtype=mx.float32))
        return summed / counts

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        encoded = self.encode(input_ids, attention_mask)
        pooled = self.pool(encoded, attention_mask)
        return self.classifier(pooled)


class TinyTransformerForMaskedLM(nn.Module):
    """Bidirectional transformer encoder with a masked-language-modeling head."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.encoder = TinyTransformerClassifier(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        encoded = self.encoder.encode(input_ids, attention_mask)
        return self.lm_head(encoded)


class TinyTransformerDecoderLM(nn.Module):
    """Tiny causal decoder trained with next-token and label-token objectives."""

    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = [EncoderBlock(config) for _ in range(config.num_layers)]
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self._position_encoding = sinusoidal_position_encoding(config.max_length, config.d_model)

    def encode(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """Encode token ids with causal self-attention."""

        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids)
        x = x + self._position_encoding[:seq_len].astype(x.dtype)[None, :, :]
        x = self.dropout(x)

        mask = causal_attention_bias(attention_mask)
        for block in self.blocks:
            x = block(x, mask)

        return self.final_norm(x)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        encoded = self.encode(input_ids, attention_mask)
        return self.lm_head(encoded)

    def label_logits(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        label_positions: mx.array,
        label_token_ids: mx.array,
    ) -> mx.array:
        """Return logits for only the negative/positive label tokens."""

        encoded = self.encode(input_ids, attention_mask)
        rows = mx.arange(input_ids.shape[0])
        label_states = encoded[rows, label_positions, :]
        vocab_logits = self.lm_head(label_states)
        return vocab_logits[:, label_token_ids]


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable scalar parameters."""

    return int(sum(math.prod(array.shape) for _, array in tree_flatten(model.trainable_parameters())))
