"""Configuration for the tiny MLX transformer experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TinyTransformerConfig:
    """Small bidirectional encoder config aimed at roughly 15k trainable parameters."""

    vocab_size: int = 256
    max_length: int = 256
    num_layers: int = 4
    d_model: int = 16
    num_heads: int = 2
    ff_dim: int = 24
    dropout: float = 0.1
    num_classes: int = 2
    pooling: str = "cls"

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serializable representation."""

        return asdict(self)
