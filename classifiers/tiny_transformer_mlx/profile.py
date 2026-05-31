"""Profile the tiny MLX transformer training step."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig
from classifiers.tiny_transformer_mlx.model import (
    TinyTransformerClassifier,
    attention_bias,
    count_trainable_parameters,
)
from classifiers.tiny_transformer_mlx.run import loss_fn
from src.sentiment.artifacts import ensure_dir, write_json
from src.sentiment.paths import OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "transformer" / "tiny_transformer_mlx_profile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-profile tiny MLX transformer shapes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--capture-shape",
        type=str,
        default=None,
        help="Optional shape to capture as BxS, for example 256x256.",
    )
    parser.add_argument(
        "--layer-breakdown",
        action="store_true",
        help="Synchronize and time individual forward-pass stages for one shape.",
    )
    return parser.parse_args()


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def make_batch(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    random_state: int,
) -> tuple[mx.array, mx.array, mx.array]:
    rng = np.random.default_rng(random_state)
    input_ids = rng.integers(4, vocab_size, size=(batch_size, sequence_length), dtype=np.int32)
    input_ids[:, 0] = 2
    input_ids[:, -1] = 3
    attention_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
    labels = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)
    return mx.array(input_ids), mx.array(attention_mask), mx.array(labels)


def benchmark_shape(
    args: argparse.Namespace,
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[str, float | int]:
    mx.random.seed(args.random_state)
    config = TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_length=sequence_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = TinyTransformerClassifier(config)
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    mx.eval(model.parameters())
    input_ids, attention_mask, labels = make_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=args.vocab_size,
        random_state=args.random_state,
    )
    mx.eval(input_ids, attention_mask, labels)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def forward() -> None:
        logits = model(input_ids, attention_mask)
        mx.eval(logits)

    def train_step() -> None:
        loss, gradients = loss_and_grad(model, input_ids, attention_mask, labels)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)

    for _ in range(args.warmup_steps):
        train_step()

    mx.reset_peak_memory()
    forward_seconds = [timed(forward) for _ in range(args.steps)]
    train_seconds = [timed(train_step) for _ in range(args.steps)]

    result = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "trainable_parameters": count_trainable_parameters(model),
        "forward_ms_mean": float(np.mean(forward_seconds) * 1000),
        "forward_ms_min": float(np.min(forward_seconds) * 1000),
        "train_step_ms_mean": float(np.mean(train_seconds) * 1000),
        "train_step_ms_min": float(np.min(train_seconds) * 1000),
        "examples_per_second_mean": float(batch_size / np.mean(train_seconds)),
        "tokens_per_second_mean": float(batch_size * sequence_length / np.mean(train_seconds)),
        "metal_active_memory_mb": float(mx.get_active_memory() / 1024 / 1024),
        "metal_peak_memory_mb": float(mx.get_peak_memory() / 1024 / 1024),
    }
    return result


def add_timing(timings: dict[str, list[float]], name: str, fn) -> mx.array:
    start = time.perf_counter()
    value = fn()
    mx.eval(value)
    timings.setdefault(name, []).append(time.perf_counter() - start)
    return value


def layer_breakdown(
    args: argparse.Namespace,
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[str, object]:
    """Synchronize after forward-pass stages to get a readable stage breakdown."""

    mx.random.seed(args.random_state)
    config = TinyTransformerConfig(
        vocab_size=args.vocab_size,
        max_length=sequence_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = TinyTransformerClassifier(config)
    model.train()
    mx.eval(model.parameters())
    input_ids, attention_mask, _ = make_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=args.vocab_size,
        random_state=args.random_state,
    )
    mx.eval(input_ids, attention_mask)

    # Warmup the kernels before timing.
    for _ in range(args.warmup_steps):
        logits = model(input_ids, attention_mask)
        mx.eval(logits)

    mx.reset_peak_memory()
    timings: dict[str, list[float]] = {}
    for _ in range(args.steps):
        x = add_timing(
            timings,
            "embedding_position_dropout",
            lambda: model.dropout(
                model.embedding(input_ids)
                + model._position_encoding[: input_ids.shape[1]][None, :, :]
            ),
        )
        mask = add_timing(timings, "attention_mask_bias", lambda: attention_bias(attention_mask))

        for block_index, block in enumerate(model.blocks, start=1):
            normalized = add_timing(
                timings,
                f"block_{block_index}_attn_norm",
                lambda block=block, x=x: block.attn_norm(x),
            )
            attended = add_timing(
                timings,
                f"block_{block_index}_attention",
                lambda block=block, normalized=normalized, mask=mask: block.attention(
                    normalized,
                    normalized,
                    normalized,
                    mask=mask,
                ),
            )
            x = add_timing(
                timings,
                f"block_{block_index}_attn_residual_dropout",
                lambda block=block, x=x, attended=attended: x + block.dropout(attended),
            )
            ffn_normalized = add_timing(
                timings,
                f"block_{block_index}_ffn_norm",
                lambda block=block, x=x: block.ffn_norm(x),
            )
            ffn_out = add_timing(
                timings,
                f"block_{block_index}_swiglu_ffn",
                lambda block=block, ffn_normalized=ffn_normalized: block.feed_forward(
                    ffn_normalized
                ),
            )
            x = add_timing(
                timings,
                f"block_{block_index}_ffn_residual_dropout",
                lambda block=block, x=x, ffn_out=ffn_out: x + block.dropout(ffn_out),
            )

        cls_state = add_timing(timings, "final_norm_cls", lambda: model.final_norm(x[:, 0, :]))
        _ = add_timing(timings, "classifier_head", lambda: model.classifier(cls_state))

    stages = []
    total_seconds = 0.0
    for name, values in timings.items():
        mean_seconds = float(np.mean(values))
        total_seconds += mean_seconds
        stages.append(
            {
                "stage": name,
                "mean_ms": mean_seconds * 1000,
                "min_ms": float(np.min(values) * 1000),
                "max_ms": float(np.max(values) * 1000),
            }
        )

    for stage in stages:
        stage["share"] = float(stage["mean_ms"] / (total_seconds * 1000))

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "trainable_parameters": count_trainable_parameters(model),
        "metal_peak_memory_mb": float(mx.get_peak_memory() / 1024 / 1024),
        "stages": stages,
    }


def should_capture(capture_shape: str | None, *, batch_size: int, sequence_length: int) -> bool:
    return capture_shape == f"{batch_size}x{sequence_length}"


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    print(f"MLX default device: {mx.default_device()}")
    print(f"Metal available: {mx.metal.is_available()}")
    print(f"Metal device info: {mx.metal.device_info()}")

    results = []
    layer_results = []
    for sequence_length in args.sequence_lengths:
        for batch_size in args.batch_sizes:
            capture_path = args.output_dir / f"capture_b{batch_size}_s{sequence_length}.gputrace"
            capture = should_capture(
                args.capture_shape,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            capture_started = False
            if capture:
                print(f"Starting Metal capture: {capture_path}")
                try:
                    mx.metal.start_capture(str(capture_path))
                    capture_started = True
                except RuntimeError as error:
                    print(
                        "Metal capture could not start. "
                        "Try rerunning with MTL_CAPTURE_ENABLED=1. "
                        f"MLX error: {error}",
                        flush=True,
                    )

            result = benchmark_shape(
                args,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

            if capture_started:
                mx.metal.stop_capture()
                result["metal_capture"] = str(capture_path)
                print(f"Stopped Metal capture: {capture_path}")

            results.append(result)
            print(
                f"B={batch_size:4d} S={sequence_length:3d} | "
                f"forward={result['forward_ms_mean']:.2f} ms | "
                f"train={result['train_step_ms_mean']:.2f} ms | "
                f"examples/s={result['examples_per_second_mean']:.1f} | "
                f"peak={result['metal_peak_memory_mb']:.1f} MB",
                flush=True,
            )

            if args.layer_breakdown:
                breakdown = layer_breakdown(
                    args,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
                layer_results.append(breakdown)
                print("Layer breakdown:")
                for stage in sorted(
                    breakdown["stages"],
                    key=lambda item: item["mean_ms"],
                    reverse=True,
                )[:12]:
                    print(
                        f"  {stage['stage']:<36} "
                        f"{stage['mean_ms']:8.3f} ms "
                        f"({stage['share'] * 100:5.1f}%)",
                        flush=True,
                    )

    payload = {
        "device": {
            "default_device": str(mx.default_device()),
            "metal_available": bool(mx.metal.is_available()),
            "metal_device_info": mx.metal.device_info(),
        },
        "profile": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
        },
        "results": results,
        "layer_breakdown": layer_results,
    }
    write_json(args.output_dir / "profile_results.json", payload)
    print(f"Saved profile results to {args.output_dir / 'profile_results.json'}")


if __name__ == "__main__":
    main()
