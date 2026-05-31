"""Masked-language-model pretraining for the tiny MLX transformer encoder."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig
from classifiers.tiny_transformer_mlx.model import TinyTransformerForMaskedLM, count_trainable_parameters
from classifiers.tiny_transformer_mlx.run import (
    batch_indices,
    build_learning_rate_schedule,
    prepare_encoded_cache,
    prepare_tokenizer,
)
from classifiers.tiny_transformer_mlx.tokenizer_utils import MASK_TOKEN, special_token_ids
from src.sentiment.artifacts import ensure_dir, write_json
from src.sentiment.data import load_kaggle_imdb, make_train_test_split
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "transformer" / "tiny_transformer_mlx_mlm_pretrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the MLX transformer with MLM.")
    parser.add_argument("--csv-path", type=Path, default=KAGGLE_IMDB_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-schedule", choices=["constant", "warmup-cosine"], default="warmup-cosine")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--quiet-encoding", action="store_true")
    parser.add_argument("--compile-step", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def make_mlm_batch(
    ids: np.ndarray,
    masks: np.ndarray,
    indices: np.ndarray,
    *,
    token_ids: dict[str, int],
    vocab_size: int,
    mask_probability: float,
    rng: np.random.Generator,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Create one masked-language-modeling batch."""

    original = ids[indices].astype(np.int32, copy=True)
    attention_mask = masks[indices].astype(np.int32, copy=False)
    masked = original.copy()

    special_ids = {
        token_ids["[PAD]"],
        token_ids["[CLS]"],
        token_ids["[SEP]"],
        token_ids[MASK_TOKEN],
    }
    candidate_mask = attention_mask.astype(bool)
    for special_id in special_ids:
        candidate_mask &= original != special_id

    selected = (rng.random(original.shape) < mask_probability) & candidate_mask
    for row_index in range(selected.shape[0]):
        if not selected[row_index].any():
            candidates = np.flatnonzero(candidate_mask[row_index])
            if len(candidates) > 0:
                selected[row_index, rng.choice(candidates)] = True

    replacement_draw = rng.random(original.shape)
    random_tokens = rng.integers(5, vocab_size, size=original.shape, dtype=np.int32)
    masked = np.where(selected & (replacement_draw < 0.8), token_ids[MASK_TOKEN], masked)
    masked = np.where(
        selected & (replacement_draw >= 0.8) & (replacement_draw < 0.9),
        random_tokens,
        masked,
    )

    return (
        mx.array(masked),
        mx.array(attention_mask),
        mx.array(original),
        mx.array(selected.astype(np.float32)),
    )


def mlm_loss_fn(
    model: TinyTransformerForMaskedLM,
    input_ids: mx.array,
    attention_mask: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> mx.array:
    logits = model(input_ids, attention_mask)
    token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    loss_sum = mx.sum(token_losses * loss_mask)
    token_count = mx.maximum(mx.sum(loss_mask), mx.array(1.0, dtype=mx.float32))
    return loss_sum / token_count


def train_one_epoch(
    model: TinyTransformerForMaskedLM,
    optimizer: optim.Optimizer,
    ids: np.ndarray,
    masks: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    token_ids: dict[str, int],
    vocab_size: int,
    mask_probability: float,
    compile_step: bool,
    log_every: int,
) -> float:
    model.train()
    rng = np.random.default_rng(seed)
    loss_and_grad = nn.value_and_grad(model, mlm_loss_fn)

    def train_step(
        input_ids: mx.array,
        attention_mask: mx.array,
        targets: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        loss, gradients = loss_and_grad(model, input_ids, attention_mask, targets, loss_mask)
        optimizer.update(model, gradients)
        return loss

    if compile_step:
        train_step = mx.compile(
            train_step,
            inputs=[model.state, optimizer.state, mx.random.state],
            outputs=[model.state, optimizer.state, mx.random.state],
        )

    total_loss = 0.0
    total_rows = 0
    batches = batch_indices(len(ids), batch_size=batch_size, rng=rng)
    for step_index, indices in enumerate(batches, start=1):
        batch = make_mlm_batch(
            ids,
            masks,
            indices,
            token_ids=token_ids,
            vocab_size=vocab_size,
            mask_probability=mask_probability,
            rng=rng,
        )
        loss = train_step(*batch)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_value = float(loss.item())
        total_loss += loss_value * len(indices)
        total_rows += len(indices)
        if log_every > 0 and step_index % log_every == 0:
            print(f"  pretrain step {step_index}/{len(batches)} | mlm_loss={loss_value:.4f}", flush=True)

    return total_loss / total_rows


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    mx.random.seed(args.random_state)

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)
    split = make_train_test_split(data, test_size=args.test_size, random_state=args.random_state)
    tokenizer = prepare_tokenizer(args, split.x_train)
    token_ids = special_token_ids(tokenizer, require_mask=True)
    arrays = prepare_encoded_cache(args, tokenizer, split)

    config = TinyTransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_length=args.max_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = TinyTransformerForMaskedLM(config)
    parameter_count = count_trainable_parameters(model)
    mx.eval(model.parameters())

    learning_rate, lr_schedule_config = build_learning_rate_schedule(
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        schedule=args.lr_schedule,
        train_rows=len(arrays["x_train_ids"]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=args.weight_decay)

    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Pretraining rows: {len(arrays['x_train_ids']):,}")
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            optimizer,
            arrays["x_train_ids"],
            arrays["x_train_mask"],
            batch_size=args.batch_size,
            seed=args.random_state + epoch,
            token_ids=token_ids,
            vocab_size=tokenizer.get_vocab_size(),
            mask_probability=args.mask_probability,
            compile_step=args.compile_step,
            log_every=args.log_every,
        )
        seconds = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "mlm_train_loss": train_loss, "seconds": seconds})
        print(f"epoch {epoch:02d} | mlm_train_loss={train_loss:.4f} | {seconds:.1f}s", flush=True)

    weights_path = args.output_dir / "mlm_model_weights.npz"
    model.save_weights(str(weights_path))
    run_config: dict[str, Any] = {
        "objective": "masked_language_modeling",
        "dataset": "Kaggle IMDb 50K Movie Reviews",
        "csv_path": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "row_limit": args.limit,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_rows": int(len(arrays["x_train_ids"])),
        "test_rows": int(len(arrays["x_test_ids"])),
        "tokenizer": {
            "type": "BPE",
            "vocab_size": tokenizer.get_vocab_size(),
            "requested_vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
            "lowercase": args.lowercase,
            "max_length": args.max_length,
            "special_token_ids": token_ids,
        },
        "model_config": config.to_dict(),
        "trainable_parameters": parameter_count,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "lr_schedule": lr_schedule_config,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "compile_step": args.compile_step,
        },
        "mask_probability": args.mask_probability,
        "weights_path": str(weights_path),
        "total_training_seconds": time.perf_counter() - start_time,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "training_history.json", {"epochs": history})
    print(f"Saved MLM pretraining artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
