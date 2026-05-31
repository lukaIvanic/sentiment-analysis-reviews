"""Train and evaluate a tiny from-scratch MLX transformer on IMDb reviews."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map
from sklearn.model_selection import train_test_split

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig
from classifiers.tiny_transformer_mlx.model import (
    TinyTransformerClassifier,
    count_trainable_parameters,
)
from classifiers.tiny_transformer_mlx.tokenizer_utils import (
    load_tokenizer,
    special_token_ids,
    train_bpe_tokenizer,
    encode_texts,
)
from src.sentiment.artifacts import ensure_dir, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_predictions
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "transformer" / "tiny_transformer_mlx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny MLX transformer from scratch on Kaggle IMDb reviews.",
    )
    parser.add_argument("--csv-path", type=Path, default=KAGGLE_IMDB_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Forward/backward micro-batch size. If smaller than --batch-size, "
            "gradients are accumulated until the effective optimizer batch is reached."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "Validation/test forward batch size. Defaults to the micro-batch size when "
            "gradient accumulation is used, otherwise to --batch-size."
        ),
    )
    parser.add_argument(
        "--parameter-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model parameter/activation dtype. Loss and reported probabilities stay float32.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "warmup-cosine"],
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-weights", type=Path, default=None)
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--quiet-encoding", action="store_true")
    parser.add_argument(
        "--compile-step",
        action="store_true",
        help="Compile the train step with mx.compile. Helps fixed-shape mini-batch runs.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a training progress line every N batches. Use 0 to disable.",
    )
    return parser.parse_args()


def build_learning_rate_schedule(
    *,
    learning_rate: float,
    min_learning_rate: float,
    schedule: str,
    train_rows: int,
    batch_size: int,
    epochs: int,
    warmup_steps: int | None,
    warmup_ratio: float,
):
    """Build an MLX optimizer learning-rate value or schedule."""

    steps_per_epoch = int(np.ceil(train_rows / batch_size))
    total_steps = max(1, steps_per_epoch * epochs)
    if schedule == "constant":
        return learning_rate, {
            "name": "constant",
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
        }

    actual_warmup_steps = warmup_steps
    if actual_warmup_steps is None:
        actual_warmup_steps = int(total_steps * warmup_ratio)
    actual_warmup_steps = max(1, min(actual_warmup_steps, total_steps - 1))
    decay_steps = max(1, total_steps - actual_warmup_steps)
    lr_schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, learning_rate, steps=actual_warmup_steps),
            optim.cosine_decay(learning_rate, decay_steps, end=min_learning_rate),
        ],
        [actual_warmup_steps],
    )
    return lr_schedule, {
        "name": "warmup-cosine",
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": actual_warmup_steps,
        "decay_steps": decay_steps,
        "min_learning_rate": min_learning_rate,
    }


def resolve_dtype(name: str) -> mx.Dtype:
    """Resolve a CLI dtype name to an MLX dtype."""

    if name == "float32":
        return mx.float32
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def validate_batching(args: argparse.Namespace) -> int:
    """Return the micro-batch size after checking accumulation settings."""

    micro_batch_size = args.micro_batch_size or args.batch_size
    if micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if micro_batch_size > args.batch_size:
        raise ValueError("--micro-batch-size cannot be larger than --batch-size")
    return micro_batch_size


def resolve_eval_batch_size(args: argparse.Namespace, micro_batch_size: int) -> int:
    """Return the validation/test batch size."""

    eval_batch_size = args.eval_batch_size
    if eval_batch_size is None:
        eval_batch_size = micro_batch_size if micro_batch_size < args.batch_size else args.batch_size
    if eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    return eval_batch_size


def metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return current == expected


def prepare_tokenizer(args: argparse.Namespace, x_train: Any) -> Any:
    tokenizer_path = args.output_dir / "tokenizer.json"
    metadata_path = args.output_dir / "tokenizer_config.json"
    metadata = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "lowercase": args.lowercase,
        "train_rows": int(len(x_train)),
    }
    if args.tokenizer_path is not None:
        metadata["tokenizer_path"] = str(args.tokenizer_path)

    if (
        tokenizer_path.exists()
        and metadata_matches(metadata_path, metadata)
        and not args.rebuild_tokenizer
    ):
        return load_tokenizer(tokenizer_path)

    if args.tokenizer_path is not None:
        tokenizer = load_tokenizer(args.tokenizer_path)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.tokenizer_path, tokenizer_path)
        write_json(metadata_path, metadata)
        return tokenizer

    tokenizer = train_bpe_tokenizer(
        (str(text) for text in x_train),
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=args.lowercase,
        length=len(x_train),
    )
    write_json(metadata_path, metadata)
    return tokenizer


def prepare_encoded_cache(
    args: argparse.Namespace,
    tokenizer: Any,
    split: Any,
) -> dict[str, np.ndarray]:
    cache_dir = ensure_dir(args.output_dir / "encoded_cache")
    data_path = cache_dir / "encoded_arrays.npz"
    metadata_path = cache_dir / "cache_config.json"
    metadata = {
        "csv_path": str(args.csv_path),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "row_limit": args.limit,
        "vocab_size": args.vocab_size,
        "max_length": args.max_length,
        "lowercase": args.lowercase,
        "train_rows": int(len(split.x_train)),
        "test_rows": int(len(split.x_test)),
        "special_token_ids": special_token_ids(tokenizer),
    }

    if data_path.exists() and metadata_matches(metadata_path, metadata) and not args.rebuild_cache:
        encoded = np.load(data_path)
        return {key: encoded[key] for key in encoded.files}

    x_train_ids, x_train_mask = encode_texts(
        tokenizer,
        split.x_train,
        max_length=args.max_length,
        show_progress=not args.quiet_encoding,
    )
    x_test_ids, x_test_mask = encode_texts(
        tokenizer,
        split.x_test,
        max_length=args.max_length,
        show_progress=not args.quiet_encoding,
    )

    arrays = {
        "x_train_ids": x_train_ids,
        "x_train_mask": x_train_mask,
        "y_train": split.y_train.to_numpy(dtype=np.int32),
        "x_test_ids": x_test_ids,
        "x_test_mask": x_test_mask,
        "y_test": split.y_test.to_numpy(dtype=np.int32),
    }
    np.savez(data_path, **arrays)
    write_json(metadata_path, metadata)
    return arrays


def load_pretrained_encoder(model: TinyTransformerClassifier, weights_path: Path) -> None:
    """Load encoder weights from either a classifier or MLM checkpoint."""

    weights = []
    for key, value in mx.load(str(weights_path)).items():
        if key.startswith("encoder."):
            weights.append((key.removeprefix("encoder."), value))
        elif key.startswith("lm_head."):
            continue
        else:
            weights.append((key, value))
    model.load_weights(weights, strict=False)


def batch_indices(
    size: int,
    *,
    batch_size: int,
    rng: np.random.Generator | None,
) -> list[np.ndarray]:
    indices = np.arange(size)
    if rng is not None:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def mx_batch(
    ids: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[mx.array, mx.array, mx.array]:
    return (
        mx.array(ids[indices].astype(np.int32, copy=False)),
        mx.array(masks[indices].astype(np.int32, copy=False)),
        mx.array(labels[indices].astype(np.int32, copy=False)),
    )


def loss_fn(
    model: TinyTransformerClassifier,
    input_ids: mx.array,
    attention_mask: mx.array,
    labels: mx.array,
) -> mx.array:
    logits = model(input_ids, attention_mask).astype(mx.float32)
    return mx.mean(nn.losses.cross_entropy(logits, labels))


def scale_tree(tree: Any, scale: float) -> Any:
    return tree_map(lambda value: value * scale, tree)


def add_trees(left: Any, right: Any) -> Any:
    return tree_map(lambda left_value, right_value: left_value + right_value, left, right)


def train_one_epoch(
    model: TinyTransformerClassifier,
    optimizer: optim.Optimizer,
    ids: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    micro_batch_size: int,
    seed: int,
    log_every: int,
    compile_step: bool,
) -> float:
    model.train()
    rng = np.random.default_rng(seed)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    total_loss = 0.0
    total_rows = 0
    use_accumulation = micro_batch_size < batch_size

    def train_step(
        batch_ids: mx.array,
        batch_masks: mx.array,
        batch_labels: mx.array,
    ) -> mx.array:
        loss, gradients = loss_and_grad(model, batch_ids, batch_masks, batch_labels)
        optimizer.update(model, gradients)
        return loss

    if compile_step and not use_accumulation:
        train_step = mx.compile(
            train_step,
            inputs=[model.state, optimizer.state, mx.random.state],
            outputs=[model.state, optimizer.state, mx.random.state],
        )

    batches = batch_indices(len(labels), batch_size=batch_size, rng=rng)
    for step_index, indices in enumerate(batches, start=1):
        batch_size_actual = len(indices)
        if use_accumulation:
            accumulated_gradients = None
            batch_loss_total = 0.0
            for micro_start in range(0, batch_size_actual, micro_batch_size):
                micro_indices = indices[micro_start : micro_start + micro_batch_size]
                batch_ids, batch_masks, batch_labels = mx_batch(ids, masks, labels, micro_indices)
                loss, gradients = loss_and_grad(model, batch_ids, batch_masks, batch_labels)
                mx.eval(loss, gradients)
                micro_size = len(micro_indices)
                batch_loss_total += float(loss.item()) * micro_size
                weighted_gradients = scale_tree(gradients, micro_size / batch_size_actual)
                if accumulated_gradients is None:
                    accumulated_gradients = weighted_gradients
                else:
                    accumulated_gradients = add_trees(accumulated_gradients, weighted_gradients)
                mx.eval(accumulated_gradients)

            optimizer.update(model, accumulated_gradients)
            mx.eval(model.parameters(), optimizer.state)
            loss_value = batch_loss_total / batch_size_actual
        else:
            batch_ids, batch_masks, batch_labels = mx_batch(ids, masks, labels, indices)
            loss = train_step(batch_ids, batch_masks, batch_labels)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_value = float(loss.item())

        total_loss += loss_value * batch_size_actual
        total_rows += batch_size_actual
        if log_every > 0 and step_index % log_every == 0:
            print(f"  train step {step_index}/{len(batches)} | loss={loss_value:.4f}", flush=True)

    return total_loss / total_rows


def predict_proba(
    model: TinyTransformerClassifier,
    ids: np.ndarray,
    masks: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels = np.zeros(len(ids), dtype=np.int32)

    for indices in batch_indices(len(ids), batch_size=batch_size, rng=None):
        batch_ids, batch_masks, _ = mx_batch(ids, masks, labels, indices)
        batch_probabilities = mx.softmax(model(batch_ids, batch_masks).astype(mx.float32), axis=-1)
        mx.eval(batch_probabilities)
        probabilities.append(np.array(batch_probabilities))

    return np.concatenate(probabilities, axis=0)


def evaluate_split(
    model: TinyTransformerClassifier,
    ids: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, float]:
    probabilities = predict_proba(model, ids, masks, batch_size=batch_size)
    predictions = probabilities.argmax(axis=1)
    loss = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    return {
        "loss": float(loss),
        "accuracy": float((predictions == labels).mean()),
    }


def split_train_validation(
    labels: np.ndarray,
    *,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    if validation_size <= 0:
        return indices, np.array([], dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_indices, validation_indices


def main() -> None:
    args = parse_args()
    micro_batch_size = validate_batching(args)
    eval_batch_size = resolve_eval_batch_size(args, micro_batch_size)
    ensure_dir(args.output_dir)
    mx.random.seed(args.random_state)

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)

    split = make_train_test_split(data, test_size=args.test_size, random_state=args.random_state)
    tokenizer = prepare_tokenizer(args, split.x_train)
    arrays = prepare_encoded_cache(args, tokenizer, split)

    model_config = TinyTransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_length=args.max_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    model = TinyTransformerClassifier(model_config)
    if args.pretrained_weights is not None:
        load_pretrained_encoder(model, args.pretrained_weights)
    parameter_dtype = resolve_dtype(args.parameter_dtype)
    if parameter_dtype != mx.float32:
        model.set_dtype(parameter_dtype)
    parameter_count = count_trainable_parameters(model)
    mx.eval(model.parameters())

    train_indices, validation_indices = split_train_validation(
        arrays["y_train"],
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    train_ids = arrays["x_train_ids"][train_indices]
    train_masks = arrays["x_train_mask"][train_indices]
    train_labels = arrays["y_train"][train_indices]

    if len(validation_indices) > 0:
        validation_ids = arrays["x_train_ids"][validation_indices]
        validation_masks = arrays["x_train_mask"][validation_indices]
        validation_labels = arrays["y_train"][validation_indices]
    else:
        validation_ids = arrays["x_test_ids"]
        validation_masks = arrays["x_test_mask"]
        validation_labels = arrays["y_test"]

    learning_rate, lr_schedule_config = build_learning_rate_schedule(
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        schedule=args.lr_schedule,
        train_rows=len(train_labels),
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )
    optimizer = optim.AdamW(
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
    )
    best_metric = -float("inf")
    best_epoch = 0
    weights_path = args.output_dir / "model_weights.npz"
    history: list[dict[str, float | int]] = []

    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training rows: {len(train_labels):,}")
    print(f"Validation rows: {len(validation_labels):,}")
    print(f"Parameter dtype: {args.parameter_dtype}")
    print(f"Optimizer batch size: {args.batch_size}")
    print(f"Micro-batch size: {micro_batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_ids,
            train_masks,
            train_labels,
            batch_size=args.batch_size,
            micro_batch_size=micro_batch_size,
            seed=args.random_state + epoch,
            log_every=args.log_every,
            compile_step=args.compile_step,
        )
        validation_metrics = evaluate_split(
            model,
            validation_ids,
            validation_masks,
            validation_labels,
            batch_size=eval_batch_size,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "seconds": epoch_seconds,
        }
        history.append(row)

        if validation_metrics["accuracy"] > best_metric:
            best_metric = validation_metrics["accuracy"]
            best_epoch = epoch
            model.save_weights(str(weights_path))

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f} | "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )

    total_seconds = time.perf_counter() - start_time
    if weights_path.exists():
        model.load_weights(str(weights_path))
        mx.eval(model.parameters())

    test_probabilities = predict_proba(
        model,
        arrays["x_test_ids"],
        arrays["x_test_mask"],
        batch_size=eval_batch_size,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    evaluation = evaluate_binary_predictions(
        arrays["y_test"],
        test_predictions,
        y_score=test_probabilities[:, 1],
        y_proba=test_probabilities,
    )

    run_config = {
        "classifier": "TinyTransformerClassifier_MLX",
        "dataset": "Kaggle IMDb 50K Movie Reviews",
        "csv_path": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "row_limit": args.limit,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "random_state": args.random_state,
        "train_rows_before_validation": int(len(split.x_train)),
        "train_rows_used": int(len(train_labels)),
        "validation_rows": int(len(validation_labels)),
        "test_rows": int(len(split.x_test)),
        "train_class_counts": class_counts(split.y_train),
        "test_class_counts": class_counts(split.y_test),
        "tokenizer": {
            "type": "BPE",
            "vocab_size": tokenizer.get_vocab_size(),
            "requested_vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
            "lowercase": args.lowercase,
            "max_length": args.max_length,
            "special_token_ids": special_token_ids(tokenizer),
            "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        },
        "model_config": model_config.to_dict(),
        "trainable_parameters": parameter_count,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "lr_schedule": lr_schedule_config,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "micro_batch_size": micro_batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": int(np.ceil(args.batch_size / micro_batch_size)),
            "parameter_dtype": args.parameter_dtype,
            "epochs": args.epochs,
            "compile_step": args.compile_step,
        },
        "pretrained_weights": str(args.pretrained_weights) if args.pretrained_weights else None,
        "best_epoch": best_epoch,
        "total_training_seconds": total_seconds,
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "training_history.json", {"epochs": history})
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])

    print(f"Saved TinyTransformer MLX artifacts to {args.output_dir}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Log-loss: {evaluation['metrics']['log_loss']:.4f}")


if __name__ == "__main__":
    main()
