"""Train a tiny causal MLX decoder to generate IMDb sentiment labels."""

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
from tqdm import tqdm

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig
from classifiers.tiny_transformer_mlx.model import (
    TinyTransformerDecoderLM,
    count_trainable_parameters,
)
from classifiers.tiny_transformer_mlx.run import (
    batch_indices,
    build_learning_rate_schedule,
    metadata_matches,
    split_train_validation,
)
from classifiers.tiny_transformer_mlx.tokenizer_utils import (
    CLS_TOKEN,
    load_tokenizer,
    special_token_ids,
    train_bpe_tokenizer,
)
from src.sentiment.artifacts import ensure_dir, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_predictions
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "transformer" / "tiny_transformer_decoder_mlx"
LABEL_TEXT_BY_CLASS = {
    0: " negative",
    1: " positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny causal MLX decoder to generate sentiment labels.",
    )
    parser.add_argument("--csv-path", type=Path, default=KAGGLE_IMDB_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "warmup-cosine"],
        default="warmup-cosine",
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lm-loss-weight", type=float, default=0.25)
    parser.add_argument("--label-loss-weight", type=float, default=1.0)
    parser.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
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
        default=500,
        help="Print a training progress line every N batches. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_label_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve the one-token negative/positive labels used by the decoder objective."""

    label_ids: dict[str, int] = {}
    for class_id, label_text in LABEL_TEXT_BY_CLASS.items():
        encoding = tokenizer.encode(label_text)
        if len(encoding.ids) != 1:
            raise ValueError(
                f"Label text {label_text!r} must encode to one token, got "
                f"{encoding.tokens} / {encoding.ids}."
            )
        label_ids[str(class_id)] = int(encoding.ids[0])
    return label_ids


def prepare_tokenizer(args: argparse.Namespace, x_train: Any) -> Any:
    tokenizer_path = args.output_dir / "tokenizer.json"
    metadata_path = args.output_dir / "tokenizer_config.json"
    metadata = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "lowercase": args.lowercase,
        "train_rows": int(len(x_train)),
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
    }

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


def encode_decoder_texts(
    tokenizer: Any,
    texts: Any,
    labels: Any,
    *,
    max_length: int,
    label_token_ids: dict[str, int],
    show_progress: bool,
) -> dict[str, np.ndarray]:
    """Encode reviews as `review tokens [CLS] label-token` causal sequences."""

    ids = special_token_ids(tokenizer)
    pad_id = ids["[PAD]"]
    cls_id = ids[CLS_TOKEN]
    review_budget = max_length - 2

    text_list = list(texts)
    label_array = np.asarray(labels, dtype=np.int32)
    sequence_ids = np.full((len(text_list), max_length), pad_id, dtype=np.uint16)
    sequence_mask = np.zeros((len(text_list), max_length), dtype=np.uint8)
    prompt_ids = np.full((len(text_list), max_length), pad_id, dtype=np.uint16)
    prompt_mask = np.zeros((len(text_list), max_length), dtype=np.uint8)
    label_positions = np.zeros(len(text_list), dtype=np.int32)

    iterator = tqdm(
        zip(text_list, label_array),
        total=len(text_list),
        desc="Encoding decoder reviews",
        disable=not show_progress,
    )

    for row_index, (text, class_id) in enumerate(iterator):
        review_ids = tokenizer.encode(str(text)).ids[:review_budget]
        label_token_id = label_token_ids[str(int(class_id))]
        prompt = review_ids + [cls_id]
        sequence = prompt + [label_token_id]
        prompt_length = len(prompt)
        sequence_length = len(sequence)

        sequence_ids[row_index, :sequence_length] = sequence
        sequence_mask[row_index, :sequence_length] = 1
        prompt_ids[row_index, :prompt_length] = prompt
        prompt_mask[row_index, :prompt_length] = 1
        label_positions[row_index] = prompt_length - 1

    return {
        "sequence_ids": sequence_ids,
        "sequence_mask": sequence_mask,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "label_positions": label_positions,
        "labels": label_array,
    }


def prepare_encoded_cache(
    args: argparse.Namespace,
    tokenizer: Any,
    split: Any,
    label_token_ids: dict[str, int],
) -> dict[str, np.ndarray]:
    cache_dir = ensure_dir(args.output_dir / "decoder_cache")
    data_path = cache_dir / "decoder_arrays.npz"
    metadata_path = cache_dir / "cache_config.json"
    metadata = {
        "csv_path": str(args.csv_path),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "row_limit": args.limit,
        "vocab_size": tokenizer.get_vocab_size(),
        "max_length": args.max_length,
        "lowercase": args.lowercase,
        "train_rows": int(len(split.x_train)),
        "test_rows": int(len(split.x_test)),
        "sequence_format": "review_tokens [CLS] label_token",
        "label_text_by_class": LABEL_TEXT_BY_CLASS,
        "label_token_ids": label_token_ids,
        "special_token_ids": special_token_ids(tokenizer),
    }

    if data_path.exists() and metadata_matches(metadata_path, metadata) and not args.rebuild_cache:
        encoded = np.load(data_path)
        return {key: encoded[key] for key in encoded.files}

    train = encode_decoder_texts(
        tokenizer,
        split.x_train,
        split.y_train,
        max_length=args.max_length,
        label_token_ids=label_token_ids,
        show_progress=not args.quiet_encoding,
    )
    test = encode_decoder_texts(
        tokenizer,
        split.x_test,
        split.y_test,
        max_length=args.max_length,
        label_token_ids=label_token_ids,
        show_progress=not args.quiet_encoding,
    )
    arrays = {
        f"x_train_{key}": value
        for key, value in train.items()
    } | {
        f"x_test_{key}": value
        for key, value in test.items()
    }
    np.savez(data_path, **arrays)
    write_json(metadata_path, metadata)
    return arrays


def mx_decoder_batch(
    sequence_ids: np.ndarray,
    sequence_mask: np.ndarray,
    labels: np.ndarray,
    label_positions: np.ndarray,
    indices: np.ndarray,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    return (
        mx.array(sequence_ids[indices].astype(np.int32, copy=False)),
        mx.array(sequence_mask[indices].astype(np.int32, copy=False)),
        mx.array(labels[indices].astype(np.int32, copy=False)),
        mx.array(label_positions[indices].astype(np.int32, copy=False)),
    )


def decoder_loss_components(
    model: TinyTransformerDecoderLM,
    sequence_ids: mx.array,
    sequence_mask: mx.array,
    labels: mx.array,
    label_positions: mx.array,
    label_token_ids: mx.array,
    *,
    lm_loss_weight: float,
    label_loss_weight: float,
) -> tuple[mx.array, mx.array, mx.array]:
    shifted_ids = sequence_ids[:, :-1]
    shifted_mask = sequence_mask[:, :-1]
    target_ids = sequence_ids[:, 1:]
    target_mask = sequence_mask[:, 1:].astype(mx.float32)
    encoded = model.encode(shifted_ids, shifted_mask)

    if lm_loss_weight > 0:
        vocab_logits = model.lm_head(encoded)
        token_losses = nn.losses.cross_entropy(
            vocab_logits.reshape(-1, model.config.vocab_size),
            target_ids.reshape(-1),
        ).reshape(target_ids.shape)
        lm_loss = mx.sum(token_losses * target_mask) / mx.maximum(
            mx.sum(target_mask),
            mx.array(1.0, dtype=mx.float32),
        )
        rows = mx.arange(sequence_ids.shape[0])
        label_logits = vocab_logits[rows, label_positions, :][:, label_token_ids]
    else:
        lm_loss = mx.array(0.0, dtype=mx.float32)
        rows = mx.arange(sequence_ids.shape[0])
        label_states = encoded[rows, label_positions, :]
        label_logits = model.lm_head(label_states)[:, label_token_ids]

    label_loss = mx.mean(nn.losses.cross_entropy(label_logits, labels))
    total_loss = (lm_loss_weight * lm_loss) + (label_loss_weight * label_loss)
    return total_loss, label_loss, lm_loss


def train_one_epoch(
    model: TinyTransformerDecoderLM,
    optimizer: optim.Optimizer,
    sequence_ids: np.ndarray,
    sequence_mask: np.ndarray,
    labels: np.ndarray,
    label_positions: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    log_every: int,
    compile_step: bool,
    label_token_ids: mx.array,
    lm_loss_weight: float,
    label_loss_weight: float,
) -> float:
    model.train()
    rng = np.random.default_rng(seed)

    def total_loss_fn(
        model: TinyTransformerDecoderLM,
        batch_ids: mx.array,
        batch_mask: mx.array,
        batch_labels: mx.array,
        batch_label_positions: mx.array,
    ) -> mx.array:
        total_loss, _, _ = decoder_loss_components(
            model,
            batch_ids,
            batch_mask,
            batch_labels,
            batch_label_positions,
            label_token_ids,
            lm_loss_weight=lm_loss_weight,
            label_loss_weight=label_loss_weight,
        )
        return total_loss

    loss_and_grad = nn.value_and_grad(model, total_loss_fn)

    def train_step(
        batch_ids: mx.array,
        batch_mask: mx.array,
        batch_labels: mx.array,
        batch_label_positions: mx.array,
    ) -> mx.array:
        loss, gradients = loss_and_grad(
            model,
            batch_ids,
            batch_mask,
            batch_labels,
            batch_label_positions,
        )
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
    batches = batch_indices(len(labels), batch_size=batch_size, rng=rng)
    for step_index, indices in enumerate(batches, start=1):
        batch_ids, batch_mask, batch_labels, batch_label_positions = mx_decoder_batch(
            sequence_ids,
            sequence_mask,
            labels,
            label_positions,
            indices,
        )
        loss = train_step(batch_ids, batch_mask, batch_labels, batch_label_positions)
        mx.eval(model.parameters(), optimizer.state, loss)
        loss_value = float(loss.item())
        batch_size_actual = len(indices)
        total_loss += loss_value * batch_size_actual
        total_rows += batch_size_actual
        if log_every > 0 and step_index % log_every == 0:
            print(f"  train step {step_index}/{len(batches)} | loss={loss_value:.4f}", flush=True)

    return total_loss / total_rows


def predict_proba(
    model: TinyTransformerDecoderLM,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    label_positions: np.ndarray,
    *,
    batch_size: int,
    label_token_ids: mx.array,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []

    for indices in batch_indices(len(prompt_ids), batch_size=batch_size, rng=None):
        batch_ids = mx.array(prompt_ids[indices].astype(np.int32, copy=False))
        batch_mask = mx.array(prompt_mask[indices].astype(np.int32, copy=False))
        batch_label_positions = mx.array(label_positions[indices].astype(np.int32, copy=False))
        logits = model.label_logits(
            batch_ids,
            batch_mask,
            batch_label_positions,
            label_token_ids,
        )
        batch_probabilities = mx.softmax(logits, axis=-1)
        mx.eval(batch_probabilities)
        probabilities.append(np.array(batch_probabilities))

    return np.concatenate(probabilities, axis=0)


def evaluate_split(
    model: TinyTransformerDecoderLM,
    sequence_ids: np.ndarray,
    sequence_mask: np.ndarray,
    prompt_ids: np.ndarray,
    prompt_mask: np.ndarray,
    labels: np.ndarray,
    label_positions: np.ndarray,
    *,
    batch_size: int,
    label_token_ids: mx.array,
    lm_loss_weight: float,
    label_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_label_loss = 0.0
    total_lm_loss = 0.0
    total_rows = 0

    for indices in batch_indices(len(labels), batch_size=batch_size, rng=None):
        batch_ids, batch_mask, batch_labels, batch_label_positions = mx_decoder_batch(
            sequence_ids,
            sequence_mask,
            labels,
            label_positions,
            indices,
        )
        loss, label_loss, lm_loss = decoder_loss_components(
            model,
            batch_ids,
            batch_mask,
            batch_labels,
            batch_label_positions,
            label_token_ids,
            lm_loss_weight=lm_loss_weight,
            label_loss_weight=label_loss_weight,
        )
        mx.eval(loss, label_loss, lm_loss)
        batch_size_actual = len(indices)
        total_loss += float(loss.item()) * batch_size_actual
        total_label_loss += float(label_loss.item()) * batch_size_actual
        total_lm_loss += float(lm_loss.item()) * batch_size_actual
        total_rows += batch_size_actual

    probabilities = predict_proba(
        model,
        prompt_ids,
        prompt_mask,
        label_positions,
        batch_size=batch_size,
        label_token_ids=label_token_ids,
    )
    predictions = probabilities.argmax(axis=1)
    return {
        "loss": total_loss / total_rows,
        "label_loss": total_label_loss / total_rows,
        "lm_loss": total_lm_loss / total_rows,
        "accuracy": float((predictions == labels).mean()),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    mx.random.seed(args.random_state)

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)

    split = make_train_test_split(data, test_size=args.test_size, random_state=args.random_state)
    tokenizer = prepare_tokenizer(args, split.x_train)
    label_token_ids_dict = resolve_label_token_ids(tokenizer)
    arrays = prepare_encoded_cache(args, tokenizer, split, label_token_ids_dict)
    label_token_ids = mx.array(
        [label_token_ids_dict["0"], label_token_ids_dict["1"]],
        dtype=mx.int32,
    )

    model_config = TinyTransformerConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_length=args.max_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = TinyTransformerDecoderLM(model_config)
    parameter_count = count_trainable_parameters(model)
    mx.eval(model.parameters())

    train_indices, validation_indices = split_train_validation(
        arrays["x_train_labels"],
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    train_sequence_ids = arrays["x_train_sequence_ids"][train_indices]
    train_sequence_mask = arrays["x_train_sequence_mask"][train_indices]
    train_labels = arrays["x_train_labels"][train_indices]
    train_label_positions = arrays["x_train_label_positions"][train_indices]

    if len(validation_indices) > 0:
        validation_sequence_ids = arrays["x_train_sequence_ids"][validation_indices]
        validation_sequence_mask = arrays["x_train_sequence_mask"][validation_indices]
        validation_prompt_ids = arrays["x_train_prompt_ids"][validation_indices]
        validation_prompt_mask = arrays["x_train_prompt_mask"][validation_indices]
        validation_labels = arrays["x_train_labels"][validation_indices]
        validation_label_positions = arrays["x_train_label_positions"][validation_indices]
    else:
        validation_sequence_ids = arrays["x_test_sequence_ids"]
        validation_sequence_mask = arrays["x_test_sequence_mask"]
        validation_prompt_ids = arrays["x_test_prompt_ids"]
        validation_prompt_mask = arrays["x_test_prompt_mask"]
        validation_labels = arrays["x_test_labels"]
        validation_label_positions = arrays["x_test_label_positions"]

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
    print(
        "Label tokens: "
        f"negative={label_token_ids_dict['0']}, positive={label_token_ids_dict['1']}",
        flush=True,
    )
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_sequence_ids,
            train_sequence_mask,
            train_labels,
            train_label_positions,
            batch_size=args.batch_size,
            seed=args.random_state + epoch,
            log_every=args.log_every,
            compile_step=args.compile_step,
            label_token_ids=label_token_ids,
            lm_loss_weight=args.lm_loss_weight,
            label_loss_weight=args.label_loss_weight,
        )
        validation_metrics = evaluate_split(
            model,
            validation_sequence_ids,
            validation_sequence_mask,
            validation_prompt_ids,
            validation_prompt_mask,
            validation_labels,
            validation_label_positions,
            batch_size=args.batch_size,
            label_token_ids=label_token_ids,
            lm_loss_weight=args.lm_loss_weight,
            label_loss_weight=args.label_loss_weight,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_label_loss": validation_metrics["label_loss"],
            "validation_lm_loss": validation_metrics["lm_loss"],
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
            f"val_label_loss={validation_metrics['label_loss']:.4f} | "
            f"val_lm_loss={validation_metrics['lm_loss']:.4f} | "
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
        arrays["x_test_prompt_ids"],
        arrays["x_test_prompt_mask"],
        arrays["x_test_label_positions"],
        batch_size=args.batch_size,
        label_token_ids=label_token_ids,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    evaluation = evaluate_binary_predictions(
        arrays["x_test_labels"],
        test_predictions,
        y_score=test_probabilities[:, 1],
        y_proba=test_probabilities,
    )

    run_config = {
        "classifier": "TinyTransformerDecoderLM_MLX",
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
            "label_text_by_class": LABEL_TEXT_BY_CLASS,
            "label_token_ids": label_token_ids_dict,
            "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        },
        "sequence_format": "review_tokens [CLS] label_token",
        "model_config": model_config.to_dict(),
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
            "lm_loss_weight": args.lm_loss_weight,
            "label_loss_weight": args.label_loss_weight,
        },
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_metric,
        "total_training_seconds": total_seconds,
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "training_history.json", {"epochs": history})
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])

    print(f"Saved TinyTransformer decoder MLX artifacts to {args.output_dir}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_metric:.4f}")
    print(f"Test accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"Test F1: {evaluation['metrics']['f1']:.4f}")
    print(f"Test ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"Test PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Test log-loss: {evaluation['metrics']['log_loss']:.4f}")


if __name__ == "__main__":
    main()
