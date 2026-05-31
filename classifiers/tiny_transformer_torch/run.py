"""Train and evaluate a tiny from-scratch PyTorch transformer on IMDb reviews."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:  # PyTorch 2.2 keeps GradScaler under torch.cuda.amp.
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from classifiers.tiny_transformer_mlx.config import TinyTransformerConfig
from classifiers.tiny_transformer_mlx.tokenizer_utils import (
    encode_texts,
    load_tokenizer,
    special_token_ids,
    train_bpe_tokenizer,
)
from classifiers.tiny_transformer_torch.model import (
    TinyTransformerClassifier,
    count_trainable_parameters,
)
from src.sentiment.artifacts import ensure_dir, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_predictions
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "transformer" / "tiny_transformer_torch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny PyTorch transformer from scratch on Kaggle IMDb reviews.",
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
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "float16", "bfloat16"],
        default="float16",
        help="Autocast dtype. Parameters stay fp32; use none for full fp32.",
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
    parser.add_argument("--rebuild-tokenizer", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--quiet-encoding", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a training progress line every N batches. Use 0 to disable.",
    )
    return parser.parse_args()


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
        "y_train": split.y_train.to_numpy(dtype=np.int64),
        "x_test_ids": x_test_ids,
        "x_test_mask": x_test_mask,
        "y_test": split.y_test.to_numpy(dtype=np.int64),
    }
    np.savez(data_path, **arrays)
    write_json(metadata_path, metadata)
    return arrays


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


def build_loader(
    ids: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(ids.astype(np.int64, copy=False)),
        torch.from_numpy(masks.astype(np.bool_, copy=False)),
        torch.from_numpy(labels.astype(np.int64, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "none" or device.type != "cuda":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested bfloat16, but this CUDA device does not report bf16 support.")
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    fused_adamw: bool,
    device: torch.device,
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if fused_adamw and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str,
    learning_rate: float,
    min_learning_rate: float,
    train_rows: int,
    batch_size: int,
    epochs: int,
    warmup_steps: int | None,
    warmup_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, dict[str, Any]]:
    steps_per_epoch = int(math.ceil(train_rows / batch_size))
    total_steps = max(1, steps_per_epoch * epochs)
    if schedule == "constant":
        return None, {
            "name": "constant",
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
        }

    actual_warmup_steps = warmup_steps
    if actual_warmup_steps is None:
        actual_warmup_steps = int(total_steps * warmup_ratio)
    actual_warmup_steps = max(1, min(actual_warmup_steps, total_steps - 1))
    decay_steps = max(1, total_steps - actual_warmup_steps)
    min_ratio = min_learning_rate / learning_rate if learning_rate > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < actual_warmup_steps:
            return (step + 1) / actual_warmup_steps
        progress = min(1.0, (step - actual_warmup_steps + 1) / decay_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), {
        "name": "warmup-cosine",
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": actual_warmup_steps,
        "decay_steps": decay_steps,
        "min_learning_rate": min_learning_rate,
    }


def create_grad_scaler(*, enabled: bool) -> GradScaler:
    try:
        return GradScaler(device="cuda", enabled=enabled)
    except TypeError:
        return GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    scaler: GradScaler | None,
    log_every: int,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_rows = 0

    for step_index, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.float(), labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_rows = int(labels.shape[0])
        loss_value = float(loss.detach().item())
        total_loss += loss_value * batch_rows
        total_rows += batch_rows

        if log_every > 0 and step_index % log_every == 0:
            print(f"  train step {step_index}/{len(loader)} | loss={loss_value:.4f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    return total_loss / total_rows


@torch.inference_mode()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []

    for input_ids, attention_mask, _ in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids, attention_mask)
        batch_probabilities = torch.softmax(logits.float(), dim=-1)
        probabilities.append(batch_probabilities.cpu().numpy())

    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(probabilities, axis=0)


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    labels: np.ndarray,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    probabilities = predict_proba(model, loader, device=device, amp_dtype=amp_dtype)
    predictions = probabilities.argmax(axis=1)
    loss = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    return {
        "loss": float(loss),
        "accuracy": float((predictions == labels).mean()),
    }


def set_runtime_options(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_state)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
        torch.backends.cudnn.benchmark = True

    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "device_capability": (
            list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None
        ),
        "allow_tf32": args.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = set_runtime_options(args, device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    eval_batch_size = args.eval_batch_size or args.batch_size

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
    model = TinyTransformerClassifier(model_config).to(device)
    parameter_count = count_trainable_parameters(model)

    if args.compile_model:
        model = torch.compile(model)

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

    train_loader = build_loader(
        train_ids,
        train_masks,
        train_labels,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.random_state,
        num_workers=args.num_workers,
        device=device,
    )
    validation_loader = build_loader(
        validation_ids,
        validation_masks,
        validation_labels,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=args.random_state,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = build_loader(
        arrays["x_test_ids"],
        arrays["x_test_mask"],
        arrays["y_test"],
        batch_size=eval_batch_size,
        shuffle=False,
        seed=args.random_state,
        num_workers=args.num_workers,
        device=device,
    )

    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fused_adamw=args.fused_adamw,
        device=device,
    )
    scheduler, lr_schedule_config = build_scheduler(
        optimizer,
        schedule=args.lr_schedule,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        train_rows=len(train_labels),
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
    )
    scaler = create_grad_scaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    best_metric = -float("inf")
    best_epoch = 0
    weights_path = args.output_dir / "model_weights.pt"
    history: list[dict[str, float | int]] = []

    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training rows: {len(train_labels):,}")
    print(f"Validation rows: {len(validation_labels):,}")
    print(f"Device: {runtime['device_name'] or runtime['device']}")
    print(f"AMP dtype: {args.amp_dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"torch.compile: {args.compile_model}")
    print(f"TF32 allowed: {args.allow_tf32}")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loader = build_loader(
            train_ids,
            train_masks,
            train_labels,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.random_state + epoch,
            num_workers=args.num_workers,
            device=device,
        )
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_every=args.log_every,
        )
        validation_metrics = evaluate_split(
            model,
            validation_loader,
            validation_labels,
            device=device,
            amp_dtype=amp_dtype,
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
            torch.save(model.state_dict(), weights_path)

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
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    test_probabilities = predict_proba(
        model,
        test_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    evaluation = evaluate_binary_predictions(
        arrays["y_test"],
        test_predictions,
        y_score=test_probabilities[:, 1],
        y_proba=test_probabilities,
    )

    if device.type == "cuda":
        runtime["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        runtime["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

    run_config = {
        "classifier": "TinyTransformerClassifier_Torch",
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
            "eval_batch_size": eval_batch_size,
            "amp_dtype": args.amp_dtype,
            "fused_adamw": args.fused_adamw,
            "epochs": args.epochs,
            "compile_model": args.compile_model,
        },
        "runtime": runtime,
        "best_epoch": best_epoch,
        "total_training_seconds": total_seconds,
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "training_history.json", {"epochs": history})
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])

    print(f"Saved TinyTransformer PyTorch artifacts to {args.output_dir}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Log-loss: {evaluation['metrics']['log_loss']:.4f}")


if __name__ == "__main__":
    main()
