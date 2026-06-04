"""Fine-tune a pretrained Hugging Face transformer on IMDb sentiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.sentiment.artifacts import ensure_dir, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_predictions
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "transformer" / "pretrained_transformer_torch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained transformer on Kaggle IMDb reviews.",
    )
    parser.add_argument("--csv-path", type=Path, default=KAGGLE_IMDB_CSV)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--truncation-strategy",
        choices=["first", "last", "head_tail_mean_logits"],
        default="first",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "none":
        return None
    if device.type not in {"cuda", "mps"}:
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested bfloat16, but this CUDA device does not support it.")
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


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


def _tokenize_first(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def _window_token_ids(
    tokenizer: Any,
    token_ids: list[int],
    *,
    max_length: int,
    side: str,
) -> tuple[list[int], list[int]]:
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    payload_length = max(1, max_length - special_count)
    if side == "first":
        payload = token_ids[:payload_length]
    elif side == "last":
        payload = token_ids[-payload_length:]
    else:
        raise ValueError(f"Unsupported side: {side}")

    ids = tokenizer.build_inputs_with_special_tokens(payload)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    mask = [1] * len(ids)
    pad_count = max_length - len(ids)
    if pad_count > 0:
        ids = ids + [pad_id] * pad_count
        mask = mask + [0] * pad_count
    return ids[:max_length], mask[:max_length]


def _tokenize_windows(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    strategy: str,
) -> dict[str, torch.Tensor]:
    first_ids: list[list[int]] = []
    first_masks: list[list[int]] = []
    second_ids: list[list[int]] = []
    second_masks: list[list[int]] = []

    for text in texts:
        ids = tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
        ids_a, mask_a = _window_token_ids(tokenizer, ids, max_length=max_length, side="first")
        first_ids.append(ids_a)
        first_masks.append(mask_a)
        if strategy == "head_tail_mean_logits":
            ids_b, mask_b = _window_token_ids(tokenizer, ids, max_length=max_length, side="last")
            second_ids.append(ids_b)
            second_masks.append(mask_b)

    tensors = {
        "input_ids": torch.tensor(first_ids, dtype=torch.long),
        "attention_mask": torch.tensor(first_masks, dtype=torch.long),
    }
    if strategy == "head_tail_mean_logits":
        tensors["tail_input_ids"] = torch.tensor(second_ids, dtype=torch.long)
        tensors["tail_attention_mask"] = torch.tensor(second_masks, dtype=torch.long)
    return tensors


def tokenize_texts(
    tokenizer: Any,
    texts: Any,
    *,
    max_length: int,
    strategy: str,
) -> dict[str, torch.Tensor]:
    text_list = [str(text) for text in texts]
    if strategy == "first":
        return _tokenize_first(tokenizer, text_list, max_length=max_length)
    if strategy in {"last", "head_tail_mean_logits"}:
        if strategy == "last":
            windowed = _tokenize_windows(
                tokenizer,
                text_list,
                max_length=max_length,
                strategy="head_tail_mean_logits",
            )
            return {
                "input_ids": windowed["tail_input_ids"],
                "attention_mask": windowed["tail_attention_mask"],
            }
        return _tokenize_windows(tokenizer, text_list, max_length=max_length, strategy=strategy)
    raise ValueError(f"Unsupported truncation strategy: {strategy}")


def build_loader(
    encoded: dict[str, torch.Tensor],
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if "tail_input_ids" in encoded:
        tensors.extend([encoded["tail_input_ids"], encoded["tail_attention_mask"]])
    tensors.append(torch.from_numpy(labels.astype(np.int64, copy=False)))

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if len(batch) == 3:
        input_ids, attention_mask, labels = batch
        inputs = {
            "input_ids": input_ids.to(device, non_blocking=True),
            "attention_mask": attention_mask.to(device, non_blocking=True),
        }
    elif len(batch) == 5:
        input_ids, attention_mask, tail_input_ids, tail_attention_mask, labels = batch
        inputs = {
            "input_ids": input_ids.to(device, non_blocking=True),
            "attention_mask": attention_mask.to(device, non_blocking=True),
            "tail_input_ids": tail_input_ids.to(device, non_blocking=True),
            "tail_attention_mask": tail_attention_mask.to(device, non_blocking=True),
        }
    else:
        raise ValueError(f"Unexpected batch size: {len(batch)} tensors")
    return inputs, labels.to(device, non_blocking=True)


def forward_logits(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if "tail_input_ids" not in inputs:
        return model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits

    head_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    tail_logits = model(
        input_ids=inputs["tail_input_ids"],
        attention_mask=inputs["tail_attention_mask"],
    ).logits
    return (head_logits + tail_logits) / 2.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    log_every: int,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_rows = 0

    for step_index, batch in enumerate(loader, start=1):
        inputs, labels = unpack_batch(batch, device=device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = forward_logits(model, inputs)
            loss = criterion(logits.float(), labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
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
    for batch in loader:
        inputs, _ = unpack_batch(batch, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = forward_logits(model, inputs)
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
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
        "f1": float(evaluate_binary_predictions(labels, predictions)["metrics"]["f1"]),
    }


def select_best_score(metrics: dict[str, float]) -> float:
    return metrics["accuracy"]


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.random_state)

    device = resolve_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device)
    eval_batch_size = args.eval_batch_size or args.batch_size

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)

    split = make_train_test_split(data, test_size=args.test_size, random_state=args.random_state)
    y_train_full = split.y_train.to_numpy(dtype=np.int64)
    train_indices, validation_indices = split_train_validation(
        y_train_full,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    x_train = split.x_train.iloc[train_indices]
    y_train = y_train_full[train_indices]
    if len(validation_indices) > 0:
        x_validation = split.x_train.iloc[validation_indices]
        y_validation = y_train_full[validation_indices]
    else:
        x_validation = split.x_test
        y_validation = split.y_test.to_numpy(dtype=np.int64)
    y_test = split.y_test.to_numpy(dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print("Tokenizing train split...", flush=True)
    train_encoded = tokenize_texts(
        tokenizer,
        x_train,
        max_length=args.max_length,
        strategy=args.truncation_strategy,
    )
    print("Tokenizing validation split...", flush=True)
    validation_encoded = tokenize_texts(
        tokenizer,
        x_validation,
        max_length=args.max_length,
        strategy=args.truncation_strategy,
    )
    print("Tokenizing test split...", flush=True)
    test_encoded = tokenize_texts(
        tokenizer,
        split.x_test,
        max_length=args.max_length,
        strategy=args.truncation_strategy,
    )

    train_loader = build_loader(
        train_encoded,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.random_state,
        num_workers=args.num_workers,
        device=device,
    )
    validation_loader = build_loader(
        validation_encoded,
        y_validation,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=args.random_state,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = build_loader(
        test_encoded,
        y_test,
        batch_size=eval_batch_size,
        shuffle=False,
        seed=args.random_state,
        num_workers=args.num_workers,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, math.ceil(len(y_train) / args.batch_size) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    best_score = -float("inf")
    best_epoch = 0
    weights_path = args.output_dir / "best_model_state.pt"
    history: list[dict[str, float | int]] = []

    print(f"Model: {args.model_name}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Training rows: {len(y_train):,}")
    print(f"Validation rows: {len(y_validation):,}")
    print(f"Test rows: {len(y_test):,}")
    print(f"Device: {device}")
    print(f"AMP dtype: {args.amp_dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Truncation strategy: {args.truncation_strategy}")

    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loader = build_loader(
            train_encoded,
            y_train,
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
            y_validation,
            device=device,
            amp_dtype=amp_dtype,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_f1": validation_metrics["f1"],
            "seconds": epoch_seconds,
        }
        history.append(row)

        score = select_best_score(validation_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), weights_path)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={validation_metrics['loss']:.4f} | "
            f"val_acc={validation_metrics['accuracy']:.4f} | "
            f"val_f1={validation_metrics['f1']:.4f} | "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )

    total_seconds = time.perf_counter() - start_time
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))

    test_probabilities = predict_proba(
        model,
        test_loader,
        device=device,
        amp_dtype=amp_dtype,
    )
    test_predictions = test_probabilities.argmax(axis=1)
    evaluation = evaluate_binary_predictions(
        y_test,
        test_predictions,
        y_score=test_probabilities[:, 1],
        y_proba=test_probabilities,
    )

    runtime: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if device.type == "cuda":
        runtime.update(
            {
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )

    run_config = {
        "classifier": "PretrainedTransformerTorch",
        "dataset": "Kaggle IMDb 50K Movie Reviews",
        "csv_path": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "row_limit": args.limit,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "random_state": args.random_state,
        "train_rows_before_validation": int(len(split.x_train)),
        "train_rows_used": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "test_rows": int(len(split.x_test)),
        "train_class_counts": class_counts(split.y_train),
        "test_class_counts": class_counts(split.y_test),
        "tokenization": {
            "tokenizer_class": tokenizer.__class__.__name__,
            "max_length": args.max_length,
            "truncation_strategy": args.truncation_strategy,
        },
        "trainable_parameters": parameter_count,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "amp_dtype": args.amp_dtype,
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

    if args.save_predictions:
        predictions = {
            "y_true": y_test.astype(int).tolist(),
            "y_pred": test_predictions.astype(int).tolist(),
            "positive_probability": test_probabilities[:, 1].astype(float).tolist(),
        }
        write_json(args.output_dir / "predictions.json", predictions)

    if args.save_model:
        model.save_pretrained(args.output_dir / "model")
        tokenizer.save_pretrained(args.output_dir / "model")
    elif weights_path.exists():
        weights_path.unlink()

    print(f"Saved pretrained transformer artifacts to {args.output_dir}")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Log-loss: {evaluation['metrics']['log_loss']:.4f}")


if __name__ == "__main__":
    main()

