"""Train/evaluate a cached prefit stacking ensemble over saved baseline pipelines."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module=r"dask\.dataframe")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.sentiment.artifacts import dump_model, ensure_dir, write_json, write_text
from src.sentiment.data import ID_TO_LABEL, load_kaggle_imdb, make_train_test_split
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "ensemble" / "stacking_classifier_prefit"
BASELINE_OUTPUT_DIR = OUTPUTS_DIR / "baselines"

BASE_MODEL_DIRS = [
    ("multinomial_nb", "multinomial_nb"),
    ("complement_nb", "complement_nb"),
    ("logistic_regression", "logistic_regression"),
    ("linear_svc", "linear_svc"),
    ("sgd_classifier", "sgd_classifier"),
    ("passive_aggressive", "passive_aggressive"),
    ("random_forest", "random_forest"),
    ("extra_trees", "extra_trees"),
    ("xgboost_classifier", "xgboost_classifier"),
    ("lightgbm_classifier", "lightgbm_classifier"),
]
TUNED_MODEL_DIRS = [
    ("multinomial_nb_tuned", "multinomial_nb_tuned_n10_cv3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a cached prefit stacking ensemble on Kaggle IMDb reviews.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=KAGGLE_IMDB_CSV,
        help="Path to Kaggle IMDb CSV.",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=BASELINE_OUTPUT_DIR,
        help="Directory containing baseline model subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run artifacts will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for cached stack features. Defaults to <output-dir>/meta_features_cache.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Saved meta-classifier path. Defaults to <output-dir>/model.joblib.",
    )
    parser.add_argument(
        "--force-refit-stack",
        action="store_true",
        help="Retrain the meta-classifier even if a saved compatible one already exists.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Recompute cached stack features even if compatible cache files exist.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of labelled data used for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the stratified train/test split and meta-classifier.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a quick smoke test.",
    )
    parser.add_argument(
        "--stack-method",
        choices=["auto", "predict_proba", "decision_function", "predict"],
        default="auto",
        help="Base-estimator method used to build stack features.",
    )
    parser.add_argument(
        "--final-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the LogisticRegression meta-classifier.",
    )
    parser.add_argument(
        "--include-tuned-nb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the tuned MultinomialNB model artifact as an extra stack member.",
    )
    return parser.parse_args()


def model_entries(include_tuned_nb: bool) -> list[tuple[str, str]]:
    """Return base model entries included in the stack."""

    if include_tuned_nb:
        return BASE_MODEL_DIRS + TUNED_MODEL_DIRS
    return BASE_MODEL_DIRS


def quiet_loaded_estimator(estimator: Any) -> Any:
    """Suppress verbose prediction logs on loaded baseline estimators."""

    classifier = estimator
    if hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
        classifier = estimator.named_steps["classifier"]

    if hasattr(classifier, "verbose"):
        classifier.verbose = 0
    if hasattr(classifier, "verbosity"):
        classifier.verbosity = 0

    return estimator


def load_prefit_estimators(
    base_model_dir: Path,
    entries: list[tuple[str, str]],
) -> list[tuple[str, str, Any]]:
    """Load already-fitted baseline pipelines from disk."""

    estimators = []
    missing = []
    for estimator_name, output_subdir in entries:
        model_path = base_model_dir / output_subdir / "model.joblib"
        if not model_path.exists():
            missing.append(str(model_path))
            continue
        estimator = quiet_loaded_estimator(joblib.load(model_path))
        estimators.append((estimator_name, str(model_path), estimator))

    if missing:
        raise FileNotFoundError(
            "Missing fitted baseline model artifact(s):\n" + "\n".join(missing)
        )

    return estimators


def select_stack_method(estimator: Any, requested: str) -> str:
    """Select the stacking method for one estimator using sklearn-like auto semantics."""

    if requested != "auto":
        if not hasattr(estimator, requested):
            raise ValueError(
                f"Estimator {estimator!r} does not expose stack method {requested!r}."
            )
        return requested

    for method in ("predict_proba", "decision_function", "predict"):
        if hasattr(estimator, method):
            return method

    raise ValueError(f"Estimator {estimator!r} has no supported stacking output method.")


def stack_output(estimator: Any, x_values: Any, method: str) -> np.ndarray:
    """Return 2D stack features for one estimator."""

    values = getattr(estimator, method)(x_values)
    values = np.asarray(values)

    if method == "predict_proba" and values.ndim == 2 and values.shape[1] == 2:
        # Match sklearn StackingClassifier's binary behavior: keep only P(class=1).
        values = values[:, 1]

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    return values.astype(np.float64, copy=False)


def cache_metadata(
    args: argparse.Namespace,
    entries: list[tuple[str, str]],
    estimator_paths: list[str],
) -> dict[str, Any]:
    """Build metadata used to validate cached stack features."""

    return {
        "csv_path": str(args.csv_path),
        "base_model_dir": str(args.base_model_dir),
        "entries": [{"name": name, "output_subdir": subdir} for name, subdir in entries],
        "estimator_paths": estimator_paths,
        "include_tuned_nb": args.include_tuned_nb,
        "stack_method_requested": args.stack_method,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "row_limit": args.limit,
    }


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    """Return all cache file paths."""

    return {
        "metadata": cache_dir / "metadata.joblib",
        "x_train_meta": cache_dir / "x_train_meta.joblib",
        "x_test_meta": cache_dir / "x_test_meta.joblib",
        "y_train": cache_dir / "y_train.joblib",
        "y_test": cache_dir / "y_test.joblib",
        "feature_info": cache_dir / "feature_info.joblib",
    }


def load_cache_if_valid(
    cache_dir: Path,
    expected_metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
    """Load cached stack features when metadata matches."""

    paths = cache_paths(cache_dir)
    if not all(path.exists() for path in paths.values()):
        return None

    try:
        metadata = joblib.load(paths["metadata"])
    except Exception:
        return None

    if metadata != expected_metadata:
        return None

    feature_info_payload = joblib.load(paths["feature_info"])
    return (
        joblib.load(paths["x_train_meta"]),
        joblib.load(paths["x_test_meta"]),
        joblib.load(paths["y_train"]),
        joblib.load(paths["y_test"]),
        feature_info_payload,
    )


def write_stack_cache(
    cache_dir: Path,
    metadata: dict[str, Any],
    x_train_meta: np.ndarray,
    x_test_meta: np.ndarray,
    y_train: Any,
    y_test: Any,
    feature_info: list[dict[str, Any]],
) -> None:
    """Persist stack features and validation metadata."""

    ensure_dir(cache_dir)
    paths = cache_paths(cache_dir)
    joblib.dump(metadata, paths["metadata"])
    joblib.dump(x_train_meta, paths["x_train_meta"])
    joblib.dump(x_test_meta, paths["x_test_meta"])
    joblib.dump(np.asarray(y_train), paths["y_train"])
    joblib.dump(np.asarray(y_test), paths["y_test"])
    joblib.dump(feature_info, paths["feature_info"])


def build_stack_features(
    estimators: list[tuple[str, str, Any]],
    x_train: Any,
    x_test: Any,
    requested_stack_method: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run saved base pipelines once and concatenate their stack features."""

    train_parts = []
    test_parts = []
    feature_info = []

    for name, model_path, estimator in estimators:
        method = select_stack_method(estimator, requested_stack_method)
        train_output = stack_output(estimator, x_train, method)
        test_output = stack_output(estimator, x_test, method)
        train_parts.append(train_output)
        test_parts.append(test_output)
        feature_info.append(
            {
                "name": name,
                "model_path": model_path,
                "stack_method": method,
                "feature_count": int(train_output.shape[1]),
            }
        )

    return np.hstack(train_parts), np.hstack(test_parts), feature_info


def load_or_build_stack_features(
    args: argparse.Namespace,
    x_train: Any,
    x_test: Any,
    y_train: Any,
    y_test: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], bool]:
    """Load cached stack features or build them from saved baseline models."""

    cache_dir = args.cache_dir or args.output_dir / "meta_features_cache"
    entries = model_entries(include_tuned_nb=args.include_tuned_nb)
    estimator_paths = [
        str(args.base_model_dir / output_subdir / "model.joblib")
        for _, output_subdir in entries
    ]
    metadata = cache_metadata(args, entries, estimator_paths)

    if not args.force_rebuild_cache:
        cached = load_cache_if_valid(cache_dir, metadata)
        if cached is not None:
            x_train_meta, x_test_meta, cached_y_train, cached_y_test, feature_info = cached
            return x_train_meta, x_test_meta, cached_y_train, cached_y_test, feature_info, True

    estimators = load_prefit_estimators(args.base_model_dir, entries)
    x_train_meta, x_test_meta, feature_info = build_stack_features(
        estimators,
        x_train,
        x_test,
        args.stack_method,
    )
    write_stack_cache(
        cache_dir,
        metadata,
        x_train_meta,
        x_test_meta,
        y_train,
        y_test,
        feature_info,
    )
    return x_train_meta, x_test_meta, np.asarray(y_train), np.asarray(y_test), feature_info, False


def load_cached_meta_model(model_path: Path, feature_info: list[dict[str, Any]]) -> Any | None:
    """Load a compatible cached meta-classifier artifact."""

    if not model_path.exists():
        return None

    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        return None
    if payload.get("model_type") != "cached_prefit_stacking":
        return None
    if payload.get("feature_info") != feature_info:
        return None

    return payload["final_estimator"]


def fit_or_load_meta_model(
    args: argparse.Namespace,
    model_path: Path,
    x_train_meta: np.ndarray,
    y_train: np.ndarray,
    feature_info: list[dict[str, Any]],
) -> tuple[LogisticRegression, bool]:
    """Load or fit the final meta-classifier."""

    if not args.force_refit_stack:
        cached_model = load_cached_meta_model(model_path, feature_info)
        if cached_model is not None:
            return cached_model, True

    final_estimator = LogisticRegression(
        C=args.final_c,
        max_iter=1000,
        penalty="l2",
        random_state=args.random_state,
        solver="liblinear",
    )
    final_estimator.fit(x_train_meta, y_train)
    dump_model(
        model_path,
        {
            "model_type": "cached_prefit_stacking",
            "final_estimator": final_estimator,
            "feature_info": feature_info,
        },
    )
    return final_estimator, False


def evaluate_meta_classifier(
    final_estimator: Any,
    x_test_meta: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the final estimator from cached stack features."""

    y_pred = final_estimator.predict(x_test_meta)
    y_proba = final_estimator.predict_proba(x_test_meta)
    y_score = y_proba[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1])),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "metrics": metrics,
        "confusion_matrix": {
            "labels": [ID_TO_LABEL[0], ID_TO_LABEL[1]],
            "matrix": cm.astype(int).tolist(),
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
            zero_division=0,
        ),
    }


def cached_class_counts(labels: Any) -> dict[str, int]:
    """Return class counts for cached numpy labels."""

    label_ids, counts = np.unique(np.asarray(labels), return_counts=True)
    return {ID_TO_LABEL[int(label_id)]: int(count) for label_id, count in zip(label_ids, counts)}


def main() -> None:
    args = parse_args()
    model_path = args.model_path or args.output_dir / "model.joblib"
    cache_dir = args.cache_dir or args.output_dir / "meta_features_cache"

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)

    split = make_train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    (
        x_train_meta,
        x_test_meta,
        y_train,
        y_test,
        feature_info,
        loaded_cached_features,
    ) = load_or_build_stack_features(
        args,
        split.x_train,
        split.x_test,
        split.y_train,
        split.y_test,
    )
    final_estimator, loaded_cached_meta_model = fit_or_load_meta_model(
        args,
        model_path,
        x_train_meta,
        y_train,
        feature_info,
    )
    evaluation = evaluate_meta_classifier(final_estimator, x_test_meta, y_test)

    run_config = {
        "classifier": "CachedPrefitStackingClassifier",
        "dataset": "Kaggle IMDb 50K Movie Reviews",
        "csv_path": str(args.csv_path),
        "base_model_dir": str(args.base_model_dir),
        "output_dir": str(args.output_dir),
        "model_path": str(model_path),
        "cache_dir": str(cache_dir),
        "loaded_cached_features": loaded_cached_features,
        "loaded_cached_meta_model": loaded_cached_meta_model,
        "force_refit_stack": args.force_refit_stack,
        "force_rebuild_cache": args.force_rebuild_cache,
        "row_limit": args.limit,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_class_counts": cached_class_counts(y_train),
        "test_class_counts": cached_class_counts(y_test),
        "ensemble": {
            "type": "cached_stacking_prefit",
            "stack_method_requested": args.stack_method,
            "uses_saved_prefit_base_models": True,
            "retrained_base_models": False,
            "cached_train_meta_features": True,
            "cached_test_meta_features": True,
            "include_tuned_nb": args.include_tuned_nb,
            "member_count": len(feature_info),
            "members": [item["name"] for item in feature_info],
            "stack_methods_selected": {
                item["name"]: item["stack_method"] for item in feature_info
            },
            "meta_feature_count": int(x_train_meta.shape[1]),
        },
        "model_params": {
            "final_estimator": {
                "class": "LogisticRegression",
                "C": args.final_c,
                "max_iter": 1000,
                "penalty": "l2",
                "solver": "liblinear",
            },
        },
        "fit_status": {
            "has_predict_proba": True,
            "log_loss_available": True,
        },
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])

    print(f"Saved cached stacking artifacts to {args.output_dir}")
    if loaded_cached_features:
        print(f"Loaded cached stack features from {cache_dir}")
    else:
        print(f"Built and cached stack features in {cache_dir}")
    if loaded_cached_meta_model:
        print(f"Loaded cached meta-classifier from {model_path}")
    else:
        print(f"Fitted meta-classifier and saved it to {model_path}")
    print("Base estimators were loaded prefit; they were not retrained.")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Log-loss: {evaluation['metrics']['log_loss']:.4f}")


if __name__ == "__main__":
    main()
