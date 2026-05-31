"""Evaluation helpers shared by classifier pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
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

from src.sentiment.data import ID_TO_LABEL


def _positive_class_scores(model: Any, x_test: Any) -> np.ndarray | None:
    """Return scores for the positive class when the estimator exposes them."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)

    return None


def _probability_matrix(model: Any, x_test: Any) -> np.ndarray | None:
    """Return class probabilities when available."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)
    return None


def evaluate_binary_predictions(
    y_test: Any,
    y_pred: Any,
    *,
    y_score: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate binary predictions with the same metrics used for sklearn models."""

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "roc_auc": None,
        "pr_auc": None,
        "log_loss": None,
    }

    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_test, y_score))

    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_test, y_proba, labels=[0, 1]))

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


def evaluate_binary_classifier(model: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    """Evaluate a fitted binary classifier with all metrics required by the course."""

    y_pred = model.predict(x_test)
    y_score = _positive_class_scores(model, x_test)
    y_proba = _probability_matrix(model, x_test)

    return evaluate_binary_predictions(
        y_test,
        y_pred,
        y_score=y_score,
        y_proba=y_proba,
    )
