"""Train and evaluate a hard VotingClassifier with independent TF-IDF pipelines."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module=r"dask\.dataframe")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.sentiment.artifacts import dump_model, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_classifier
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "ensemble" / "hard_voting_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a hard VotingClassifier over all trained model families.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=KAGGLE_IMDB_CSV,
        help="Path to Kaggle IMDb CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where run artifacts will be written.",
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
        help="Random seed for the stratified train/test split and model members.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a quick smoke test.",
    )
    parser.add_argument(
        "--tree-n-estimators",
        type=int,
        default=100,
        help="Number of trees for RandomForestClassifier and ExtraTreesClassifier.",
    )
    parser.add_argument(
        "--boosting-n-estimators",
        type=int,
        default=200,
        help="Number of boosted trees for XGBoostClassifier and LightGBMClassifier.",
    )
    return parser.parse_args()


def build_tfidf(
    *,
    max_features: int | None = 50_000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for one ensemble member."""

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def make_pipeline(classifier: Any, *, tfidf: TfidfVectorizer | None = None) -> Pipeline:
    """Build one complete text-to-classifier member pipeline."""

    return Pipeline(
        steps=[
            ("tfidf", tfidf or build_tfidf()),
            ("classifier", classifier),
        ]
    )


def build_classifier(args: argparse.Namespace) -> VotingClassifier:
    """Build the hard-voting ensemble over independent TF-IDF pipelines."""

    estimators = [
        ("multinomial_nb", make_pipeline(MultinomialNB(alpha=1.0))),
        (
            "multinomial_nb_tuned",
            make_pipeline(
                MultinomialNB(alpha=0.5),
                tfidf=build_tfidf(max_features=80_000, min_df=3, max_df=0.9),
            ),
        ),
        ("complement_nb", make_pipeline(ComplementNB(alpha=1.0))),
        (
            "logistic_regression",
            make_pipeline(
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    penalty="l2",
                    random_state=args.random_state,
                    solver="liblinear",
                )
            ),
        ),
        (
            "linear_svc",
            make_pipeline(
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    random_state=args.random_state,
                    dual="auto",
                )
            ),
        ),
        (
            "sgd_classifier",
            make_pipeline(
                SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=args.random_state,
                    n_jobs=-1,
                )
            ),
        ),
        (
            "passive_aggressive",
            make_pipeline(
                PassiveAggressiveClassifier(
                    C=1.0,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=args.random_state,
                    n_jobs=-1,
                )
            ),
        ),
        (
            "random_forest",
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=args.tree_n_estimators,
                    criterion="gini",
                    max_depth=None,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=args.random_state,
                )
            ),
        ),
        (
            "extra_trees",
            make_pipeline(
                ExtraTreesClassifier(
                    n_estimators=args.tree_n_estimators,
                    criterion="gini",
                    max_depth=None,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=args.random_state,
                )
            ),
        ),
        (
            "xgboost_classifier",
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=args.boosting_n_estimators,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=args.random_state,
                    verbosity=1,
                )
            ),
        ),
        (
            "lightgbm_classifier",
            make_pipeline(
                LGBMClassifier(
                    objective="binary",
                    boosting_type="gbdt",
                    metric="binary_logloss",
                    n_estimators=args.boosting_n_estimators,
                    learning_rate=0.1,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    force_col_wise=True,
                    n_jobs=-1,
                    random_state=args.random_state,
                    verbosity=-1,
                )
            ),
        ),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="hard",
        n_jobs=None,
    )


def add_vote_fraction_metrics(
    classifier: VotingClassifier,
    x_test: Any,
    y_test: Any,
    metrics: dict[str, float | int | None],
) -> dict[str, float | int | None]:
    """Add score-like metrics from positive vote fractions.

    Hard voting does not expose calibrated probabilities. The positive vote
    fraction is still useful as an ordinal score, so we keep it under explicit
    vote_fraction_* names instead of pretending it is predict_proba output.
    """

    member_predictions = np.asarray(classifier.transform(x_test))
    if member_predictions.shape[0] != len(y_test) and member_predictions.shape[1] == len(y_test):
        member_predictions = member_predictions.T

    positive_vote_fraction = (member_predictions == 1).mean(axis=1)
    pseudo_proba = np.column_stack([1.0 - positive_vote_fraction, positive_vote_fraction])
    clipped_pseudo_proba = np.clip(pseudo_proba, 1e-15, 1.0 - 1e-15)

    metrics["vote_fraction_roc_auc"] = float(roc_auc_score(y_test, positive_vote_fraction))
    metrics["vote_fraction_pr_auc"] = float(average_precision_score(y_test, positive_vote_fraction))
    metrics["vote_fraction_log_loss_clipped"] = float(
        log_loss(y_test, clipped_pseudo_proba, labels=[0, 1])
    )
    metrics["vote_tie_count"] = int(np.sum(positive_vote_fraction == 0.5))
    metrics["ensemble_member_count"] = int(member_predictions.shape[1])
    return metrics


def fitted_member_summary(classifier: VotingClassifier) -> dict[str, dict[str, Any]]:
    """Return fitted feature counts and estimator class names for each member."""

    summary = {}
    for name, estimator in classifier.named_estimators_.items():
        vectorizer = estimator.named_steps["tfidf"]
        member_classifier = estimator.named_steps["classifier"]
        summary[name] = {
            "classifier_class": member_classifier.__class__.__name__,
            "actual_tfidf_features": int(len(vectorizer.vocabulary_)),
            "tfidf": {
                "max_features": vectorizer.max_features,
                "min_df": vectorizer.min_df,
                "max_df": vectorizer.max_df,
                "ngram_range": list(vectorizer.ngram_range),
                "sublinear_tf": vectorizer.sublinear_tf,
                "strip_accents": vectorizer.strip_accents,
                "lowercase": vectorizer.lowercase,
            },
        }
    return summary


def main() -> None:
    args = parse_args()

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)

    split = make_train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    classifier = build_classifier(args)
    classifier.fit(split.x_train, split.y_train)

    evaluation = evaluate_binary_classifier(classifier, split.x_test, split.y_test)
    add_vote_fraction_metrics(classifier, split.x_test, split.y_test, evaluation["metrics"])

    run_config = {
        "classifier": "VotingClassifier",
        "dataset": "Kaggle IMDb 50K Movie Reviews",
        "csv_path": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "row_limit": args.limit,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_rows": int(len(split.x_train)),
        "test_rows": int(len(split.x_test)),
        "train_class_counts": class_counts(split.y_train),
        "test_class_counts": class_counts(split.y_test),
        "ensemble": {
            "type": "hard_voting",
            "member_count": len(classifier.estimators),
            "members": [name for name, _ in classifier.estimators],
            "uses_independent_tfidf_pipelines": True,
            "includes_tuned_multinomial_nb_variant": True,
        },
        "members": fitted_member_summary(classifier),
        "model_params": {
            "tree_n_estimators": args.tree_n_estimators,
            "boosting_n_estimators": args.boosting_n_estimators,
        },
        "fit_status": {
            "has_predict_proba": hasattr(classifier, "predict_proba"),
            "log_loss_available": hasattr(classifier, "predict_proba"),
            "standard_roc_auc_available": False,
            "vote_fraction_scores_available": True,
        },
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])
    dump_model(args.output_dir / "model.joblib", classifier)

    print(f"Saved hard VotingClassifier artifacts to {args.output_dir}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"Vote-fraction ROC-AUC: {evaluation['metrics']['vote_fraction_roc_auc']:.4f}")
    print(f"Vote-fraction PR-AUC: {evaluation['metrics']['vote_fraction_pr_auc']:.4f}")
    print(
        "Vote-fraction clipped log-loss: "
        f"{evaluation['metrics']['vote_fraction_log_loss_clipped']:.4f}"
    )
    print(f"Members: {evaluation['metrics']['ensemble_member_count']}")
    print(f"Vote ties: {evaluation['metrics']['vote_tie_count']}")


if __name__ == "__main__":
    main()
