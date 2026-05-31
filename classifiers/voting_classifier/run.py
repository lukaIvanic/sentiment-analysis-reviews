"""Train and evaluate a TF-IDF + VotingClassifier sentiment ensemble."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module=r"dask\.dataframe")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.sentiment.artifacts import dump_model, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_classifier
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "ensemble" / "voting_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a soft VotingClassifier ensemble on Kaggle IMDb reviews.",
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
        help="Random seed for the stratified train/test split and ensemble members.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a quick smoke test.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore terms appearing in fewer than this many documents.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Ignore terms appearing in more than this fraction of documents.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        choices=[1, 2],
        help="Use unigrams only or unigrams plus bigrams.",
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=1000,
        help="Maximum optimizer iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--nb-alpha",
        type=float,
        default=0.5,
        help="Laplace smoothing value for MultinomialNB.",
    )
    parser.add_argument(
        "--lightgbm-n-estimators",
        type=int,
        default=200,
        help="Number of boosted trees for LightGBMClassifier.",
    )
    parser.add_argument(
        "--lightgbm-learning-rate",
        type=float,
        default=0.1,
        help="Boosting learning rate for LightGBMClassifier.",
    )
    parser.add_argument(
        "--lightgbm-num-leaves",
        type=int,
        default=31,
        help="Maximum leaves per LightGBM tree.",
    )
    parser.add_argument(
        "--lightgbm-min-child-samples",
        type=int,
        default=20,
        help="Minimum data points required in a LightGBM leaf.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=None,
        metavar=("LOGISTIC", "NB", "LIGHTGBM"),
        help="Optional soft-voting weights for logistic, multinomial NB, and LightGBM.",
    )
    return parser.parse_args()


def build_classifier(args: argparse.Namespace) -> VotingClassifier:
    """Build the soft-voting classifier."""

    estimators = [
        (
            "logistic_regression",
            LogisticRegression(
                C=args.logistic_c,
                max_iter=args.logistic_max_iter,
                penalty="l2",
                random_state=args.random_state,
                solver="liblinear",
            ),
        ),
        ("multinomial_nb", MultinomialNB(alpha=args.nb_alpha)),
        (
            "lightgbm_classifier",
            LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                metric="binary_logloss",
                n_estimators=args.lightgbm_n_estimators,
                learning_rate=args.lightgbm_learning_rate,
                num_leaves=args.lightgbm_num_leaves,
                max_depth=-1,
                min_child_samples=args.lightgbm_min_child_samples,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                force_col_wise=True,
                n_jobs=-1,
                random_state=args.random_state,
                verbosity=-1,
            ),
        ),
    ]

    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=args.weights,
        n_jobs=None,
        flatten_transform=True,
    )


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the shared-vectorizer ensemble pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, args.ngram_max),
                    min_df=args.min_df,
                    max_df=args.max_df,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", build_classifier(args)),
        ]
    )


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

    pipeline = build_pipeline(args)
    pipeline.fit(split.x_train, split.y_train)

    evaluation = evaluate_binary_classifier(pipeline, split.x_test, split.y_test)

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
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
        "tfidf": {
            "max_features": args.max_features,
            "actual_features": int(len(vectorizer.vocabulary_)),
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_range": [1, args.ngram_max],
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "lowercase": True,
        },
        "ensemble": {
            "type": "soft_voting",
            "members": [name for name, _ in classifier.estimators],
            "weights": args.weights,
        },
        "model_params": {
            "logistic_regression": {
                "C": args.logistic_c,
                "max_iter": args.logistic_max_iter,
                "penalty": "l2",
                "solver": "liblinear",
            },
            "multinomial_nb": {
                "alpha": args.nb_alpha,
            },
            "lightgbm_classifier": {
                "objective": "binary",
                "boosting_type": "gbdt",
                "metric": "binary_logloss",
                "n_estimators": args.lightgbm_n_estimators,
                "learning_rate": args.lightgbm_learning_rate,
                "num_leaves": args.lightgbm_num_leaves,
                "max_depth": -1,
                "min_child_samples": args.lightgbm_min_child_samples,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "force_col_wise": True,
            },
        },
        "fit_status": {
            "has_predict_proba": hasattr(pipeline, "predict_proba"),
            "log_loss_available": hasattr(pipeline, "predict_proba"),
        },
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])
    dump_model(args.output_dir / "model.joblib", pipeline)

    print(f"Saved VotingClassifier artifacts to {args.output_dir}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Log-loss: {evaluation['metrics']['log_loss']:.4f}")


if __name__ == "__main__":
    main()
