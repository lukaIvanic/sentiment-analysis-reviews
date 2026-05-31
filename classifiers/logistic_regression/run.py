"""Train and evaluate a TF-IDF + LogisticRegression sentiment classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.sentiment.artifacts import dump_model, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_classifier
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "baselines" / "logistic_regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate LogisticRegression on Kaggle IMDb reviews.",
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
        help="Random seed for the stratified train/test split.",
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
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength. Smaller values regularize more strongly.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum optimizer iterations.",
    )
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the sklearn pipeline for this classifier."""

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
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    max_iter=args.max_iter,
                    penalty="l2",
                    random_state=args.random_state,
                    solver="liblinear",
                ),
            ),
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

    run_config = {
        "classifier": "LogisticRegression",
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
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_range": [1, args.ngram_max],
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "lowercase": True,
        },
        "model_params": {
            "C": args.c,
            "max_iter": args.max_iter,
            "penalty": "l2",
            "solver": "liblinear",
        },
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])
    dump_model(args.output_dir / "model.joblib", pipeline)

    print(f"Saved LogisticRegression artifacts to {args.output_dir}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()

