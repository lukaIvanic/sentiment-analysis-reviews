"""Train and evaluate a TF-IDF + RandomForestClassifier sentiment classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.sentiment.artifacts import dump_model, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_classifier
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "baselines" / "random_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RandomForestClassifier on Kaggle IMDb reviews.",
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
        help="Random seed for the stratified train/test split and forest.",
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
        "--n-estimators",
        type=int,
        default=100,
        help="Number of decision trees in the forest.",
    )
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy", "log_loss"],
        default="gini",
        help="Split quality criterion.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of each tree. Defaults to unlimited depth.",
    )
    parser.add_argument(
        "--forest-max-features",
        choices=["sqrt", "log2"],
        default="sqrt",
        help="Number of candidate features considered at each tree split.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples required at each leaf node.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity passed to RandomForestClassifier.",
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
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    criterion=args.criterion,
                    max_depth=args.max_depth,
                    max_features=args.forest_max_features,
                    min_samples_leaf=args.min_samples_leaf,
                    n_jobs=-1,
                    random_state=args.random_state,
                    verbose=args.verbose,
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

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    run_config = {
        "classifier": "RandomForestClassifier",
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
        "model_params": {
            "n_estimators": args.n_estimators,
            "criterion": args.criterion,
            "max_depth": args.max_depth,
            "max_features": args.forest_max_features,
            "min_samples_leaf": args.min_samples_leaf,
            "n_jobs": -1,
        },
        "fit_status": {
            "n_estimators": int(len(classifier.estimators_)),
            "has_predict_proba": hasattr(pipeline, "predict_proba"),
            "log_loss_available": hasattr(pipeline, "predict_proba"),
        },
    }

    write_json(args.output_dir / "run_config.json", run_config)
    write_json(args.output_dir / "metrics.json", evaluation["metrics"])
    write_json(args.output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(args.output_dir / "classification_report.txt", evaluation["classification_report"])
    dump_model(args.output_dir / "model.joblib", pipeline)

    print(f"Saved RandomForestClassifier artifacts to {args.output_dir}")
    print(f"Accuracy: {evaluation['metrics']['accuracy']:.4f}")
    print(f"F1: {evaluation['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {evaluation['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC: {evaluation['metrics']['pr_auc']:.4f}")
    print(f"Log-loss: {evaluation['metrics']['log_loss']:.4f}")
    print(f"Trees: {len(classifier.estimators_)}")


if __name__ == "__main__":
    main()
