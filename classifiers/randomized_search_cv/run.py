"""Run RandomizedSearchCV coverage for the required TF-IDF classifiers."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.sentiment.artifacts import dump_model, ensure_dir, write_json, write_text
from src.sentiment.data import class_counts, load_kaggle_imdb, make_train_test_split
from src.sentiment.metrics import evaluate_binary_classifier
from src.sentiment.paths import KAGGLE_IMDB_CSV, OUTPUTS_DIR


warnings.filterwarnings(
    "ignore",
    message=r"\s*Dask dataframe query planning is disabled.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "searches" / "randomized_search_cv_required_n5_cv3"

COMMON_TFIDF_SPACE: dict[str, list[Any]] = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_df": [0.9, 0.95],
    "tfidf__max_features": [30_000, 50_000],
    "tfidf__sublinear_tf": [True, False],
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    estimator: Any
    model_space: dict[str, list[Any]]

    @property
    def param_distributions(self) -> dict[str, list[Any]]:
        return {**COMMON_TFIDF_SPACE, **self.model_space}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run broad RandomizedSearchCV coverage for the ten classical "
            "TF-IDF classifiers required by the course project."
        ),
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
        help="Directory where CV-search artifacts will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model keys to run, or 'all'.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="Random parameter samples per classifier.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds per sampled parameter setting.",
    )
    parser.add_argument(
        "--scoring",
        default="f1",
        help="Metric optimized by RandomizedSearchCV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of labelled data used for held-out testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for splits, model randomness, and RandomizedSearchCV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--search-jobs",
        type=int,
        default=1,
        help="Parallel jobs used by RandomizedSearchCV.",
    )
    parser.add_argument(
        "--estimator-jobs",
        type=int,
        default=4,
        help="Parallel jobs used inside estimators that support n_jobs.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity passed to RandomizedSearchCV.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Also write each fitted best estimator as model.joblib.",
    )
    return parser.parse_args()


def build_model_specs(random_state: int, estimator_jobs: int) -> dict[str, ModelSpec]:
    return {
        "multinomial_nb": ModelSpec(
            key="multinomial_nb",
            display_name="MultinomialNB",
            estimator=MultinomialNB(),
            model_space={
                "classifier__alpha": [0.1, 0.3, 0.5, 1.0, 2.0],
            },
        ),
        "complement_nb": ModelSpec(
            key="complement_nb",
            display_name="ComplementNB",
            estimator=ComplementNB(),
            model_space={
                "classifier__alpha": [0.1, 0.3, 0.5, 1.0, 2.0],
            },
        ),
        "logistic_regression": ModelSpec(
            key="logistic_regression",
            display_name="LogisticRegression",
            estimator=LogisticRegression(
                max_iter=1000,
                penalty="l2",
                random_state=random_state,
                solver="liblinear",
            ),
            model_space={
                "classifier__C": [0.3, 0.6, 1.0, 2.0, 4.0],
            },
        ),
        "linear_svc": ModelSpec(
            key="linear_svc",
            display_name="LinearSVC",
            estimator=LinearSVC(
                dual="auto",
                max_iter=5000,
                random_state=random_state,
            ),
            model_space={
                "classifier__C": [0.3, 0.6, 1.0, 2.0, 4.0],
            },
        ),
        "sgd_classifier": ModelSpec(
            key="sgd_classifier",
            display_name="SGDClassifier",
            estimator=SGDClassifier(
                max_iter=1000,
                random_state=random_state,
                tol=1e-3,
            ),
            model_space={
                "classifier__alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                "classifier__loss": ["hinge", "modified_huber", "log_loss"],
                "classifier__penalty": ["l2", "elasticnet"],
            },
        ),
        "passive_aggressive": ModelSpec(
            key="passive_aggressive",
            display_name="PassiveAggressiveClassifier",
            estimator=PassiveAggressiveClassifier(
                max_iter=1000,
                random_state=random_state,
                tol=1e-3,
            ),
            model_space={
                "classifier__C": [0.1, 0.3, 0.6, 1.0, 2.0],
                "classifier__loss": ["hinge", "squared_hinge"],
            },
        ),
        "random_forest": ModelSpec(
            key="random_forest",
            display_name="RandomForestClassifier",
            estimator=RandomForestClassifier(
                n_jobs=estimator_jobs,
                random_state=random_state,
                verbose=0,
            ),
            model_space={
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 60, 120],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__n_estimators": [100, 200],
            },
        ),
        "extra_trees": ModelSpec(
            key="extra_trees",
            display_name="ExtraTreesClassifier",
            estimator=ExtraTreesClassifier(
                n_jobs=estimator_jobs,
                random_state=random_state,
                verbose=0,
            ),
            model_space={
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 60, 120],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__n_estimators": [100, 200],
            },
        ),
        "xgboost_classifier": ModelSpec(
            key="xgboost_classifier",
            display_name="XGBoostClassifier",
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=estimator_jobs,
                random_state=random_state,
                verbosity=0,
            ),
            model_space={
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 4, 6],
                "classifier__n_estimators": [100, 200],
                "classifier__reg_lambda": [0.5, 1.0, 2.0],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
        "lightgbm_classifier": ModelSpec(
            key="lightgbm_classifier",
            display_name="LightGBMClassifier",
            estimator=LGBMClassifier(
                boosting_type="gbdt",
                force_col_wise=True,
                metric="binary_logloss",
                n_jobs=estimator_jobs,
                objective="binary",
                random_state=random_state,
                verbosity=-1,
            ),
            model_space={
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__min_child_samples": [10, 20, 40],
                "classifier__n_estimators": [100, 200],
                "classifier__num_leaves": [15, 31, 63],
                "classifier__reg_lambda": [0.5, 1.0, 2.0],
                "classifier__subsample": [0.8, 1.0],
                "classifier__subsample_freq": [1],
            },
        ),
    }


def select_specs(args: argparse.Namespace, specs: dict[str, ModelSpec]) -> list[ModelSpec]:
    if args.models == ["all"] or "all" in args.models:
        return list(specs.values())

    unknown = sorted(set(args.models).difference(specs))
    if unknown:
        known = ", ".join(specs.keys())
        raise ValueError(f"Unknown model key(s): {unknown}. Known keys: {known}")

    return [specs[key] for key in args.models]


def build_pipeline(spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            ("classifier", spec.estimator),
        ],
    )


def count_candidates(param_distributions: dict[str, list[Any]]) -> int:
    total = 1
    for values in param_distributions.values():
        total *= len(values)
    return total


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def format_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def best_params_short(params: dict[str, Any]) -> str:
    ordered = {key: params[key] for key in sorted(params)}
    return json.dumps(jsonable(ordered), sort_keys=True)


def make_summary_markdown(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# RandomizedSearchCV coverage summary",
        "",
        (
            f"Run configuration: `n_iter={args.n_iter}`, `cv={args.cv}`, "
            f"`scoring={args.scoring}`, `random_state={args.random_state}`."
        ),
        "",
        "| Classifier | Best CV F1 | Test accuracy | Test F1 | ROC-AUC | PR-AUC | MCC | Log-loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            "| {classifier} | {best_cv_score} | {accuracy} | {f1} | {roc_auc} | "
            "{pr_auc} | {mcc} | {log_loss} |".format(
                classifier=row["classifier"],
                best_cv_score=format_metric(row["best_cv_score"]),
                accuracy=format_metric(row["accuracy"]),
                f1=format_metric(row["f1"]),
                roc_auc=format_metric(row["roc_auc"]),
                pr_auc=format_metric(row["pr_auc"]),
                mcc=format_metric(row["mcc"]),
                log_loss=format_metric(row["log_loss"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best parameter samples",
            "",
            "| Classifier | Best sampled parameters |",
            "|---|---|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(f"| {row['classifier']} | `{row['best_params']}` |")

    lines.append("")
    return "\n".join(lines)


def run_search_for_spec(
    spec: ModelSpec,
    args: argparse.Namespace,
    split: Any,
) -> dict[str, Any]:
    model_output_dir = ensure_dir(args.output_dir / spec.key)
    param_distributions = spec.param_distributions
    effective_n_iter = min(args.n_iter, count_candidates(param_distributions))

    search = RandomizedSearchCV(
        estimator=build_pipeline(spec),
        param_distributions=param_distributions,
        n_iter=effective_n_iter,
        scoring=args.scoring,
        cv=args.cv,
        random_state=args.random_state,
        n_jobs=args.search_jobs,
        verbose=args.verbose,
        refit=True,
        return_train_score=True,
    )

    start_time = time.perf_counter()
    search.fit(split.x_train, split.y_train)
    elapsed_seconds = time.perf_counter() - start_time

    best_estimator = search.best_estimator_
    evaluation = evaluate_binary_classifier(best_estimator, split.x_test, split.y_test)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(model_output_dir / "search_results.csv", index=False)

    write_json(model_output_dir / "metrics.json", evaluation["metrics"])
    write_json(model_output_dir / "confusion_matrix.json", evaluation["confusion_matrix"])
    write_text(model_output_dir / "classification_report.txt", evaluation["classification_report"])

    if args.save_models:
        dump_model(model_output_dir / "model.joblib", best_estimator)

    vectorizer = best_estimator.named_steps["tfidf"]
    run_config = {
        "classifier": spec.display_name,
        "model_key": spec.key,
        "dataset": "Kaggle IMDb 50K Movie Reviews",
        "csv_path": str(args.csv_path),
        "output_dir": str(model_output_dir),
        "row_limit": args.limit,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_rows": int(len(split.x_train)),
        "test_rows": int(len(split.x_test)),
        "train_class_counts": class_counts(split.y_train),
        "test_class_counts": class_counts(split.y_test),
        "tuning": {
            "enabled": True,
            "search_class": "RandomizedSearchCV",
            "n_iter_requested": args.n_iter,
            "n_iter_effective": effective_n_iter,
            "cv": args.cv,
            "scoring": args.scoring,
            "search_jobs": args.search_jobs,
            "estimator_jobs": args.estimator_jobs,
            "candidate_space_size": count_candidates(param_distributions),
            "elapsed_seconds": elapsed_seconds,
            "best_score": float(search.best_score_),
            "best_params": jsonable(search.best_params_),
        },
        "tfidf": {
            "actual_features": int(len(vectorizer.vocabulary_)),
            "strip_accents": "unicode",
            "lowercase": True,
        },
    }
    write_json(model_output_dir / "run_config.json", run_config)

    metrics = evaluation["metrics"]
    return {
        "model_key": spec.key,
        "classifier": spec.display_name,
        "best_cv_score": float(search.best_score_),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "mcc": metrics["mcc"],
        "log_loss": metrics["log_loss"],
        "elapsed_seconds": elapsed_seconds,
        "n_iter_effective": effective_n_iter,
        "cv": args.cv,
        "best_params": best_params_short(search.best_params_),
        "artifact_dir": str(model_output_dir),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    specs_by_key = build_model_specs(args.random_state, args.estimator_jobs)
    selected_specs = select_specs(args, specs_by_key)

    data = load_kaggle_imdb(args.csv_path)
    if args.limit is not None:
        data = data.sample(n=args.limit, random_state=args.random_state).reset_index(drop=True)

    split = make_train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    run_config = {
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
        "search": {
            "search_class": "RandomizedSearchCV",
            "n_iter": args.n_iter,
            "cv": args.cv,
            "scoring": args.scoring,
            "search_jobs": args.search_jobs,
            "estimator_jobs": args.estimator_jobs,
        },
        "models": [spec.key for spec in selected_specs],
    }
    write_json(args.output_dir / "run_config.json", run_config)

    rows = []
    for index, spec in enumerate(selected_specs, start=1):
        print(f"[{index}/{len(selected_specs)}] RandomizedSearchCV for {spec.display_name}")
        rows.append(run_search_for_spec(spec, args, split))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    write_text(args.output_dir / "summary.md", make_summary_markdown(summary, args))

    print(f"Saved RandomizedSearchCV summary to {args.output_dir / 'summary.csv'}")
    print(summary.loc[:, ["classifier", "best_cv_score", "accuracy", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
