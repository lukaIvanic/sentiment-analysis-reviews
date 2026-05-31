"""Dataset loading and splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.sentiment.paths import KAGGLE_IMDB_CSV


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class DatasetSplit:
    """Container for one reproducible train/test split."""

    x_train: pd.Series
    x_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def load_kaggle_imdb(csv_path: Path = KAGGLE_IMDB_CSV) -> pd.DataFrame:
    """Load the Kaggle IMDb CSV and normalize columns for binary classification."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing Kaggle IMDb CSV at {csv_path}. "
            "Download/extract the dataset before running this pipeline."
        )

    data = pd.read_csv(csv_path)
    required_columns = {"review", "sentiment"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    data = data.loc[:, ["review", "sentiment"]].copy()
    data["review"] = data["review"].fillna("").astype(str)
    data["sentiment"] = data["sentiment"].astype(str).str.lower().str.strip()

    unknown_labels = sorted(set(data["sentiment"]) - set(LABEL_TO_ID))
    if unknown_labels:
        raise ValueError(f"Unexpected sentiment labels: {unknown_labels}")

    data["label"] = data["sentiment"].map(LABEL_TO_ID).astype(int)
    return data


def make_train_test_split(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a stratified split so positive/negative balance is preserved."""

    x_train, x_test, y_train, y_test = train_test_split(
        data["review"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    return DatasetSplit(
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def class_counts(labels: pd.Series) -> dict[str, int]:
    """Return class counts with human-readable labels."""

    counts = labels.value_counts().sort_index()
    return {ID_TO_LABEL[int(label_id)]: int(count) for label_id, count in counts.items()}
