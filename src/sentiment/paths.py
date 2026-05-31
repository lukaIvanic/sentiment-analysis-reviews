"""Project paths used by training scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

KAGGLE_IMDB_CSV = RAW_DATA_DIR / "kaggle_imdb_50k" / "IMDB Dataset.csv"

