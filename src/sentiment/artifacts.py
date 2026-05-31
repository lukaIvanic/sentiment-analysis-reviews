"""Small helpers for saving run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text file."""

    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def dump_model(path: Path, model: Any) -> None:
    """Persist a fitted sklearn pipeline."""

    ensure_dir(path.parent)
    joblib.dump(model, path)

