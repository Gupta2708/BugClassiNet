"""Configuration and dataset helpers shared by trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.settings import load_yaml


def training_config(path: str | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Overlay an optional YAML mapping onto explicit safe defaults."""
    return {**defaults, **(load_yaml(path) if path else {})}


def read_split(path: str | None, data_dir: str | None, split: str) -> pd.DataFrame:
    """Read a requested Parquet split, failing loudly when absent."""
    if path:
        source = Path(path)
    else:
        root = Path(data_dir or "")
        preferred = {"train": "train_clean.parquet", "validation": "validation_clean.parquet"}
        source = root / preferred.get(split, f"{split}.parquet")
    if not source.is_file():
        raise FileNotFoundError(f"Required {split} Parquet is missing: {source}")
    return pd.read_parquet(source)
