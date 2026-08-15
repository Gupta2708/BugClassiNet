"""Processed-data validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bugclassinet.constants import REQUIRED_TEXT_COLUMNS
from bugclassinet.data.deduplicate import duplicate_overlap


def validate_frame(frame: pd.DataFrame, expected_labels: set[str] | None = None) -> None:
    """Raise informative errors for unusable processed datasets."""
    missing = [column for column in REQUIRED_TEXT_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            f"Required processed columns missing: {missing}; available: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["text"].isna().any() or frame["text"].str.strip().eq("").any():
        raise ValueError("Dataset contains missing or blank model inputs")
    if expected_labels and not set(frame["canonical_label"].unique()).issubset(expected_labels):
        raise ValueError(
            f"Unexpected labels: {sorted(set(frame['canonical_label']) - expected_labels)}"
        )


def validate_parquet(path: str | Path, expected_labels: set[str] | None = None) -> pd.DataFrame:
    """Read and validate a Parquet frame."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Processed data missing: {source}")
    frame = pd.read_parquet(source)
    validate_frame(frame, expected_labels)
    return frame


def validate_nlbse_variants(directory: str | Path) -> dict[str, int]:
    """Validate benchmark/clean NLBSE variants and their official-test isolation."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Processed NLBSE directory does not exist: {root}")
    required = {
        "train_benchmark": root / "train_benchmark.parquet",
        "validation": root / "validation.parquet",
        "train_clean": root / "train_clean.parquet",
        "validation_clean": root / "validation_clean.parquet",
        "test": root / "test.parquet",
    }
    frames = {name: validate_parquet(path) for name, path in required.items()}
    clean_overlap = duplicate_overlap(frames["train_clean"], frames["test"])
    if not clean_overlap.empty:
        raise ValueError(f"Clean train overlaps official test: {len(clean_overlap)} rows")
    validation_overlap = duplicate_overlap(frames["validation_clean"], frames["test"])
    if not validation_overlap.empty:
        raise ValueError(f"Clean validation overlaps official test: {len(validation_overlap)} rows")
    for train_name, validation_name in (
        ("train_benchmark", "validation"),
        ("train_clean", "validation_clean"),
    ):
        overlap = set(frames[train_name]["issue_id"]) & set(frames[validation_name]["issue_id"])
        if overlap:
            raise ValueError(f"{train_name}/{validation_name} have ID overlap")
    return {name: len(frame) for name, frame in frames.items()}
