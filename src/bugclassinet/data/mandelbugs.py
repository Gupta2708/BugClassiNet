"""Mandelbugs dataset guards used by Stage 2 and 3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_mandelbugs(path: str | Path, expected: set[str]) -> pd.DataFrame:
    """Load a Mandelbugs Parquet file and validate text, labels, and project metadata."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required Mandelbugs Parquet is missing: {source}")
    frame = pd.read_parquet(source)
    required = {"text", "canonical_label", "project"}
    if missing := required - set(frame.columns):
        raise ValueError(
            f"Mandelbugs missing columns {sorted(missing)}; available={list(frame.columns)}"
        )
    unexpected = set(frame["canonical_label"].unique()) - expected
    if unexpected:
        raise ValueError(f"Unexpected Mandelbugs labels: {sorted(unexpected)}")
    return frame
