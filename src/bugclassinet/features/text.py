"""Safe model-input text helpers."""

from __future__ import annotations

import pandas as pd


def require_model_text(frame: pd.DataFrame) -> pd.Series:
    """Return the existing text Series without making a million-row copy."""
    if "text" not in frame:
        raise ValueError("Model input requires a text column")
    values = frame["text"]
    if values.isna().any():
        raise ValueError("Model input contains missing text")
    if any(not isinstance(value, str) or not value.strip() for value in values.array):
        raise ValueError("Model input contains blank text")
    return values
