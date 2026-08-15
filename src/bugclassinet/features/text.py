"""Safe model-input text helpers."""

from __future__ import annotations

import pandas as pd


def require_model_text(frame: pd.DataFrame) -> pd.Series:
    """Return text input while preventing accidental label leakage columns."""
    if "text" not in frame:
        raise ValueError("Model input requires a text column")
    values = frame["text"].fillna("").astype(str)
    if values.str.strip().eq("").any():
        raise ValueError("Model input contains blank text")
    return values
