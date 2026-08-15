"""Small error-analysis tables."""

from __future__ import annotations

import pandas as pd


def misclassifications(frame: pd.DataFrame, truth: str = "canonical_label") -> pd.DataFrame:
    """Return only wrongly classified rows from a prediction table."""
    if truth not in frame or "prediction" not in frame:
        raise ValueError("Prediction table must include truth and prediction columns")
    return frame.loc[frame[truth] != frame["prediction"]].copy()
