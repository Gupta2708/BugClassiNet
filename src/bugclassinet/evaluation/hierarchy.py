"""Hierarchical predictions and aggregate routing evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bugclassinet.evaluation.metrics import classification_metrics


def evaluate_hierarchy(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate final hierarchy output when gold final labels are available."""
    required = {"final_label", "prediction"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Hierarchy evaluation missing {sorted(missing)}")
    return classification_metrics(frame["final_label"].tolist(), frame["prediction"].tolist())
