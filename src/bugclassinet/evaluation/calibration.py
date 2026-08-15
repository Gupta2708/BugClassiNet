"""Confidence abstention helpers."""

from __future__ import annotations

import numpy as np


def apply_abstention(labels: list[str], probabilities: np.ndarray, threshold: float) -> list[str]:
    """Replace predictions below a configured confidence with ABSTAIN."""
    if not 0 <= threshold < 1:
        raise ValueError("Abstention threshold must be in [0, 1)")
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths")
    return [
        label if float(row.max()) >= threshold else "ABSTAIN"
        for label, row in zip(labels, probabilities, strict=True)
    ]
