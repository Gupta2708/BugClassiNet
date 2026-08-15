"""Classification metric calculation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Report reproducible aggregate, per-class, and confusion-matrix metrics."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metrics need equal non-empty prediction and target sequences")
    labels = sorted(set(y_true) | set(y_pred))
    scores = {}
    for average in ("macro", "micro", "weighted"):
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        scores[f"{average}_f1"] = float(f1)
    return {
        **scores,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
