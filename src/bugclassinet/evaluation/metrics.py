"""Classification metric calculation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report reproducible aggregate, per-class, and confusion-matrix metrics."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metrics need equal non-empty prediction and target sequences")
    ordered_labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    if not ordered_labels or len(ordered_labels) != len(set(ordered_labels)):
        raise ValueError("Metric labels must be a non-empty sequence of unique values")
    unexpected = sorted((set(y_true) | set(y_pred)) - set(ordered_labels))
    if unexpected:
        raise ValueError(f"Observed labels are absent from the requested order: {unexpected}")
    scores = {}
    for average in ("macro", "micro", "weighted"):
        scores[f"{average}_f1"] = float(
            f1_score(
                y_true,
                y_pred,
                labels=ordered_labels,
                average=average,
                zero_division=0,
            )
        )
    return {
        **scores,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": ordered_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ordered_labels).tolist(),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=ordered_labels,
            output_dict=True,
            zero_division=0,
        ),
    }
