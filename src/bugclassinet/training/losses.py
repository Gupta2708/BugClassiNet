"""Loss documentation and calculation helpers."""

from __future__ import annotations


def balanced_class_weights(labels: list[str]) -> dict[str, float]:
    """Return inverse-frequency normalized weights for weighted cross entropy."""
    if not labels:
        raise ValueError("Cannot calculate class weights from no labels")
    counts = {label: labels.count(label) for label in set(labels)}
    total = len(labels)
    return {label: total / (len(counts) * count) for label, count in counts.items()}
