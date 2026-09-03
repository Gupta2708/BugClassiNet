"""Detailed, artifact-producing Stage-1 evaluation reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.utils.io import write_json

STAGE1_REPORT_LABELS = ("BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION")


def _classification_report_frame(report: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered_rows = [*STAGE1_REPORT_LABELS, "accuracy", "macro avg", "weighted avg"]
    for name in ordered_rows:
        value = report.get(name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            row = {"class": name, **value}
        else:
            row = {
                "class": name,
                "precision": float(value),
                "recall": float(value),
                "f1-score": float(value),
                "support": report.get("weighted avg", {}).get("support", 0),
            }
        rows.append(row)
    return pd.DataFrame(rows, columns=["class", "precision", "recall", "f1-score", "support"])


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _print_report(
    metrics: Mapping[str, Any],
    per_class: pd.DataFrame,
    matrix: pd.DataFrame,
    counts: pd.DataFrame,
    full_report: pd.DataFrame,
) -> None:
    print("\n=== OVERALL METRICS ===")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Micro-F1: {metrics['micro_f1']:.6f}")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.6f}")
    print(f"MCC: {metrics['mcc']:.6f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.6f}")
    if metrics.get("eval_loss") is not None:
        print(f"Eval loss: {metrics['eval_loss']:.6f}")

    print("\n=== PER-CLASS METRICS ===")
    for row in per_class.itertuples(index=False):
        print(f"\n{row[0]}")
        print(f"Precision : {row[1]:.6f}")
        print(f"Recall    : {row[2]:.6f}")
        print(f"F1        : {row[3]:.6f}")
        print(f"Support   : {int(row[4])}")

    print("\n=== CONFUSION MATRIX ===")
    print(matrix.to_string())
    print("\n=== PREDICTION COUNTS ===")
    print("\nTRUE CLASS COUNTS")
    print(counts.loc[:, ["class", "true_count"]].to_string(index=False))
    print("\nPREDICTED CLASS COUNTS")
    print(counts.loc[:, ["class", "predicted_count"]].to_string(index=False))
    print("\n=== FULL CLASSIFICATION REPORT ===")
    print(full_report.to_string(index=False))


def write_stage1_evaluation(
    output_dir: str | Path,
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    label_to_id: Mapping[str, int],
    *,
    eval_loss: float | None = None,
    logits: np.ndarray | None = None,
    score_label_order: Sequence[str] | None = None,
    issue_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute, print, and save one complete Stage-1 evaluation package."""
    truth = list(true_labels)
    predictions = list(predicted_labels)
    if len(truth) != len(predictions):
        raise ValueError("Evaluation targets and predictions have different row counts")
    if set(label_to_id) != set(STAGE1_REPORT_LABELS):
        raise ValueError(f"Stage-1 label mapping must contain exactly {list(STAGE1_REPORT_LABELS)}")

    metrics = classification_metrics(truth, predictions, labels=STAGE1_REPORT_LABELS)
    if eval_loss is not None:
        metrics["eval_loss"] = float(eval_loss)

    report = metrics["per_class"]
    per_class = pd.DataFrame(
        [
            {
                "class": label,
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in STAGE1_REPORT_LABELS
        ]
    )
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"TRUE_{label}" for label in STAGE1_REPORT_LABELS],
        columns=[f"PRED_{label}" for label in STAGE1_REPORT_LABELS],
    )
    counts = pd.DataFrame(
        {
            "class": STAGE1_REPORT_LABELS,
            "true_count": [truth.count(label) for label in STAGE1_REPORT_LABELS],
            "predicted_count": [predictions.count(label) for label in STAGE1_REPORT_LABELS],
        }
    )
    full_report = _classification_report_frame(report)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "metrics.json", metrics)
    per_class.to_csv(target / "per_class_metrics.csv", index=False)
    matrix.to_csv(target / "confusion_matrix.csv", index_label="true_label")
    full_report.to_csv(target / "classification_report.csv", index=False)
    counts.to_csv(target / "prediction_counts.csv", index=False)

    rows: dict[str, Any] = {
        "true_label": truth,
        "predicted_label": predictions,
        "true_label_id": [label_to_id[label] for label in truth],
        "predicted_label_id": [label_to_id[label] for label in predictions],
    }
    if issue_ids is not None:
        if len(issue_ids) != len(truth):
            raise ValueError("Issue ID count differs from evaluation row count")
        rows = {"issue_id": list(issue_ids), **rows}
    if logits is not None:
        values = np.asarray(logits, dtype=np.float32)
        score_labels = list(score_label_order or STAGE1_REPORT_LABELS)
        if values.shape != (len(truth), len(score_labels)):
            raise ValueError(
                "Logit shape does not match rows/classes: "
                f"shape={values.shape}, rows={len(truth)}, labels={score_labels}"
            )
        probabilities = _probabilities(values)
        for index, label in enumerate(score_labels):
            rows[f"logit_{label}"] = values[:, index]
        for index, label in enumerate(score_labels):
            rows[f"probability_{label}"] = probabilities[:, index]
    pd.DataFrame(rows).to_parquet(target / "predictions.parquet", index=False)

    _print_report(metrics, per_class, matrix, counts, full_report)
    return metrics
