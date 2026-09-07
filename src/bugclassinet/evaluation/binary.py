"""Derived BUG versus NON_BUG evaluation for a frozen four-class model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from bugclassinet.data.transformer import load_parquet_dataset
from bugclassinet.evaluation.frozen_stage1 import FrozenPredictions, FrozenStage1Evaluator
from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json
from bugclassinet.utils.reproducibility import git_commit, utc_timestamp

BINARY_LABELS = ("BUG", "NON_BUG")


def binary_ground_truth(labels: list[str]) -> list[str]:
    """Collapse the canonical Stage-1 truth labels without fitting a model."""
    allowed = {"BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION"}
    unexpected = sorted(set(labels) - allowed)
    if unexpected:
        raise ValueError(f"Cannot derive binary labels from: {unexpected}")
    return ["BUG" if label == "BUG" else "NON_BUG" for label in labels]


def select_macro_f1_threshold(
    true_binary: list[str], p_bug: np.ndarray
) -> tuple[float, pd.DataFrame]:
    """Select a BUG threshold by validation Macro-F1 with deterministic ties."""
    scores = np.asarray(p_bug, dtype=np.float64)
    if len(true_binary) != len(scores) or not len(scores):
        raise ValueError("Threshold selection requires equal non-empty validation arrays")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("BUG probabilities must be finite and between zero and one")
    truth = np.asarray([label == "BUG" for label in true_binary], dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    cumulative_tp = np.cumsum(sorted_truth)
    cumulative_fp = np.cumsum(1 - sorted_truth)
    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], np.array([True])])
    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = cumulative_fp[group_ends].astype(np.float64)
    positives = float(truth.sum())
    negatives = float(len(truth) - truth.sum())
    fn = positives - tp
    tn = negatives - fp
    bug_denominator = (2 * tp) + fp + fn
    non_bug_denominator = (2 * tn) + fp + fn
    bug_f1 = np.divide(2 * tp, bug_denominator, out=np.zeros_like(tp), where=bug_denominator > 0)
    non_bug_f1 = np.divide(
        2 * tn,
        non_bug_denominator,
        out=np.zeros_like(tn),
        where=non_bug_denominator > 0,
    )
    table = pd.DataFrame(
        {
            "threshold": sorted_scores[group_ends],
            "macro_f1": (bug_f1 + non_bug_f1) / 2,
            "bug_f1": bug_f1,
            "non_bug_f1": non_bug_f1,
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "tn": tn.astype(int),
            "fn": fn.astype(int),
        }
    )
    best_macro = table["macro_f1"].max()
    candidates = table.loc[np.isclose(table["macro_f1"], best_macro)].copy()
    candidates["distance_from_0_5"] = (candidates["threshold"] - 0.5).abs()
    selected = candidates.sort_values(
        ["distance_from_0_5", "threshold"], ascending=[True, True], kind="stable"
    ).iloc[0]
    table["selected"] = np.isclose(table["threshold"], float(selected["threshold"])) & np.isclose(
        table["macro_f1"], best_macro
    )
    return float(selected["threshold"]), table


def _binary_metrics(truth: list[str], predicted: list[str], p_bug: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(truth, predicted, labels=BINARY_LABELS)
    numeric_truth = np.asarray([label == "BUG" for label in truth], dtype=np.int64)
    metrics["bug_pr_auc"] = float(average_precision_score(numeric_truth, p_bug))
    metrics["roc_auc"] = float(roc_auc_score(numeric_truth, p_bug))
    metrics["brier_score"] = float(brier_score_loss(numeric_truth, p_bug))
    return metrics


def _strategy_predictions(
    result: FrozenPredictions,
    p_bug: np.ndarray,
    threshold: float,
) -> dict[str, list[str]]:
    return {
        "collapsed_argmax": binary_ground_truth(result.predicted_labels),
        "p_bug_0.5": ["BUG" if score >= 0.5 else "NON_BUG" for score in p_bug],
        "p_bug_validation_threshold": [
            "BUG" if score >= threshold else "NON_BUG" for score in p_bug
        ],
    }


def _report_rows(split: str, strategy: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"split": split, "strategy": strategy, "class": label, **metrics["per_class"][label]}
        for label in BINARY_LABELS
    ]


def _matrix_rows(split: str, strategy: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "strategy": strategy,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": metrics["confusion_matrix"][true_index][predicted_index],
        }
        for true_index, true_label in enumerate(BINARY_LABELS)
        for predicted_index, predicted_label in enumerate(BINARY_LABELS)
    ]


def _parquet_dataset(path: Path, cache_dir: Path) -> Any:
    schema = pq.read_schema(path).names
    columns = [column for column in ("issue_id", "text", "canonical_label") if column in schema]
    return load_parquet_dataset(path, columns, cache_dir)


def evaluate_stage1_binary(
    model_dir: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate two binary derivations using validation-only threshold selection."""
    validation_source = Path(validation_path).resolve()
    test_source = Path(test_path).resolve()
    destination = Path(output_dir).resolve()
    for source in (validation_source, test_source):
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation dataset does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    runtime_dir = destination.parent / ".bugclassinet-cache" / f"{destination.name}-binary"
    evaluator = FrozenStage1Evaluator(model_dir, runtime_dir, batch_size)
    bug_id = evaluator.label_to_id["BUG"]

    validation = evaluator.predict(
        _parquet_dataset(validation_source, runtime_dir / "raw-validation"),
        "binary-validation",
        [
            column
            for column in ("issue_id", "canonical_label")
            if column in pq.read_schema(validation_source).names
        ],
    )
    validation_truth = binary_ground_truth(validation.true_labels)
    validation_p_bug = validation.probabilities[:, bug_id]
    selected_threshold, threshold_table = select_macro_f1_threshold(
        validation_truth, validation_p_bug
    )

    test = evaluator.predict(
        _parquet_dataset(test_source, runtime_dir / "raw-test"),
        "binary-test",
        [
            column
            for column in ("issue_id", "canonical_label")
            if column in pq.read_schema(test_source).names
        ],
    )
    test_truth = binary_ground_truth(test.true_labels)
    test_p_bug = test.probabilities[:, bug_id]
    all_metrics: dict[str, dict[str, Any]] = {}
    report_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for split, result, truth, probabilities in (
        ("validation", validation, validation_truth, validation_p_bug),
        ("test", test, test_truth, test_p_bug),
    ):
        split_metrics: dict[str, Any] = {}
        for strategy, predictions in _strategy_predictions(
            result, probabilities, selected_threshold
        ).items():
            metrics = _binary_metrics(truth, predictions, probabilities)
            split_metrics[strategy] = metrics
            report_rows.extend(_report_rows(split, strategy, metrics))
            matrix_rows.extend(_matrix_rows(split, strategy, metrics))
        all_metrics[split] = split_metrics

    threshold_table.to_csv(destination / "threshold_selection.csv", index=False)
    pd.DataFrame(report_rows).to_csv(destination / "classification_report.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(destination / "confusion_matrix.csv", index=False)
    write_json(destination / "validation_metrics.json", all_metrics["validation"])
    write_json(destination / "test_metrics.json", all_metrics["test"])
    summary = {
        "experiment": "Derived BUG vs NON_BUG evaluation of frozen Stage-1 model",
        "training_invoked": False,
        "threshold_selection_split": "validation",
        "threshold_selection_criterion": "maximum binary Macro-F1",
        "selected_p_bug_threshold": selected_threshold,
        "test": {
            strategy: {
                key: metrics[key]
                for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc")
            }
            for strategy, metrics in all_metrics["test"].items()
        },
    }
    write_json(destination / "summary.json", summary)

    strategy_predictions = _strategy_predictions(test, test_p_bug, selected_threshold)
    prediction_rows: dict[str, Any] = {
        "true_multiclass_label": test.true_labels,
        "pred_multiclass_label": test.predicted_labels,
        "true_binary_label": test_truth,
        "pred_binary_collapsed": strategy_predictions["collapsed_argmax"],
        "p_bug": test_p_bug,
        "pred_binary_p05": strategy_predictions["p_bug_0.5"],
        "pred_binary_validation_threshold": strategy_predictions["p_bug_validation_threshold"],
    }
    if "issue_id" in test.metadata:
        prediction_rows = {"issue_id": test.metadata["issue_id"].tolist(), **prediction_rows}
    pd.DataFrame(prediction_rows).to_parquet(destination / "predictions.parquet", index=False)

    parameter_hash_after = evaluator.verify_unchanged()
    manifest = {
        **evaluator.identity(),
        "model_parameter_sha256_after": parameter_hash_after,
        "model_parameters_unchanged": True,
        "git_commit": git_commit(),
        "validation_path": str(validation_source),
        "validation_sha256": sha256_file(validation_source),
        "validation_rows": len(validation.true_labels),
        "test_path": str(test_source),
        "test_sha256": sha256_file(test_source),
        "test_rows": len(test.true_labels),
        "selected_threshold": selected_threshold,
        "threshold_selected_from": "validation only",
        "evaluation_timestamp": utc_timestamp(),
    }
    write_json(destination / "run_manifest.json", manifest)
    print("\n=== DERIVED BINARY EVALUATION ===")
    print(f"Validation-selected p_bug threshold: {selected_threshold:.8f}")
    for strategy, metrics in all_metrics["test"].items():
        bug = metrics["per_class"]["BUG"]
        non_bug = metrics["per_class"]["NON_BUG"]
        print(f"\n{strategy}")
        print(
            f"Accuracy={metrics['accuracy']:.6f} Macro-F1={metrics['macro_f1']:.6f} "
            f"Weighted-F1={metrics['weighted_f1']:.6f} "
            f"Balanced Accuracy={metrics['balanced_accuracy']:.6f} MCC={metrics['mcc']:.6f}"
        )
        print(
            f"BUG P/R/F1/support={bug['precision']:.6f}/{bug['recall']:.6f}/"
            f"{bug['f1-score']:.6f}/{int(bug['support'])}"
        )
        print(
            "NON_BUG P/R/F1/support="
            f"{non_bug['precision']:.6f}/{non_bug['recall']:.6f}/"
            f"{non_bug['f1-score']:.6f}/{int(non_bug['support'])}"
        )
        print(
            f"BUG PR-AUC={metrics['bug_pr_auc']:.6f} ROC-AUC={metrics['roc_auc']:.6f} "
            f"Brier={metrics['brier_score']:.6f}"
        )
        print(np.asarray(metrics["confusion_matrix"]))
    return summary
