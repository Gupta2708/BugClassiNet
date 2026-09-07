"""Frozen NLBSE 2023 to NLBSE 2024 cross-dataset evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from bugclassinet.data.nlbse2024 import load_nlbse2024_csv
from bugclassinet.evaluation.frozen_stage1 import FrozenStage1Evaluator
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json
from bugclassinet.utils.reproducibility import git_commit, utc_timestamp

NLBSE2024_TARGET_LABELS = ("BUG", "ENHANCEMENT", "QUESTION")
NATIVE_PREDICTION_LABELS = ("BUG", "ENHANCEMENT", "QUESTION", "DOCUMENTATION")


def resolve_stage1_label_ids(label_to_id: dict[str, int]) -> dict[str, int]:
    """Validate and resolve all Stage-1 IDs from persisted model metadata."""
    required = {"BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION"}
    if set(label_to_id) != required:
        raise ValueError(f"Frozen Stage-1 label metadata is incompatible: {label_to_id}")
    values = [int(label_to_id[label]) for label in required]
    if len(set(values)) != 4 or sorted(values) != list(range(4)):
        raise ValueError(f"Frozen Stage-1 label IDs must be unique and contiguous: {label_to_id}")
    return {label: int(label_to_id[label]) for label in required}


def project_three_class_probabilities(
    probabilities: np.ndarray, label_to_id: dict[str, int]
) -> tuple[np.ndarray, list[str]]:
    """Exclude DOCUMENTATION, renormalize, and argmax in canonical target order."""
    mapping = resolve_stage1_label_ids(label_to_id)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected four-class probabilities, got shape={values.shape}")
    indices = [mapping[label] for label in NLBSE2024_TARGET_LABELS]
    projected = values[:, indices]
    denominators = projected.sum(axis=1, keepdims=True)
    if (denominators <= 0).any() or not np.isfinite(denominators).all():
        raise ValueError("Cannot renormalize invalid projected probabilities")
    projected = projected / denominators
    predicted = [NLBSE2024_TARGET_LABELS[index] for index in np.argmax(projected, axis=1)]
    return projected.astype(np.float32), predicted


def _transfer_metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Transfer metrics require equal non-empty inputs")
    unexpected_truth = sorted(set(truth) - set(NLBSE2024_TARGET_LABELS))
    unexpected_predictions = sorted(set(predicted) - set(NATIVE_PREDICTION_LABELS))
    if unexpected_truth or unexpected_predictions:
        raise ValueError(
            "Unexpected transfer labels: "
            f"truth={unexpected_truth}, predictions={unexpected_predictions}"
        )
    report = classification_report(
        truth,
        predicted,
        labels=list(NLBSE2024_TARGET_LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = pd.crosstab(
        pd.Categorical(truth, categories=NLBSE2024_TARGET_LABELS),
        pd.Categorical(predicted, categories=NATIVE_PREDICTION_LABELS),
        dropna=False,
    )
    documentation_count = int(sum(label == "DOCUMENTATION" for label in predicted))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(NLBSE2024_TARGET_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(NLBSE2024_TARGET_LABELS),
                average="micro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(NLBSE2024_TARGET_LABELS),
                average="weighted",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "per_class": {label: report[label] for label in NLBSE2024_TARGET_LABELS},
        "true_class_counts": {label: truth.count(label) for label in NLBSE2024_TARGET_LABELS},
        "predicted_class_counts": {
            label: predicted.count(label) for label in NATIVE_PREDICTION_LABELS
        },
        "documentation_prediction_count": documentation_count,
        "documentation_prediction_rate": documentation_count / len(predicted),
        "confusion_matrix": {
            "true_labels": list(NLBSE2024_TARGET_LABELS),
            "predicted_labels": list(NATIVE_PREDICTION_LABELS),
            "matrix": matrix.to_numpy().tolist(),
        },
    }


def _classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"class": label, **metrics["per_class"][label]} for label in NLBSE2024_TARGET_LABELS]
    )


def _confusion_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    confusion = metrics["confusion_matrix"]
    return pd.DataFrame(
        confusion["matrix"],
        index=[f"TRUE_{label}" for label in confusion["true_labels"]],
        columns=[f"PRED_{label}" for label in confusion["predicted_labels"]],
    )


def _per_repo_metrics(
    metadata: pd.DataFrame, truth: list[str], predicted: list[str]
) -> pd.DataFrame:
    rows = pd.DataFrame({"repo": metadata["repo"].tolist(), "truth": truth, "pred": predicted})
    reports = []
    for repo, group in rows.groupby("repo", sort=True):
        metrics = _transfer_metrics(group["truth"].tolist(), group["pred"].tolist())
        reports.append(
            {
                "repo": repo,
                "rows": len(group),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mcc": metrics["mcc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "documentation_prediction_count": metrics["documentation_prediction_count"],
                "documentation_prediction_rate": metrics["documentation_prediction_rate"],
            }
        )
    return pd.DataFrame(reports)


def evaluate_nlbse2024(
    model_dir: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run native and projected frozen cross-dataset transfer protocols."""
    try:
        from datasets import Dataset
    except ImportError as error:
        raise ImportError(
            "NLBSE 2024 evaluation requires `pip install -e .[transformers]`."
        ) from error

    source = Path(data_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    frame, audit = load_nlbse2024_csv(source)
    write_json(destination / "dataset_audit.json", audit)
    metadata_columns = [
        column
        for column in (
            "source_row_id",
            "repo",
            "created_at",
            "original_label",
            "canonical_label",
            "text_hash",
        )
        if column in frame
    ]
    dataset = Dataset.from_pandas(
        frame[["text", *metadata_columns]],
        preserve_index=False,
    )
    del frame

    runtime_dir = destination.parent / ".bugclassinet-cache" / f"{destination.name}-nlbse2024"
    evaluator = FrozenStage1Evaluator(model_dir, runtime_dir, batch_size)
    label_ids = resolve_stage1_label_ids(evaluator.label_to_id)
    result = evaluator.predict(dataset, "nlbse2024-test", metadata_columns)
    native_metrics = _transfer_metrics(result.true_labels, result.predicted_labels)
    projected_probabilities, projected_predictions = project_three_class_probabilities(
        result.probabilities, label_ids
    )
    projected_metrics = _transfer_metrics(result.true_labels, projected_predictions)

    native_repo = _per_repo_metrics(result.metadata, result.true_labels, result.predicted_labels)
    projected_repo = _per_repo_metrics(result.metadata, result.true_labels, projected_predictions)
    native_metrics["cross_repo_macro_f1"] = float(native_repo["macro_f1"].mean())
    projected_metrics["cross_repo_macro_f1"] = float(projected_repo["macro_f1"].mean())

    for name, metrics, per_repo in (
        ("native", native_metrics, native_repo),
        ("projected_3class", projected_metrics, projected_repo),
    ):
        protocol_dir = destination / name
        protocol_dir.mkdir(parents=True, exist_ok=True)
        write_json(protocol_dir / "overall_metrics.json", metrics)
        per_repo.to_csv(protocol_dir / "per_repo_metrics.csv", index=False)
        _classification_report_frame(metrics).to_csv(
            protocol_dir / "classification_report.csv", index=False
        )
        _confusion_frame(metrics).to_csv(
            protocol_dir / "confusion_matrix.csv", index_label="true_label"
        )

    comparison = pd.DataFrame(
        [
            {
                "protocol": protocol,
                **{
                    key: metrics[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "micro_f1",
                        "weighted_f1",
                        "mcc",
                        "balanced_accuracy",
                        "cross_repo_macro_f1",
                        "documentation_prediction_count",
                        "documentation_prediction_rate",
                    )
                },
            }
            for protocol, metrics in (
                ("native_strict_four_output", native_metrics),
                ("projected_three_class_post_hoc", projected_metrics),
            )
        ]
    )
    comparison.to_csv(destination / "comparison.csv", index=False)

    predictions: dict[str, Any] = {
        "repo": result.metadata["repo"].tolist(),
        "original_label": result.metadata["original_label"].tolist(),
        "canonical_true_label": result.true_labels,
        "text_hash": result.metadata["text_hash"].tolist(),
        "pred_native": result.predicted_labels,
        "pred_projected": projected_predictions,
        "out_of_taxonomy_native": [label == "DOCUMENTATION" for label in result.predicted_labels],
    }
    if "created_at" in result.metadata:
        predictions["created_at"] = result.metadata["created_at"].tolist()
    for label, column in (
        ("BUG", "p_bug"),
        ("DOCUMENTATION", "p_documentation"),
        ("ENHANCEMENT", "p_enhancement"),
        ("QUESTION", "p_question"),
    ):
        predictions[column] = result.probabilities[:, label_ids[label]]
    for index, label in enumerate(NLBSE2024_TARGET_LABELS):
        predictions[f"p_projected_{label.lower()}"] = projected_probabilities[:, index]
    pd.DataFrame(predictions).to_parquet(destination / "predictions.parquet", index=False)

    parameter_hash_after = evaluator.verify_unchanged()
    manifest = {
        **evaluator.identity(),
        "experiment": "Frozen NLBSE 2023 -> NLBSE 2024 cross-dataset transfer",
        "protocols": ["native strict four-output", "projected three-class post-hoc"],
        "issues_train_used_for_fitting": False,
        "test_used_for_tuning": False,
        "model_parameter_sha256_after": parameter_hash_after,
        "model_parameters_unchanged": True,
        "git_commit": git_commit(),
        "dataset_path": str(source),
        "dataset_sha256": sha256_file(source),
        "rows": len(result.true_labels),
        "evaluation_timestamp": utc_timestamp(),
    }
    write_json(destination / "run_manifest.json", manifest)
    print("\n=== FROZEN NLBSE 2023 -> NLBSE 2024 CROSS-DATASET TRANSFER ===")
    print(comparison.to_string(index=False))
    print("\nNative per-repository metrics")
    print(native_repo.to_string(index=False))
    print("\nProjected three-class post-hoc per-repository metrics")
    print(projected_repo.to_string(index=False))
    return {
        "experiment": manifest["experiment"],
        "training_invoked": False,
        "native": native_metrics,
        "projected_3class": projected_metrics,
    }
