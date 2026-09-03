import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

from bugclassinet.evaluation.reporting import STAGE1_REPORT_LABELS, write_stage1_evaluation


def test_stage1_report_has_fixed_order_correct_metrics_and_artifacts(tmp_path, capsys) -> None:
    truth = ["BUG", "BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION"]
    predicted = ["BUG", "DOCUMENTATION", "DOCUMENTATION", "BUG", "ENHANCEMENT"]
    label_to_id = {label: index for index, label in enumerate(STAGE1_REPORT_LABELS)}
    logits = np.arange(20, dtype=np.float32).reshape(5, 4)

    metrics = write_stage1_evaluation(
        tmp_path,
        truth,
        predicted,
        label_to_id,
        eval_loss=0.75,
        logits=logits,
        score_label_order=STAGE1_REPORT_LABELS,
        issue_ids=[f"issue-{index}" for index in range(len(truth))],
    )

    assert metrics["labels"] == list(STAGE1_REPORT_LABELS)
    assert metrics["accuracy"] == pytest.approx(accuracy_score(truth, predicted))
    assert metrics["micro_f1"] == pytest.approx(
        f1_score(truth, predicted, labels=STAGE1_REPORT_LABELS, average="micro")
    )
    assert metrics["macro_f1"] == pytest.approx(
        f1_score(truth, predicted, labels=STAGE1_REPORT_LABELS, average="macro")
    )
    assert metrics["weighted_f1"] == pytest.approx(
        f1_score(truth, predicted, labels=STAGE1_REPORT_LABELS, average="weighted")
    )
    assert metrics["mcc"] == pytest.approx(matthews_corrcoef(truth, predicted))
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, predicted))
    assert metrics["eval_loss"] == 0.75

    saved_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    for name in (
        "accuracy",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "balanced_accuracy",
    ):
        assert saved_metrics[name] == pytest.approx(metrics[name])

    per_class = pd.read_csv(tmp_path / "per_class_metrics.csv")
    assert per_class["class"].tolist() == list(STAGE1_REPORT_LABELS)
    assert per_class.loc[per_class["class"] == "QUESTION", "f1"].item() == 0.0
    assert per_class.loc[per_class["class"] == "BUG", "f1"].item() == pytest.approx(0.5)

    matrix = pd.read_csv(tmp_path / "confusion_matrix.csv", index_col="true_label")
    assert matrix.index.tolist() == [f"TRUE_{label}" for label in STAGE1_REPORT_LABELS]
    assert matrix.columns.tolist() == [f"PRED_{label}" for label in STAGE1_REPORT_LABELS]
    assert matrix.loc["TRUE_BUG", "PRED_DOCUMENTATION"] == 1
    assert matrix.loc["TRUE_ENHANCEMENT", "PRED_BUG"] == 1
    assert matrix.loc["TRUE_QUESTION", "PRED_ENHANCEMENT"] == 1

    counts = pd.read_csv(tmp_path / "prediction_counts.csv")
    assert counts["true_count"].sum() == len(truth)
    assert counts["predicted_count"].sum() == len(truth)
    assert counts.loc[counts["class"] == "QUESTION", "predicted_count"].item() == 0

    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(predictions) == len(truth)
    assert {
        "true_label",
        "predicted_label",
        "true_label_id",
        "predicted_label_id",
        "logit_BUG",
        "probability_QUESTION",
    } <= set(predictions.columns)
    assert np.allclose(
        predictions[[f"probability_{label}" for label in STAGE1_REPORT_LABELS]].sum(axis=1),
        1.0,
    )

    output = capsys.readouterr().out
    assert "=== OVERALL METRICS ===" in output
    assert "=== PER-CLASS METRICS ===" in output
    assert "=== CONFUSION MATRIX ===" in output
    assert "=== PREDICTION COUNTS ===" in output
    assert "TRUE CLASS COUNTS" in output
    assert "PREDICTED CLASS COUNTS" in output
    assert "=== FULL CLASSIFICATION REPORT ===" in output


def test_stage1_report_rejects_noncanonical_label_mapping(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly"):
        write_stage1_evaluation(tmp_path, ["BUG"], ["BUG"], {"BUG": 0})
