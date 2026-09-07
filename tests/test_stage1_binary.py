from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from bugclassinet.cli import build_parser
from bugclassinet.evaluation.binary import binary_ground_truth, select_macro_f1_threshold
from bugclassinet.training import commands


def test_binary_ground_truth_maps_only_bug_to_bug() -> None:
    assert binary_ground_truth(["BUG", "DOCUMENTATION", "ENHANCEMENT", "QUESTION"]) == [
        "BUG",
        "NON_BUG",
        "NON_BUG",
        "NON_BUG",
    ]


def test_threshold_selection_maximizes_validation_macro_f1_deterministically() -> None:
    truth = ["BUG", "BUG", "NON_BUG", "NON_BUG"]
    probabilities = np.array([0.9, 0.6, 0.55, 0.1])
    threshold, table = select_macro_f1_threshold(truth, probabilities)
    predicted = ["BUG" if score >= threshold else "NON_BUG" for score in probabilities]

    assert threshold == pytest.approx(0.6)
    assert f1_score(truth, predicted, average="macro") == pytest.approx(1.0)
    assert table.loc[table["selected"], "threshold"].tolist() == [pytest.approx(0.6)]


def test_new_evaluation_commands_dispatch_without_training(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        commands,
        "binary_evaluation",
        lambda *args: calls.append(("binary", args)) or {"training_invoked": False},
    )
    monkeypatch.setattr(
        commands,
        "nlbse2024_evaluation",
        lambda *args: calls.append(("external", args)) or {"training_invoked": False},
    )
    parser = build_parser()
    binary = parser.parse_args(
        [
            "evaluate-stage1-binary",
            "--model",
            "model",
            "--validation",
            "validation.parquet",
            "--test",
            "test.parquet",
            "--output-dir",
            str(tmp_path / "binary"),
        ]
    )
    external = parser.parse_args(
        [
            "evaluate-nlbse2024",
            "--model",
            "model",
            "--data",
            "issues_test.csv",
            "--output-dir",
            str(tmp_path / "external"),
            "--batch-size",
            "16",
        ]
    )
    binary.func(binary)
    external.func(external)

    assert [name for name, _ in calls] == ["binary", "external"]
    assert calls[0][1][-1] is None
    assert calls[1][1][-1] == 16
