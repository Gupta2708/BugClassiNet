from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bugclassinet.data.harmonize import construct_issue_text
from bugclassinet.data.nlbse2024 import load_nlbse2024_csv
from bugclassinet.evaluation.nlbse2024 import (
    _transfer_metrics,
    project_three_class_probabilities,
    resolve_stage1_label_ids,
)


def _write_csv(tmp_path, labels: list[str], titles=None, bodies=None):
    rows = len(labels)
    path = tmp_path / "issues_test.csv"
    pd.DataFrame(
        {
            "repo": [f"repo-{index % 2}" for index in range(rows)],
            "created_at": ["2024-01-01"] * rows,
            "label": labels,
            "title": titles if titles is not None else ["title"] * rows,
            "body": bodies if bodies is not None else ["body"] * rows,
        }
    ).to_csv(path, index=False)
    return path


def test_nlbse2024_mapping_and_stage1_text_construction(tmp_path) -> None:
    source = _write_csv(tmp_path, ["bug", "feature", "question"])
    frame, audit = load_nlbse2024_csv(source)

    assert frame["canonical_label"].tolist() == ["BUG", "ENHANCEMENT", "QUESTION"]
    assert frame["original_label"].tolist() == ["bug", "feature", "question"]
    assert frame["text"].iloc[0] == construct_issue_text("title", "body")
    assert audit["rows"] == 3
    assert sum(audit["label_counts"].values()) == 3


def test_nlbse2024_rejects_unexpected_label(tmp_path) -> None:
    source = _write_csv(tmp_path, ["bug", "documentation"])
    with pytest.raises(ValueError, match="Unexpected NLBSE 2024 labels"):
        load_nlbse2024_csv(source)


def test_nlbse2024_null_title_and_body_preserve_rows(tmp_path) -> None:
    source = _write_csv(
        tmp_path,
        ["bug", "question"],
        titles=[None, "Why?"],
        bodies=["Crash", None],
    )
    frame, audit = load_nlbse2024_csv(source)

    assert len(frame) == 2
    assert frame["text"].notna().all()
    assert audit["missing_title_count"] == 1
    assert audit["missing_body_count"] == 1


def test_stage1_ids_are_resolved_from_metadata_and_projection_excludes_docs() -> None:
    mapping = {"QUESTION": 0, "BUG": 1, "DOCUMENTATION": 2, "ENHANCEMENT": 3}
    resolved = resolve_stage1_label_ids(mapping)
    assert resolved["BUG"] == 1

    probabilities = np.array([[0.1, 0.2, 0.6, 0.1], [0.6, 0.1, 0.2, 0.1]])
    projected, predicted = project_three_class_probabilities(probabilities, mapping)
    assert projected.shape == (2, 3)
    assert np.allclose(projected.sum(axis=1), 1.0)
    assert predicted == ["BUG", "QUESTION"]


def test_native_documentation_prediction_is_retained_and_counted() -> None:
    metrics = _transfer_metrics(
        ["BUG", "ENHANCEMENT", "QUESTION"],
        ["DOCUMENTATION", "ENHANCEMENT", "QUESTION"],
    )

    assert metrics["documentation_prediction_count"] == 1
    assert metrics["documentation_prediction_rate"] == pytest.approx(1 / 3)
    assert metrics["predicted_class_counts"]["DOCUMENTATION"] == 1
    assert metrics["confusion_matrix"]["predicted_labels"][-1] == "DOCUMENTATION"
