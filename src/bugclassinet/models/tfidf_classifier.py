"""Reproducible, memory-conscious TF-IDF LinearSVC baseline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bugclassinet.features.text import require_model_text
from bugclassinet.features.tfidf import (
    SparseMatrixLogger,
    TfidfFeatureConfig,
    make_tfidf_features,
)
from bugclassinet.utils.io import write_json


def resolved_tfidf_config(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve feature and classifier defaults for recording with artifacts."""
    values = values or {}
    feature_config = TfidfFeatureConfig.from_mapping(values)
    resolved = {
        **feature_config.as_dict(),
        "class_weight": values.get("class_weight", "balanced"),
        "classifier_max_iter": int(values.get("classifier_max_iter", 5_000)),
        "seed": int(values.get("seed", 42)),
    }
    for key in (
        "max_train_samples",
        "source_training_rows",
        "training_rows",
        "training_class_counts",
    ):
        if key in values:
            resolved[key] = values[key]
    return resolved


def build_tfidf_pipeline(values: Mapping[str, Any] | None = None) -> Pipeline:
    """Build sparse bounded TF-IDF followed by a balanced LinearSVC."""
    resolved = resolved_tfidf_config(values)
    features = TfidfFeatureConfig.from_mapping(resolved)
    return Pipeline(
        [
            ("features", make_tfidf_features(features)),
            ("matrix_audit", SparseMatrixLogger()),
            (
                "classifier",
                LinearSVC(
                    class_weight=resolved["class_weight"],
                    max_iter=resolved["classifier_max_iter"],
                    dual="auto",
                ),
            ),
        ]
    )


def train_tfidf(
    frame: pd.DataFrame,
    output_path: str | Path,
    config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Fit and persist a sparse Stage 1 baseline from canonical data."""
    if "canonical_label" not in frame:
        raise ValueError("TF-IDF training requires canonical_label")
    pipeline = build_tfidf_pipeline(config)
    pipeline.fit(require_model_text(frame), frame["canonical_label"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    write_json(target.parent / "training_config.json", resolved_tfidf_config(config))
    return pipeline


def load_tfidf(path: str | Path) -> Pipeline:
    """Load a persisted TF-IDF pipeline or fail loudly."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TF-IDF model does not exist: {source}")
    return joblib.load(source)
