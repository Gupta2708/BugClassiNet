"""Reproducible TF-IDF LinearSVC baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bugclassinet.features.text import require_model_text
from bugclassinet.features.tfidf import make_tfidf_features


def build_tfidf_pipeline() -> Pipeline:
    """Build the specified word+character TF-IDF and balanced LinearSVC pipeline."""
    return Pipeline(
        [("features", make_tfidf_features()), ("classifier", LinearSVC(class_weight="balanced"))]
    )


def train_tfidf(frame: pd.DataFrame, output_path: str | Path) -> Pipeline:
    """Fit and persist a Stage 1 baseline from canonical data."""
    if "canonical_label" not in frame:
        raise ValueError("TF-IDF training requires canonical_label")
    pipeline = build_tfidf_pipeline()
    pipeline.fit(require_model_text(frame), frame["canonical_label"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    return pipeline


def load_tfidf(path: str | Path) -> Pipeline:
    """Load a persisted TF-IDF pipeline or fail loudly."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TF-IDF model does not exist: {source}")
    return joblib.load(source)
