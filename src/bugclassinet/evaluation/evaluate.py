"""Artifact-producing model evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bugclassinet.evaluation.metrics import classification_metrics
from bugclassinet.features.text import require_model_text
from bugclassinet.utils.io import write_json

LOGGER = logging.getLogger(__name__)


def predict_in_batches(model: Any, texts: pd.Series, batch_size: int = 1_000) -> np.ndarray:
    """Predict in bounded batches to keep large sparse TF-IDF transforms in memory."""
    if batch_size <= 0:
        raise ValueError("prediction batch_size must be positive")
    predictions: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        LOGGER.info("Predicting rows %d-%d of %d", start + 1, stop, total)
        predictions.append(np.asarray(model.predict(texts.iloc[start:stop])))
    return np.concatenate(predictions) if predictions else np.array([], dtype=str)


def evaluate_classifier(
    model: Any, frame: pd.DataFrame, output_dir: str | Path, prediction_batch_size: int = 1_000
) -> dict[str, Any]:
    """Evaluate a fitted classifier and write metrics plus row-level predictions."""
    if "canonical_label" not in frame:
        raise ValueError("Evaluation requires canonical_label")
    predictions = predict_in_batches(model, require_model_text(frame), prediction_batch_size)
    metrics = classification_metrics(frame["canonical_label"].tolist(), list(predictions))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "metrics.json", metrics)
    rows = frame.copy()
    rows["prediction"] = predictions
    rows.to_parquet(target / "predictions.parquet", index=False)
    return metrics
