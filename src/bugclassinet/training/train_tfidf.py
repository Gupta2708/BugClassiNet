"""TF-IDF training package entry point."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from bugclassinet.evaluation.evaluate import evaluate_classifier
from bugclassinet.models.tfidf_classifier import train_tfidf
from bugclassinet.settings import load_yaml
from bugclassinet.training.common import read_split

LOGGER = logging.getLogger(__name__)


def _stratified_limit(
    frame: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    """Select an exact, deterministic class-stratified training subset."""
    if maximum is not None and maximum <= 0:
        raise ValueError("--max-train-samples must be positive")
    if maximum is None or len(frame) <= maximum:
        return frame
    if "canonical_label" not in frame:
        raise ValueError("Stratified TF-IDF sampling requires canonical_label")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=maximum, random_state=seed)
    selected, _ = next(splitter.split(frame, frame["canonical_label"]))
    return frame.iloc[selected]


def run(
    data_dir: str | None,
    train: str | None,
    validation: str | None,
    output_dir: str,
    config_path: str | None = None,
    max_train_samples: int | None = None,
) -> dict[str, object]:
    """Train on sparse features, release train data, then evaluate full validation."""
    config = load_yaml(config_path) if config_path else {}
    training = read_split(
        train,
        data_dir,
        "train",
        columns=["text", "canonical_label"],
    )
    source_training_rows = len(training)
    seed = int(config.get("seed", 42))
    training = _stratified_limit(training, max_train_samples, seed)
    class_counts = {
        str(label): int(count)
        for label, count in training["canonical_label"].value_counts().items()
    }
    LOGGER.info(
        "Using %d of %d TF-IDF training rows (stratified seed=%d)",
        len(training),
        source_training_rows,
        seed,
    )
    LOGGER.info("TF-IDF training class counts=%s", class_counts)
    destination = Path(output_dir)
    run_config = {
        **config,
        "seed": seed,
        "max_train_samples": max_train_samples,
        "source_training_rows": source_training_rows,
        "training_rows": len(training),
        "training_class_counts": class_counts,
    }
    model = train_tfidf(training, destination / "model.joblib", run_config)
    del training
    gc.collect()
    validation_data = read_split(validation, data_dir, "validation")
    LOGGER.info("Evaluating all %d validation rows in batches", len(validation_data))
    return evaluate_classifier(model, validation_data, destination / "validation")
