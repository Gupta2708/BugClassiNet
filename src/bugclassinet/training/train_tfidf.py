"""TF-IDF training package entry point."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from bugclassinet.evaluation.evaluate import evaluate_classifier
from bugclassinet.models.tfidf_classifier import train_tfidf
from bugclassinet.settings import load_yaml
from bugclassinet.training.common import read_split

LOGGER = logging.getLogger(__name__)


def run(
    data_dir: str | None,
    train: str | None,
    validation: str | None,
    output_dir: str,
    config_path: str | None = None,
) -> dict[str, object]:
    """Train on sparse features, release train data, then evaluate full validation."""
    config = load_yaml(config_path) if config_path else {}
    training = read_split(
        train,
        data_dir,
        "train",
        columns=["text", "canonical_label"],
    )
    LOGGER.info("Loaded %d TF-IDF training rows", len(training))
    destination = Path(output_dir)
    model = train_tfidf(training, destination / "model.joblib", config)
    del training
    gc.collect()
    validation_data = read_split(validation, data_dir, "validation")
    LOGGER.info("Evaluating all %d validation rows in batches", len(validation_data))
    return evaluate_classifier(model, validation_data, destination / "validation")
