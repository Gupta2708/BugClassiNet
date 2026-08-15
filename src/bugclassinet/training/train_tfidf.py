"""TF-IDF training package entry point."""

from __future__ import annotations

from pathlib import Path

from bugclassinet.evaluation.evaluate import evaluate_classifier
from bugclassinet.models.tfidf_classifier import train_tfidf
from bugclassinet.training.common import read_split


def run(
    data_dir: str | None, train: str | None, validation: str | None, output_dir: str
) -> dict[str, object]:
    """Train baseline then evaluate on validation without touching official test data."""
    training = read_split(train, data_dir, "train")
    validation_data = read_split(validation, data_dir, "validation")
    destination = Path(output_dir)
    model = train_tfidf(training, destination / "model.joblib")
    return evaluate_classifier(model, validation_data, destination / "validation")
