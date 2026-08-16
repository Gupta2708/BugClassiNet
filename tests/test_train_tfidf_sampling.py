from types import SimpleNamespace

import pandas as pd
import pytest

from bugclassinet.training import commands
from bugclassinet.training import train_tfidf as training_module
from bugclassinet.training.train_tfidf import _stratified_limit


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"report {index}" for index in range(100)],
            "canonical_label": ["BUG"] * 80 + ["QUESTION"] * 20,
        }
    )


def test_tfidf_limit_is_exact_deterministic_and_stratified() -> None:
    first = _stratified_limit(_training_frame(), 50, seed=42)
    second = _stratified_limit(_training_frame(), 50, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 50
    assert first["canonical_label"].value_counts().to_dict() == {"BUG": 40, "QUESTION": 10}


def test_tfidf_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        _stratified_limit(_training_frame(), 0, seed=42)


def test_tfidf_command_preserves_full_validation(monkeypatch) -> None:
    args = SimpleNamespace(
        data_dir=None,
        train="train.parquet",
        validation="validation.parquet",
        output_dir="outputs/tfidf",
        config="tfidf.yaml",
        max_train_samples=200_000,
        max_eval_samples=1,
    )

    monkeypatch.setattr(commands.tfidf, "run", lambda *args: {})
    with pytest.raises(ValueError, match="complete validation"):
        commands.train_tfidf(args)


def test_tfidf_command_forwards_training_limit(monkeypatch) -> None:
    captured = {}
    args = SimpleNamespace(
        data_dir=None,
        train="train.parquet",
        validation="validation.parquet",
        output_dir="outputs/tfidf",
        config="tfidf.yaml",
        max_train_samples=200_000,
        max_eval_samples=None,
    )

    def fake_run(*values):
        captured["values"] = values
        return {}

    monkeypatch.setattr(commands.tfidf, "run", fake_run)
    commands.train_tfidf(args)
    assert captured["values"][-1] == 200_000


def test_tfidf_run_limits_only_training(monkeypatch, tmp_path) -> None:
    train = _training_frame()
    validation = pd.DataFrame(
        {
            "text": [f"validation {index}" for index in range(7)],
            "canonical_label": ["BUG"] * 5 + ["QUESTION"] * 2,
        }
    )
    observed = {}

    def fake_read_split(path, data_dir, split, columns=None):
        return train.copy() if split == "train" else validation.copy()

    def fake_train_tfidf(frame, output_path, config):
        observed["training_rows"] = len(frame)
        return object()

    def fake_evaluate(model, frame, output_path):
        observed["validation_rows"] = len(frame)
        return {"rows": len(frame)}

    monkeypatch.setattr(training_module, "read_split", fake_read_split)
    monkeypatch.setattr(training_module, "train_tfidf", fake_train_tfidf)
    monkeypatch.setattr(training_module, "evaluate_classifier", fake_evaluate)

    result = training_module.run(
        None,
        "train.parquet",
        "validation.parquet",
        str(tmp_path),
        max_train_samples=50,
    )

    assert observed == {"training_rows": 50, "validation_rows": 7}
    assert result == {"rows": 7}
