from pathlib import Path

import pandas as pd
import pytest

from bugclassinet.training import train_stage1 as module


def test_stage1_forwards_paths_and_limits_without_loading_pandas(tmp_path, monkeypatch) -> None:
    train = tmp_path / "train_clean.parquet"
    validation = tmp_path / "validation_clean.parquet"
    train.write_bytes(b"fixture")
    validation.write_bytes(b"fixture")
    captured = {}

    def fail_pandas_read(*args, **kwargs):
        raise AssertionError("Stage 1 must not materialize Parquet through pandas")

    def fake_train_transformer(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(pd, "read_parquet", fail_pandas_read)
    monkeypatch.setattr(module, "train_transformer", fake_train_transformer)

    result = module.run(
        None,
        str(train),
        str(validation),
        str(tmp_path / "model"),
        None,
        None,
        max_train_samples=500_000,
        max_eval_samples=None,
    )

    assert result == {"ok": True}
    assert captured["args"][0:2] == (Path(train), Path(validation))
    assert captured["kwargs"] == {"max_train_samples": 500_000}


def test_stage1_rejects_reduced_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="complete validation"):
        module.run(
            None,
            str(tmp_path / "train.parquet"),
            str(tmp_path / "validation.parquet"),
            str(tmp_path / "model"),
            None,
            None,
            max_eval_samples=100,
        )
