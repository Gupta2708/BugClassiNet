from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bugclassinet.models import transformer_classifier as module
from bugclassinet.models.transformer_classifier import TransformerTrainingConfig, train_transformer


@pytest.mark.parametrize("cuda_available", [False, True])
def test_stage1_trainer_uses_safe_model_dtype(tmp_path, monkeypatch, cuda_available: bool) -> None:
    class Dataset:
        @staticmethod
        def from_dict(values):
            return values

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] for _ in texts], "attention_mask": [[1] for _ in texts]}

        def save_pretrained(self, output):
            return None

    class Model:
        def __init__(self) -> None:
            self.float_calls = 0

        def float(self):
            self.float_calls += 1
            return self

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, **kwargs):
            return None

        def evaluate(self):
            return {"eval_macro_f1": 1.0}

        def save_model(self, output):
            return None

        def predict(self, dataset):
            return SimpleNamespace(predictions=np.array([[1.0, 0.0], [0.0, 1.0]]))

    created = {}

    def make_model(*args, **kwargs):
        created["model"] = Model()
        return created["model"]

    def make_arguments(**kwargs):
        created["arguments"] = kwargs
        return object()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        tensor=lambda *args, **kwargs: np.array(args[0]),
        float=float,
    )
    fake_auto_model = SimpleNamespace(from_pretrained=make_model)
    fake_auto_tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(
        module,
        "_dependencies",
        lambda: (
            fake_torch,
            Dataset,
            fake_auto_model,
            fake_auto_tokenizer,
            lambda **kwargs: object(),
            Trainer,
            make_arguments,
        ),
    )
    frame = pd.DataFrame({"text": ["bug", "docs"], "canonical_label": ["BUG", "DOCUMENTATION"]})
    metrics = train_transformer(
        frame, frame, tmp_path, TransformerTrainingConfig(model_name="tiny", epochs=1)
    )
    assert metrics["eval_macro_f1"] == 1.0
    assert created["model"].float_calls == int(cuda_available)
    assert created["arguments"]["fp16"] is cuda_available
    assert (tmp_path / "validation_predictions.parquet").is_file()
