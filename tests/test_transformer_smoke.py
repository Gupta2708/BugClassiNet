from types import SimpleNamespace

import numpy as np
import pandas as pd

from bugclassinet.models import transformer_classifier as module
from bugclassinet.models.transformer_classifier import TransformerTrainingConfig, train_transformer


def test_stage1_trainer_runs_one_mocked_cpu_step(tmp_path, monkeypatch) -> None:
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
        pass

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

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        tensor=lambda *args, **kwargs: np.array(args[0]),
        float=float,
    )
    fake_auto_model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model())
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
            lambda **kwargs: object(),
        ),
    )
    frame = pd.DataFrame({"text": ["bug", "docs"], "canonical_label": ["BUG", "DOCUMENTATION"]})
    metrics = train_transformer(
        frame, frame, tmp_path, TransformerTrainingConfig(model_name="tiny", epochs=1)
    )
    assert metrics["eval_macro_f1"] == 1.0
    assert (tmp_path / "validation_predictions.parquet").is_file()
