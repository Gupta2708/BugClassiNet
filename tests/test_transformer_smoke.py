from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bugclassinet.models import transformer_classifier as module
from bugclassinet.models.transformer_classifier import TransformerTrainingConfig, train_transformer


class FakeDataset:
    """Small HF-like dataset that exercises disk-backed tokenization orchestration."""

    map_calls: list[dict[str, object]] = []

    def __init__(self, rows: list[dict[str, object]], fingerprint: str = "fixture") -> None:
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []
        self._fingerprint = fingerprint

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def from_pandas(cls, frame, preserve_index):
        assert preserve_index is False
        return cls(frame.to_dict(orient="records"))

    def map(self, function, **kwargs):
        self.map_calls.append(kwargs)
        batch_size = int(kwargs["batch_size"])
        mapped_rows: list[dict[str, object]] = []
        for start in range(0, len(self), batch_size):
            rows = self.rows[start : start + batch_size]
            batch = {column: [row[column] for row in rows] for column in self.column_names}
            encoded = function(batch, **kwargs["fn_kwargs"])
            mapped_rows.extend(
                {column: values[index] for column, values in encoded.items()}
                for index in range(len(rows))
            )
        return FakeDataset(mapped_rows, fingerprint=str(kwargs["new_fingerprint"]))

    def select_columns(self, columns):
        return FakeDataset(
            [{column: row[column] for column in columns} for row in self.rows],
            fingerprint=self._fingerprint,
        )

    def add_column(self, name, values):
        rows = [dict(row) for row in self.rows]
        for row, value in zip(rows, values, strict=True):
            row[name] = value
        return FakeDataset(rows, fingerprint=self._fingerprint)

    def to_parquet(self, path):
        import pandas as pd

        pd.DataFrame(self.rows).to_parquet(path, index=False)


def test_dataframe_adapter_preserves_validation_metadata(monkeypatch) -> None:
    monkeypatch.setattr(module, "_dependencies", lambda: (None, FakeDataset))
    frame = pd.DataFrame(
        {
            "issue_id": ["v1"],
            "project": ["alpha"],
            "text": ["bug"],
            "canonical_label": ["BUG"],
        }
    )

    training = module._dataframe_dataset(frame, retain_metadata=False)
    validation = module._dataframe_dataset(frame, retain_metadata=True)

    assert training.column_names == ["text", "canonical_label"]
    assert validation.column_names == frame.columns.tolist()


@pytest.mark.parametrize("cuda_available", [False, True])
def test_stage1_trainer_uses_memory_safe_tokenization(
    tmp_path, monkeypatch, cuda_available: bool
) -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, texts, **kwargs):
            self.calls.append(kwargs)
            input_ids = [[101, len(text), 102] for text in texts]
            return {
                "input_ids": input_ids,
                "attention_mask": [[1] * len(values) for values in input_ids],
            }

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
            self.train_dataset = kwargs["train_dataset"]
            created["eval_dataset"] = kwargs["eval_dataset"]

        def train(self, **kwargs):
            return None

        def evaluate(self):
            return {"eval_macro_f1": 1.0}

        def save_model(self, output):
            return None

        def predict(self, dataset):
            created["predict_dataset"] = dataset
            return SimpleNamespace(predictions=np.array([[1.0, 0.0], [0.0, 1.0]]))

    created: dict[str, object] = {}

    def make_model(*args, **kwargs):
        created["model"] = Model()
        return created["model"]

    def make_arguments(**kwargs):
        created["arguments"] = kwargs
        return SimpleNamespace(n_gpu=1, world_size=1, parallel_mode="not_distributed")

    def make_collator(**kwargs):
        created["collator"] = kwargs
        return object()

    tokenizer = Tokenizer()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        tensor=lambda *args, **kwargs: np.array(args[0]),
        float=float,
    )
    fake_auto_model = SimpleNamespace(from_pretrained=make_model)
    fake_auto_tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        module,
        "_dependencies",
        lambda: (
            fake_torch,
            FakeDataset,
            fake_auto_model,
            fake_auto_tokenizer,
            make_collator,
            lambda **kwargs: object(),
            Trainer,
            make_arguments,
        ),
    )

    def inspect_dataset(dataset, split_name, required_columns):
        assert required_columns <= set(dataset.column_names)
        counts: dict[str, int] = {}
        for row in dataset.rows:
            label = str(row["canonical_label"])
            counts[label] = counts.get(label, 0) + 1
        return counts

    monkeypatch.setattr(module, "inspect_dataset", inspect_dataset)
    FakeDataset.map_calls.clear()
    train = FakeDataset(
        [
            {"text": "bug", "canonical_label": "BUG"},
            {"text": "docs", "canonical_label": "DOCUMENTATION"},
        ],
        fingerprint="train",
    )
    validation = FakeDataset(
        [
            {
                "issue_id": "v1",
                "project": "alpha",
                "text": "bug",
                "canonical_label": "BUG",
            },
            {
                "issue_id": "v2",
                "project": "beta",
                "text": "docs",
                "canonical_label": "DOCUMENTATION",
            },
        ],
        fingerprint="validation",
    )

    metrics = train_transformer(
        train,
        validation,
        tmp_path,
        TransformerTrainingConfig(model_name="tiny", epochs=1),
    )

    assert metrics["eval_macro_f1"] == 1.0
    assert created["model"].float_calls == int(cuda_available)
    assert created["arguments"]["fp16"] is cuda_available
    assert created["arguments"]["dataloader_num_workers"] == 0
    assert created["arguments"]["dataloader_persistent_workers"] is False
    assert created["collator"] == {
        "tokenizer": tokenizer,
        "padding": True,
        "return_tensors": "pt",
    }
    assert created["predict_dataset"] is created["eval_dataset"]
    assert len(FakeDataset.map_calls) == 2
    assert all(call["batched"] is True for call in FakeDataset.map_calls)
    assert all(call["batch_size"] == 512 for call in FakeDataset.map_calls)
    assert all(call["writer_batch_size"] == 512 for call in FakeDataset.map_calls)
    assert all(call["keep_in_memory"] is False for call in FakeDataset.map_calls)
    assert all(call["remove_columns"] for call in FakeDataset.map_calls)
    assert tokenizer.calls
    assert all(
        call == {"truncation": True, "max_length": 256, "padding": False}
        for call in tokenizer.calls
    )
    assert (tmp_path / "validation_predictions.parquet").is_file()
    predictions = pd.read_parquet(tmp_path / "validation_predictions.parquet")
    assert predictions.columns.tolist() == [
        "issue_id",
        "project",
        "canonical_label",
        "prediction",
    ]
