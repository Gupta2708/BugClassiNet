import pandas as pd

from bugclassinet.evaluation.evaluate import predict_in_batches


class RecordingModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, texts: pd.Series) -> list[str]:
        self.batch_sizes.append(len(texts))
        return ["BUG"] * len(texts)


def test_prediction_is_batched() -> None:
    model = RecordingModel()
    predictions = predict_in_batches(model, pd.Series(["a"] * 2_501), batch_size=1_000)
    assert model.batch_sizes == [1_000, 1_000, 501]
    assert len(predictions) == 2_501
