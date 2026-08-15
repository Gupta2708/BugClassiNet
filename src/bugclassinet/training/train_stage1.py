"""Stage 1 transformer training entry point."""

from __future__ import annotations

import pandas as pd

from bugclassinet.models.transformer_classifier import TransformerTrainingConfig, train_transformer
from bugclassinet.training.common import read_split, training_config


def _limit(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    """Take a deterministic bounded subset for an explicit smoke run."""
    if maximum is None or len(frame) <= maximum:
        return frame
    if maximum <= 0:
        raise ValueError("Sample limits must be positive")
    return frame.sample(n=maximum, random_state=seed).reset_index(drop=True)


def run(
    data_dir: str | None,
    train: str | None,
    validation: str | None,
    output_dir: str,
    config_path: str | None,
    checkpoint: str | None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
    """Train DeBERTa Stage 1 with a configurable, resumable implementation."""
    config = training_config(config_path, {"model_name": "microsoft/deberta-v3-small"})
    allowed = set(TransformerTrainingConfig.__dataclass_fields__)
    values = {key: value for key, value in config.items() if key in allowed}
    if max_steps is not None:
        values["max_steps"] = max_steps
    model_config = TransformerTrainingConfig(**values)
    return train_transformer(
        _limit(read_split(train, data_dir, "train"), max_train_samples, model_config.seed),
        _limit(read_split(validation, data_dir, "validation"), max_eval_samples, model_config.seed),
        output_dir,
        model_config,
        checkpoint,
    )
