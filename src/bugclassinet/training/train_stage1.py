"""Stage 1 transformer training entry point."""

from __future__ import annotations

from bugclassinet.models.transformer_classifier import TransformerTrainingConfig, train_transformer
from bugclassinet.training.common import resolve_split_path, training_config


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
    if max_eval_samples is not None:
        raise ValueError("Stage-1 evaluation always uses the complete validation split")
    config = training_config(config_path, {"model_name": "microsoft/deberta-v3-small"})
    allowed = set(TransformerTrainingConfig.__dataclass_fields__)
    values = {key: value for key, value in config.items() if key in allowed}
    if max_steps is not None:
        values["max_steps"] = max_steps
    model_config = TransformerTrainingConfig(**values)
    return train_transformer(
        resolve_split_path(train, data_dir, "train"),
        resolve_split_path(validation, data_dir, "validation"),
        output_dir,
        model_config,
        checkpoint,
        max_train_samples=max_train_samples,
    )
