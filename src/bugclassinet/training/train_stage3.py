"""Stage 3 ModernBERT training entry point."""

from __future__ import annotations

from bugclassinet.models.stage3 import prepare_stage3
from bugclassinet.models.transformer_classifier import TransformerTrainingConfig, train_transformer
from bugclassinet.training.common import read_split, training_config
from bugclassinet.training.sampling import group_cross_validation_splits


def run(
    data_dir: str | None,
    train: str | None,
    validation: str | None,
    output_dir: str,
    config_path: str | None,
    checkpoint: str | None,
) -> dict[str, object]:
    """Train ARB/NAM from a Stage 2 checkpoint with a fresh classification head."""
    training, valid = (
        prepare_stage3(read_split(train, data_dir, "train")),
        prepare_stage3(read_split(validation, data_dir, "validation")),
    )
    group_cross_validation_splits(training, min(5, training["project"].nunique()))
    values = training_config(
        config_path, {"model_name": checkpoint or "answerdotai/ModernBERT-base"}
    )
    allowed = set(TransformerTrainingConfig.__dataclass_fields__)
    return train_transformer(
        training,
        valid,
        output_dir,
        TransformerTrainingConfig(
            **{key: value for key, value in values.items() if key in allowed}
        ),
    )
