"""Stage 1 DeBERTa configuration helpers."""

from __future__ import annotations

from bugclassinet.models.transformer_classifier import TransformerTrainingConfig


def stage1_config(**kwargs: object) -> TransformerTrainingConfig:
    """Create a Stage 1 config with the required DeBERTa default."""
    return TransformerTrainingConfig(model_name="microsoft/deberta-v3-small", **kwargs)
