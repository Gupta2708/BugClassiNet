"""Stage 3 ARB/NAM dataset construction."""

from __future__ import annotations

import pandas as pd

from bugclassinet.models.transformer_classifier import TransformerTrainingConfig


def prepare_stage3(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only MAN subclasses for a new classification head."""
    required = {"canonical_label", "text", "project"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Stage 3 data missing columns: {sorted(missing)}")
    data = frame.loc[frame["canonical_label"].isin({"ARB", "NAM"})].copy()
    if data.empty:
        raise ValueError("Stage 3 requires at least one ARB or NAM report")
    return data


def stage3_config(**kwargs: object) -> TransformerTrainingConfig:
    """Create a Stage 3 ModernBERT config (with a new output head)."""
    return TransformerTrainingConfig(model_name="answerdotai/ModernBERT-base", **kwargs)
