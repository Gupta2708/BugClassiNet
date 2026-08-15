"""Stage 2 BOH/MAN label construction."""

from __future__ import annotations

import pandas as pd

from bugclassinet.models.transformer_classifier import TransformerTrainingConfig


def prepare_stage2(frame: pd.DataFrame) -> pd.DataFrame:
    """Map NAM and ARB to MAN, retaining BOH reports only."""
    required = {"canonical_label", "text", "project"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Stage 2 data missing columns: {sorted(missing)}")
    data = frame.loc[frame["canonical_label"].isin({"BOH", "MAN", "NAM", "ARB"})].copy()
    data["canonical_label"] = data["canonical_label"].replace({"NAM": "MAN", "ARB": "MAN"})
    if set(data["canonical_label"].unique()) - {"BOH", "MAN"}:
        raise ValueError("Stage 2 labels must be BOH or MAN")
    return data


def stage2_config(**kwargs: object) -> TransformerTrainingConfig:
    """Create a Stage 2 ModernBERT config."""
    return TransformerTrainingConfig(model_name="answerdotai/ModernBERT-base", **kwargs)
