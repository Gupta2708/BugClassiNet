"""Stage 3 ARB/NAM dataset construction."""

from __future__ import annotations

import pandas as pd

from bugclassinet.models.transformer_classifier import TransformerTrainingConfig

STAGE3_LABELS = ("ARB", "NAM")
_EVIDENCE_TEXT_COLUMNS = {"initial": "text_initial", "full": "text_full"}


def _stage3_label_column(frame: pd.DataFrame) -> str:
    """Prefer the research Stage-3 label, falling back to the canonical schema."""
    for column in ("stage3_label", "canonical_label"):
        if column in frame.columns:
            return column
    raise ValueError(
        "Stage 3 data needs a stage3_label or canonical_label column; "
        f"found {sorted(frame.columns)}"
    )


def _stage3_text_column(frame: pd.DataFrame, evidence_mode: str) -> str:
    """Resolve the evidence text column, accepting already-canonical inputs."""
    if evidence_mode not in _EVIDENCE_TEXT_COLUMNS:
        raise ValueError(
            f"Invalid evidence_mode={evidence_mode!r}; "
            f"expected one of {sorted(_EVIDENCE_TEXT_COLUMNS)}"
        )
    preferred = _EVIDENCE_TEXT_COLUMNS[evidence_mode]
    if preferred in frame.columns:
        return preferred
    if "text" in frame.columns:
        return "text"
    raise ValueError(
        f"Stage 3 evidence_mode={evidence_mode!r} needs a {preferred} or text column; "
        f"found {sorted(frame.columns)}"
    )


def prepare_stage3(frame: pd.DataFrame, evidence_mode: str = "initial") -> pd.DataFrame:
    """Keep only ARB/NAM reports and project them onto the canonical schema.

    Accepts either the canonical ``canonical_label``/``text`` schema or the
    research Stage-3 schema (``stage3_label`` plus ``text_initial``/``text_full``).
    The input frame is never mutated, and classes outside ARB/NAM are dropped
    rather than remapped.
    """
    if "project" not in frame.columns:
        raise ValueError("Stage 3 data missing columns: ['project']")
    label_column = _stage3_label_column(frame)
    text_column = _stage3_text_column(frame, evidence_mode)

    data = frame.copy()
    data["canonical_label"] = data[label_column].astype("string")
    data["text"] = data[text_column].astype("string")
    data = data.loc[data["canonical_label"].isin(STAGE3_LABELS)].copy()
    if data.empty:
        raise ValueError("Stage 3 requires at least one ARB or NAM report")
    if data["text"].isna().any() or data["text"].str.strip().eq("").any():
        raise ValueError(f"Stage 3 evidence column {text_column!r} contains blank text")
    data["canonical_label"] = data["canonical_label"].astype(str)
    data["text"] = data["text"].astype(str)
    return data


def stage3_config(**kwargs: object) -> TransformerTrainingConfig:
    """Create a Stage 3 ModernBERT config (with a new output head)."""
    return TransformerTrainingConfig(model_name="answerdotai/ModernBERT-base", **kwargs)
