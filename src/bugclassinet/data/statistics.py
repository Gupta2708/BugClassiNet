"""Dataset audit statistics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataset_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-ready, non-modeling dataset statistics."""
    return {
        "rows": len(frame),
        "class_counts": frame["canonical_label"].value_counts().sort_index().to_dict(),
        "repositories": int(frame["repository"].nunique()) if "repository" in frame else None,
        "duplicate_hashes": int(frame["text_hash"].duplicated().sum())
        if "text_hash" in frame
        else None,
    }
