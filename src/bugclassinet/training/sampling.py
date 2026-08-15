"""Group-aware cross-validation split utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold


def group_cross_validation_splits(
    frame: pd.DataFrame, folds: int = 5
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return project-disjoint folds for Mandelbugs experiments."""
    if "project" not in frame:
        raise ValueError("Group-aware cross-validation requires a project column")
    projects = frame["project"].dropna().unique()
    if len(projects) < folds:
        raise ValueError(
            "Need at least "
            f"{folds} projects for group-aware cross-validation; found {len(projects)}"
        )
    splitter = GroupKFold(n_splits=folds)
    return [
        (frame.iloc[a].copy(), frame.iloc[b].copy())
        for a, b in splitter.split(frame, groups=frame["project"])
    ]
