"""Leakage-conscious validation split creation."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_validation_split(
    frame: pd.DataFrame, validation_fraction: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split official training data by repository where practical, otherwise stratified."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if "canonical_label" not in frame or "issue_id" not in frame:
        raise ValueError("Split requires canonical_label and issue_id columns")
    if frame["issue_id"].duplicated().any():
        raise ValueError("Input contains duplicate issue_id values")
    groups = frame.get("repository")
    if groups is not None and groups.notna().all() and groups.nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(frame, groups=groups))
        strategy = "repository_grouped"
    else:
        counts = frame["canonical_label"].value_counts()
        stratify = frame["canonical_label"] if (counts >= 2).all() else None
        train_idx, val_idx = train_test_split(
            range(len(frame)), test_size=validation_fraction, random_state=seed, stratify=stratify
        )
        strategy = "stratified_no_repository_metadata"
    train, validation = frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()
    overlap = set(train["issue_id"]) & set(validation["issue_id"])
    if overlap:
        raise RuntimeError(
            f"Split integrity failure: IDs appear in both splits: {sorted(overlap)[:5]}"
        )
    return train.reset_index(drop=True), validation.reset_index(drop=True), strategy
