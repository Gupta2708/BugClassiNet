"""Deterministic text-hash deduplication."""

from __future__ import annotations

import pandas as pd


def deduplicate_train(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the first instance of each exact normalized text hash in train data."""
    if "text_hash" not in frame:
        raise ValueError("Cannot deduplicate without text_hash")
    result = frame.drop_duplicates(subset="text_hash", keep="first").copy()
    return result.reset_index(drop=True), len(frame) - len(result)


def duplicate_overlap(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return official-test rows whose normalized text appears in the left frame."""
    if "text_hash" not in left or "text_hash" not in right:
        raise ValueError("Cannot compare duplicates without text_hash")
    return right.loc[right["text_hash"].isin(set(left["text_hash"]))].copy()


def exclude_test_overlap(
    train: pd.DataFrame, official_test: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Remove train rows that exactly match an official-test normalized text hash.

    The official test frame is read-only: this function filters only a copied
    training frame and raises if the postcondition is not met.
    """
    overlap = duplicate_overlap(train, official_test)
    clean = train.loc[~train["text_hash"].isin(set(official_test["text_hash"]))].copy()
    if not duplicate_overlap(clean, official_test).empty:
        raise RuntimeError(
            "Test-overlap cleaning failed: clean training still overlaps official test"
        )
    return clean.reset_index(drop=True), len(overlap)


def ensure_unique_issue_ids(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Create stable internal IDs when a dataset's source issue IDs are only locally unique.

    The original source ID remains available in ``source_issue_id``.  This is applied
    after text deduplication, so the source ID plus normalized text hash identifies a
    retained record without dropping semantically distinct reports.
    """
    required = {"issue_id", "text_hash"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Cannot ensure unique issue IDs; missing columns: {sorted(missing)}")
    if not frame["issue_id"].duplicated().any():
        return frame, 0
    data = frame.copy()
    data["source_issue_id"] = data["issue_id"].astype(str)
    data["issue_id"] = data["source_issue_id"] + ":" + data["text_hash"].str[:16]
    if data["issue_id"].duplicated().any():
        raise RuntimeError("Could not create unique internal issue IDs after deduplication")
    return data, int(data["source_issue_id"].duplicated(keep=False).sum())
