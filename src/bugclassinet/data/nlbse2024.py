"""Strict loader and audit for the external NLBSE 2024 issue dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.data.harmonize import (
    clean_text,
    construct_issue_text,
    normalised_text_hash,
)
from bugclassinet.data.schema import infer_issue_schema

NLBSE2024_LABEL_MAPPING = {
    "bug": "BUG",
    "feature": "ENHANCEMENT",
    "question": "QUESTION",
}


def _created_at_column(columns: list[str]) -> str | None:
    normalized = {column.lower().strip().replace("-", "_"): column for column in columns}
    return normalized.get("created_at")


def load_nlbse2024_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load NLBSE 2024 without dropping, sampling, or rebalancing any row."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NLBSE 2024 CSV does not exist: {source}")
    raw = pd.read_csv(source)
    schema = infer_issue_schema(raw.columns)
    missing_semantics = [
        name
        for name, value in {
            "repo": schema.repository,
            "label": schema.original_label,
            "title": schema.title,
            "body": schema.body,
        }.items()
        if value is None
    ]
    if missing_semantics:
        raise ValueError(
            "NLBSE 2024 requires repo, label, title, and body columns; "
            f"missing semantic columns: {missing_semantics}; available={list(raw.columns)}"
        )

    created_at = _created_at_column(list(raw.columns))
    missing_title = int(raw[schema.title].isna().sum())
    missing_body = int(raw[schema.body].isna().sum())
    result = pd.DataFrame(
        {
            "repo": raw[schema.repository].map(clean_text),
            "original_label": raw[schema.original_label].map(clean_text),
            "title": raw[schema.title].map(clean_text),
            "body": raw[schema.body].map(clean_text),
        }
    )
    if created_at is not None:
        result.insert(1, "created_at", raw[created_at])
    result["canonical_label"] = result["original_label"].str.casefold().map(NLBSE2024_LABEL_MAPPING)
    unexpected = sorted(result.loc[result["canonical_label"].isna(), "original_label"].unique())
    if unexpected:
        raise ValueError(f"Unexpected NLBSE 2024 labels: {unexpected}")
    result["text"] = construct_issue_text(result["title"], result["body"])
    result["text_hash"] = result["text"].map(normalised_text_hash)
    result["source_row_id"] = [str(index) for index in range(len(result))]

    label_repo = pd.crosstab(result["canonical_label"], result["repo"], dropna=False)
    audit = {
        "rows": len(result),
        "source_columns": list(raw.columns),
        "normalized_columns": list(result.columns),
        "missing_title_count": missing_title,
        "missing_body_count": missing_body,
        "duplicate_exact_text_count": int(result["text_hash"].duplicated().sum()),
        "label_counts": result["canonical_label"].value_counts().sort_index().to_dict(),
        "repository_counts": result["repo"].value_counts().sort_index().to_dict(),
        "label_by_repository": {
            label: {repo: int(count) for repo, count in row.items()}
            for label, row in label_repo.iterrows()
        },
    }
    return result, audit
