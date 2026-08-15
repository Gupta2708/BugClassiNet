"""Issue-text construction and canonical label mapping."""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from bugclassinet.constants import STAGE1_LABELS
from bugclassinet.data.schema import IssueSchema


def clean_text(value: object) -> str:
    """Normalize missing values and whitespace without altering content semantics."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def canonicalize_stage1_label(value: object) -> str | None:
    """Map known NLBSE-style values to canonical Stage 1 labels."""
    label = clean_text(value).lower()
    if "bug" == label or label.startswith("bug,") or " bug" in label:
        return "BUG"
    if "feature" in label or "enhancement" in label:
        return "ENHANCEMENT"
    if "question" in label:
        return "QUESTION"
    if "documentation" in label or label == "docs" or "doc" in label:
        return "DOCUMENTATION"
    return None


def normalised_text_hash(text: str) -> str:
    """Hash case-folded whitespace-normalized input text for exact duplicate detection."""
    normalised = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def harmonize_issues(frame: pd.DataFrame, schema: IssueSchema) -> pd.DataFrame:
    """Preserve source data and append canonical fields required by the pipeline."""
    data = frame.copy()
    title = data[schema.title].map(clean_text) if schema.title else pd.Series("", index=data.index)
    body = data[schema.body].map(clean_text) if schema.body else pd.Series("", index=data.index)
    usable = (title != "") | (body != "")
    data = data.loc[usable].copy()
    title, body = title.loc[usable], body.loc[usable]
    data["title"] = title
    data["body"] = body
    data["text"] = "[TITLE]\n" + title + "\n\n[DESCRIPTION]\n" + body
    data["original_label"] = data[schema.original_label].map(clean_text)
    data["canonical_label"] = data["original_label"].map(canonicalize_stage1_label)
    unknown = sorted(data.loc[data["canonical_label"].isna(), "original_label"].unique())
    if unknown:
        raise ValueError(f"Unmapped Stage 1 labels found: {unknown}")
    data["binary_label"] = data["canonical_label"].eq("BUG").map({True: "BUG", False: "NON_BUG"})
    data["text_hash"] = data["text"].map(normalised_text_hash)
    data["issue_id"] = (
        data[schema.issue_id].astype(str) if schema.issue_id else data.index.astype(str)
    )
    if schema.repository:
        data["repository"] = data[schema.repository].map(clean_text)
    if not set(data["canonical_label"].unique()).issubset(STAGE1_LABELS):
        raise ValueError("Canonical labels failed validation")
    return data.reset_index(drop=True)
