"""CSV schema inference for imperfect issue datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass


class SchemaInferenceError(ValueError):
    """Raised when required semantic columns cannot be mapped safely."""


@dataclass(frozen=True)
class IssueSchema:
    """Detected source column mapping."""

    issue_id: str | None
    title: str | None
    body: str | None
    original_label: str
    repository: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


_CANDIDATES = {
    "issue_id": ("issue_id", "id", "issue number", "issue_number", "number"),
    "title": ("title", "issue_title", "summary", "subject"),
    "body": ("body", "description", "issue_body", "content", "text"),
    "original_label": ("label", "labels", "class", "classification", "category", "type"),
    "repository": ("repository", "repo", "project", "project_name", "full_name"),
}


def _normalise(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def _match(columns: Iterable[str], key: str) -> str | None:
    indexed = {_normalise(column): column for column in columns}
    for candidate in _CANDIDATES[key]:
        if candidate in indexed:
            return indexed[candidate]
    return None


def infer_issue_schema(columns: Iterable[str]) -> IssueSchema:
    """Infer semantic issue columns and fail if text/label requirements are unmet."""
    available = list(columns)
    detected = {field: _match(available, field) for field in _CANDIDATES}
    if not detected["original_label"] or not (detected["title"] or detected["body"]):
        raise SchemaInferenceError(
            "Could not infer required issue schema. Need a label and title or body column. "
            f"Available columns: {available}. Detected: {detected}."
        )
    return IssueSchema(**detected)
