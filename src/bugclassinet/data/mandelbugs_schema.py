"""Stage-2 schema, identifiers, and leakage-safe evidence construction."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

PROJECTS = ("Linux", "MySQL", "HTTPD", "AXIS")
CLASSES = ("BOH", "NAM", "ARB", "UNK")
LABELS = ("BOH", "MANDELBUG")
STAGE2_MAPPING = {"BOH": "BOH", "NAM": "MANDELBUG", "ARB": "MANDELBUG", "UNK": None}
SUBCLASSES = {"N/A", "NAU", "ARU", "LAG", "ENV", "TIM", "SEQ", "MEM", "STO", "LOG", "NUM", "TOT"}
TEXT_VERSION = "mandelbugs-evidence-v1"


def string_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None or pd.isna(value) else str(value)


def project_name(value: Any) -> str:
    key = re.sub(r"[^a-z]", "", string_value(value).lower())
    aliases = {
        "linux": "Linux",
        "linuxkernel": "Linux",
        "mysql": "MySQL",
        "httpd": "HTTPD",
        "apachehttpd": "HTTPD",
        "axis": "AXIS",
        "apacheaxis": "AXIS",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported project: {value!r}")
    return aliases[key]


def bug_identifier(project: str, value: Any) -> str:
    raw = string_value(value).strip()
    if project == "AXIS":
        raw = raw.removeprefix("AXIS-")
    if not re.fullmatch(r"\d+(?:\.0+)?", raw):
        raise ValueError(f"Missing or invalid bug ID: {value!r}")
    number = int(raw.split(".")[0])
    if number <= 0:
        raise ValueError(f"Bug ID must be positive: {value!r}")
    return f"AXIS-{number}" if project == "AXIS" else str(number)


def issue_key(project: str, bug_id: Any) -> str:
    return f"{project}:{bug_identifier(project, bug_id)}"


def construct_evidence(record: dict[str, Any]) -> tuple[str, str]:
    """Only allowlisted issue content enters evidence; metadata never does."""
    fields = {
        key: string_value(record.get(key)).replace("\r\n", "\n").strip()
        for key in ("title", "initial_description", "environment_text", "comments_text")
    }
    initial = (
        f"[TITLE]\n{fields['title']}\n\n[DESCRIPTION]\n{fields['initial_description']}"
        f"\n\n[ENVIRONMENT]\n{fields['environment_text']}"
    )
    return initial, f"{initial}\n\n[COMMENTS]\n{fields['comments_text']}"


def validate_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "project",
        "subsystem",
        "bug_id",
        "original_class",
        "original_subclass",
        "source_arff",
    }
    if missing := required - set(frame):
        raise ValueError(f"Label table missing columns: {sorted(missing)}")
    result = frame.copy()
    result["project"] = result["project"].map(project_name)
    result["bug_id"] = [
        bug_identifier(p, b) for p, b in zip(result.project, result.bug_id, strict=True)
    ]
    if not set(result.original_class).issubset(CLASSES):
        raise ValueError("Invalid original_class in label table")
    if not set(result.original_subclass.dropna()).issubset(SUBCLASSES):
        raise ValueError("Invalid original_subclass in label table")
    result["issue_key"] = [
        issue_key(p, b) for p, b in zip(result.project, result.bug_id, strict=True)
    ]
    result["stage2_label"] = result.original_class.map(STAGE2_MAPPING)
    result["stage3_label"] = result.original_class.where(result.original_class.isin(["NAM", "ARB"]))
    return result
