"""Mandelbugs dataset guards used by Stage 2 and 3."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.data.mandelbugs_schema import (
    CLASSES,
    STAGE2_MAPPING,
    SUBCLASSES,
    bug_identifier,
    string_value,
)
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json
from bugclassinet.utils.reproducibility import git_commit, utc_timestamp

LABEL_COLUMNS = [
    "project",
    "subsystem",
    "bug_id",
    "original_class",
    "original_subclass",
    "source_arff",
    "source_row",
    "source_line",
    "source_dataset",
    "source_doi",
    "stage2_label",
    "stage3_label",
]
FAILURE_COLUMNS = ["source_arff", "source_row", "source_line", "raw_row", "error"]


def infer_project_subsystem(path: str | Path) -> tuple[str, str]:
    stem = Path(path).stem.lower()
    for prefix, project in (
        ("apache_httpd_", "HTTPD"),
        ("linux_", "Linux"),
        ("mysql_", "MySQL"),
        ("axis_", "AXIS"),
        ("apache_axis_", "AXIS"),
    ):
        if stem.startswith(prefix) and stem[len(prefix) :]:
            return project, stem[len(prefix) :]
    raise ValueError(f"Cannot infer project/subsystem from ARFF filename: {path}")


def _fields(line: str) -> list[str]:
    # Official files are dense ARFF; preserve nominal values, remove syntax whitespace.
    quote = "'" if line.lstrip().startswith("'") else '"'
    return [
        value.strip().strip("'\"")
        for value in next(csv.reader([line], skipinitialspace=True, quotechar=quote, strict=True))
    ]


def parse_arff(path: str | Path, source_name: str | None = None) -> tuple[list, list, list]:
    """Parse the dataset's dense ARFF dialect and return rows, failures, warnings.

    BugID is deliberately treated as an identifier: the official AXIS ARFF has
    an incorrect numeric declaration. That documented anomaly is reported.
    Sparse/multiline ARFF is rejected explicitly, never silently skipped.
    """
    source = Path(path)
    name = source_name or source.name
    rows, failures, warnings = [], [], []
    attributes: list[tuple[str, str]] = []
    in_data = False
    ordinal = 0
    project, subsystem = infer_project_subsystem(source)
    for line_number, raw in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            match = re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)", line, re.I)
            if match:
                attributes.append((next(v for v in match.groups()[:3] if v), match.group(4)))
            elif line.lower() == "@data":
                names = [re.sub(r"[^a-z]", "", k.lower()) for k, _ in attributes]
                if len(names) != len(set(names)) or set(names) not in (
                    {"bugid", "bugclass"},
                    {"bugid", "bugclass", "bugsubclass"},
                ):
                    raise ValueError(
                        "Expected unique BugID, BugClass and optional BugSubclass attributes"
                    )
                in_data = True
            elif not line.lower().startswith("@relation"):
                raise ValueError(f"Unsupported ARFF header {name}:{line_number}: {line}")
            continue
        ordinal += 1
        try:
            fields = _fields(line)
            if len(fields) != len(attributes) or len(fields) not in (2, 3):
                raise ValueError("Wrong field count or unsupported sparse ARFF row")
            indexed = {
                re.sub(r"[^a-z]", "", k.lower()): v
                for (k, _), v in zip(attributes, fields, strict=True)
            }
            if set(indexed) - {"bugid", "bugclass", "bugsubclass"} or not {
                "bugid",
                "bugclass",
            } <= set(indexed):
                raise ValueError("Expected BugID, BugClass and optional BugSubclass attributes")
            bug_id = bug_identifier(project, indexed["bugid"])
            label = string_value(indexed["bugclass"])
            subclass = string_value(indexed.get("bugsubclass", "")) or None
            if label not in CLASSES or (subclass is not None and subclass not in SUBCLASSES):
                raise ValueError(f"Illegal class/subclass: {label!r}/{subclass!r}")
            for (attribute, declaration), value in zip(attributes, fields, strict=True):
                if declaration.strip().startswith("{"):
                    allowed = _fields(declaration.strip()[1:-1])
                    if value not in allowed:
                        raise ValueError(
                            f"{attribute} value {value!r} absent from ARFF declaration"
                        )
                if (
                    attribute.lower() == "bugid"
                    and declaration.lower() in {"numeric", "integer", "real"}
                    and project == "AXIS"
                ):
                    warnings.append(
                        {
                            "source_arff": name,
                            "source_row": ordinal,
                            "warning": "AXIS textual identifier in numeric BugID attribute",
                        }
                    )
            rows.append(
                dict(
                    zip(
                        LABEL_COLUMNS,
                        [
                            project,
                            subsystem,
                            bug_id,
                            label,
                            subclass,
                            name,
                            ordinal,
                            line_number,
                            "mandelbugs_oss",
                            "10.5281/zenodo.581660",
                            STAGE2_MAPPING[label],
                            label if label in {"NAM", "ARB"} else None,
                        ],
                        strict=True,
                    )
                )
            )
        except (ValueError, csv.Error) as error:
            failures.append(
                dict(
                    zip(FAILURE_COLUMNS, [name, ordinal, line_number, raw, str(error)], strict=True)
                )
            )
    if not in_data:
        raise ValueError(f"ARFF has no @data section: {name}")
    return rows, failures, warnings


def audit_mandelbugs(raw_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root, out = Path(raw_dir), Path(output_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".arff")
    if not files:
        raise ValueError(f"No ARFF files found beneath {root}")
    out.mkdir(parents=True, exist_ok=True)
    rows, failures, warnings = [], [], []
    for path in files:
        name = path.relative_to(root).as_posix()
        try:
            parsed, bad, notes = parse_arff(path, name)
            rows.extend(parsed)
            failures.extend(bad)
            warnings.extend(notes)
        except (ValueError, UnicodeError) as error:
            failures.append(dict(zip(FAILURE_COLUMNS, [name, 0, 0, "", str(error)], strict=True)))
    labels = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    labels.to_parquet(out / "labels.parquet", index=False)
    duplicates = labels.loc[labels.duplicated(["project", "bug_id"], keep=False)].copy()
    duplicates["across_files"] = (
        duplicates.groupby(["project", "bug_id"]).source_arff.transform("nunique") > 1
    )
    duplicates["conflicting_class"] = (
        duplicates.groupby(["project", "bug_id"]).original_class.transform("nunique") > 1
    )
    duplicates.to_csv(out / "duplicate_ids.csv", index=False)
    pd.DataFrame(failures, columns=FAILURE_COLUMNS).to_csv(out / "parse_failures.csv", index=False)
    for name, keys in {
        "class": ["original_class"],
        "project": ["project"],
        "project_class": ["project", "original_class"],
        "subclass": ["original_class", "original_subclass"],
    }.items():
        labels.groupby(keys, dropna=False).size().rename("rows").reset_index().to_csv(
            out / f"{name}_distribution.csv", index=False
        )
    audit = {
        "arff_files": len(files),
        "total_rows": len(labels) + sum(f["source_row"] > 0 for f in failures),
        "valid_rows": len(labels),
        "unique_issue_ids": len(labels[["project", "bug_id"]].drop_duplicates()),
        "class_counts": labels.original_class.value_counts().to_dict(),
        "by_project": labels.groupby(["project", "original_class"])
        .size()
        .reset_index(name="rows")
        .to_dict("records"),
        "by_subsystem": labels.groupby(["project", "subsystem", "original_class"])
        .size()
        .reset_index(name="rows")
        .to_dict("records"),
        "subclasses": labels.original_subclass.fillna("<missing>").value_counts().to_dict(),
        "duplicate_rows": len(duplicates),
        "duplicate_id_groups": len(duplicates.groupby(["project", "bug_id"])),
        "duplicate_rows_across_files": int(duplicates.across_files.sum()),
        "malformed_rows_or_files": len(failures),
        "missing_or_invalid_ids": sum("bug id" in f["error"].lower() for f in failures),
        "warnings": warnings,
        "schema_valid": not failures,
    }
    write_json(out / "dataset_audit.json", audit)
    write_json(
        out / "source_manifest.json",
        {
            "source_doi": "10.5281/zenodo.581660",
            "source_dataset": "mandelbugs_oss",
            "raw_dir": str(root.resolve()),
            "raw_files": {p.relative_to(root).as_posix(): sha256_file(p) for p in files},
            "timestamp": utc_timestamp(),
            "git_commit": git_commit(),
            "training_invoked": False,
            "parser": "mandelbugs-dense-arff-v1",
        },
    )
    if failures:
        raise ValueError(
            f"ARFF audit found {len(failures)} failures; "
            f"inspect {out / 'parse_failures.csv'} before continuing"
        )
    return audit


def load_mandelbugs(path: str | Path, expected: set[str]) -> pd.DataFrame:
    """Load a Mandelbugs Parquet file and validate text, labels, and project metadata."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required Mandelbugs Parquet is missing: {source}")
    frame = pd.read_parquet(source)
    required = {"text", "canonical_label", "project"}
    if missing := required - set(frame.columns):
        raise ValueError(
            f"Mandelbugs missing columns {sorted(missing)}; available={list(frame.columns)}"
        )
    unexpected = set(frame["canonical_label"].unique()) - expected
    if unexpected:
        raise ValueError(f"Unexpected Mandelbugs labels: {sorted(unexpected)}")
    return frame
