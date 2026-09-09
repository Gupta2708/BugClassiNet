"""Join official labels to retrieved evidence and quarantine ambiguities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.data.harmonize import normalised_text_hash
from bugclassinet.data.mandelbugs_enrich import coverage_readiness
from bugclassinet.data.mandelbugs_schema import TEXT_VERSION, construct_evidence, validate_labels
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json


def prepare_mandelbugs(
    labels_path: str | Path, reports_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    labels = validate_labels(pd.read_parquet(labels_path))
    reports = pd.read_parquet(reports_path)
    required = {
        "issue_key",
        "retrieval_status",
        "retrieval_source",
        "tracker_url",
        "title",
        "initial_description",
        "environment_text",
        "comments_text",
    }
    if missing := required - set(reports):
        raise ValueError(f"Reports missing columns: {sorted(missing)}")
    if set(reports.issue_key) - set(labels.issue_key):
        raise ValueError("Reports contain identities absent from official labels")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # One retrieval per identity; repeated source annotations must have identical evidence.
    reports = reports[sorted(required)].drop_duplicates()
    if reports.issue_key.duplicated().any():
        raise ValueError("Conflicting report evidence for the same issue_key")
    joined = labels.merge(reports, on="issue_key", how="left", validate="many_to_one")
    joined["retrieval_status"] = joined.retrieval_status.fillna("NOT_FOUND")
    joined["retrieval_source"] = joined.retrieval_source.fillna("AUTO")
    joined["tracker_url"] = joined.tracker_url.fillna("")
    evidence = [construct_evidence(row) for row in joined.to_dict("records")]
    joined["text_initial"] = [item[0] for item in evidence]
    joined["text_full"] = [item[1] for item in evidence]
    joined["content_sha256"] = joined.text_initial.map(
        lambda v: hashlib.sha256(v.encode()).hexdigest()
    )
    joined["full_content_sha256"] = joined.text_full.map(
        lambda v: hashlib.sha256(v.encode()).hexdigest()
    )
    for mode in ("initial", "full"):
        joined[f"normalized_{mode}_sha256"] = joined[f"text_{mode}"].map(normalised_text_hash)
    joined[joined.original_class == "UNK"].to_parquet(out / "stage2_unknown.parquet", index=False)
    identity_conflicts = joined.groupby("issue_key").original_class.transform("nunique") > 1
    usable = (joined.retrieval_status == "SUCCESS") & joined.original_class.isin(
        ["BOH", "NAM", "ARB"]
    )
    usable &= joined.initial_description.fillna("").str.strip().ne("")
    candidate = joined[usable].copy()
    conflict = identity_conflicts.loc[candidate.index].copy()
    for mode in ("initial", "full"):
        conflict |= (
            candidate.groupby(f"normalized_{mode}_sha256").stage2_label.transform("nunique") > 1
        )
    candidate[conflict].to_csv(out / "conflicting_labels.csv", index=False)
    joined[identity_conflicts].to_csv(out / "conflicting_ids.csv", index=False)
    candidates = candidate[~conflict].copy()
    repeated = candidates.duplicated("issue_key", keep="first")
    candidates[repeated].to_csv(out / "duplicate_annotations.csv", index=False)
    ready = candidates[~repeated].copy()
    duplicate_text = ready.duplicated("normalized_initial_sha256", keep=False) | ready.duplicated(
        "normalized_full_sha256", keep=False
    )
    ready[duplicate_text].to_csv(out / "duplicate_text.csv", index=False)
    joined[~usable & joined.original_class.ne("UNK")].to_csv(
        out / "unenriched_classified.csv", index=False
    )
    ready.to_parquet(out / "stage2.parquet", index=False)
    readiness = coverage_readiness(labels, joined, ready)
    pd.DataFrame(readiness["projects"]).to_csv(out / "stage2_readiness.csv", index=False)
    write_json(out / "stage2_readiness.json", readiness)
    audit = {
        "label_rows": len(labels),
        "stage2_rows": len(ready),
        "unknown_rows": int(joined.original_class.eq("UNK").sum()),
        "conflicting_label_rows_quarantined": int(conflict.sum()),
        "conflicting_id_rows": int(identity_conflicts.sum()),
        "repeated_annotations_removed": int(repeated.sum()),
        "duplicate_text_rows": int(duplicate_text.sum()),
        "unenriched_classified_rows": int((~usable & joined.original_class.ne("UNK")).sum()),
        "class_counts": ready.stage2_label.value_counts().to_dict(),
        "by_project": ready.groupby(["project", "stage2_label"])
        .size()
        .reset_index(name="rows")
        .to_dict("records"),
        "labels_sha256": sha256_file(labels_path),
        "reports_sha256": sha256_file(reports_path),
        "stage2_sha256": sha256_file(out / "stage2.parquet"),
        "text_version": TEXT_VERSION,
        "duplicate_policy": (
            "same-label text retained; cross-partition text purged from training per fold"
        ),
        "training_invoked": False,
        "readiness": readiness,
    }
    write_json(out / "stage2_dataset_audit.json", audit)
    return audit
