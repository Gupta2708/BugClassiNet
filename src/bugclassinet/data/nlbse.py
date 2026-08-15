"""NLBSE archive preparation workflow."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.data.archive import extract_tar_gz
from bugclassinet.data.deduplicate import (
    deduplicate_train,
    duplicate_overlap,
    ensure_unique_issue_ids,
    exclude_test_overlap,
)
from bugclassinet.data.harmonize import harmonize_issues
from bugclassinet.data.schema import infer_issue_schema
from bugclassinet.data.split import make_validation_split
from bugclassinet.data.statistics import dataset_statistics
from bugclassinet.data.validate import validate_frame
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json

LOGGER = logging.getLogger(__name__)


def _read_archive(archive: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="bugclassinet-nlbse-") as temp_dir:
        csvs = extract_tar_gz(archive, temp_dir)
        if not csvs:
            raise ValueError(f"No CSV files found in {archive}")
        discovered = []
        frames = []
        for csv in csvs:
            raw = pd.read_csv(csv)
            schema = infer_issue_schema(raw.columns)
            LOGGER.info(
                "Discovered %s columns=%s schema=%s", csv.name, list(raw.columns), schema.as_dict()
            )
            frames.append(harmonize_issues(raw, schema))
            discovered.append(
                {
                    "file": str(csv.relative_to(temp_dir)),
                    "columns": list(raw.columns),
                    "schema": schema.as_dict(),
                }
            )
        return pd.concat(frames, ignore_index=True), {"files": discovered}


def _balanced_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    labels = sorted(frame["canonical_label"].unique())
    per_class = maximum // len(labels)
    parts = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby("canonical_label")
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def prepare_nlbse(
    train_archive: str | Path,
    test_archive: str | Path,
    output_dir: str | Path = "data/processed/nlbse2023",
    sample_dir: str | Path = "data/samples",
    validation_fraction: float = 0.15,
    seed: int = 42,
    sample_size: int = 20_000,
) -> dict[str, Any]:
    """Build canonical NLBSE Parquet splits from untouched train/test archives."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_raw, train_discovery = _read_archive(train_archive)
    test, test_discovery = _read_archive(test_archive)
    deduped, removed = deduplicate_train(train_raw)
    deduped, source_id_rows_disambiguated = ensure_unique_issue_ids(deduped)
    if source_id_rows_disambiguated:
        LOGGER.warning(
            "Disambiguated %d rows with non-unique source issue IDs using normalized text hashes",
            source_id_rows_disambiguated,
        )
    overlap = duplicate_overlap(deduped, test)
    train_benchmark, validation, benchmark_strategy = make_validation_split(
        deduped, validation_fraction, seed
    )
    clean_candidates, overlap_rows_removed = exclude_test_overlap(deduped, test)
    train_clean, validation_clean, clean_strategy = make_validation_split(
        clean_candidates, validation_fraction, seed
    )
    outputs = {
        "train_benchmark": train_benchmark,
        "validation": validation,
        "train_clean": train_clean,
        "validation_clean": validation_clean,
        "test": test,
    }
    for name, frame in outputs.items():
        validate_frame(frame)
        frame.to_parquet(out / f"{name}.parquet", index=False)
    sample_path = Path(sample_dir) / "nlbse2023_development.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    _balanced_sample(train_clean, sample_size, seed).to_parquet(sample_path, index=False)
    label_mapping = {label.lower(): label for label in sorted(deduped["canonical_label"].unique())}
    write_json(out / "label_mapping.json", label_mapping)
    stats = {
        "train_benchmark": dataset_statistics(train_benchmark),
        "validation": dataset_statistics(validation),
        "train_clean": dataset_statistics(train_clean),
        "validation_clean": dataset_statistics(validation_clean),
        "test": dataset_statistics(test),
        "deduplicated_train_rows_removed": removed,
        "source_issue_id_rows_disambiguated": source_id_rows_disambiguated,
        "official_test_overlap_rows_before_cleaning": len(overlap),
        "rows_removed_for_official_test_overlap": overlap_rows_removed,
        "official_test_overlap_rows_after_cleaning": len(duplicate_overlap(train_clean, test)),
        "benchmark_split_strategy": benchmark_strategy,
        "clean_split_strategy": clean_strategy,
    }
    write_json(out / "dataset_statistics.json", stats)
    manifest = {
        "source_archives": {
            str(train_archive): sha256_file(train_archive),
            str(test_archive): sha256_file(test_archive),
        },
        "processed_files": {
            name: sha256_file(out / name)
            for name in (
                "train_benchmark.parquet",
                "validation.parquet",
                "train_clean.parquet",
                "validation_clean.parquet",
                "test.parquet",
                "label_mapping.json",
                "dataset_statistics.json",
            )
        },
        "discovery": {"train": train_discovery, "test": test_discovery},
        "official_test_overlap_before_cleaning": overlap[["issue_id", "text_hash"]].to_dict(
            "records"
        ),
        "official_test_rows": len(test),
    }
    write_json(out / "data_manifest.json", manifest)
    return {"output_dir": str(out), "statistics": stats, "sample": str(sample_path)}
