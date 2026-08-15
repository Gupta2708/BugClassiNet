import tarfile
from pathlib import Path

import pandas as pd

from bugclassinet.data.nlbse import prepare_nlbse
from bugclassinet.data.validate import validate_nlbse_variants


def _archive(path: Path, rows: list[dict[str, object]]) -> Path:
    csv = path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    with tarfile.open(path, "w:gz") as handle:
        handle.add(csv, arcname="nested/issues.csv")
    return path


def test_prepare_nlbse_end_to_end(tmp_path: Path) -> None:
    labels = ["bug", "feature", "question", "documentation"]
    train_rows = [
        {"id": i, "title": f"{label} title {i}", "description": f"body {i}", "label": label}
        for i, label in enumerate(labels * 5)
    ]
    train_rows.append({"id": 99, "title": "test bug", "description": "test body", "label": "bug"})
    train_archive = _archive(tmp_path / "train.tar.gz", train_rows)
    test_archive = _archive(
        tmp_path / "test.tar.gz",
        [{"id": 100, "title": "test bug", "description": "test body", "label": "bug"}],
    )
    output = tmp_path / "processed"
    prepare_nlbse(
        train_archive, test_archive, output, tmp_path / "samples", validation_fraction=0.2
    )
    assert {path.name for path in output.iterdir()} >= {
        "train_benchmark.parquet",
        "validation.parquet",
        "train_clean.parquet",
        "validation_clean.parquet",
        "test.parquet",
        "data_manifest.json",
    }
    benchmark, valid = (
        pd.read_parquet(output / "train_benchmark.parquet"),
        pd.read_parquet(output / "validation.parquet"),
    )
    clean, clean_valid, official_test = (
        pd.read_parquet(output / "train_clean.parquet"),
        pd.read_parquet(output / "validation_clean.parquet"),
        pd.read_parquet(output / "test.parquet"),
    )
    assert not (set(benchmark.issue_id) & set(valid.issue_id))
    assert not (set(clean.issue_id) & set(clean_valid.issue_id))
    assert len(benchmark) + len(valid) == 21  # Benchmark rows are preserved.
    assert len(clean) + len(clean_valid) == 20
    assert not set(clean.text_hash) & set(official_test.text_hash)
    assert official_test[["id", "title", "body", "original_label"]].to_dict("records") == [
        {"id": 100, "title": "test bug", "body": "test body", "original_label": "bug"}
    ]
    assert validate_nlbse_variants(output)["train_clean"] == len(clean)
