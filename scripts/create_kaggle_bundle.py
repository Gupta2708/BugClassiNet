"""Stage individual processed files for upload with the Kaggle CLI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json

REQUIRED_FILES = (
    "train_clean.parquet",
    "validation_clean.parquet",
    "test.parquet",
    "train_benchmark.parquet",
    "validation.parquet",
    "label_mapping.json",
    "data_manifest.json",
    "dataset_statistics.json",
)


def create_upload_directory(processed_dir: str | Path, output_dir: str | Path) -> Path:
    """Copy required artifacts into a flat, checksum-verified upload directory."""
    source, destination = Path(processed_dir), Path(output_dir)
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot stage upload: required files are missing: {missing}")
    checksums = {name: sha256_file(source / name) for name in REQUIRED_FILES}
    checksum_path = source / "checksums.json"
    write_json(checksum_path, checksums)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*REQUIRED_FILES, "checksums.json"):
        print(f"Staging {name}")
        shutil.copy2(source / name, destination / name)
    return destination


def main() -> None:
    """Parse arguments and create a Kaggle CLI upload directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/nlbse2023")
    parser.add_argument("--output-dir", default="kaggle/upload/bugclassinet-nlbse-v1")
    args = parser.parse_args()
    print(create_upload_directory(args.processed_dir, args.output_dir))


if __name__ == "__main__":
    main()
