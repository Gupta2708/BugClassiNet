"""Safe archive inspection and extraction."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd


def extract_tar_gz(archive: str | Path, destination: str | Path) -> list[Path]:
    """Extract a tar archive after rejecting path traversal entries."""
    source, target = Path(archive), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"Archive does not exist: {source}")
    target.mkdir(parents=True, exist_ok=True)
    resolved = target.resolve()
    with tarfile.open(source, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target / member.name).resolve()
            if not member_path.is_relative_to(resolved):
                raise ValueError(f"Unsafe archive member rejected: {member.name}")
        tar.extractall(target, filter="data")
    return sorted(target.rglob("*.csv"))


def inspect_archive(archive: str | Path) -> list[dict[str, object]]:
    """Return CSV paths and columns contained in an archive."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bugclassinet-inspect-") as temp_dir:
        csvs = extract_tar_gz(archive, temp_dir)
        if not csvs:
            raise ValueError(f"No CSV files found in archive: {archive}")
        return [
            {
                "file": str(csv.relative_to(temp_dir)),
                "columns": list(pd.read_csv(csv, nrows=0).columns),
            }
            for csv in csvs
        ]
