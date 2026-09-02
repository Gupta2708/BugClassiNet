"""Memory-conscious Arrow dataset helpers for Transformer training."""

from __future__ import annotations

import gc
import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

_INSPECTION_BATCH_SIZE = 8_192
_LABEL_BATCH_SIZE = 65_536
_SAMPLE_FINGERPRINT_VERSION = "ordered-training-sample-v1"


def _dataset_class() -> Any:
    try:
        from datasets import Dataset
    except ImportError as error:
        raise ImportError(
            "Transformer datasets require `pip install -e .[transformers]`."
        ) from error
    return Dataset


def load_parquet_dataset(
    path: str | Path,
    columns: Sequence[str],
    cache_dir: str | Path,
) -> Any:
    """Load projected Parquet columns into a disk-backed Arrow Dataset.

    ``Dataset.from_parquet`` prepares an Arrow cache and memory-maps it when
    ``keep_in_memory`` is false.  Column projection is deliberately applied at
    the Parquet reader, before unused issue fields can enter host memory.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required Parquet is missing: {source}")
    selected_columns = list(dict.fromkeys(columns))
    if not selected_columns:
        raise ValueError("At least one Parquet column must be requested")

    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_class().from_parquet(
        str(source),
        columns=selected_columns,
        cache_dir=str(destination),
        keep_in_memory=False,
    )
    missing = set(selected_columns) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Parquet is missing requested columns: {sorted(missing)}")
    return dataset


def _compact_label_ids(dataset: Any) -> np.ndarray:
    """Scan only canonical labels into a compact integer vector."""
    if "canonical_label" not in dataset.column_names:
        raise ValueError("Stratified sampling requires canonical_label")

    label_dataset = dataset.select_columns(["canonical_label"])
    encoded = np.empty(len(dataset), dtype=np.int32)
    label_to_id: dict[str, int] = {}
    offset = 0
    for batch in label_dataset.iter(batch_size=_LABEL_BATCH_SIZE):
        labels = batch["canonical_label"]
        stop = offset + len(labels)
        if stop > len(encoded):
            raise ValueError("Dataset yielded more labels than its reported length")
        for position, label in enumerate(labels, start=offset):
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Stratified sampling found a missing or blank canonical_label")
            encoded[position] = label_to_id.setdefault(label, len(label_to_id))
        offset = stop
    if offset != len(encoded):
        raise ValueError(
            f"Dataset reported {len(encoded)} rows but yielded {offset} canonical labels"
        )
    return encoded


def _safe_cache_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return safe or "split"


def stable_sample_fingerprint(dataset: Any) -> str:
    """Hash the ordered training identities without materializing Python row copies."""
    identity_columns = (
        ["issue_id"] if "issue_id" in dataset.column_names else ["text", "canonical_label"]
    )
    identity = dataset.select_columns(identity_columns)
    digest = hashlib.sha256()
    digest.update(_SAMPLE_FINGERPRINT_VERSION.encode("utf-8"))
    digest.update(len(dataset).to_bytes(8, "big", signed=False))
    for batch in identity.iter(batch_size=_LABEL_BATCH_SIZE):
        batch_size = len(batch[identity_columns[0]])
        for position in range(batch_size):
            for column in identity_columns:
                value = str(batch[column][position]).encode("utf-8")
                digest.update(len(value).to_bytes(8, "big", signed=False))
                digest.update(value)
    return digest.hexdigest()


def stratified_limit(
    dataset: Any,
    maximum: int | None,
    seed: int,
    cache_dir: str | Path,
    split_name: str,
) -> Any:
    """Return an exact deterministic natural-distribution subset.

    Only the compact label vector and index arrays are resident during
    sampling.  The selected-index mapping is written to Arrow on disk instead
    of being retained as a large Python list.
    """
    if maximum is not None and maximum <= 0:
        raise ValueError("Sample limits must be positive")
    if maximum is None or len(dataset) <= maximum:
        return dataset

    label_ids = _compact_label_ids(dataset)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=maximum, random_state=seed)
    placeholder = np.zeros(len(dataset), dtype=np.uint8)
    selected, discarded = next(splitter.split(placeholder, label_ids))

    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selection_hash = hashlib.sha256(selected.tobytes()).hexdigest()[:16]
    cache_file = destination / (
        f"{_safe_cache_component(split_name)}-n{maximum}-seed{seed}-{selection_hash}.arrow"
    )
    limited = dataset.select(
        selected,
        indices_cache_file_name=str(cache_file.resolve()),
        keep_in_memory=False,
        writer_batch_size=10_000,
    )

    del discarded, label_ids, placeholder, selected
    gc.collect()
    return limited


def inspect_dataset(
    dataset: Any,
    name: str,
    required_columns: Sequence[str] = ("text", "canonical_label"),
) -> dict[str, int]:
    """Validate a raw split in bounded batches and return sorted class counts."""
    if not name or not name.strip():
        raise ValueError("Dataset name must be nonblank")
    available = set(dataset.column_names)
    required = set(required_columns) | {"text", "canonical_label"}
    missing = required - available
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError(f"{name} is empty")

    counts: Counter[str] = Counter()
    observed_rows = 0
    inspection = dataset.select_columns(["text", "canonical_label"])
    for batch in inspection.iter(batch_size=_INSPECTION_BATCH_SIZE):
        texts = batch["text"]
        labels = batch["canonical_label"]
        if len(texts) != len(labels):
            raise ValueError(f"{name} yielded different text and label batch sizes")
        for text, label in zip(texts, labels, strict=True):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{name} contains missing or blank text")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{name} contains missing or blank canonical_label")
            counts[label] += 1
        observed_rows += len(texts)

    if observed_rows != len(dataset):
        raise ValueError(f"{name} reported {len(dataset)} rows but yielded {observed_rows}")
    return dict(sorted(counts.items()))
