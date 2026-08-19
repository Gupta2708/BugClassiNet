"""Compare legacy and memory-safe Stage-1 preprocessing peak RSS.

Run exactly one mode per process so the operating-system high-water mark is
attributable to that implementation.  A deterministic tokenizer approximates
variable-length word-piece output without downloading a model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from bugclassinet.data.transformer import (
    inspect_dataset,
    load_parquet_dataset,
    stratified_limit,
)
from bugclassinet.models.transformer_classifier import (
    TransformerTrainingConfig,
    _tokenize_dataset,
)
from bugclassinet.utils.memory import memory_snapshot

_BYTES_PER_GIB = 1024**3


class DeterministicFakeTokenizer:
    """Produce stable token IDs while implementing the tokenizer call contract."""

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool,
        max_length: int,
        padding: bool | str,
    ) -> dict[str, list[list[int]]]:
        if not truncation:
            raise ValueError("The benchmark requires truncation")

        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Benchmark input contains missing or blank text")

            available = max(max_length - 2, 0)
            pieces = text.split()[:available]
            token_ids = [1]
            token_ids.extend(3 + (zlib.crc32(piece.encode("utf-8")) % 30_000) for piece in pieces)
            if max_length > 1:
                token_ids.append(2)
            token_ids = token_ids[:max_length]
            attention_mask = [1] * len(token_ids)

            if padding == "max_length":
                padding_length = max_length - len(token_ids)
                token_ids.extend([0] * padding_length)
                attention_mask.extend([0] * padding_length)
            elif padding not in (False, None):
                raise ValueError(f"Unsupported fake-tokenizer padding mode: {padding!r}")

            input_ids.append(token_ids)
            attention_masks.append(attention_mask)

        return {"input_ids": input_ids, "attention_mask": attention_masks}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _require_datasets() -> Any:
    try:
        import datasets
    except ImportError as error:
        raise SystemExit(
            "Missing optional dependency `datasets`; install with "
            "`python -m pip install -e .[transformers]`."
        ) from error
    datasets.disable_progress_bars()
    return datasets


def _label_mapping(labels: list[Any]) -> dict[str, int]:
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Benchmark input contains missing or blank canonical_label")
    return {label: index for index, label in enumerate(sorted(set(labels)))}


def _legacy(
    source: Path,
    sample_size: int,
    seed: int,
    max_length: int,
    tokenizer: DeterministicFakeTokenizer,
    dataset_class: Any,
) -> tuple[Any, int, dict[str, Any]]:
    """Reproduce the previous eager pandas/list/globally-padded path."""
    all_rows = pd.read_parquet(source)
    if "text" not in all_rows or "canonical_label" not in all_rows:
        raise ValueError("Parquet must contain text and canonical_label")

    if len(all_rows) > sample_size:
        sampled = all_rows.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        del all_rows
        gc.collect()
    else:
        sampled = all_rows

    texts = sampled["text"].tolist()
    labels = sampled["canonical_label"].tolist()
    label_to_id = _label_mapping(labels)
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    encoded["labels"] = [label_to_id[label] for label in labels]
    tokenized = dataset_class.from_dict(encoded)

    # The former trainer retained both its sampled DataFrame argument and the
    # in-memory Arrow token table, but the temporary Python lists left scope.
    del encoded, labels, texts
    gc.collect()
    details = {
        "loaded_columns": "all",
        "padding": "max_length",
        "storage": "in-memory Dataset.from_dict",
        "retained_sample_dataframe": True,
        "sample_dataframe_bytes": int(sampled.memory_usage(index=True, deep=True).sum()),
        "features": list(tokenized.column_names),
    }
    return tokenized, len(sampled), details


def _memory_safe(
    source: Path,
    sample_size: int,
    seed: int,
    max_length: int,
    tokenizer: DeterministicFakeTokenizer,
    cache_dir: Path,
    tokenization_batch_size: int,
    writer_batch_size: int,
) -> tuple[Any, int, dict[str, Any]]:
    """Exercise the projected, stratified, disk-backed production path."""
    raw = load_parquet_dataset(
        source,
        ["text", "canonical_label"],
        cache_dir / "raw",
    )
    sampled = stratified_limit(raw, sample_size, seed, cache_dir / "selection", "train")
    counts = inspect_dataset(sampled, "benchmark train")
    label_to_id = {label: index for index, label in enumerate(sorted(counts))}

    token_cache_dir = cache_dir / "tokens"
    token_cache_dir.mkdir(parents=True, exist_ok=True)
    config = TransformerTrainingConfig(
        model_name="benchmark/deterministic-fake-tokenizer",
        max_length=max_length,
        seed=seed,
        tokenization_batch_size=tokenization_batch_size,
        tokenization_writer_batch_size=writer_batch_size,
        dataset_cache_dir=str(cache_dir),
    )
    tokenized = _tokenize_dataset(
        sampled,
        tokenizer,
        label_to_id,
        config,
        token_cache_dir,
        "benchmark-train",
    )
    selected_rows = len(sampled)

    del raw, sampled
    gc.collect()
    details = {
        "loaded_columns": ["text", "canonical_label"],
        "padding": False,
        "storage": "cache-backed Arrow map",
        "cache_dir": str(cache_dir.resolve()),
        "class_counts": counts,
        "features": list(tokenized.column_names),
    }
    return tokenized, selected_rows, details


def _default_cache_dir(source: Path, sample_size: int, seed: int) -> Path:
    root = Path(tempfile.gettempdir()) / "bugclassinet-stage1-memory-benchmark"
    return root / f"{source.stem}-n{sample_size}-seed{seed}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("legacy", "memory-safe"))
    parser.add_argument("--parquet", required=True, type=Path, help="Stage-1 training Parquet")
    parser.add_argument(
        "--sample-size",
        "--rows",
        dest="sample_size",
        type=_positive_int,
        default=20_000,
        help="Rows retained after loading (default: 20000)",
    )
    parser.add_argument("--max-length", type=_positive_int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--tokenization-batch-size", type=_positive_int, default=512)
    parser.add_argument("--writer-batch-size", type=_positive_int, default=512)
    return parser


def main() -> None:
    args = _parser().parse_args()
    source = args.parquet.resolve()
    if not source.is_file():
        raise SystemExit(f"Parquet file does not exist: {source}")

    datasets = _require_datasets()
    cache_dir = args.cache_dir or _default_cache_dir(source, args.sample_size, args.seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = DeterministicFakeTokenizer()
    started = time.perf_counter()
    initial = memory_snapshot()

    if args.mode == "legacy":
        tokenized, selected_rows, details = _legacy(
            source,
            args.sample_size,
            args.seed,
            args.max_length,
            tokenizer,
            datasets.Dataset,
        )
    else:
        tokenized, selected_rows, details = _memory_safe(
            source,
            args.sample_size,
            args.seed,
            args.max_length,
            tokenizer,
            cache_dir,
            args.tokenization_batch_size,
            args.writer_batch_size,
        )

    # Materialize one row so accidental lazy construction cannot make the
    # memory-safe path appear artificially cheap.
    if selected_rows:
        _ = tokenized[0]
    final = memory_snapshot()
    output = {
        "mode": args.mode,
        "pid": os.getpid(),
        "parquet": str(source),
        "requested_sample_size": args.sample_size,
        "selected_rows": selected_rows,
        "max_length": args.max_length,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "initial_rss_bytes": initial["rss_bytes"],
        "final_rss_bytes": final["rss_bytes"],
        "peak_rss_bytes": final["peak_rss_bytes"],
        "final_rss_gib": (
            None if final["rss_bytes"] is None else final["rss_bytes"] / _BYTES_PER_GIB
        ),
        "peak_rss_gib": (
            None if final["peak_rss_bytes"] is None else final["peak_rss_bytes"] / _BYTES_PER_GIB
        ),
        "arrow_allocated_bytes": final["arrow_bytes"],
        "details": details,
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
