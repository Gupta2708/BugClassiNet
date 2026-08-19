import logging
import sys
from types import SimpleNamespace

import pytest

from bugclassinet.data import transformer as data_module
from bugclassinet.utils import memory as memory_module


class FakeDataset:
    from_parquet_calls = []
    select_calls = []

    def __init__(self, rows, fingerprint="fixture"):
        self.rows = list(rows)
        self.column_names = list(self.rows[0]) if self.rows else ["text", "canonical_label"]
        self._fingerprint = fingerprint

    def __len__(self):
        return len(self.rows)

    @classmethod
    def from_parquet(cls, path, **kwargs):
        cls.from_parquet_calls.append((path, kwargs))
        return cls(
            [{column: f"{column}-value" for column in kwargs["columns"]}],
            fingerprint="parquet-fixture",
        )

    def select_columns(self, columns):
        return FakeDataset(
            [{column: row[column] for column in columns} for row in self.rows],
            fingerprint=self._fingerprint,
        )

    def iter(self, batch_size):
        for start in range(0, len(self.rows), batch_size):
            rows = self.rows[start : start + batch_size]
            yield {column: [row[column] for row in rows] for column in self.column_names}

    def select(self, indices, **kwargs):
        index_values = [int(index) for index in indices]
        FakeDataset.select_calls.append((index_values, kwargs))
        return FakeDataset(
            [self.rows[index] for index in index_values],
            fingerprint=f"{self._fingerprint}-selected",
        )


@pytest.fixture(autouse=True)
def clear_fake_calls():
    FakeDataset.from_parquet_calls.clear()
    FakeDataset.select_calls.clear()


def _rows():
    return [
        {
            "row_id": index,
            "text": f"report {index}",
            "canonical_label": "BUG" if index < 80 else "QUESTION",
        }
        for index in range(100)
    ]


def test_load_parquet_dataset_projects_columns_to_disk_cache(tmp_path, monkeypatch):
    source = tmp_path / "train.parquet"
    source.write_bytes(b"fixture")
    cache = tmp_path / "cache"
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))

    loaded = data_module.load_parquet_dataset(source, ["text", "canonical_label", "text"], cache)

    assert len(loaded) == 1
    _, kwargs = FakeDataset.from_parquet_calls[-1]
    assert kwargs == {
        "columns": ["text", "canonical_label"],
        "cache_dir": str(cache),
        "keep_in_memory": False,
    }


def test_load_parquet_dataset_rejects_missing_file_and_empty_projection(tmp_path):
    with pytest.raises(FileNotFoundError, match="Parquet is missing"):
        data_module.load_parquet_dataset(tmp_path / "missing.parquet", ["text"], tmp_path)

    source = tmp_path / "train.parquet"
    source.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="At least one"):
        data_module.load_parquet_dataset(source, [], tmp_path)


def test_stratified_limit_is_exact_deterministic_and_disk_backed(tmp_path):
    source = FakeDataset(_rows())

    first = data_module.stratified_limit(source, 50, 42, tmp_path, "train clean")
    first_ids = [row["row_id"] for row in first.rows]
    second = data_module.stratified_limit(source, 50, 42, tmp_path, "train clean")

    assert len(first) == 50
    assert first_ids == [row["row_id"] for row in second.rows]
    assert data_module.inspect_dataset(first, "training") == {"BUG": 40, "QUESTION": 10}
    _, options = FakeDataset.select_calls[0]
    assert options["keep_in_memory"] is False
    assert options["writer_batch_size"] == 10_000
    assert options["indices_cache_file_name"].endswith(".arrow")
    assert "train-clean-n50-seed42" in options["indices_cache_file_name"]


def test_stratified_limit_validates_limit_and_bypasses_unneeded_sampling(tmp_path):
    source = FakeDataset(_rows())
    assert data_module.stratified_limit(source, None, 42, tmp_path, "train") is source
    assert data_module.stratified_limit(source, len(source), 42, tmp_path, "train") is source
    with pytest.raises(ValueError, match="positive"):
        data_module.stratified_limit(source, 0, 42, tmp_path, "train")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"text": " ", "canonical_label": "BUG"}, "blank text"),
        ({"text": "valid", "canonical_label": None}, "blank canonical_label"),
    ],
)
def test_inspect_dataset_rejects_invalid_rows(row, message):
    with pytest.raises(ValueError, match=message):
        data_module.inspect_dataset(FakeDataset([row]), "training")


def test_inspect_dataset_checks_required_columns():
    source = FakeDataset([{"text": "valid", "canonical_label": "BUG"}])
    with pytest.raises(ValueError, match="issue_id"):
        data_module.inspect_dataset(source, "validation", required_columns=["issue_id"])


def test_memory_checkpoint_logs_native_and_arrow_bytes(caplog, monkeypatch):
    monkeypatch.setattr(memory_module, "_current_rss_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr(memory_module, "_peak_rss_bytes", lambda: 3 * 1024**3)
    monkeypatch.setattr(memory_module, "_arrow_allocated_bytes", lambda: 512 * 1024**2)

    with caplog.at_level(logging.INFO):
        snapshot = memory_module.log_memory_checkpoint("after_sampling", rows=500_000)

    assert snapshot == {
        "rss_bytes": 2 * 1024**3,
        "peak_rss_bytes": 3 * 1024**3,
        "arrow_bytes": 512 * 1024**2,
    }
    assert "checkpoint=after_sampling" in caplog.text
    assert "rss_gib=2.000" in caplog.text
    assert "rows=500000" in caplog.text


def test_memory_checkpoint_degrades_when_measurements_are_unavailable(caplog, monkeypatch):
    monkeypatch.setattr(memory_module, "_current_rss_bytes", lambda: None)
    monkeypatch.setattr(memory_module, "_peak_rss_bytes", lambda: None)
    monkeypatch.setattr(memory_module, "_arrow_allocated_bytes", lambda: None)

    with caplog.at_level(logging.INFO):
        snapshot = memory_module.log_memory_checkpoint("before_load")

    assert snapshot == {"rss_bytes": None, "peak_rss_bytes": None, "arrow_bytes": None}
    assert "rss_gib=unavailable" in caplog.text
