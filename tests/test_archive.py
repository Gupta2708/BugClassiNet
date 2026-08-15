import tarfile
from pathlib import Path

from bugclassinet.data.archive import inspect_archive


def test_inspect_archive(tmp_path: Path) -> None:
    csv = tmp_path / "issues.csv"
    csv.write_text("id,title,description,label\n1,t,b,bug\n", encoding="utf-8")
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(csv, arcname="nested/issues.csv")
    inspected = inspect_archive(archive)
    assert inspected[0]["columns"] == ["id", "title", "description", "label"]
