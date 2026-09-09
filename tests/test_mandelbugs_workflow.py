"""Small offline fixtures only: never train on research data or contact trackers."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bugclassinet.data.mandelbugs import audit_mandelbugs, infer_project_subsystem, parse_arff
from bugclassinet.data.mandelbugs_prepare import prepare_mandelbugs
from bugclassinet.data.mandelbugs_schema import construct_evidence, string_value, validate_labels

HEADER = """@relation fixture
@attribute BugID numeric
@attribute BugClass {BOH,NAM,ARB,UNK}
@attribute BugSubclass {N/A,ENV,MEM}
@data
"""


def labels_fixture():
    return validate_labels(
        pd.DataFrame(
            [
                {
                    "project": "Linux",
                    "subsystem": "ipv4",
                    "bug_id": str(i),
                    "original_class": label,
                    "original_subclass": "N/A",
                    "source_arff": "linux_ipv4.arff",
                }
                for i, label in enumerate(["BOH", "NAM", "ARB", "UNK"], 1)
            ]
        )
    )


def test_arff_mapping_optional_subclass_and_audit(tmp_path):
    raw = tmp_path / "raw" / "nested"
    raw.mkdir(parents=True)
    (raw / "axis_soap.arff").write_text(
        HEADER + "AXIS-1,BOH,N/A\nAXIS-2,NAM,ENV\nAXIS-3,ARB,MEM\nAXIS-4,UNK,N/A\n"
    )
    audit = audit_mandelbugs(raw.parent, tmp_path / "audit")
    assert audit["class_counts"] == {"BOH": 1, "NAM": 1, "ARB": 1, "UNK": 1}
    rows = pd.read_parquet(tmp_path / "audit/labels.parquet")
    assert rows.stage2_label.fillna("NULL").tolist() == ["BOH", "MANDELBUG", "MANDELBUG", "NULL"]
    assert rows.stage3_label.fillna("NULL").tolist() == ["NULL", "NAM", "ARB", "NULL"]
    assert rows.original_subclass.tolist() == ["N/A", "ENV", "MEM", "N/A"]
    assert rows.source_arff.unique().tolist() == ["nested/axis_soap.arff"]
    assert rows.bug_id.iloc[0] == "AXIS-1"
    assert len(audit["warnings"]) == 4
    (raw / "linux_ipv4.arff").write_text(
        "@relation test\n@attribute BugID numeric\n@attribute BugClass {BOH}\n@data\n1,BOH\n"
    )
    assert parse_arff(raw / "linux_ipv4.arff")[0][0]["original_subclass"] is None
    assert string_value(b"BOH") == "BOH"
    assert infer_project_subsystem("apache_httpd_mod_ssl.arff") == ("HTTPD", "mod_ssl")
    assert infer_project_subsystem("mysql_optimizer.arff") == ("MySQL", "optimizer")


def test_arff_duplicates_and_failures_are_never_silent(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "mysql_optimizer.arff").write_text(HEADER + "1,BOH,N/A\n1,NAM,ENV\n?,BOH,N/A\n")
    (raw / "mysql_replication.arff").write_text(HEADER + "1,BOH,N/A\n")
    with pytest.raises(ValueError, match="failures"):
        audit_mandelbugs(raw, tmp_path / "out")
    failures = pd.read_csv(tmp_path / "out/parse_failures.csv")
    assert len(failures) == 1 and failures.source_row.iloc[0] == 3
    duplicates = pd.read_csv(tmp_path / "out/duplicate_ids.csv")
    assert len(duplicates) == 3
    assert duplicates.across_files.all() and duplicates.conflicting_class.all()


def test_evidence_excludes_metadata_and_preserves_initial_full_boundary():
    record = {
        "title": "Crash",
        "initial_description": "Steps",
        "environment_text": "OS",
        "comments_text": "Later fix",
        "project": "SECRET_PROJECT",
        "original_class": "NAM",
        "original_subclass": "ENV",
        "status": "SECRET_STATUS",
        "resolution": "SECRET_RESOLUTION",
    }
    initial, full = construct_evidence(record)
    assert "Later fix" not in initial and "Later fix" in full
    assert "SECRET" not in full and "NAM" not in full
    assert "[ENVIRONMENT]\nOS" in initial


def test_prepare_unknown_failed_duplicate_and_conflicting_text(tmp_path):
    labels = labels_fixture()
    # Duplicate same source annotation retained in labels, collapsed only in prepared data.
    labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    labels.to_parquet(tmp_path / "labels.parquet")
    reports = []
    for row in labels.drop_duplicates("issue_key").to_dict("records"):
        reports.append(
            {
                **row,
                "retrieval_status": "SUCCESS",
                "retrieval_source": "AUTO",
                "tracker_url": "https://example.invalid",
                "title": "Crash",
                "initial_description": "same" if row["bug_id"] in {"1", "2"} else "different",
                "comments_text": "",
                "environment_text": "",
            }
        )
    pd.DataFrame(reports).to_parquet(tmp_path / "reports.parquet")
    audit = prepare_mandelbugs(
        tmp_path / "labels.parquet", tmp_path / "reports.parquet", tmp_path / "out"
    )
    assert audit["unknown_rows"] == 1 and audit["conflicting_label_rows_quarantined"] == 3
    ready = pd.read_parquet(tmp_path / "out/stage2.parquet")
    assert ready.bug_id.tolist() == ["3"] and ready.stage2_label.tolist() == ["MANDELBUG"]
    assert pd.read_parquet(tmp_path / "out/stage2_unknown.parquet").original_class.tolist() == [
        "UNK"
    ]
    reports[1]["initial_description"] = "unique second"
    reports[2]["retrieval_status"] = "NOT_FOUND"
    pd.DataFrame(reports).to_parquet(tmp_path / "reports.parquet")
    audit = prepare_mandelbugs(
        tmp_path / "labels.parquet", tmp_path / "reports.parquet", tmp_path / "out2"
    )
    assert audit["stage2_rows"] == 2 and audit["repeated_annotations_removed"] == 1
    assert audit["unenriched_classified_rows"] == 1


@pytest.fixture
def retrieval():
    pytest.importorskip("bs4")
    pytest.importorskip("requests")
    from bugclassinet.data import mandelbugs_enrich

    return mandelbugs_enrich


class Session:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = iter(responses), 0, {}

    def get(self, *args, **kwargs):
        self.calls += 1
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def response(status=200, content=b"", headers=None):
    return SimpleNamespace(status_code=status, content=content, headers=headers or {})


BUGZILLA = (
    b'<title>Bug 1</title><span id="short_desc_nonedit_display">Crash</span>'
    b'<pre class="bz_comment_text">Initial</pre><pre class="bz_comment_text">Comment</pre>'
    b'<div id="resolution">DO NOT INCLUDE</div>'
)


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "AUTH_REQUIRED"),
        (403, "AUTH_REQUIRED"),
        (429, "RATE_LIMITED"),
        (404, "NOT_FOUND"),
        (302, "UNSUPPORTED"),
    ],
)
def test_retrieval_status_and_host_circuit(tmp_path, retrieval, status, expected):
    session = Session([response(status, headers={"Retry-After": "120"})])
    fetch = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None)
    assert fetch.retrieve("Linux", "1")["retrieval_status"] == expected
    assert fetch.retrieve("Linux", "1")["retrieval_status"] == expected
    if status in {401, 403, 429}:
        assert fetch.retrieve("Linux", "2")["retrieval_status"] == expected
    assert session.calls == 1


def test_raw_cache_retry_and_integrity(tmp_path, retrieval):
    import requests

    session = Session(
        [requests.ConnectionError("offline"), response(503), response(content=BUGZILLA)]
    )
    delays = []
    fetch = retrieval.CachedRetriever(tmp_path, session=session, sleeper=delays.append)
    record = fetch.retrieve("Linux", "1")
    assert record["retrieval_status"] == "SUCCESS" and record["comments_text"] == "Comment"
    assert record["error"] == ""
    assert delays == [1.0, 2, 4]
    assert fetch.retrieve("Linux", "1") == record and session.calls == 3
    next(tmp_path.glob("*.html")).write_text("changed")
    with pytest.raises(ValueError, match="Corrupt"):
        fetch.retrieve("Linux", "1")


def test_adapter_content_and_manual_priority(retrieval):
    jira = (
        b'<h1 id="summary-val">Crash</h1><div id="description-val">Initial</div>'
        b'<div id="environment-val">OS</div><div id="status-val">SECRET</div>'
    )
    mysql = b"<h1>Bug #1: Crash</h1><pre>Description: initial</pre><pre>Later</pre>"
    for project, html in [("AXIS", jira), ("MySQL", mysql), ("HTTPD", BUGZILLA)]:
        result = retrieval.parse_tracker_html(project, html)
        assert result["retrieval_status"] == "SUCCESS"
        assert "SECRET" not in construct_evidence(result)[1]
    authentication_bug = b"<title>Authentication fails</title>" + BUGZILLA
    assert (
        retrieval.parse_tracker_html("Linux", authentication_bug)["retrieval_status"] == "SUCCESS"
    )
    assert (
        retrieval.parse_tracker_html("Linux", b"<title>Access denied</title>")["retrieval_status"]
        == "AUTH_REQUIRED"
    )
    assert (
        retrieval.parse_tracker_html("Linux", b"<p>Wrong layout</p>")["retrieval_status"]
        == "PARSE_ERROR"
    )
    assert (
        retrieval.parse_tracker_html("Linux", b'<span id="short_desc">Only title</span>')[
            "retrieval_status"
        ]
        == "EMPTY_CONTENT"
    )
    good = {"retrieval_status": "SUCCESS", "title": "prior"}
    assert retrieval.merge_manual(good, {"title": "replacement"}) == good
    assert (
        retrieval.merge_manual({"retrieval_status": "AUTH_REQUIRED"}, {"title": "manual"})[
            "retrieval_source"
        ]
        == "MANUAL_IMPORT"
    )


def test_offline_manual_enrichment_and_successful_rerun(tmp_path, monkeypatch, retrieval):
    import requests

    labels = labels_fixture()
    labels.to_parquet(tmp_path / "labels.parquet")
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    manual = pd.DataFrame(
        [
            {
                "project": "Linux",
                "bug_id": "1",
                "title": "Manual",
                "initial_description": "Evidence",
                "comments_text": "",
                "environment_text": "",
                "tracker_url": "https://bugzilla.kernel.org/show_bug.cgi?id=1",
            }
        ]
    )
    manual.to_csv(manual_dir / "reports.csv", index=False)
    monkeypatch.setattr(
        requests.Session, "get", lambda *a, **k: pytest.fail("Offline must not access network")
    )
    arguments = (tmp_path / "labels.parquet", tmp_path / "out", tmp_path / "cache")
    audit = retrieval.enrich_mandelbugs(
        *arguments, manual_dir=manual_dir, offline=True, retry_failures=True
    )
    assert audit["unique_status_counts"] == {"UNSUPPORTED": 3, "SUCCESS": 1}
    manual.loc[0, "initial_description"] = "Replacement"
    manual.to_csv(manual_dir / "reports.csv", index=False)
    retrieval.enrich_mandelbugs(*arguments, manual_dir=manual_dir, offline=True)
    reports = pd.read_parquet(tmp_path / "out/reports.parquet")
    assert reports.loc[reports.bug_id == "1", "initial_description"].iloc[0] == "Evidence"
    assert (
        json.loads((tmp_path / "out/retrieval_audit.json").read_text())["training_invoked"] is False
    )
