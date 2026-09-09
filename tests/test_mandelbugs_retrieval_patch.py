"""Secret-safe HTTPD API, portable prior reports, and readiness fixtures."""

import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from bugclassinet.data import mandelbugs_enrich as retrieval
from bugclassinet.data.mandelbugs_schema import construct_evidence, validate_labels
from bugclassinet.utils.checksums import sha256_file


def response(status=200, payload=None, headers=None):
    content = json.dumps(payload or {}).encode()
    return SimpleNamespace(status_code=status, content=content, headers=headers or {})


class RecordingSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def label_table(projects=("Linux", "MySQL", "HTTPD", "AXIS")):
    rows = []
    for project in projects:
        for bug_id, label in (("1", "BOH"), ("2", "NAM"), ("3", "UNK")):
            rows.append(
                {
                    "project": project,
                    "subsystem": "component",
                    "bug_id": bug_id,
                    "original_class": label,
                    "original_subclass": "N/A",
                    "source_arff": f"{project}.arff",
                }
            )
    return validate_labels(pd.DataFrame(rows))


def success_reports(labels):
    return pd.DataFrame(
        [
            {
                **row,
                "retrieval_status": "SUCCESS",
                "retrieval_source": "AUTO_REMOTE",
                "tracker_url": "https://example.invalid/issue",
                "title": "Issue",
                "initial_description": "Reproduction steps",
                "comments_text": "Later comment",
                "environment_text": "OS",
            }
            for row in labels.to_dict("records")
        ]
    )


def test_historical_mysql_fixture_parses_content_and_environment():
    html = b"""
    <title>MySQL Bugs: #21704: Renaming column does not update FK definition</title>
    <table>
      <tr><th>Submitted:</th><td>17 Aug 2006 22:21</td></tr>
      <tr><th>OS:</th><td>Linux (Linux)</td><th>CPU Architecture:</th><td>Any</td></tr>
      <tr><th>Status:</th><td>Closed</td></tr>
    </table>
    <div class="comment"><pre>Description:\nInitial reproduction</pre></div>
    <div class="comment"><pre>Public follow-up</pre></div>
    """
    record = retrieval.parse_tracker_html("MySQL", html)
    assert record["title"] == "Renaming column does not update FK definition"
    assert record["initial_description"] == "Description:\nInitial reproduction"
    assert record["comments_text"] == "Public follow-up"
    assert record["environment_text"] == "Operating system: Linux (Linux)\nHardware: Any"
    assert record["tracker_operating_system"] == "Linux (Linux)"
    assert record["tracker_platform"] == "Any"
    assert record["created_at"] == "17 Aug 2006 22:21"
    assert "Closed" not in construct_evidence(record)[1]


def test_httpd_missing_key_is_auth_required_without_request(tmp_path, monkeypatch):
    monkeypatch.delenv(retrieval.APACHE_API_KEY_ENV, raising=False)
    session = RecordingSession([])
    fetch = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None)
    record = fetch.retrieve("HTTPD", "7441")
    assert record["retrieval_status"] == "AUTH_REQUIRED"
    assert record["credential_supplied"] is False
    assert session.calls == []


def test_httpd_api_key_success_is_secret_safe(tmp_path, monkeypatch, caplog):
    secret = "test-secret-value"
    monkeypatch.setenv(retrieval.APACHE_API_KEY_ENV, secret)
    bug = {
        "bugs": [
            {
                "id": 7441,
                "summary": "HTTPD crash",
                "creation_time": "2000-01-01T00:00:00Z",
                "op_sys": "Linux",
                "platform": "x86",
                "component": "Core",
                "status": "RESOLVED",
                "resolution": "FIXED",
            }
        ]
    }
    comments = {
        "bugs": {
            "7441": {
                "comments": [
                    {"id": 1, "count": 0, "text": "Initial report", "is_private": False},
                    {"id": 2, "count": 1, "text": "Private", "is_private": True},
                    {"id": 3, "count": 2, "text": "Public follow-up", "is_private": False},
                ]
            }
        }
    }
    session = RecordingSession([response(payload=bug), response(payload=comments)])
    caplog.set_level(logging.DEBUG)
    fetch = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None)
    record = fetch.retrieve("HTTPD", "7441")
    assert record["retrieval_status"] == "SUCCESS"
    assert record["initial_description"] == "Initial report"
    assert record["comments_text"] == "Public follow-up"
    assert record["environment_text"] == "Operating system: Linux\nHardware: x86"
    assert record["tracker_component"] == "Core"
    assert all(call[1]["params"] == {"Bugzilla_api_key": secret} for call in session.calls)
    assert all("Bugzilla_api_key" not in call[0] for call in session.calls)
    assert all("/rest.cgi/bug/7441" in call[0] for call in session.calls)
    saved = "\n".join(path.read_text(errors="ignore") for path in tmp_path.iterdir())
    assert secret not in saved and secret not in caplog.text and secret not in json.dumps(record)
    evidence = construct_evidence(record)[1]
    assert "RESOLVED" not in evidence and "FIXED" not in evidence and "Private" not in evidence


@pytest.mark.parametrize(
    "status,expected",
    [(401, "AUTH_FAILED"), (403, "AUTH_FAILED"), (404, "NOT_FOUND"), (429, "RATE_LIMITED")],
)
def test_httpd_api_statuses(tmp_path, monkeypatch, status, expected):
    monkeypatch.setenv(retrieval.APACHE_API_KEY_ENV, "short-test-key")
    session = RecordingSession([response(status=status, headers={"Retry-After": "60"})])
    fetch = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None)
    record = fetch.retrieve("HTTPD", "7441")
    assert record["retrieval_status"] == expected
    assert "short-test-key" not in json.dumps(record)


def test_httpd_api_json_auth_error_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv(retrieval.APACHE_API_KEY_ENV, "never-persist-this")
    session = RecordingSession(
        [response(payload={"error": True, "code": 306, "message": "invalid key"})]
    )
    record = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None).retrieve(
        "HTTPD", "7441"
    )
    assert record["retrieval_status"] == "AUTH_FAILED"
    assert "never-persist-this" not in "\n".join(
        path.read_text(errors="ignore") for path in tmp_path.iterdir()
    )


def test_httpd_api_json_invalid_bug_is_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv(retrieval.APACHE_API_KEY_ENV, "short-test-key")
    session = RecordingSession(
        [response(payload={"error": True, "code": 101, "message": "Invalid Bug ID"})]
    )
    record = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None).retrieve(
        "HTTPD", "7441"
    )
    assert record["retrieval_status"] == "NOT_FOUND"
    assert len(session.calls) == 1


def test_httpd_api_never_caches_a_response_that_echoes_the_key(tmp_path, monkeypatch):
    secret = "response-echo-test-key"
    monkeypatch.setenv(retrieval.APACHE_API_KEY_ENV, secret)
    session = RecordingSession([response(payload={"message": secret})])
    record = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None).retrieve(
        "HTTPD", "7441"
    )
    assert record["retrieval_status"] == "AUTH_FAILED"
    assert secret not in "\n".join(path.read_text(errors="ignore") for path in tmp_path.iterdir())


def test_httpd_secret_never_appears_in_enrichment_manifest(tmp_path, monkeypatch):
    secret = "manifest-secret-test"
    monkeypatch.setenv(retrieval.APACHE_API_KEY_ENV, secret)
    labels = label_table(("HTTPD",))
    labels.to_parquet(tmp_path / "labels.parquet")
    bug = {"bugs": [{"id": 1, "summary": "Issue"}]}
    comments = {
        "bugs": {
            "1": {
                "comments": [{"id": 1, "count": 0, "text": "Initial report", "is_private": False}]
            }
        }
    }
    session = RecordingSession([response(payload=bug), response(payload=comments)])
    monkeypatch.setattr(retrieval.requests, "Session", lambda: session)

    retrieval.enrich_mandelbugs(
        tmp_path / "labels.parquet",
        tmp_path / "reports",
        tmp_path / "cache",
        manual_dir=None,
        only_issue="HTTPD:1",
    )
    persisted = "\n".join(
        path.read_text(errors="ignore")
        for root in (tmp_path / "cache", tmp_path / "reports")
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".parquet"
    )
    assert secret not in persisted


def test_prior_report_reused_without_network_and_audited(tmp_path, monkeypatch):
    labels = label_table(("MySQL",))
    labels.to_parquet(tmp_path / "labels.parquet")
    prior = success_reports(labels)
    prior.loc[prior.bug_id == "1", "initial_description"] = "Local MySQL evidence"
    prior.to_parquet(tmp_path / "local_reports.parquet")
    monkeypatch.setattr(
        retrieval.CachedRetriever,
        "retrieve",
        lambda *args, **kwargs: pytest.fail("A prior success must prevent network retrieval"),
    )
    audit = retrieval.enrich_mandelbugs(
        tmp_path / "labels.parquet",
        tmp_path / "merged",
        tmp_path / "cache",
        prior_reports=[tmp_path / "local_reports.parquet"],
        only_issue="MySQL:1",
    )
    result = pd.read_parquet(tmp_path / "merged/reports.parquet").iloc[0]
    assert result.retrieval_source == "PRIOR_REPORT"
    assert result.initial_description == "Local MySQL evidence"
    imported = audit["prior_reports"][0]
    assert imported["sha256"] == sha256_file(tmp_path / "local_reports.parquet")
    assert imported["rows"] == 3 and imported["reused_successful_records"] == 1
    assert "C:\\" not in json.dumps(audit)

    retrieval.enrich_mandelbugs(
        tmp_path / "labels.parquet",
        tmp_path / "merged",
        tmp_path / "cache",
        prior_reports=[tmp_path / "local_reports.parquet"],
        only_issue="MySQL:1",
        offline=True,
    )
    rerun = pd.read_parquet(tmp_path / "merged/reports.parquet").iloc[0]
    assert rerun.prior_report_file == "local_reports.parquet"
    assert rerun.prior_report_sha256 == sha256_file(tmp_path / "local_reports.parquet")


def test_failed_prior_record_never_replaces_success(tmp_path):
    labels = label_table(("MySQL",))
    good = success_reports(labels.iloc[[0]])
    good.loc[:, "initial_description"] = "Complete local evidence"
    failed = good.copy()
    failed.loc[:, "retrieval_status"] = "AUTH_REQUIRED"
    failed.loc[:, "initial_description"] = ""
    good.to_parquet(tmp_path / "good.parquet")
    failed.to_parquet(tmp_path / "failed.parquet")

    imported, _ = retrieval.load_prior_reports(
        [tmp_path / "good.parquet", tmp_path / "failed.parquet"],
        set(labels.issue_key),
    )
    assert imported["MySQL:1"]["retrieval_status"] == "SUCCESS"
    assert imported["MySQL:1"]["initial_description"] == "Complete local evidence"


def test_only_project_limits_retrieval_to_mysql(tmp_path, monkeypatch):
    labels = label_table(("Linux", "MySQL"))
    labels.to_parquet(tmp_path / "labels.parquet")
    calls = []

    def retrieve(_self, project, bug_id, _retry=False):
        calls.append((project, bug_id))
        return {
            "retrieval_status": "SUCCESS",
            "retrieval_source": "AUTO_REMOTE",
            "tracker_url": "https://example.invalid/issue",
            "title": "Issue",
            "initial_description": "Evidence",
        }

    monkeypatch.setattr(retrieval.CachedRetriever, "retrieve", retrieve)
    audit = retrieval.enrich_mandelbugs(
        tmp_path / "labels.parquet",
        tmp_path / "mysql",
        tmp_path / "cache",
        manual_dir=None,
        only_project="MySQL",
    )
    assert calls == [("MySQL", "1"), ("MySQL", "2"), ("MySQL", "3")]
    assert audit["only_project"] == "MySQL"
    assert set(pd.read_parquet(tmp_path / "mysql/reports.parquet").project) == {"MySQL"}


def test_manual_upgrades_empty_but_does_not_degrade_richer_success():
    empty = {"retrieval_status": "EMPTY_CONTENT", "retrieval_source": "AUTO_REMOTE"}
    manual = {"initial_description": "Manual evidence", "title": "Manual"}
    assert retrieval.merge_manual(empty, manual)["retrieval_source"] == "MANUAL_IMPORT"
    rich = {
        "retrieval_status": "SUCCESS",
        "retrieval_source": "AUTO_REMOTE",
        "title": "Rich",
        "initial_description": "A substantially richer automatic description",
        "comments_text": "Useful follow-up",
        "environment_text": "Linux",
    }
    assert retrieval.merge_manual(rich, manual) == rich


def test_four_project_readiness_requires_both_classes():
    labels = label_table()
    reports = success_reports(labels)
    ready = retrieval.coverage_readiness(labels, reports)
    assert ready["recommended_for_four_project_lopo"] is True
    missing = retrieval.coverage_readiness(
        labels[labels.project.ne("HTTPD")], reports[reports.project.ne("HTTPD")]
    )
    assert missing["all_four_projects_present"] is False
    mysql_nam = labels.project.eq("MySQL") & labels.original_class.eq("NAM")
    only_boh = retrieval.coverage_readiness(
        labels, reports[~reports.issue_key.isin(labels[mysql_nam].issue_key)]
    )
    assert only_boh["both_stage2_classes_each_project"] is False
    assert only_boh["recommended_for_four_project_lopo"] is False
