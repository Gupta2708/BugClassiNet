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


MYSQL_ARCHIVED_PAGE = b"""
<html><head><title>MySQL Bugs: #21704: Renaming column</title></head>
<body><h1>Bug #21704: Renaming column</h1>
<table><tr><td>Submitted:</td><td>17 Aug 2006 22:21</td></tr>
<tr><td>OS:</td><td>Linux</td></tr><tr><td>CPU Architecture:</td><td>Any</td></tr></table>
<pre>Description:
Renaming a column does not update the foreign key definition.</pre>
<pre>A later public comment.</pre>
</body></html>
"""


def raw_response(status=200, content=b"", headers=None):
    return SimpleNamespace(status_code=status, content=content, headers=headers or {})


def test_web_archive_fallback_recovers_a_blocked_mysql_page(tmp_path):
    """A 403 from the origin falls back to the archive without retrying the origin."""
    session = RecordingSession(
        [
            raw_response(403, b"<html><title>Technical Difficulties</title></html>"),
            raw_response(200, json.dumps([["timestamp"], ["20080311100430"]]).encode()),
            raw_response(200, MYSQL_ARCHIVED_PAGE),
        ]
    )
    record = retrieval.CachedRetriever(
        tmp_path, session=session, sleeper=lambda _: None, allow_web_archive=True
    ).retrieve("MySQL", "21704")
    assert record["retrieval_status"] == "SUCCESS"
    assert record["retrieval_source"] == "WEB_ARCHIVE"
    assert record["archive_timestamp"] == "20080311100430"
    assert "foreign key definition" in record["initial_description"]
    # The blocked origin is contacted once, then never again.
    hosts = [url.split("/")[2] for url, _ in session.calls]
    assert hosts.count("bugs.mysql.com") == 1
    assert hosts.count(retrieval.WEB_ARCHIVE_HOST) == 2


def test_web_archive_is_opt_in_and_never_reached_by_default(tmp_path):
    session = RecordingSession([raw_response(403, b"<html><title>Denied</title></html>")])
    record = retrieval.CachedRetriever(tmp_path, session=session, sleeper=lambda _: None).retrieve(
        "MySQL", "21704"
    )
    assert record["retrieval_status"] == "AUTH_REQUIRED"
    assert all(retrieval.WEB_ARCHIVE_HOST not in url for url, _ in session.calls)


def test_archived_login_page_is_not_accepted_as_evidence(tmp_path):
    session = RecordingSession(
        [
            raw_response(403, b"<html><title>Denied</title></html>"),
            raw_response(200, json.dumps([["timestamp"], ["20080311100430"]]).encode()),
            raw_response(200, b"<html><title>Log in</title><body>Sign in</body></html>"),
        ]
    )
    record = retrieval.CachedRetriever(
        tmp_path, session=session, sleeper=lambda _: None, allow_web_archive=True
    ).retrieve("MySQL", "21704")
    assert record["retrieval_status"] != "SUCCESS"


def test_missing_snapshot_is_recorded_without_masking_the_origin_block(tmp_path):
    session = RecordingSession(
        [
            raw_response(403, b"<html><title>Denied</title></html>"),
            raw_response(200, b"[]"),
        ]
    )
    record = retrieval.CachedRetriever(
        tmp_path, session=session, sleeper=lambda _: None, allow_web_archive=True
    ).retrieve("MySQL", "21704")
    # The origin's own block is the more informative status and wins the merge,
    # but the archive outcome is still recorded for the audit.
    assert record["retrieval_status"] == "AUTH_REQUIRED"
    assert record["web_archive_status"] == "NOT_FOUND"


def test_blocked_status_is_never_masked_by_a_stale_not_found():
    """The regression that reported rate-limited HTTPD retrievals as NOT_FOUND."""
    stale = {"retrieval_status": "NOT_FOUND", "retrieval_source": "AUTO_REMOTE"}
    blocked = {"retrieval_status": "RATE_LIMITED", "retrieval_source": "AUTO_REMOTE"}
    # Order must not decide the outcome: the transient failure wins either way.
    assert retrieval.choose_best_record([stale, blocked])["retrieval_status"] == "RATE_LIMITED"
    assert retrieval.choose_best_record([blocked, stale])["retrieval_status"] == "RATE_LIMITED"


@pytest.mark.parametrize(
    "transient", ["RATE_LIMITED", "AUTH_REQUIRED", "AUTH_FAILED", "NETWORK_ERROR"]
)
def test_every_transient_failure_outranks_not_found(transient):
    stale = {"retrieval_status": "NOT_FOUND", "retrieval_source": "AUTO_REMOTE"}
    fresh = {"retrieval_status": transient, "retrieval_source": "AUTO_REMOTE"}
    assert retrieval.choose_best_record([stale, fresh])["retrieval_status"] == transient


def test_usable_evidence_still_beats_every_failure():
    good = {
        "retrieval_status": "SUCCESS",
        "retrieval_source": "WEB_ARCHIVE",
        "initial_description": "a real report body",
    }
    blocked = {"retrieval_status": "RATE_LIMITED", "retrieval_source": "AUTO_REMOTE"}
    assert retrieval.choose_best_record([blocked, good])["retrieval_status"] == "SUCCESS"


def bugzilla_payloads(returned_id, comment_key):
    bug = {"bugs": [{"id": returned_id, "summary": "Summary", "component": "core"}]}
    comments = {
        "bugs": {comment_key: {"comments": [{"count": 0, "id": 1, "text": "Initial report."}]}}
    }
    return bug, comments


def test_alias_or_moved_bug_is_kept_and_recorded():
    """A canonical id differing from the requested id is evidence, not a PARSE_ERROR."""
    bug, comments = bugzilla_payloads(9999, "9999")
    parsed = retrieval.parse_bugzilla_rest("1234", bug, comments)
    assert parsed["retrieval_status"] == "SUCCESS"
    assert parsed["initial_description"] == "Initial report."
    assert parsed["tracker_resolved_bug_id"] == "9999"
    assert parsed["tracker_id_substituted"] is True


def test_exact_id_match_is_not_flagged_as_substituted():
    bug, comments = bugzilla_payloads(1234, "1234")
    parsed = retrieval.parse_bugzilla_rest("1234", bug, comments)
    assert parsed["retrieval_status"] == "SUCCESS"
    assert parsed["tracker_id_substituted"] is False


def test_response_without_an_identifier_is_still_rejected():
    bug = {"bugs": [{"summary": "Summary"}]}
    with pytest.raises(ValueError):
        retrieval.parse_bugzilla_rest("1234", bug, {"bugs": {}})


def test_multiple_bugs_in_one_response_are_still_rejected():
    bug = {"bugs": [{"id": 1}, {"id": 2}]}
    with pytest.raises(ValueError):
        retrieval.parse_bugzilla_rest("1", bug, {"bugs": {}})


LOGIN_CAPTURE = b"<html><title>Log in to MySQL Bugs</title><body>Please log in</body></html>"


def test_snapshot_walk_skips_login_captures_and_keeps_the_first_real_report(tmp_path):
    """The recovery for captures where the crawler hit a login wall."""
    index = json.dumps(
        [["timestamp"], ["20080728061450"], ["20080801044712"], ["20080928070640"]]
    ).encode()
    session = RecordingSession(
        [
            raw_response(403, b"<html><title>Technical Difficulties</title></html>"),
            raw_response(200, index),
            raw_response(200, LOGIN_CAPTURE),
            raw_response(200, LOGIN_CAPTURE),
            raw_response(200, MYSQL_ARCHIVED_PAGE),
        ]
    )
    record = retrieval.CachedRetriever(
        tmp_path, session=session, sleeper=lambda _: None, allow_web_archive=True
    ).retrieve("MySQL", "38185")
    assert record["retrieval_status"] == "SUCCESS"
    # The capture that actually parsed is the one recorded, not the earliest.
    assert record["archive_timestamp"] == "20080928070640"
    assert "foreign key definition" in record["initial_description"]
    attempts = json.loads(record["web_archive_attempts"])
    assert [item["status"] for item in attempts] == [
        "EMPTY_CONTENT",
        "EMPTY_CONTENT",
        "SUCCESS",
    ]


def test_snapshot_walk_is_bounded_and_reports_exhaustion(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "WEB_ARCHIVE_MAX_SNAPSHOTS", 3)
    index = json.dumps([["timestamp"], ["1"], ["2"], ["3"], ["4"], ["5"]]).encode()
    session = RecordingSession(
        [
            raw_response(403, b"<html><title>Denied</title></html>"),
            raw_response(200, index),
            *[raw_response(200, LOGIN_CAPTURE) for _ in range(3)],
        ]
    )
    record = retrieval.CachedRetriever(
        tmp_path, session=session, sleeper=lambda _: None, allow_web_archive=True
    ).retrieve("MySQL", "38185")
    assert record["web_archive_status"] == "EMPTY_CONTENT"
    # Exactly the bounded number of captures is fetched, never all of them.
    assert len(json.loads(record["web_archive_attempts"])) == 3


def test_snapshot_walk_stops_immediately_when_the_archive_rate_limits(tmp_path):
    index = json.dumps([["timestamp"], ["1"], ["2"], ["3"]]).encode()
    session = RecordingSession(
        [
            raw_response(403, b"<html><title>Denied</title></html>"),
            raw_response(200, index),
            raw_response(200, LOGIN_CAPTURE),
            raw_response(429, b"", {"Retry-After": "60"}),
        ]
    )
    record = retrieval.CachedRetriever(
        tmp_path, session=session, sleeper=lambda _: None, allow_web_archive=True
    ).retrieve("MySQL", "38185")
    # Rate limiting is not evidence that the remaining captures are unusable.
    assert record["web_archive_status"] == "RATE_LIMITED"
    assert len(json.loads(record["web_archive_attempts"])) == 1
