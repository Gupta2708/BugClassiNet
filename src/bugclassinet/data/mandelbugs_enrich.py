"""Polite, cached historical issue retrieval and manual-import fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from bugclassinet.data.mandelbugs_schema import (
    TEXT_VERSION,
    bug_identifier,
    construct_evidence,
    issue_key,
    project_name,
    string_value,
    validate_labels,
)
from bugclassinet.utils.checksums import sha256_file
from bugclassinet.utils.io import write_json
from bugclassinet.utils.reproducibility import git_commit, utc_timestamp

LOGGER = logging.getLogger(__name__)
USER_AGENT = (
    "BugClassiNet-Next/0.1 (academic Mandelbugs issue-text retrieval; cached, rate-limited)"
)
STATUSES = {
    "SUCCESS",
    "NOT_FOUND",
    "AUTH_REQUIRED",
    "AUTH_FAILED",
    "RATE_LIMITED",
    "NETWORK_ERROR",
    "PARSE_ERROR",
    "UNSUPPORTED",
    "EMPTY_CONTENT",
}
TEXT_FIELDS = ("title", "initial_description", "comments_text", "environment_text")
PARSER_VERSION = "historical-tracker-v2"
APACHE_API_KEY_ENV = "BUGCLASSINET_APACHE_BUGZILLA_API_KEY"
# ASF does not expose the optional /rest rewrite: that path returns an HTML
# 404. Use Bugzilla's native REST script entry point for this installation.
APACHE_REST_ROOT = "https://bz.apache.org/bugzilla/rest.cgi"
SAFE_TRACKER_METADATA = ("tracker_component", "tracker_operating_system", "tracker_platform")


def tracker_url(project: str, bug_id: str) -> str:
    bug_id = bug_identifier(project, bug_id)
    templates = {
        "Linux": "https://bugzilla.kernel.org/show_bug.cgi?id={}",
        "MySQL": "https://bugs.mysql.com/bug.php?id={}",
        "HTTPD": "https://bz.apache.org/bugzilla/show_bug.cgi?id={}",
        "AXIS": "https://issues.apache.org/jira/browse/{}",
    }
    if project not in templates:
        raise ValueError(f"Unsupported tracker project: {project}")
    return templates[project].format(bug_id)


def _text(node: Any) -> str:
    return node.get_text("\n", strip=True) if node is not None else ""


def _mysql_field(soup: BeautifulSoup, label: str) -> str:
    """Read one historical MySQL metadata cell without copying the whole table."""
    wanted = label.casefold().rstrip(":")
    for row in soup.select("tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        for index, cell in enumerate(cells[:-1]):
            if _text(cell).casefold().rstrip(":") == wanted:
                return _text(cells[index + 1])
    return ""


def parse_tracker_html(project: str, content: bytes) -> dict[str, Any]:
    """Extract allowlisted content nodes, never entire pages or metadata tables."""
    soup = BeautifulSoup(content, "html.parser")
    page_title = _text(soup.title).casefold()
    looks_blocked = any(
        word in page_title
        for word in (
            "access denied",
            "verify you are human",
            "making sure you're not a bot",
            "just a moment",
            "sign in",
            "log in",
            "authentication",
        )
    )
    # Real bug summaries can themselves discuss authentication/login failures.
    has_issue_evidence = soup.select_one(".bz_comment_text, #description-val") is not None
    if (looks_blocked and not has_issue_evidence) or soup.select_one(
        "#challenge-form, #anubis_challenge, .g-recaptcha"
    ):
        return {"retrieval_status": "AUTH_REQUIRED", "error": "Authentication or anti-bot page"}
    if any(word in page_title for word in ("not found", "invalid bug", "does not exist")):
        return {"retrieval_status": "NOT_FOUND", "error": page_title}
    title = description = environment = created = ""
    comments: list[str] = []
    if project in {"Linux", "HTTPD"}:
        title = _text(soup.select_one("#short_desc_nonedit_display, #short_desc"))
        descriptions = soup.select(".bz_comment_text")
        if descriptions:
            description = _text(descriptions[0])
            comments = [_text(node) for node in descriptions[1:]]
        created = _text(soup.select_one(".bz_comment_time"))
    elif project == "AXIS":
        title = _text(soup.select_one("#summary-val"))
        description = _text(soup.select_one("#description-val"))
        environment = _text(soup.select_one("#environment-val"))
        comments = [_text(node) for node in soup.select(".issue-data-block .action-body")]
        created_node = soup.select_one("#created-val time")
        created = created_node.get("datetime", "") if created_node else ""
    elif project == "MySQL":
        # MySQL's report content is held in preformatted comment blocks. Only
        # the first block is the initial report, never the status/resolution table.
        heading = _text(soup.select_one("h1"))
        if ":" in heading:
            title = heading.split(":", 1)[1].strip()
        if not title:
            match = re.search(r"MySQL Bugs:\s*#\d+:\s*(.+)", _text(soup.title), re.I)
            title = match.group(1).strip() if match else ""
        descriptions = soup.select(".comment pre, #bugpage pre, pre")
        if descriptions:
            description = _text(descriptions[0])
            comments = [_text(node) for node in descriptions[1:]]
        if not title:
            for row in soup.select("tr"):
                cells = row.find_all(["td", "th"], recursive=False)
                if len(cells) >= 2 and _text(cells[0]).rstrip(":").lower() in {
                    "synopsis",
                    "summary",
                }:
                    title = _text(cells[1])
        operating_system = _mysql_field(soup, "OS")
        hardware = _mysql_field(soup, "CPU Architecture")
        environment = "\n".join(
            value
            for value in (
                f"Operating system: {operating_system}" if operating_system else "",
                f"Hardware: {hardware}" if hardware else "",
            )
            if value
        )
        created = _mysql_field(soup, "Submitted")
    else:
        return {"retrieval_status": "UNSUPPORTED", "error": f"No parser for {project}"}
    if not title and not description:
        return {"retrieval_status": "PARSE_ERROR", "error": "Recognized issue content nodes absent"}
    if not description.strip():
        return {
            "retrieval_status": "EMPTY_CONTENT",
            "error": "No initial description extracted",
            "title": title,
        }
    result = {
        "retrieval_status": "SUCCESS",
        "error": "",
        "title": title,
        "initial_description": description,
        "comments_text": "\n\n".join(comments),
        "environment_text": environment,
        "created_at": created,
    }
    if project == "MySQL":
        result.update(
            tracker_operating_system=operating_system,
            tracker_platform=hardware,
        )
    return result


def parse_bugzilla_rest(
    expected_bug_id: str, bug_payload: dict[str, Any], comments_payload: dict[str, Any]
) -> dict[str, Any]:
    """Parse documented Bugzilla bug/comment responses, excluding private comments."""
    bugs = bug_payload.get("bugs")
    if not isinstance(bugs, list) or len(bugs) != 1 or str(bugs[0].get("id")) != expected_bug_id:
        raise ValueError("Bugzilla bug response did not contain the requested issue")
    bug = bugs[0]
    comment_group = comments_payload.get("bugs", {}).get(expected_bug_id, {})
    comments = comment_group.get("comments")
    if not isinstance(comments, list):
        raise ValueError("Bugzilla comments response has an unexpected schema")
    public = [item for item in comments if not item.get("is_private", False)]
    public.sort(key=lambda item: (int(item.get("count", 0)), int(item.get("id", 0))))
    descriptions = [string_value(item.get("text")).strip() for item in public]
    initial = next(
        (text for item, text in zip(public, descriptions, strict=True) if item.get("count") == 0),
        "",
    )
    later = [
        text
        for item, text in zip(public, descriptions, strict=True)
        if item.get("count") != 0 and text
    ]
    operating_system = string_value(bug.get("op_sys")).strip()
    platform = string_value(bug.get("platform")).strip()
    environment = "\n".join(
        value
        for value in (
            f"Operating system: {operating_system}" if operating_system else "",
            f"Hardware: {platform}" if platform else "",
        )
        if value
    )
    title = string_value(bug.get("summary")).strip()
    if not title and not initial:
        return {"retrieval_status": "PARSE_ERROR", "error": "Bugzilla content fields absent"}
    if not initial:
        return {
            "retrieval_status": "EMPTY_CONTENT",
            "error": "No public initial description returned",
            "title": title,
        }
    return {
        "retrieval_status": "SUCCESS",
        "error": "",
        "title": title,
        "initial_description": initial,
        "comments_text": "\n\n".join(later),
        "environment_text": environment,
        "created_at": string_value(bug.get("creation_time")),
        "tracker_component": string_value(bug.get("component")),
        "tracker_operating_system": operating_system,
        "tracker_platform": platform,
    }


def _bugzilla_payload_failure(payload: dict[str, Any]) -> dict[str, Any]:
    """Map documented JSON error objects without retaining server/credential text."""
    try:
        code = int(payload.get("code", -1))
    except (TypeError, ValueError):
        code = -1
    if code == 101:  # Bugzilla WebService: Invalid Bug ID.
        return {
            "retrieval_status": "NOT_FOUND",
            "error": "Bugzilla API reports that the issue does not exist",
        }
    return {
        "retrieval_status": "AUTH_FAILED",
        "error": "Bugzilla API rejected the authenticated request",
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".json.tmp")
    write_json(temporary, value)
    temporary.replace(path)


class CachedRetriever:
    """Serial retrieval; 401/403/429 stop further requests to that host in this run."""

    def __init__(
        self,
        cache_dir: str | Path,
        sleep_seconds: float = 1.0,
        retries: int = 3,
        session: Any = None,
        sleeper: Any = time.sleep,
    ) -> None:
        if sleep_seconds < 0 or retries < 1:
            raise ValueError("sleep_seconds must be nonnegative and retries positive")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.delay, self.retries = sleep_seconds, retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.sleep = sleeper
        self.blocked: dict[str, str] = {}

    def retrieve(self, project: str, bug_id: str, retry_failures: bool = False) -> dict[str, Any]:
        url = tracker_url(project, bug_id)
        token = hashlib.sha256(url.encode()).hexdigest()
        metadata_path = self.cache / f"{token}.json"
        if metadata_path.exists():
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            if prior.get("tracker_url") != url:
                raise ValueError("Cache URL identity mismatch")
            has_new_httpd_key = project == "HTTPD" and bool(os.environ.get(APACHE_API_KEY_ENV))
            should_retry_auth = has_new_httpd_key and prior["retrieval_status"] == "AUTH_REQUIRED"
            if prior["retrieval_status"] == "SUCCESS" or (
                not retry_failures and not should_retry_auth
            ):
                self._validate_raw_cache(prior, token)
                return prior
        if project == "HTTPD":
            result = self._retrieve_httpd(bug_identifier(project, bug_id), url, token)
        else:
            result = self._retrieve_html(project, url, token)
        assert result["retrieval_status"] in STATUSES
        _atomic_json(metadata_path, result)
        return result

    def _validate_raw_cache(self, record: dict[str, Any], token: str) -> None:
        raw_responses = record.get("raw_responses", [])
        for item in raw_responses:
            path = self.cache / item["file"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise ValueError(f"Corrupt raw response cache: {path}")
        # Backward compatibility with the first HTML cache format.
        if record.get("raw_sha256"):
            path = self.cache / f"{token}.html"
            if not path.is_file() or sha256_file(path) != record["raw_sha256"]:
                raise ValueError(f"Corrupt raw response cache: {path}")

    def _base_result(self, url: str) -> dict[str, Any]:
        return {
            "tracker_url": url,
            "retrieval_source": "AUTO_REMOTE",
            "retrieved_at": utc_timestamp(),
            "parser_version": PARSER_VERSION,
        }

    def _retrieve_html(self, project: str, url: str, token: str) -> dict[str, Any]:
        raw_path = self.cache / f"{token}.html"
        host = urlparse(url).netloc
        result = self._base_result(url)
        if host in self.blocked:
            result.update(
                retrieval_status=self.blocked[host],
                error="Host circuit open; use manual import or retry later",
            )
        else:
            for attempt in range(self.retries):
                self.sleep(self.delay if attempt == 0 else max(self.delay, 2**attempt))
                try:
                    response = self.session.get(url, timeout=(10, 45), allow_redirects=False)
                    result["http_status"] = response.status_code
                    if response.status_code in (401, 403, 429):
                        status = "RATE_LIMITED" if response.status_code == 429 else "AUTH_REQUIRED"
                        self.blocked[host] = status
                        result.update(
                            retrieval_status=status,
                            error=f"HTTP {response.status_code}; no bypass attempted",
                            retry_after=response.headers.get("Retry-After"),
                        )
                        break
                    if response.status_code in (404, 410):
                        result.update(
                            retrieval_status="NOT_FOUND", error=f"HTTP {response.status_code}"
                        )
                        break
                    if 300 <= response.status_code < 400:
                        result.update(
                            retrieval_status="UNSUPPORTED",
                            error="Redirect requires manual verification",
                            redirect_location=response.headers.get("Location"),
                        )
                        break
                    if response.status_code >= 500:
                        result.update(
                            retrieval_status="NETWORK_ERROR", error=f"HTTP {response.status_code}"
                        )
                        continue
                    if response.status_code != 200:
                        result.update(
                            retrieval_status="NETWORK_ERROR", error=f"HTTP {response.status_code}"
                        )
                        break
                    # Cache every HTTP-success response, including parse failures.
                    temporary = raw_path.with_suffix(".html.tmp")
                    temporary.write_bytes(response.content)
                    temporary.replace(raw_path)
                    result.update(raw_sha256=sha256_file(raw_path), raw_cache_file=raw_path.name)
                    result.update(parse_tracker_html(project, response.content))
                    if result["retrieval_status"] == "AUTH_REQUIRED":
                        self.blocked[host] = "AUTH_REQUIRED"
                    break
                except requests.RequestException as error:
                    result.update(retrieval_status="NETWORK_ERROR", error=str(error))
        return result

    def _api_request(
        self, endpoint: str, api_key: str, raw_path: Path
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Request one API resource; never return or persist credential-bearing URLs."""
        host = urlparse(endpoint).netloc
        if host in self.blocked:
            return None, {
                "retrieval_status": self.blocked[host],
                "error": "Host circuit open; use a valid API key or retry later",
            }
        for attempt in range(self.retries):
            self.sleep(self.delay if attempt == 0 else max(self.delay, 2**attempt))
            try:
                response = self.session.get(
                    endpoint,
                    params={"Bugzilla_api_key": api_key},
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    timeout=(10, 45),
                    allow_redirects=False,
                )
            except requests.RequestException:
                if attempt + 1 == self.retries:
                    return None, {
                        "retrieval_status": "NETWORK_ERROR",
                        "error": "Network request failed",
                    }
                continue
            status = response.status_code
            if status in (401, 403):
                self.blocked[host] = "AUTH_FAILED"
                return None, {
                    "retrieval_status": "AUTH_FAILED",
                    "error": f"Bugzilla API authentication failed (HTTP {status})",
                    "http_status": status,
                }
            if status == 429:
                self.blocked[host] = "RATE_LIMITED"
                return None, {
                    "retrieval_status": "RATE_LIMITED",
                    "error": "Bugzilla API rate limited the request",
                    "http_status": status,
                    "retry_after": response.headers.get("Retry-After"),
                }
            if status in (404, 410):
                return None, {
                    "retrieval_status": "NOT_FOUND",
                    "error": f"Bugzilla API returned HTTP {status}",
                    "http_status": status,
                }
            if 300 <= status < 400:
                return None, {
                    "retrieval_status": "UNSUPPORTED",
                    "error": "Bugzilla API redirect requires manual verification",
                    "http_status": status,
                }
            if status >= 500:
                if attempt + 1 < self.retries:
                    continue
                return None, {
                    "retrieval_status": "NETWORK_ERROR",
                    "error": f"Bugzilla API returned HTTP {status}",
                    "http_status": status,
                }
            if status != 200:
                return None, {
                    "retrieval_status": "NETWORK_ERROR",
                    "error": f"Bugzilla API returned HTTP {status}",
                    "http_status": status,
                }
            if api_key.encode() in response.content:
                return None, {
                    "retrieval_status": "AUTH_FAILED",
                    "error": "Bugzilla response rejected by credential-safety check",
                    "http_status": status,
                }
            temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(raw_path)
            return response, None
        raise AssertionError("unreachable")

    def _retrieve_httpd(self, bug_id: str, tracker: str, token: str) -> dict[str, Any]:
        result = self._base_result(tracker)
        result.update(
            authentication="api_key_environment",
            credential_supplied=bool(os.environ.get(APACHE_API_KEY_ENV)),
        )
        api_key = os.environ.get(APACHE_API_KEY_ENV, "")
        if not api_key:
            result.update(
                retrieval_status="AUTH_REQUIRED",
                error=f"Set {APACHE_API_KEY_ENV} for documented Apache Bugzilla API access",
            )
            return result
        bug_endpoint = f"{APACHE_REST_ROOT}/bug/{bug_id}"
        comments_endpoint = f"{bug_endpoint}/comment"
        bug_path = self.cache / f"{token}.bug.json"
        comments_path = self.cache / f"{token}.comments.json"
        bug_response, failure = self._api_request(bug_endpoint, api_key, bug_path)
        if failure:
            result.update(failure)
            return result
        result["raw_responses"] = [
            {"kind": "bug", "file": bug_path.name, "sha256": sha256_file(bug_path)}
        ]
        try:
            bug_payload = json.loads(bug_response.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            result.update(
                retrieval_status="PARSE_ERROR", error="Bugzilla API returned invalid JSON"
            )
            return result
        if bug_payload.get("error"):
            failure = _bugzilla_payload_failure(bug_payload)
            if failure["retrieval_status"] == "AUTH_FAILED":
                self.blocked[urlparse(bug_endpoint).netloc] = "AUTH_FAILED"
            result.update(failure)
            return result
        comments_response, failure = self._api_request(comments_endpoint, api_key, comments_path)
        if failure:
            result.update(failure)
            return result
        result["raw_responses"].append(
            {"kind": "comments", "file": comments_path.name, "sha256": sha256_file(comments_path)}
        )
        result["api_endpoints"] = [bug_endpoint, comments_endpoint]
        try:
            comments_payload = json.loads(comments_response.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            result.update(
                retrieval_status="PARSE_ERROR", error="Bugzilla API returned invalid JSON"
            )
            return result
        if comments_payload.get("error"):
            failure = _bugzilla_payload_failure(comments_payload)
            if failure["retrieval_status"] == "AUTH_FAILED":
                self.blocked[urlparse(comments_endpoint).netloc] = "AUTH_FAILED"
            result.update(failure)
            return result
        try:
            result.update(parse_bugzilla_rest(bug_id, bug_payload, comments_payload))
        except (TypeError, ValueError, KeyError):
            result.update(
                retrieval_status="PARSE_ERROR", error="Unexpected Bugzilla API response schema"
            )
        return result


def load_manual_imports(directory: str | Path | None) -> tuple[dict[str, dict], dict[str, str]]:
    records: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    if directory is None or not Path(directory).exists():
        return records, hashes
    root = Path(directory)
    for source in sorted(root.rglob("*")):
        if source.suffix.lower() not in {".csv", ".parquet"}:
            continue
        frame = (
            pd.read_csv(source, dtype=str, keep_default_na=False)
            if source.suffix.lower() == ".csv"
            else pd.read_parquet(source)
        )
        required = {"project", "bug_id", "tracker_url", *TEXT_FIELDS}
        if missing := required - set(frame):
            raise ValueError(f"Manual file {source} missing {sorted(missing)}")
        source_name = source.relative_to(root).as_posix()
        hashes[source_name] = sha256_file(source)
        for row in frame.to_dict("records"):
            project = project_name(row["project"])
            key = issue_key(project, row["bug_id"])
            record = {
                field: string_value(row.get(field))
                for field in (*TEXT_FIELDS, "tracker_url", "created_at")
            }
            record["manual_source_file"] = source_name
            record["manual_source_sha256"] = hashes[source_name]
            if not record["initial_description"].strip():
                raise ValueError(f"Manual import has empty description: {key}")
            if key in records and records[key] != record:
                raise ValueError(f"Conflicting manual imports for {key}")
            records[key] = record
    return records, hashes


def merge_manual(existing: dict[str, Any], manual: dict[str, Any] | None) -> dict[str, Any]:
    """Backward-compatible helper using the same content-quality merge policy."""
    if manual is None:
        return existing
    candidate = {
        **manual,
        "retrieval_status": "SUCCESS",
        "retrieval_source": "MANUAL_IMPORT",
        "retrieved_at": utc_timestamp(),
        "error": "",
    }
    return choose_best_record([existing, candidate])


def _usable(record: dict[str, Any]) -> bool:
    return record.get("retrieval_status") == "SUCCESS" and bool(
        string_value(record.get("initial_description")).strip()
    )


def _quality(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    status_rank = {
        "SUCCESS": 5,
        "EMPTY_CONTENT": 4,
        "AUTH_REQUIRED": 3,
        "AUTH_FAILED": 3,
        "NOT_FOUND": 2,
        "RATE_LIMITED": 2,
        "NETWORK_ERROR": 1,
        "PARSE_ERROR": 1,
        "UNSUPPORTED": 0,
    }.get(record.get("retrieval_status"), -1)
    values = [string_value(record.get(field)).strip() for field in TEXT_FIELDS]
    source_rank = {
        "AUTO_REMOTE": 4,
        "PRIOR_REPORT": 3,
        "MANUAL_IMPORT": 2,
        "AUTO": 1,
    }.get(record.get("retrieval_source"), 0)
    return (
        _usable(record),
        status_rank,
        sum(bool(value) for value in values),
        # Avoid replacing a stable success for trivial formatting differences.
        # A new candidate must add fields or a substantial block of content.
        sum(map(len, values)) // 256,
        source_rank,
    )


def choose_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose usable/richer evidence first, with source only as deterministic tie-breaker."""
    if not records:
        raise ValueError("At least one retrieval candidate is required")
    return max(records, key=_quality).copy()


def _safe_import_record(row: dict[str, Any], source: str) -> dict[str, Any]:
    record = {
        field: string_value(row.get(field))
        for field in (*TEXT_FIELDS, "created_at", "tracker_url", *SAFE_TRACKER_METADATA)
    }
    record.update(
        retrieval_status=string_value(row.get("retrieval_status")),
        retrieval_source=source,
        upstream_retrieval_source=string_value(row.get("retrieval_source")),
        retrieved_at=string_value(row.get("retrieved_at")),
        error=string_value(row.get("error")),
    )
    return record


def _preserve_safe_provenance(record: dict[str, Any], row: dict[str, Any]) -> None:
    """Retain only portable provenance from this tool's own existing output."""
    for field in (
        "prior_report_file",
        "prior_report_sha256",
        "manual_source_file",
        "manual_source_sha256",
        "upstream_retrieval_source",
    ):
        value = string_value(row.get(field))
        if value:
            record[field] = value


def load_prior_reports(
    paths: list[str | Path] | None, valid_issue_keys: set[str]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    audits = []
    for source in paths or []:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Prior reports file does not exist: {path}")
        frame = pd.read_parquet(path)
        required = {"project", "bug_id", "retrieval_status", *TEXT_FIELDS}
        if missing := required - set(frame):
            raise ValueError(f"Prior reports {path.name} missing columns: {sorted(missing)}")
        digest = sha256_file(path)
        imported = 0
        for row in frame.to_dict("records"):
            project = project_name(row["project"])
            key = issue_key(project, row["bug_id"])
            if key not in valid_issue_keys or string_value(row["retrieval_status"]) != "SUCCESS":
                continue
            candidate = _safe_import_record(row, "PRIOR_REPORT")
            if not _usable(candidate):
                continue
            candidate.update(prior_report_file=path.name, prior_report_sha256=digest)
            records[key] = (
                choose_best_record([records[key], candidate]) if key in records else candidate
            )
            imported += 1
        audits.append(
            {
                "file_name": path.name,
                "sha256": digest,
                "rows": len(frame),
                "successful_rows_eligible": imported,
                "reused_successful_records": 0,
            }
        )
    return records, audits


def coverage_readiness(
    labels: pd.DataFrame, reports: pd.DataFrame, prepared: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Summarize per-project usable evidence; readiness requires two classes in every project."""
    unique_reports = reports.sort_values("issue_key", kind="stable").drop_duplicates("issue_key")
    success = unique_reports[
        unique_reports.retrieval_status.eq("SUCCESS")
        & unique_reports.initial_description.fillna("").str.strip().ne("")
    ]
    if prepared is None:
        joined = labels.merge(success[["issue_key"]], on="issue_key", how="inner")
        conflicts = joined.groupby("issue_key").original_class.transform("nunique") > 1
        usable = joined[~conflicts & joined.original_class.isin(["BOH", "NAM", "ARB"])]
        usable = usable.drop_duplicates("issue_key")
    else:
        usable = prepared.drop_duplicates("issue_key")
    rows = []
    for project in ("Linux", "MySQL", "HTTPD", "AXIS"):
        source = labels[labels.project.eq(project)]
        project_success = success[success.project.eq(project)]
        project_usable = usable[usable.project.eq(project)]
        counts = project_usable.original_class.value_counts()
        unique_ids = source.issue_key.nunique()
        rows.append(
            {
                "project": project,
                "source_rows": len(source),
                "unique_issue_ids": unique_ids,
                "successful_reports": project_success.issue_key.nunique(),
                "classified_successful_reports": project_usable.issue_key.nunique(),
                "BOH_usable": int(counts.get("BOH", 0)),
                "NAM_usable": int(counts.get("NAM", 0)),
                "ARB_usable": int(counts.get("ARB", 0)),
                "MANDELBUG_usable": int(counts.get("NAM", 0) + counts.get("ARB", 0)),
                "UNK_count": int(source.original_class.eq("UNK").sum()),
                "coverage_rate": (
                    float(project_success.issue_key.nunique() / unique_ids) if unique_ids else 0.0
                ),
            }
        )
    all_present = all(row["unique_issue_ids"] > 0 for row in rows)
    both_classes = all(row["BOH_usable"] > 0 and row["MANDELBUG_usable"] > 0 for row in rows)
    return {
        "projects": rows,
        "all_four_projects_present": all_present,
        "both_stage2_classes_each_project": both_classes,
        "recommended_for_four_project_lopo": all_present and both_classes,
    }


def enrich_mandelbugs(
    labels_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    sleep_seconds: float = 1.0,
    manual_dir: str | Path | None = "data/manual_enrichment",
    retry_failures: bool = False,
    offline: bool = False,
    prior_reports: list[str | Path] | None = None,
    only_issue: str | None = None,
    only_project: str | None = None,
) -> dict[str, Any]:
    all_labels = validate_labels(pd.read_parquet(labels_path))
    labels = all_labels
    if only_issue and only_project:
        raise ValueError("--only-issue and --only-project cannot be used together")
    if only_project:
        selected_project = project_name(only_project)
        labels = all_labels[all_labels.project.eq(selected_project)].copy()
        if labels.empty:
            raise ValueError(f"Selected project is absent from official labels: {selected_project}")
    if only_issue:
        if ":" not in only_issue:
            raise ValueError("--only-issue must use PROJECT:BUG_ID, for example MySQL:21704")
        project, bug_id = only_issue.split(":", 1)
        selected_key = issue_key(project_name(project), bug_id)
        labels = all_labels[all_labels.issue_key.eq(selected_key)].copy()
        if labels.empty:
            raise ValueError(f"Selected issue is absent from official labels: {selected_key}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manual, manual_hashes = load_manual_imports(manual_dir)
    extraneous = set(manual) - set(all_labels.issue_key)
    if extraneous:
        raise ValueError(f"Manual issue IDs absent from official labels: {sorted(extraneous)}")
    imported, prior_audits = load_prior_reports(prior_reports, set(all_labels.issue_key))
    retriever = CachedRetriever(cache_dir, sleep_seconds)
    existing: dict[str, dict[str, Any]] = {}
    if (out / "reports.parquet").exists():
        prior = pd.read_parquet(out / "reports.parquet")
        if (only_issue or only_project) and set(prior.issue_key) - set(labels.issue_key):
            raise ValueError("Use a separate output directory for filtered retrieval runs")
        for row in prior.to_dict("records"):
            key = string_value(row.get("issue_key"))
            if key in set(all_labels.issue_key):
                candidate = _safe_import_record(row, string_value(row.get("retrieval_source")))
                _preserve_safe_provenance(candidate, row)
                existing[key] = (
                    choose_best_record([existing[key], candidate]) if key in existing else candidate
                )
    results = {}
    for row in labels.drop_duplicates("issue_key").to_dict("records"):
        key = row["issue_key"]
        candidates = [record for record in (existing.get(key), imported.get(key)) if record]
        if key in manual:
            candidates.append(
                {
                    **manual[key],
                    "retrieval_status": "SUCCESS",
                    "retrieval_source": "MANUAL_IMPORT",
                    "retrieved_at": utc_timestamp(),
                    "error": "",
                }
            )
        cache_file = Path(cache_dir) / (
            hashlib.sha256(tracker_url(row["project"], row["bug_id"]).encode()).hexdigest()
            + ".json"
        )
        # A legitimate prior/manual/current success prevents unnecessary network access.
        if not any(_usable(candidate) for candidate in candidates):
            if cache_file.exists() or not offline:
                candidates.append(
                    retriever.retrieve(
                        row["project"], row["bug_id"], retry_failures and not offline
                    )
                )
            else:
                candidates.append(
                    {
                        "retrieval_status": "UNSUPPORTED",
                        "retrieval_source": "AUTO_REMOTE",
                        "tracker_url": tracker_url(row["project"], row["bug_id"]),
                        "error": "No cached response; offline/manual mode",
                    }
                )
        result = choose_best_record(candidates)
        sources = sorted(
            {string_value(candidate.get("retrieval_source")) for candidate in candidates}
        )
        result["merge_candidate_sources"] = "|".join(filter(None, sources))
        for field in (*TEXT_FIELDS, "created_at", "error"):
            result.setdefault(field, "")
        initial, full = construct_evidence(result)
        result.update(
            text_initial=initial,
            text_full=full,
            content_sha256=hashlib.sha256(initial.encode()).hexdigest(),
            full_content_sha256=hashlib.sha256(full.encode()).hexdigest(),
        )
        results[key] = result
        if result.get("retrieval_source") == "PRIOR_REPORT":
            for item in prior_audits:
                if item["sha256"] == result.get("prior_report_sha256"):
                    item["reused_successful_records"] += 1
        LOGGER.info(
            "Retrieved %s status=%s source=%s",
            key,
            result["retrieval_status"],
            result["retrieval_source"],
        )
    reports = pd.DataFrame(
        [{**row, **results[row["issue_key"]]} for row in labels.to_dict("records")]
    )
    temporary = out / "reports.parquet.tmp"
    reports.to_parquet(temporary, index=False)
    temporary.replace(out / "reports.parquet")
    failures = reports[reports.retrieval_status != "SUCCESS"]
    failures[["issue_key", "project", "bug_id", "tracker_url", "retrieval_status", "error"]].to_csv(
        out / "retrieval_failures.csv", index=False
    )
    reports.groupby(["project", "retrieval_status"]).size().reset_index(name="rows").to_csv(
        out / "retrieval_by_project.csv", index=False
    )
    reports[["issue_key", "content_sha256", "full_content_sha256"]].to_csv(
        out / "content_hashes.csv", index=False
    )
    readiness = coverage_readiness(labels, reports)
    pd.DataFrame(readiness["projects"]).to_csv(out / "stage2_readiness.csv", index=False)
    write_json(out / "stage2_readiness.json", readiness)
    audit = {
        "rows": len(reports),
        "unique_issues": len(results),
        "status_counts": reports.retrieval_status.value_counts().to_dict(),
        "unique_status_counts": pd.Series([r["retrieval_status"] for r in results.values()])
        .value_counts()
        .to_dict(),
        "labels_sha256": sha256_file(labels_path),
        "manual_files": manual_hashes,
        "prior_reports": prior_audits,
        "only_issue": only_issue,
        "only_project": only_project,
        "readiness": readiness,
        "text_version": TEXT_VERSION,
        "parser_version": PARSER_VERSION,
        "timestamp": utc_timestamp(),
        "git_commit": git_commit(),
        "training_invoked": False,
    }
    write_json(out / "retrieval_audit.json", audit)
    return audit
