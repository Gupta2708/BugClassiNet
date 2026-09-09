"""Polite, cached historical issue retrieval and manual-import fallback."""

from __future__ import annotations

import hashlib
import json
import logging
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
    "RATE_LIMITED",
    "NETWORK_ERROR",
    "PARSE_ERROR",
    "UNSUPPORTED",
    "EMPTY_CONTENT",
}
TEXT_FIELDS = ("title", "initial_description", "comments_text", "environment_text")
PARSER_VERSION = "historical-tracker-html-v1"


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
    return {
        "retrieval_status": "SUCCESS",
        "error": "",
        "title": title,
        "initial_description": description,
        "comments_text": "\n\n".join(comments),
        "environment_text": environment,
        "created_at": created,
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
        raw_path = self.cache / f"{token}.html"
        if metadata_path.exists():
            prior = json.loads(metadata_path.read_text(encoding="utf-8"))
            if prior.get("tracker_url") != url:
                raise ValueError("Cache URL identity mismatch")
            if prior["retrieval_status"] == "SUCCESS" or not retry_failures:
                if prior.get("raw_sha256") and (
                    not raw_path.exists() or sha256_file(raw_path) != prior["raw_sha256"]
                ):
                    raise ValueError(f"Corrupt raw response cache: {raw_path}")
                return prior
        host = urlparse(url).netloc
        result: dict[str, Any] = {
            "tracker_url": url,
            "retrieval_source": "AUTO",
            "retrieved_at": utc_timestamp(),
            "parser_version": PARSER_VERSION,
        }
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
                    result.update(
                        raw_sha256=sha256_file(raw_path), raw_cache_file=str(raw_path.resolve())
                    )
                    result.update(parse_tracker_html(project, response.content))
                    if result["retrieval_status"] == "AUTH_REQUIRED":
                        self.blocked[host] = "AUTH_REQUIRED"
                    break
                except requests.RequestException as error:
                    result.update(retrieval_status="NETWORK_ERROR", error=str(error))
        assert result["retrieval_status"] in STATUSES
        _atomic_json(metadata_path, result)
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
        hashes[str(source.resolve())] = sha256_file(source)
        for row in frame.to_dict("records"):
            project = project_name(row["project"])
            key = issue_key(project, row["bug_id"])
            record = {
                field: string_value(row.get(field))
                for field in (*TEXT_FIELDS, "tracker_url", "created_at")
            }
            record["manual_source_file"] = str(source.resolve())
            record["manual_source_sha256"] = hashes[str(source.resolve())]
            if not record["initial_description"].strip():
                raise ValueError(f"Manual import has empty description: {key}")
            if key in records and records[key] != record:
                raise ValueError(f"Conflicting manual imports for {key}")
            records[key] = record
    return records, hashes


def merge_manual(existing: dict[str, Any], manual: dict[str, Any] | None) -> dict[str, Any]:
    """Successful automatic/prior records always win; manual fills retrieval failures."""
    if existing.get("retrieval_status") == "SUCCESS" or manual is None:
        return existing
    return {
        **existing,
        **manual,
        "retrieval_status": "SUCCESS",
        "retrieval_source": "MANUAL_IMPORT",
        "retrieved_at": utc_timestamp(),
        "error": "",
    }


def enrich_mandelbugs(
    labels_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    sleep_seconds: float = 1.0,
    manual_dir: str | Path | None = "data/manual_enrichment",
    retry_failures: bool = False,
    offline: bool = False,
) -> dict[str, Any]:
    labels = validate_labels(pd.read_parquet(labels_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manual, manual_hashes = load_manual_imports(manual_dir)
    extraneous = set(manual) - set(labels.issue_key)
    if extraneous:
        raise ValueError(f"Manual issue IDs absent from official labels: {sorted(extraneous)}")
    retriever = CachedRetriever(cache_dir, sleep_seconds)
    prior_success = {}
    if (out / "reports.parquet").exists():
        prior = pd.read_parquet(out / "reports.parquet")
        # Keep successful prior evidence, including manual imports. Labels below
        # are always rejoined from the current official label table.
        for record in prior[prior.retrieval_status == "SUCCESS"].to_dict("records"):
            prior_success[record["issue_key"]] = {
                field: value for field, value in record.items() if field not in labels.columns
            }
    results = {}
    for row in labels.drop_duplicates("issue_key").to_dict("records"):
        key = row["issue_key"]
        cache_file = Path(cache_dir) / (
            hashlib.sha256(tracker_url(row["project"], row["bug_id"]).encode()).hexdigest()
            + ".json"
        )
        if key in prior_success:
            result = prior_success[key]
        elif cache_file.exists() or (not offline and key not in manual):
            result = retriever.retrieve(
                row["project"], row["bug_id"], retry_failures and not offline
            )
        else:
            result = {
                "retrieval_status": "UNSUPPORTED",
                "retrieval_source": "AUTO",
                "tracker_url": tracker_url(row["project"], row["bug_id"]),
                "error": "No cached response; offline/manual mode",
            }
        result = merge_manual(result, manual.get(key))
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
    audit = {
        "rows": len(reports),
        "unique_issues": len(results),
        "status_counts": reports.retrieval_status.value_counts().to_dict(),
        "unique_status_counts": pd.Series([r["retrieval_status"] for r in results.values()])
        .value_counts()
        .to_dict(),
        "labels_sha256": sha256_file(labels_path),
        "manual_files": manual_hashes,
        "text_version": TEXT_VERSION,
        "parser_version": PARSER_VERSION,
        "timestamp": utc_timestamp(),
        "git_commit": git_commit(),
        "training_invoked": False,
    }
    write_json(out / "retrieval_audit.json", audit)
    return audit
