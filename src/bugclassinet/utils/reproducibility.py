"""Small reproducibility metadata helpers."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone


def utc_timestamp() -> str:
    """Return a stable UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_commit() -> str | None:
    """Return the current Git commit when evaluation runs inside a checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None
