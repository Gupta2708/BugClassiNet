"""Low-overhead process and Arrow memory reporting.

The training jobs run on Linux in Kaggle, where ``/proc`` and ``resource`` are
available without adding another package to the environment.  The helpers also
degrade to ``None`` on platforms that cannot expose a measurement; memory
instrumentation must never prevent an experiment from running.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

LOGGER = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024**3


def _current_rss_bytes() -> int | None:
    """Return current resident bytes using an optional package or ``/proc``."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError, AttributeError):
        pass

    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    # Linux reports VmRSS in KiB.
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _peak_rss_bytes() -> int | None:
    """Return peak resident bytes when the operating system exposes them."""
    try:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux (including Kaggle) reports KiB.
        return maximum if sys.platform == "darwin" else maximum * 1024
    except (ImportError, OSError, ValueError, AttributeError):
        pass

    try:
        import psutil

        details = psutil.Process(os.getpid()).memory_info()
        maximum = getattr(details, "peak_wset", None)
        return int(maximum) if maximum is not None else None
    except (ImportError, OSError, AttributeError):
        return None


def _arrow_allocated_bytes() -> int | None:
    """Return bytes currently held by Arrow's allocator, if Arrow is installed."""
    try:
        import pyarrow as pa

        return int(pa.total_allocated_bytes())
    except (ImportError, OSError, AttributeError):
        return None


def memory_snapshot() -> dict[str, int | None]:
    """Capture native-process and Arrow allocator memory in bytes."""
    return {
        "rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "arrow_bytes": _arrow_allocated_bytes(),
    }


def _format_gib(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / _BYTES_PER_GIB:.3f}"


def log_memory_checkpoint(
    checkpoint: str,
    *,
    logger: logging.Logger | None = None,
    **context: Any,
) -> dict[str, int | None]:
    """Log and return a memory snapshot for a named training checkpoint.

    Extra context is rendered with ``repr`` so callers can include row counts,
    split names, and cache sizes without this utility depending on Datasets.
    """
    if not checkpoint or not checkpoint.strip():
        raise ValueError("Memory checkpoint name must be nonblank")

    snapshot = memory_snapshot()
    details = " ".join(f"{key}={value!r}" for key, value in sorted(context.items()))
    (logger or LOGGER).info(
        "Memory checkpoint=%s rss_gib=%s peak_rss_gib=%s arrow_gib=%s%s",
        checkpoint,
        _format_gib(snapshot["rss_bytes"]),
        _format_gib(snapshot["peak_rss_bytes"]),
        _format_gib(snapshot["arrow_bytes"]),
        f" {details}" if details else "",
    )
    return snapshot


def log_memory(
    logger: logging.Logger,
    checkpoint: str,
    **context: Any,
) -> dict[str, int | None]:
    """Compatibility-friendly logger-first wrapper used by training modules."""
    return log_memory_checkpoint(checkpoint, logger=logger, **context)
