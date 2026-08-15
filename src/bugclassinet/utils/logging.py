"""Logging configuration."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure a predictable console logger once."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
