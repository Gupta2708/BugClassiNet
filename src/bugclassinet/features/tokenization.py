"""Hugging Face tokenization helpers."""

from __future__ import annotations

from typing import Any


def tokenize_texts(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, Any]:
    """Tokenize with explicit truncation and fixed maximum length."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    return tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")
