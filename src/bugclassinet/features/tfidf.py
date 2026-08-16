"""Memory-conscious sparse TF-IDF feature construction."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

LOGGER = logging.getLogger(__name__)

FeatureMode = Literal["word", "word_char"]


@dataclass(frozen=True)
class TfidfFeatureConfig:
    """Bounded sparse feature settings suitable for million-row corpora."""

    feature_mode: FeatureMode = "word_char"
    word_ngram_range: tuple[int, int] = (1, 2)
    word_max_features: int = 200_000
    word_min_df: int | float = 3
    word_max_df: int | float = 0.98
    char_ngram_range: tuple[int, int] = (3, 5)
    char_max_features: int = 150_000
    char_min_df: int | float = 5
    char_max_df: int | float = 0.98
    sublinear_tf: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> TfidfFeatureConfig:
        """Create validated feature settings from a YAML-style mapping."""
        values = values or {}
        fields = cls.__dataclass_fields__
        selected = {key: value for key, value in values.items() if key in fields}
        for key in ("word_ngram_range", "char_ngram_range"):
            if key in selected:
                selected[key] = tuple(selected[key])
        config = cls(**selected)
        if config.feature_mode not in {"word", "word_char"}:
            raise ValueError("feature_mode must be 'word' or 'word_char'")
        if config.word_max_features <= 0 or config.char_max_features <= 0:
            raise ValueError("TF-IDF max_features values must be positive")
        return config

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable resolved settings."""
        return asdict(self)


def _word_vectorizer(config: TfidfFeatureConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=config.word_ngram_range,
        max_features=config.word_max_features,
        min_df=config.word_min_df,
        max_df=config.word_max_df,
        sublinear_tf=config.sublinear_tf,
        dtype=np.float32,
    )


def _char_vectorizer(config: TfidfFeatureConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.char_ngram_range,
        max_features=config.char_max_features,
        min_df=config.char_min_df,
        max_df=config.char_max_df,
        sublinear_tf=config.sublinear_tf,
        dtype=np.float32,
    )


def make_tfidf_features(
    config: TfidfFeatureConfig | None = None,
) -> TfidfVectorizer | FeatureUnion:
    """Build bounded word-only or word+character sparse TF-IDF features."""
    config = config or TfidfFeatureConfig()
    word = _word_vectorizer(config)
    if config.feature_mode == "word":
        return word
    return FeatureUnion(
        [("word", word), ("char", _char_vectorizer(config))],
        n_jobs=1,
    )


def sparse_matrix_nbytes(matrix: sparse.spmatrix) -> int:
    """Estimate bytes owned by a SciPy sparse matrix without densifying it."""
    if not sparse.issparse(matrix):
        raise TypeError("TF-IDF produced a dense matrix; refusing to continue")
    arrays = (
        getattr(matrix, name)
        for name in ("data", "indices", "indptr", "row", "col")
        if hasattr(matrix, name)
    )
    return sum(array.nbytes for array in arrays)


class SparseMatrixLogger(TransformerMixin, BaseEstimator):
    """Log sparse matrix dimensions and memory without copying feature data."""

    def _log(self, matrix: sparse.spmatrix, phase: str) -> None:
        memory_gib = sparse_matrix_nbytes(matrix) / (1024**3)
        LOGGER.info(
            "TF-IDF %s matrix shape=%s features=%d dtype=%s format=%s nnz=%d "
            "estimated_sparse_memory=%.3f GiB",
            phase,
            matrix.shape,
            matrix.shape[1],
            matrix.dtype,
            matrix.getformat(),
            matrix.nnz,
            memory_gib,
        )

    def fit(self, X: sparse.spmatrix, y: Any = None) -> SparseMatrixLogger:
        self._log(X, "fit")
        return self

    def transform(self, X: sparse.spmatrix) -> sparse.spmatrix:
        self._log(X, "transform")
        return X

    def fit_transform(
        self, X: sparse.spmatrix, y: Any = None, **fit_params: Any
    ) -> sparse.spmatrix:
        self._log(X, "fit")
        return X
