import numpy as np
import pytest
from scipy import sparse

from bugclassinet.features.tfidf import SparseMatrixLogger, sparse_matrix_nbytes


def test_sparse_memory_estimate_counts_sparse_buffers() -> None:
    matrix = sparse.csr_matrix(np.eye(3, dtype=np.float32))
    expected = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert sparse_matrix_nbytes(matrix) == expected
    assert SparseMatrixLogger().fit_transform(matrix) is matrix


def test_dense_feature_matrix_is_rejected() -> None:
    with pytest.raises(TypeError, match="dense matrix"):
        sparse_matrix_nbytes(np.eye(3, dtype=np.float32))
