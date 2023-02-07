import numpy as np
import pytest
from ensemble_tabpfn.utils import aggregate_arrays, chunker


def test_chunker_1_chunk():
    X = np.random.rand(100)
    chunk_size = 1000
    expected = 1
    chunks = chunker(X, chunk_size)
    assert len(chunks) == expected
    assert chunks[-1] == (0,100)


def test_chunker_imperfect_chunks():
    # Num samples cannot be divided into chunks of equal sizes
    # len(chunks[-1]) < chunk_size

    X = np.random.rand(188)
    chunk_size = 20
    expected = 10
    chunks = chunker(X, chunk_size)
    X[180:200]
    assert len(chunks) == expected
    assert chunks[-1] == (180, 188)


def test_chunker_perfect_chunks():
    # Num samples can be perfectly divided into chunks of
    # len(chunk_size)
    X = np.random.rand(200)
    chunk_size = 20
    expected = 10
    chunks = chunker(X, chunk_size)
    X[180:200]
    assert len(chunks) == expected
    assert chunks[-1] == (180, 200)


def test_aggregate_1_chunk_1_ensemble():
    expected = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    X = [[expected]]
    print(X, len(X[0][0]))
    result = aggregate_arrays(X)
    assert result.tolist() == expected.tolist()


def test_aggregate_n_chunks_1_ensemble():
    pass


def test_aggregate_n_chunks_n_ensembles():
    pass
