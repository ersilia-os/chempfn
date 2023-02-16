import numpy as np
import pytest
from ensemble_tabpfn.utils import aggregate_arrays, chunker


def test_chunker_1_chunk():
    X = np.random.rand(100)
    chunk_size = 1000
    expected = 1
    chunks = chunker(X, chunk_size)
    assert len(chunks) == expected
    assert chunks[-1] == (0, 100)


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
    arr = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    X = [[arr]]
    expected = arr
    result = aggregate_arrays(X)
    assert np.array_equal(result, expected)


def test_aggregate_n_even_chunks_1_ensemble():
    arr = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    X = [[arr, arr, arr]]
    expected = np.concatenate([arr, arr, arr], axis=0)
    result = aggregate_arrays(X)
    assert np.array_equal(result, expected)


def test_aggregate_n_uneven_chunks_1_ensemble():
    arr = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    uneven_arr = np.array([1, 0, 0, 1, 1, 0])
    X = [[arr, arr, arr, uneven_arr]]
    expected = np.concatenate([arr, arr, arr, uneven_arr], axis=0)
    result = aggregate_arrays(X)
    assert np.array_equal(result, expected)


def test_aggregate_1_chunk_n_ensembles():
    arr1 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    arr2 = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    arr3 = arr1
    X = [[arr1], [arr2], [arr3]]
    expected = arr1
    result = aggregate_arrays(X)
    assert np.array_equal(result, expected)


def test_aggregate_n_even_chunks_n_ensembles():
    arr1 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    arr2 = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    X = [[arr1, arr2, arr1, arr2], [arr1, arr2, arr1, arr2], [arr1, arr2, arr1, arr2]]
    expected = np.concatenate([arr1, arr2, arr1, arr2], axis=0)
    result = aggregate_arrays(X)
    assert np.array_equal(result, expected)


@pytest.mark.skip("Flatten array before getting average")
def test_aggregate_n_uneven_chunks_n_ensembles():
    arr1 = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    arr2 = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    uneven_arr = np.array([1, 0, 0, 1, 1, 0])
    X = [[arr1, arr2, uneven_arr], [arr1, arr2, uneven_arr], [arr1, arr2, uneven_arr]]
    expected = np.concatenate([arr1, arr2, uneven_arr], axis=0)
    result = aggregate_arrays(X)
    assert np.array_equal(result, expected)