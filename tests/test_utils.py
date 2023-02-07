from ensemble_tabpfn.utils import aggregate_arrays, chunker


def test_chunker_1_chunk():
    # Num samples less than chunk size
    pass


def test_chunker_imperfect_chunks():
    # Num samples cannot be divided into chunks of equal sizes
    # len(chunks[-1]) < chunk_size

    pass


def test_chunker_perfect_chunks():
    # Num samples can be perfectly divided into chunks of
    # len(chunk_size)

    pass


def test_aggregate_1_chunk_1_ensemble():
    pass


def test_aggregate_n_chunks_1_ensemble():
    pass


def test_aggregate_n_chunks_n_ensembles():
    pass
