import numpy as np
import math


def chunker(X, chunk_size):
    """Returns start and end indices of chunks

    Args:
        X (_type_): _description_
        chunk_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_samples = X.shape[0]
    num_chunks = math.ceil(num_samples / chunk_size)

    if num_chunks == 1:
        chunks = [(0, num_samples)]
        return chunks

    chunks = []
    for chunk in range(num_chunks):   
        beg = chunk * chunk_size
        end = num_samples if chunk==num_chunks-1 else (chunk + 1) * chunk_size
        chunks.append((beg, end))
    return chunks


def aggregate_arrays(X):
    X = np.array(X, dtype="object")
    ensembles, chunks, samples = X.shape
    if ensembles == 1 and chunks == 1:
        return X.reshape(samples)
    return X
