from enum import IntEnum
import numpy as np
import math

#TODO (@dhanshree) Update documentation

class ShapeCount(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    UNDEF = -1


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
        end = num_samples if chunk == num_chunks - 1 else (chunk + 1) * chunk_size
        chunks.append((beg, end))
    return chunks


#TODO(@dhanshree) Maybe we can replace this with some decorators
def aggregate_arrays(X):
    """_summary_

    Args:
        X (_type_): _description_

    Returns:
        _type_: _description_
    """
    X = np.array(X, dtype="object")
    arr_shape = X.shape

    # The array is 3-D, ie it has equal sized chunks across all ensembles
    if len(arr_shape) == ShapeCount.THREE:
        ensembles, chunks, samples = arr_shape

    # The array is 2-D with possibly uneven chunks across all ensembles.
    # This is usually the last chunk
    else:
        ensembles, chunks = arr_shape
        samples = ShapeCount.UNDEF

    # When there's only one ensemble, we have to only deal with individual chunks
    if ensembles == ShapeCount.ONE:
        # When there is only one chunk in this ensemble, simply flatten the array
        if chunks == ShapeCount.ONE:
            return X.reshape(samples)

        # When there is more than one chunk and all the chunks are of the same size
        if samples != ShapeCount.UNDEF:
            return X.reshape(chunks * samples)

        # Finally, when there is more than one chunk and the chunks are uneven in length,
        # we concatente all the chunks and flatten the array.
        return np.concatenate([X[0][chunk] for chunk in range(chunks)], axis=0)

    # When there is more than one ensemble
    else:
        if chunks == ShapeCount.ONE:
            return np.round(np.mean(X, axis=0, dtype=np.float64).reshape(samples))

        if samples != ShapeCount.UNDEF:
            return np.round(np.mean(X, axis=0, dtype=np.float64).reshape(chunks*samples))
        
        #TODO (@dhanshree) Flatten array before gettinga average
        # X = np.round(np.mean(X, axis=0, dtype=np.float64))
        # return np.concatenate([X[0][chunk] for chunk in range(chunks)], axis=0)
