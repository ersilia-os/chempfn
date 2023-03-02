from enum import IntEnum
import numpy as np
import math
from typing import List


class Result:
    """Stores the result from EnsembleTabPFN predictions"""

    def __init__(self) -> None:
        self.raw_preds: List[np.ndarray] = []
        self.raw_probs: List[np.ndarray] = []
        self.preds: np.ndarray
        self.probs: np.ndarray

    def aggregate(self)-> None:
        """Aggregates results across ensembles.
        """
        Y = np.array(self.raw_preds)
        P = np.array(self.raw_probs)

        print(Y.shape, P.shape)
        ensembles, samples = Y.shape

        if ensembles == 1:
            self.probs = P.reshape(samples)
            self.preds = Y.reshape(samples)
        else:
            self.probs = np.round(np.mean(P, axis=0, dtype=np.float64))
            self.preds = np.round(np.mean(Y, axis=0, dtype=np.float64))


class TabPFNConstants(IntEnum):
    """Constants tracking TabPFN's limitations.

    """
    MAX_INP_SIZE: int = 1000
    MAX_FEAT_SIZE: int = 100