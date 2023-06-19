from enum import IntEnum
import numpy as np
import math
from typing import List


class Ensemble:
    """Stores the indexes of an ensemble comprising of data and features"""

    __slots__ = ["data_indices", "feat_samplers"]

    def __init__(self, data_indices, feat_samplers) -> None:
        self.data_indices = data_indices
        self.feat_samplers = feat_samplers

    

class Result:
    """Stores the result from EnsembleTabPFN predictions"""

    def __init__(self) -> None:
        self.raw_probs: List[np.ndarray] = []

    def aggregate(self) -> None:
        """Aggregates results across ensembles."""
        self.probs = np.mean(self.raw_probs, axis=0, dtype=np.float64)
        self.preds = np.argmax(self.probs, axis=-1)


class TabPFNConstants(IntEnum):
    """Constants mapping TabPFN's limitations."""

    MAX_INP_SIZE: int = 1000
    MAX_FEAT_SIZE: int = 100
