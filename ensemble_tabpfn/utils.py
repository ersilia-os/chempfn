from enum import IntEnum
import numpy as np
import math
from typing import List


class Ensemble:
    """Stores the indexes of an ensemble comprising of data and features"""

    # __slots__ = ["data_indices", "feat_samplers"]

    def __init__(self, data, data_indices, feat_samplers) -> None:
        self.data = data
        self.data_indices = data_indices
        self.feat_samplers = feat_samplers


class Result:
    """Stores the result from EnsembleTabPFN predictions"""

    def __init__(
        self, samples: int, classes: int, tolerance: float = 1e-4, patience: int = 5
    ) -> None:
        """_summary_

        Parameters
        ----------
        samples : int
            Samples in the test dataset
        classes : int
            Number of classes in the dataset
        tolerance : float, optional
            Tolerance for no improvements in predicted probabilities, by default 1e-4
        patience : int, optional
            Number of epochs for which no improvement is tolerated, by default 5
        """
        self.tolerance = tolerance
        self.patience = patience
        self.freeze = np.zeros(samples, dtype=bool)
        self.no_change_count = np.zeros(samples, dtype=int)
        self.ensembles = np.ones(
            samples
        )  # Initial condition, to prevent divide by zero error
        self.prob_mean = np.zeros((samples, classes), dtype=np.float64)

    def compare_preds(self, curr_mean) -> None:
        """Compares current and previous mean predictions."""
        no_change = np.any(
            np.abs(self.prob_mean - curr_mean) < self.tolerance, axis=1
        )
        self.no_change_count += no_change
        self.prob_mean[~no_change] = curr_mean[~no_change]
        self.ensembles[~no_change] += 1
        self.freeze[self.no_change_count >= self.patience] = True

    @property
    def probs(self) -> np.ndarray:
        """Returns the final predictions."""
        return self.prob_mean

    @property
    def preds(self) -> np.ndarray:
        """Returns the final predictions."""
        return np.argmax(self.probs, axis=1)


class TabPFNConstants(IntEnum):
    """Constants mapping TabPFN's limitations."""

    MAX_INP_SIZE: int = 1000
    MAX_FEAT_SIZE: int = 100
