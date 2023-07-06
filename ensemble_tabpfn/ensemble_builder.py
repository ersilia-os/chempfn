from typing import List, Optional

import numpy as np

from .utils import TabPFNConstants
from .samplers import DataSampler, FeatureSampler, get_data_sampler
from .samplers.features import BaseSampler


class Ensemble:
    """Stores the indexes of an ensemble comprising of data and features"""

    __slots__ = ["data_indices", "feat_samplers"]

    def __init__(
        self, data_indices: List[int], feat_samplers: List[BaseSampler]
    ) -> None:
        self.data_indices = data_indices
        self.feat_samplers = feat_samplers


class EnsembleBuilder:
    def __init__(
        self,
        max_iters: int = 100,
        random_state: int = 42,
        n_samples: int = TabPFNConstants.MAX_INP_SIZE,
        n_features: int = TabPFNConstants.MAX_FEAT_SIZE,
        data_sampler: str = "bootstrap",
    ) -> None:
        if not (n_samples <= TabPFNConstants.MAX_INP_SIZE):
            raise ValueError(
                f"n_samples must be less than or equal to {TabPFNConstants.MAX_INP_SIZE}, got {n_samples}"
            )

        if not (n_features <= TabPFNConstants.MAX_FEAT_SIZE):
            raise ValueError(
                f"n_features must be less than or equal to {TabPFNConstants.MAX_FEAT_SIZE}, got {n_features}"
            )
        self.max_iters: int = max_iters
        self.data_sampler: DataSampler = get_data_sampler(sampler_type=data_sampler)(
            n_samples=n_samples
        )
        self.feature_sampler = FeatureSampler(n_features=n_features)
        self.random_state: Optional[int] = random_state

    def _data_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: Optional[int] = None,
    ):
        _x, _y, indices = self.data_sampler.sample(X, y, random_state=random_state)
        return (_x, _y, indices)

    def _feat_subsample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: bool = False,
    ) -> List[np.ndarray]:
        if transform:
            return self.feature_sampler.reduce(X)
        return self.feature_sampler.sample(X, y)  # type: ignore

    def _generate_ensembles(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        # Implement early stopping
        ensembles = []
        for _iter in range(self.max_iters):
            _x, _y, indices = self._data_subsample(X, y, random_state=self.random_state)
            self._feat_subsample(_x, _y)
            ensemble = Ensemble(
                data_indices=indices,
                feat_samplers=self.feature_sampler.get_samplers(),
            )
            ensembles.append(ensemble)

        return ensembles

    def build(self, X: np.ndarray, y: np.ndarray):
        return self._generate_ensembles(X, y)