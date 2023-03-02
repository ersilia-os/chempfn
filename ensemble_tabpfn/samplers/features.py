from typing import Optional, Type, List

import numpy as np
from lol import LOL
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectKBest, chi2

from ..utils import TabPFNConstants

ALL_SAMPLERS = ["pca", "lrp", "selectk", "cluster", "random"]


class FeatureSampler:
    """Interface for all feature samplers.

    A sampler is a scikit learn BaseEstimator type. All feature sub samplers
    must implement this and set sampler attribute appropriately.
    """

    def __init__(
        self,
        n_features: int = TabPFNConstants.MAX_FEAT_SIZE,
        fit_with_y: bool = False,
    ) -> None:
        """Constructor for FeatureSampler interface.

        Parameters
        ----------
        n_features : int, optional
            Number of features to sample from the feature space, by default TabPFNConstants.MAX_FEAT_SIZE
        fit_with_y : bool, optional
            Some feature extraction or selection methods require the target variable y to be present. Set true for fitting with y, by default False.
        """
        self.n_features: int = n_features
        self.fit_with_y: bool = fit_with_y
        self.sampler: BaseEstimator

    def _validate_sampler(self) -> None:
        if self.sampler is None:
            raise NotImplementedError

    def sample(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Transform the dataset to maximum number of features by sampling.

        Parameters
        ----------
        X : np.ndarray
           The input data to be transformed, of shape (n_samples, n_features)
        y : np.ndarray
            The target variable corresponding to X, of shape (n_samples,)
        transform: bool, optional
            Transform the data on a sampler that has already been fit. By default it is False when a sampler is being fit on an ensemble's data. Set this to true when reducing features in the test data during prediction.
        Returns
        -------
        np.ndarray
           The transformed data of shape (n_samples, n_features)
        """
        self._validate_sampler()

        if self.fit_with_y:
            return self.sampler.fit_transform(X, y)
        else:
            return self.sampler.fit_transform(X)

    def reduce(self, X: np.ndarray) -> np.ndarray:
        self._validate_sampler()
        try:
            check_is_fitted(self.sampler)
            return self.sampler.transform(X)
        except NotFittedError:
            raise ValueError(
                f"The {self.__class__.__name__} has not been fit on ensemble data. Fit the sampler before attempting to transform unseen data."
            )


class SelectKSampler(FeatureSampler):
    """Select K Best according to chi2 scores."""

    def __init__(
        self, n_features: int = TabPFNConstants.MAX_FEAT_SIZE
    ) -> None:
        super().__init__(n_features, fit_with_y=True)
        self.sampler = SelectKBest(chi2, k=self.n_features)


class ClusterSampler(FeatureSampler):
    """Performs heirarchical clustering to agglomerate similar features."""

    def __init__(
        self, n_features: int = TabPFNConstants.MAX_FEAT_SIZE
    ) -> None:
        super().__init__(n_features)
        self.sampler = FeatureAgglomeration(n_clusters=self.n_features)


class LRPSampler(FeatureSampler):
    """Low Rank Project sampling to reduce dimensionality in feature space."""

    def __init__(
        self, n_features: int = TabPFNConstants.MAX_FEAT_SIZE
    ) -> None:
        super().__init__(n_features, fit_with_y=True)
        self.sampler = LOL(n_components=self.n_features)


class PCASampler(FeatureSampler):
    """PCA Sampling to reduce dimensionality in feature space."""

    def __init__(
        self, n_features: int = TabPFNConstants.MAX_FEAT_SIZE
    ) -> None:
        super().__init__(n_features)
        self.sampler = PCA(n_components=self.n_features)


class RandomSampler(FeatureSampler):
    """Randomly selects self.n_features from the feature space."""

    def __init__(
        self, n_features: int = TabPFNConstants.MAX_FEAT_SIZE
    ) -> None:
        super().__init__(n_features)
        self.is_fit: bool = False
        self.indices: List[int] = []
        self.sampler = "RandomSampler"

    def sample(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            self.indices = np.random.choice(
                X.shape[1], self.n_features
            ).tolist()
            self.is_fit = True
        return X[:, self.indices]

    def reduce(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise ValueError(
                "The RandomSampler has not been fit on ensemble data. Fit the sampler before attempting to transform unseen data."
            )
        return X[:, self.indices]


sampler_map = {
    "pca": PCASampler,
    "selectk": LRPSampler,
    "lrp": SelectKSampler,
    "cluster": ClusterSampler,
    "random": RandomSampler,
}


def get_feature_sampler(sampler_type: str) -> Type[FeatureSampler]:
    if sampler_type not in ALL_SAMPLERS:
        raise ValueError(
            f"Invalid feature sampler provided. Must be one of {ALL_SAMPLERS}"
        )
    feat_sampler = sampler_map[sampler_type]
    return feat_sampler
