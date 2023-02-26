import numpy as np
from lol import LOL
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectKBest, chi2
from typing import Optional, Type, List

ALL_SAMPLERS = ["pca", "lrp", "selectk", "cluster", "random"]
N_FEATURES = 100


class FeatureSampler:
    def __init__(self, fit_with_y: Optional[bool] = False) -> None:
        self.fit_with_y = fit_with_y
        self.sampler: BaseEstimator

    def _validate_sampler(self) -> None:
        if self.sampler is None:
            raise NotImplementedError

    def sample(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._validate_sampler()
        if self.fit_with_y:
            return self.sampler.fit_transform(X, y)
        else:
            return self.sampler.fit_transform(X)


class SelectKSampler(FeatureSampler):
    def __init__(self) -> None:
        super().__init__(fit_with_y=True)
        self.sampler = SelectKBest(chi2, k=N_FEATURES)


class ClusterSampler(FeatureSampler):
    def __init__(self) -> None:
        super().__init__()
        self.sampler = FeatureAgglomeration(n_clusters=N_FEATURES)


class LRPSampler(FeatureSampler):
    def __init__(self) -> None:
        super().__init__(fit_with_y=True)
        self.sampler = LOL(n_components=N_FEATURES)


class PCASampler(FeatureSampler):
    def __init__(self) -> None:
        super().__init__()
        self.sampler = PCA(n_components=N_FEATURES)


class RandomSampler(FeatureSampler):
    def __init__(self) -> None:
        super().__init__()
        self.is_fit: bool = False
        self.indices: List[int] = []

    def sample(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            self.indices = np.random.choice(X.shape[1], N_FEATURES).tolist()
            return X[self.indices]
        else:
            return X[self.indices]


sampler_map = {
    "pca": PCASampler,
    "selectk": LRPSampler,
    "lrp": SelectKSampler,
    "kmeans": ClusterSampler,
    "randomized": RandomSampler,
}


def get_feature_sampler(sampler_type: str) -> Type[FeatureSampler]:
    if sampler_type not in ALL_SAMPLERS:
        raise ValueError(
            f"Invalid feature sampler provided. Must be one of {ALL_SAMPLERS}"
        )
    feat_sampler = sampler_map[sampler_type]
    return feat_sampler
