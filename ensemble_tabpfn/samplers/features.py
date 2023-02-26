import numpy as np
from lol import LOL
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectKBest, chi2

# TODO consolidate constants
ALL_SAMPLERS = ["pca", "lrp", "selectk", "cluster", "random"]
N_FEATURES = 100


class BaseSampler:
    def __init__(self, fit_with_y=False) -> None:
        self.fit_with_y = fit_with_y
        self.sampler = None

    def _validate_sampler(self) -> None:
        if self.sampler is None:
            raise NotImplementedError

    def sample(self, X, y):
        self._validate_sampler()
        if self.fit_with_y:
            return self.sampler.fit_transform(X, y)
        else:
            return self.sampler.fit_transform(X)


class SelectKSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(fit_with_y=True)
        self.sampler = SelectKBest(chi2, k=N_FEATURES)


class ClusterSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__()
        self.sampler = FeatureAgglomeration(n_clusters=N_FEATURES)


class LRPSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(fit_with_y=True)
        self.sampler = LOL(n_components=N_FEATURES)


class PCASampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__()
        self.sampler = PCA(n_components=N_FEATURES)


class RandomSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__()
        self.is_fit = False
        self.indices = []

    def sample(self, X, y):
        if not self.is_fit:
            self.indices = np.random.choice(X.shape[1], self.n_features)
        else:
            return X[self.indices]


sampler_map = {
    "pca": PCASampler,
    "selectk": LRPSampler,
    "lrp": SelectKSampler,
    "kmeans": ClusterSampler,
    "randomized": RandomSampler,
}


def sample_features(sampler: str, X, y):
    sampler = sampler_map[sampler]
    return sampler