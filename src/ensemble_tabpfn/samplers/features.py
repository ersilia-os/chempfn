import numpy as np
from lol import LOL
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2

#TODO consolidate constants
ALL_SAMPLERS = ["pca", "lrp", "selectk", "kmeans", "random"]
N_FEATURES = 100


class BaseSampler:

    def __init__(self, n_features=N_FEATURES, fit_with_y=False) -> None:
        self.n_features = n_features
        self.fit_with_y = fit_with_y

    def sample(self, X, y):
        if self.fit_with_y:
            return self.sampler.fit_transform(X,y)
        else:
            return self.sampler.fit_transform(X)

class SelectKSampler(BaseSampler):

    def __init__(self) -> None:
        super().__init__(N_FEATURES, fit_with_y=True)
        self.sampler = SelectKBest(chi2, k=self.n_features)
        
class KMeansSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(N_FEATURES, fit_with_y=True)
        self.sampler = KMeans(n_clusters=N_FEATURES)

class LRPSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(N_FEATURES, fit_with_y=True)
        self.sampler = LOL(n_components=self.n_features)


class PCASampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(N_FEATURES, fit_with_y=False)
        self.sampler = PCA(n_components=self.n_features)

class RandomSampler(BaseSampler):
    def __init__(self) -> None:
        super().__init__(N_FEATURES, fit_with_y=False)
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
    "kmeans": KMeansSampler,
    "randomized": RandomSampler,
}


def sample_features(sampler: str, X, y):
    sampler = sampler_map[sampler]
    return sampler
