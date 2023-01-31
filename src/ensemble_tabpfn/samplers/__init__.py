from sklearn.utils import resample


class BootstrapSampler:
    def __init__(self):
        pass

    def sample(self, X, y, stratify=None, n_samples: int = 1000, replace: bool = True):
        if stratify is not None:
            return resample(
                X, y, n_samples=n_samples, replace=replace, stratify=stratify
            )
        else:
            return resample(X, y, n_samples=n_samples, replace=replace)
