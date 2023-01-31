import math
import numpy as np
from tabpfn import TabPFNClassifier

from samplers import BootstrapSampler

# TODO: Inherit from sklearn BaseEstimator and ClassifierMixin
class EnsembleTabPFN:
    def __init__(self):
        self.model = TabPFNClassifier(device="cuda", N_ensemble_configurations=4)
        self._sampler = BootstrapSampler()
        self._num_iters = 1000
        self._is_fitted = False
        self._ensemble = []

    # TODO convert to getting indices only; perhaps we can measure storage difference between storing data and indices
    def _subsample(self, X, y):
        _x, _y = self._sampler.sample(X, y, stratify=y)
        assert len(_x) == 1000
        assert len(_y) == 1000
        return (_x, _y)

    def _generate_ensemble(self, X, y):
        for i in range(self._num_iters):
            data = self._subsample(X, y)
            self._ensemble.append(data)

    def _chunk_data(self, X, y):
        pass

    def fit(self, X, y):
        self._is_fitted = True
        self._generate_ensemble(X, y)

    # TODO Parametrize chunk size
    def predict(self, X):
        # Perform data chunking so that we don't run into any OOM issues
        # TODO What is a good estimate for chunk size? Or should we let the user decide?
        total_data = len(X)
        chunk_size = 2000
        num_chunks = math.ceil(total_data / chunk_size)
        preds = []
        for chunk in range(num_chunks):
            X_chunk = X[(chunk * chunk_size) : (chunk + 1) * chunk_size]
            for data in self._ensemble:
                _x, _y = data
                self.model.fit(_x, _y)
                preds.append(self.model.predict(X_chunk))

        preds = np.array(preds)
        return np.round(np.mean(preds, axis=0))
