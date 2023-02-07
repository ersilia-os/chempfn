import math

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import numpy as np
import torch
from tabpfn import TabPFNClassifier

from samplers import get_sampler
from utils import _chunker, _aggregate_arrays

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TABPFN_MAX_INP_SIZE = 1000


class EnsembleTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iters=100,
        sampler="bootstrap",
        n_ensemble_configurations=4,
        chunk_size=2000,
    ):
        assert sampler in ["bootstrap"], "Sampler not recognized"
        self.sampler = get_sampler(sampler_type=sampler)()
        self.max_iters = max_iters
        self.n_ensemble_configurations = n_ensemble_configurations
        self.chunk_size = chunk_size

    def _subsample(self, X, y, stratify=None):
        _x, _y = self._sampler.sample(X, y, stratify=stratify)
        assert len(_x) <= _TABPFN_MAX_INP_SIZE
        assert len(_y) <= _TABPFN_MAX_INP_SIZE
        return (_x, _y)

    def _generate_ensemble(self, X, y, stratify=None):
        for i in range(self._num_iters):
            data = self._subsample(X, y)
            self._ensembles.append(data)

    def _chunk_data(self, X):
        return _chunker(X, self.chunk_size)

    def fit(self, X, y, stratify=None):
        X, y = check_X_y(X, y, force_all_finite=False)
        self.ensembles_ = []
        if X.shape[0] < _TABPFN_MAX_INP_SIZE:
            # If the input size is smaller than what TabPFN can
            # work with, then generating ensembles is not required
            self.ensembles_.append((X, y))
            return
        self._generate_ensemble(X, y)

    def _predict(self, X, return_prob=False):

        model = TabPFNClassifier(
            device=DEVICE, N_ensemble_configurations=self.n_ensemble_configurations
        )

        result = []
        chunks = self._chunk_data(X)
        for data in self.ensembles_:
            for start, end in chunks:
                X_chunk = X[start:end]
                _x, _y = data
                model.fit(_x, _y)
                result.append(
                    (model.predict(X_chunk, return_winning_probability=return_prob))
                )

        return result

    @staticmethod
    def _aggregate_preds(preds):
        preds = _aggregate_arrays(preds)
        return preds

    def predict(self, X):
        check_is_fitted(self, attributes="ensembles_")
        preds = self._predict(X)
        preds = self._aggregate_preds(preds)
        return preds

    def predict_proba(self, X):
        check_is_fitted(self, attributes="ensembles_")
        y, p = self._predict(X, return_prob=True)
        preds = self._aggregate_preds(y)
        probs = self._aggregate_preds(p)
        return preds, probs
