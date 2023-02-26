from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch
from tabpfn import TabPFNClassifier

from .samplers import get_data_sampler
from .samplers.features import sample_features
from .utils import aggregate_arrays, chunker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TABPFN_MAX_INP_SIZE = 1000
_TABPFN_MAX_FEAT = 100


class EnsembleTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iters=100,
        data_sampler="bootstrap",
        feature_sampler="selectk",
        n_ensemble_configurations=4,
        chunk_size=2000,
    ):
        self.data_sampler = get_data_sampler(sampler_type=data_sampler)()
        self.feature_sampler = sample_features(sampler=feature_sampler)
        self.max_iters = max_iters
        self.n_ensemble_configurations = n_ensemble_configurations
        self.chunk_size = chunk_size

    def _data_subsample(self, X, y, stratify=None):
        _x, _y = self.data_sampler.sample(X, y, stratify=stratify)
        assert len(_x) <= _TABPFN_MAX_INP_SIZE
        assert len(_y) <= _TABPFN_MAX_INP_SIZE
        return (_x, _y)
    
    def _feat_subsample(self, X, y):
        self.feat_transformer_ = self.feature_sampler(X, y)

    def _generate_ensemble(self, X, y, stratify=None):
        iter = 0
        while iter < self.max_iters:
            data = self._subsample(X, y)
            self.ensembles_.append(data)
            iter += 1

    def _chunk_data(self, X):
        return chunker(X, self.chunk_size)

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

        sample_features = True if X.shape[1] > _TABPFN_MAX_FEAT else False

        combined_result = []
        chunks = self._chunk_data(X)
        for data in self.ensembles_:
            ensemble_result = []
            for start, end in chunks:
                X_chunk = X[start:end]
                _x, _y = data
                if sample_features:
                    _x = self.feature_sampler.sample(_x, _y)
                model.fit(_x, _y)
                ensemble_result.append(
                    (model.predict(X_chunk, return_winning_probability=return_prob))
                )
            combined_result.append(ensemble_result)

        return combined_result

    @staticmethod
    def _aggregate_preds(preds):
        preds = aggregate_arrays(preds)
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
