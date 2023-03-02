from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch
from tabpfn import TabPFNClassifier
from typing import List, Optional

from .samplers import get_data_sampler, DataSampler
from .samplers import get_feature_sampler, FeatureSampler
from .utils import Result, TabPFNConstants

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EnsembleTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iters: int = 100,
        data_sampler: str = "bootstrap",
        n_samples: int = TabPFNConstants.MAX_INP_SIZE,
        n_features: int = TabPFNConstants.MAX_INP_SIZE,
        feature_sampler: str = "selectk",
        n_ensemble_configurations: int = 4,
    ) -> None:
        """Ensemble TabPFN estimator class that performs data transformations to work with TabPFN.

        For training data of shape (n_samples, n_features) where n_samples exceeds 1000
        and n_features exceeds 100, creates data sub-sample ensembles and performs
        dimensionality reduction or feature extraction on each sub-sample to generate
        predictions for test data. The ensemble predictions are aggregated to return
        predictions for the target variable.

        Parameters
        ----------
        max_iters : int, optional
            Number of subsampling iterations to run on the training data, by default 100 subsampling iterations are run.
            The higher the number of subsampling iterations, the slower the prection time will be.
        data_sampler : str, optional
            Data sampler to use for subsampling data. By default, bootstrap sampling is used with replacement.
        n_samples: int, optional
            Number of data samples to inlcude per ensemble, by default 1000. It should always be less than or equal to 1000.
        n_features: int, optional
            Number of features to include per ensemble, by default 100. It should always be less than or equal to 100.
        feature_sampler : str, optional
            Feature subsampler to use. One of {"pca", "lrp", "selectk", "cluster", "random"}, default: selectk. By default, selectk with chi2 scoring is used.
        n_ensemble_configurations : int, optional
            Ensemble configuration in TabPFN classifier, by default 4. A highe value will slow down prediction.
        """

        self.data_sampler: DataSampler = get_data_sampler(
            sampler_type=data_sampler
        )(n_samples=n_samples)
        self.feature_sampler: FeatureSampler = get_feature_sampler(
            sampler_type=feature_sampler
        )(n_features=n_features)
        self.max_iters: int = max_iters
        self.n_ensemble_configurations: int = n_ensemble_configurations

    def _data_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: Optional[np.ndarray] = None,
    ):
        _x, _y = self.data_sampler.sample(X, y, stratify=stratify)
        assert len(_x) <= _TABPFN_MAX_INP_SIZE
        assert len(_y) <= _TABPFN_MAX_INP_SIZE
        return (_x, _y)

    def _feat_subsample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: bool = False,
    ) -> np.ndarray:
        if transform:
            return self.feature_sampler.reduce(X)
        return self.feature_sampler.sample(X, y)  # type: ignore

    def _generate_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: Optional[np.ndarray] = None,
    ):
        iter = 0
        while iter < self.max_iters:
            data = self._data_subsample(X, y, stratify=stratify)
            self.ensembles_.append(data)
            iter += 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Generate ensembles to use during prediction

        Parameters
        ----------
        X : np.ndarray
            Training samples of shape (n_samples, n_features)
        y : np.ndarray
            Target variable of shape(n_samples)
        """
        X, y = check_X_y(X, y, force_all_finite=False)
        self.ensembles_ = []
        if X.shape[0] < _TABPFN_MAX_INP_SIZE:
            # If the input size is smaller than what TabPFN can
            # work with, then generating ensembles is not required
            self.ensembles_.append((X, y))
            return

        self._generate_ensemble(X, y)

    def _predict(self, X: np.ndarray, return_prob: bool = True) -> Result:
        model = TabPFNClassifier(
            device=DEVICE,
            N_ensemble_configurations=self.n_ensemble_configurations,
        )

        check_is_fitted(self, attributes="ensembles_")
        result = Result()
        sample_features = True if X.shape[1] > _TABPFN_MAX_FEAT else False

        # For each data ensembles, sample features if needed
        # Fit TabPFN on ensemble of samples from the training data
        # Generate results for the test data.
        for data in self.ensembles_:
            _x, _y = data
            if sample_features:
                _x_new = self._feat_subsample(_x, _y)
            model.fit(_x_new, _y)
            X_new = self._feat_subsample(X, transform=True)
            pred, prob = model.predict(
                X_new, return_winning_probability=return_prob
            )
            result.raw_preds.append(pred)
            result.raw_probs.append(prob)

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for input samples X

        Predict class for input samples by fitting TabPFN on ensembles generated during call to fit.
        Then aggregate results for all ensembles.
        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features)

        Returns
        -------
        y : np.ndarray of shape (n_samples,)
            The predicted classes for X by aggregating results across ensembles.
        """
        result = self._predict(X)
        result.aggregate()
        return result.preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability estimates for input samples X

        Get probability estimates for input samples by fitting TabPFN on ensembles generated during call to fit.
        Then aggregate results for all ensembles.
        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features)

        Returns
        -------
        y : np.ndarray of shape (n_samples,)
            The probability estimates for X by aggregating results across ensembles.
        """
        result = self._predict(X)
        result.aggregate
        return result.probs
