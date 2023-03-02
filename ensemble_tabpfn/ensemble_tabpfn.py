from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch
from tabpfn import TabPFNClassifier
from typing import List, Optional

from .samplers import get_data_sampler, DataSampler
from .samplers import FeatureSampler
from .utils import Result, TabPFNConstants

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EnsembleTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iters: int = 100,
        data_sampler: str = "bootstrap",
        n_samples: int = TabPFNConstants.MAX_INP_SIZE,
        n_features: int = TabPFNConstants.MAX_INP_SIZE,
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
        n_ensemble_configurations : int, optional
            Ensemble configuration in TabPFN classifier, by default 4. A highe value will slow down prediction.
        """

        if not (n_samples <= TabPFNConstants.MAX_INP_SIZE):
            raise ValueError(
                f"n_samples must be less than or equal to {TabPFNConstants.MAX_INP_SIZE}"
            )

        if not (n_features <= TabPFNConstants.MAX_FEAT_SIZE):
            raise ValueError(
                f"n_features must be less than or equal to {TabPFNConstants.MAX_FEAT_SIZE}"
            )

        self.n_samples = n_samples
        self.n_features = n_features
        self.data_sampler: DataSampler = get_data_sampler(
            sampler_type=data_sampler
        )(n_samples=n_samples)
        self.feature_sampler = FeatureSampler(n_features=n_features)
        self.max_iters: int = max_iters
        self.n_ensemble_configurations: int = n_ensemble_configurations

    def _data_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: Optional[np.ndarray] = None,
    ):
        _x, _y = self.data_sampler.sample(X, y, stratify=stratify)
        return (_x, _y)

    def _feat_subsample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: bool = False,
    ) -> List[np.ndarray]:
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
        if X.shape[0] < self.n_samples:
            # If the input size is smaller than what TabPFN can
            # work with, then generating ensembles is not required
            self.ensembles_.append((X, y))
            return

        self._generate_ensemble(X, y)

    def _predict_with_data_ensembles(
        self, X: np.ndarray, model: TabPFNClassifier, return_prob: bool = True
    ) -> Result:
        result = Result()
        for data in self.ensembles_:
            _x, _y = data
            model.fit(_x, _y)
            pred, prob = model.predict(
                X, return_winning_probability=return_prob
            )
            result.raw_preds.append(pred)
            result.raw_probs.append(prob)
        return result

    def _predict_with_data_and_feature_ensembles(
        self, X: np.ndarray, model: TabPFNClassifier, return_prob: bool = True
    ) -> Result:
        result = Result()
        for data in self.ensembles_:
            _x, _y = data
            feature_result = Result()
            train_x_sampled_features = self._feat_subsample(_x, _y)
            test_x_sampled_features = self._feat_subsample(X, transform=True)
            for train_new, test_new in zip(
                train_x_sampled_features, test_x_sampled_features
            ):
                model.fit(train_new, _y)
                pred, prob = model.predict(
                    test_new, return_winning_probability=return_prob
                )
                feature_result.raw_preds.append(pred)
                feature_result.raw_probs.append(prob)
            feature_result.aggregate()
            result.raw_preds.append(feature_result.preds)
            result.raw_probs.append(feature_result.probs)

        return result

    def _predict(self, X: np.ndarray) -> Result:
        """Runs TabPFN predictions on ensembles of data samples and feature samples

        Checks whether features need to be sampled depending on the input data
        shape and the n_features specified during instantiation.
        Parameters
        ----------
        X : np.ndarray
            The test input of the shape (n_samples, m_features)

        Returns
        -------
        Result
            Result object containing the raw predictions and raw probabilities across ensembles.
        """
        model = TabPFNClassifier(
            device=DEVICE,
            N_ensemble_configurations=self.n_ensemble_configurations,
        )

        check_is_fitted(self, attributes="ensembles_")

        sample_features = True if X.shape[1] > self.n_features else False

        if sample_features:
            result = self._predict_with_data_and_feature_ensembles(X, model)

        else:
            result = self._predict_with_data_ensembles(X, model)

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
