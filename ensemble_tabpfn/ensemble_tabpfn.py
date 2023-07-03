from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch
from tabpfn import TabPFNClassifier
from typing import List, Optional
import pickle
import pandas as pd


from .samplers import get_data_sampler, DataSampler
from .samplers import FeatureSampler
from .utils import Result, TabPFNConstants, Ensemble

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EnsembleTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iters: int = 100,
        data_sampler: str = "bootstrap",
        n_samples: int = TabPFNConstants.MAX_INP_SIZE,
        n_features: int = TabPFNConstants.MAX_FEAT_SIZE,
        random_state: Optional[int] = None,
        early_stopping_rounds: int = 5,
        tolerance: float = 1e-4,
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
        random_state : int, optional
            Random state to use for reproducibility in data and feature subsampling, by default None.
        early_stopping_rounds : int, optional
            Number of rounds to wait for no improvement in validation loss before stopping training, by default 5.
        tolerance : float, optional
            Tolerance for early stopping, by default 1e-4.
        n_ensemble_configurations : int, optional
            Ensemble configuration in TabPFN classifier, by default 4. A highe value will slow down prediction.
        """

        if not (n_samples <= TabPFNConstants.MAX_INP_SIZE):
            raise ValueError(
                f"n_samples must be less than or equal to {TabPFNConstants.MAX_INP_SIZE}, got {n_samples}"
            )

        if not (n_features <= TabPFNConstants.MAX_FEAT_SIZE):
            raise ValueError(
                f"n_features must be less than or equal to {TabPFNConstants.MAX_FEAT_SIZE}, got {n_features}"
            )

        self.n_samples = n_samples
        self.n_features = n_features
        self.data_sampler: DataSampler = get_data_sampler(sampler_type=data_sampler)(
            n_samples=n_samples
        )
        self.feature_sampler = FeatureSampler(n_features=n_features)
        self.max_iters: int = max_iters
        self.random_state: Optional[int] = random_state
        self.early_stopping_rounds: int = early_stopping_rounds
        self.tolerance: float = tolerance
        self.n_ensemble_configurations: int = n_ensemble_configurations

    def _data_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: Optional[int] = None,
    ):
        _x, _y, indices = self.data_sampler.sample(X, y, random_state=random_state)
        return (_x, _y, indices)

    def _feat_subsample(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        transform: bool = False,
    ) -> List[np.ndarray]:
        if transform:
            return self.feature_sampler.reduce(X)
        return self.feature_sampler.sample(X, y)  # type: ignore

    def _generate_ensembles(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        # Implement early stopping
        for _iter in range(self.max_iters):
            _x, _y, indices = self._data_subsample(X, y, random_state=self.random_state)
            self._feat_subsample(_x, _y)
            ensemble = Ensemble(
                data=(_x, _y), data_indices=indices, feat_samplers=self.feature_sampler.get_samplers()
            )
            self.ensembles_.append(ensemble)

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
        self.ensembles_: List[Ensemble] = []
        self._generate_ensembles(X, y)
        self.classes_ = np.unique(y).size

    def save_model(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        return pickle.load(open(path, "rb"))

    def _predict(self, X: np.ndarray) -> Result:

        check_is_fitted(self, attributes="ensembles_")

        # TODO Move to init
        model = TabPFNClassifier(
            device=DEVICE,
            N_ensemble_configurations=self.n_ensemble_configurations,
        )

        result = Result(
            samples=X.shape[0],
            classes=self.classes_,
            tolerance=self.tolerance,
            patience=self.early_stopping_rounds,
        )

        # Set initial values
        indices = np.arange(X.shape[0])

        for itr in range(self.max_iters):
            print(f"==========Evaluating ensemble {itr}============")
            _x, _y = self.ensembles_[itr].data
            self.feature_sampler.samplers = self.ensembles_[itr].feat_samplers
            train_x_sampled_features = self._feat_subsample(_x, _y)
            test_x_sampled_features = self._feat_subsample(X, transform=True)
            for train_new, test_new in zip(
                train_x_sampled_features, test_x_sampled_features
            ):

                model.fit(train_new, _y)
                p = model.predict_proba(test_new[indices])

                curr_mean = result.prob_mean.copy()
                curr_mean[indices] = (p + result.prob_mean[indices]) / (
                    result.ensembles[indices] + 1
                )[:, None]
                result.compare_preds(curr_mean)

                indices = np.arange(X.shape[0])[result.freeze == False]
                print(f"Remaining samples: {len(indices)}")

                if len(indices) == 0:
                    break
            if len(indices) == 0:
                break
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
        return result.probs
