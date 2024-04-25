import logging
import pickle
from typing import Optional

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB as BaselineClassifier
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch
from tabpfn import TabPFNClassifier

from .result import Result
from .ensemble_builder import EnsembleBuilder
from .samplers import FeatureSampler

# TODO maybe add support for mps backend for Macs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: abstract to utils
logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)


class EnsembleTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_iters: int = 10,
        random_state: Optional[int] = None,
        early_stopping_rounds: int = 5,
        tolerance: float = 1e-2,
        n_ensemble_configurations: int = 4,
        verbose: bool = True,  # TODO: very hacky, there should be a better way to do this
        baseline: bool = False,
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
            Number of rounds to wait for no improvement in validation loss before stopping training, by default 5. This sets the patience.
        tolerance : float, optional
            Tolerance for early stopping, by default 1e-4.
        n_ensemble_configurations : int, optional
            Ensemble configuration in TabPFN classifier, by default 4. A highe value will slow down prediction.
        baseline: bool, optional
            Use a simple baseline classifier instead of TabPFN, by default False
        """
        if verbose:
            logger.setLevel(logging.DEBUG)

        self.max_iters: int = max_iters
        self.random_state: Optional[int] = random_state
        self.early_stopping_rounds: int = early_stopping_rounds
        self.tolerance: float = tolerance
        self.n_ensemble_configurations: int = n_ensemble_configurations
        logger.debug(f"Device: {DEVICE}")
        if not baseline:
            self.model = TabPFNClassifier(
                device=DEVICE,
                N_ensemble_configurations=self.n_ensemble_configurations,
            )
        else:
            self.model = BaselineClassifier()

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
        bldr = EnsembleBuilder(self.max_iters, self.random_state)
        self.ensembles_ = bldr.build(X, y)
        self.classes_ = np.unique(y).size

    def save_model(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.ensembles_, f)

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.ensembles_ = pickle.load(open(path, "rb"))
        return model

    def _predict(self, X: np.ndarray) -> Result:
        check_is_fitted(self, attributes="ensembles_")

        feature_sampler = FeatureSampler()

        result = Result(
            samples=X.shape[0],
            classes=self.classes_,
            tolerance=self.tolerance,
            patience=self.early_stopping_rounds,
        )

        # Set initial values
        indices = np.arange(X.shape[0])

        for itr in range(self.max_iters):
            logger.debug(f"==========Evaluating ensemble {itr}============")
            train_x_sampled_features, _y = self.ensembles_[itr].data
            feature_sampler.samplers = self.ensembles_[itr].feat_samplers
            test_x_sampled_features = feature_sampler.reduce(X)
            for train_new, test_new in zip(
                train_x_sampled_features, test_x_sampled_features
            ):
                self.model.fit(train_new, _y)
                p = self.model.predict_proba(test_new[indices])

                curr_mean = result.prob_mean.copy()
                curr_mean[indices] = (p + result.prob_mean[indices]) / (
                    result.ensembles[indices] + 1
                )[:, None]
                result.compare_preds(curr_mean)

                indices = np.arange(X.shape[0])[result.freeze == False]
                logger.debug(f"Remaining samples: {len(indices)}")

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
