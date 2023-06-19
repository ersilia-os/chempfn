import numpy as np
from typing import Optional, List, Type
from ..utils import TabPFNConstants

ALL_SAMPLERS = ["bootstrap"]


class DataSampler:
    def __init__(self, n_samples: int = TabPFNConstants.MAX_INP_SIZE) -> None:
        """Base DataSampler class.

        Parameters
        ----------
        n_samples : int, optional
            Number of input examples to be included in a sample, by default TabPFNConstants.MAX_INP_SIZE
        """
        self.n_samples = n_samples

    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: Optional[int] = None,
    ) -> List[np.ndarray]:
        raise NotImplementedError


class BootstrapSampler(DataSampler):
    """Perform bootstrap sampling on data with replacement"""

    def __init__(self, n_samples: int = TabPFNConstants.MAX_INP_SIZE) -> None:
        super().__init__(n_samples=n_samples)

    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        replace: bool = True,
        random_state: Optional[int] = None,
    ) -> List[np.ndarray]:
        np.random.seed(random_state)
        indices = np.random.choice(
            X,
            n_samples=self.n_samples,
            replace=replace
        )
        return [X[indices], y[indices], indices]


def get_data_sampler(sampler_type: str) -> Type[DataSampler]:
    if sampler_type not in ALL_SAMPLERS:
        raise ValueError(
            f"Invalid data sampler provided. Must be one of {ALL_SAMPLERS}"
        )
    return {"bootstrap": BootstrapSampler}[sampler_type]
