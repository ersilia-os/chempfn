import numpy as np
from sklearn.utils import resample
from typing import Optional, List, Type

ALL_SAMPLERS = ["bootstrap"]


class DataSampler:
    def __init__(self) -> None:
        pass

    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: Optional[np.ndarray] = None,
        n_samples: int = 1000,
    ) -> List[np.ndarray]:
        raise NotImplementedError


class BootstrapSampler(DataSampler):
    def __init__(self) -> None:
        super().__init__()

    def sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stratify: Optional[np.ndarray] = None,
        n_samples: int = 1000,
        replace: bool = True
    ) -> List[np.ndarray]:
        return resample(
            X, y, n_samples=n_samples, replace=replace, stratify=stratify
        )


def get_data_sampler(sampler_type: str) -> Type[DataSampler]:
    if sampler_type not in ALL_SAMPLERS:
        raise ValueError(
            f"Invalid data sampler provided. Must be one of {ALL_SAMPLERS}"
        )
    return {"bootstrap": BootstrapSampler}[sampler_type]