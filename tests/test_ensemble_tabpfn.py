import os

import pytest
import numpy as np

from chempfn import EnsembleTabPFN
from chempfn.ensemble_builder import EnsembleBuilder

@pytest.fixture(scope="module")
def ensembles():
    X = np.random.rand(2000, 200)
    y = np.random.rand(2000)
    
    ensemble_builder = EnsembleBuilder(max_iters=10)
    ensembles = ensemble_builder.build(X, y)
    yield ensembles


class TestEnsembleTabPFN:
    SAVED_MODEL_PATH = "tests/artifacts/ensemble.pkl"
    def test_estimator_creation(self):
        """Test that the estimator can be instantiated."""
        ensemble_tabpfn = EnsembleTabPFN()
        assert ensemble_tabpfn is not None

    def test_estimator_save(self, ensembles):
        """Test that the estimator can be saved."""
        ensemble_tabpfn = EnsembleTabPFN()
        ensemble_tabpfn.ensembles_ = ensembles
        ensemble_tabpfn.save_model(self.SAVED_MODEL_PATH)
        assert os.path.exists(self.SAVED_MODEL_PATH)

    def test_estimator_load(self):
        """Test that the estimator can be loaded."""
        ensemble_tabpfn = EnsembleTabPFN.load_model(self.SAVED_MODEL_PATH)
        assert ensemble_tabpfn is not None
        assert len(ensemble_tabpfn.ensembles_) == 10
