import os
from operator import itemgetter

import pytest
import numpy as np
from tdc.single_pred import ADME

from chempfn import ChemPFN


@pytest.fixture(scope="module")
def data():
    data = ADME(name="HIA_Hou")
    split = data.get_split()
    train, test, valid = itemgetter("train", "test", "valid")(split)
    yield (train, test)


class TestChemPFN:
    def test_estimator_creation(self):
        """Test that the estimator can be instantiated."""
        chempfn = ChemPFN()
        assert chempfn.eosce is not None
        assert chempfn.etpfn is not None

    def test_chempfn_fit(self, data):
        """Test that the estimator can be fit."""
        chempfn = ChemPFN()
        train, test = data
        chempfn.fit(train["Drug"].to_list(), train["Y"].to_list())
        assert chempfn.etpfn.ensembles_ is not None

    def test_chempfn_predict(self, data):
        """Test that the estimator can predict."""
        chempfn = ChemPFN()
        train, test = data
        chempfn.fit(train["Drug"].to_list(), train["Y"].to_list())
        preds = chempfn.predict(test.Drug.to_list())
        assert preds is not None
        print(preds.head(10))
