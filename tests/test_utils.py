import numpy as np
import pytest
from chempfn.result import Result


@pytest.mark.skip(reason="Implementation Changed")
#TODO Update tests
class TestAggregate:
    raw_preds = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    raw_probs = np.array([0.7, 0.2, 0.2, 0.6, 0.7, 0.3, 0.4, 0.4, 0.8, 0.9])

    def test_aggregate_1_ensemble(self):
        epfn_res = Result()
        epfn_res.raw_preds = [self.raw_preds]
        epfn_res.raw_probs = [self.raw_probs]
        expected_preds = self.raw_preds
        expected_probs = self.raw_probs
        epfn_res.aggregate()
        assert np.array_equal(epfn_res.preds, expected_preds)
        assert np.array_equiv(epfn_res.probs, expected_probs)
 
    def test_aggregate_n_ensemble(self):
        epfn_res = Result()
        epfn_res.raw_preds = [self.raw_preds]*3
        epfn_res.raw_probs = [self.raw_probs]*3
        expected_preds = self.raw_preds
        expected_probs = self.raw_probs
        epfn_res.aggregate()
        assert np.array_equal(epfn_res.preds, expected_preds)
        # assert np.array_equal(epfn_res.probs, expected_probs)