from unittest import TestCase

from numpy.testing import assert_almost_equal

from statkit.power import sample_size_roc_auc


class TestPower(TestCase):
    def test_sample_size_roc_auc(self):
        """Test with quoted values from paper."""
        assert_almost_equal(
            sample_size_roc_auc(0.825, 0.90, alpha=0.05, power=0.8)[0], 176, decimal=0
        )
        assert_almost_equal(
            sample_size_roc_auc(0.825, 0.90, alpha=0.05, power=0.9)[0], 239, decimal=0
        )
        assert_almost_equal(
            sample_size_roc_auc(0.825, 0.90, alpha=0.05, power=0.95)[0], 298, decimal=0
        )
