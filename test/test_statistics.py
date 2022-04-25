from unittest import TestCase
from numpy import ones_like

from numpy import array
from numpy.testing import assert_almost_equal

from statkit.statistics import paired_permutation_test


def mean_estimator(_, y_pred) -> float:
    """Compute mean of `y_pred` and discard `y_test`."""
    return y_pred.mean()


class TestBootstrap(TestCase):
    """Check bootstrap method with exact solutions."""

    def setUp(self):
        """Generate normal distributed data."""
        self.treatment = [
            28.44,
            29.32,
            31.22,
            29.58,
            30.34,
            28.76,
            29.21,
            30.4,
            31.12,
            31.78,
            27.58,
            31.57,
            30.73,
            30.43,
            30.31,
            30.32,
            29.18,
            29.52,
            29.22,
            30.56,
        ]
        self.control = [
            33.51,
            30.63,
            32.38,
            32.52,
            29.41,
            30.93,
            49.78,
            28.96,
            35.77,
            31.42,
            30.76,
            30.6,
            23.64,
            30.54,
            47.78,
            31.98,
            34.52,
            32.42,
            31.32,
            40.72,
        ]

    def test_with_mean_estimator(self):
        """Test a mean metric so that it coincides with conventional bootstrap."""
        # Compare with values computed using mlxtend.
        p_mlxtend = {
            "two-sided": 0.011898810118988102,
            "less": 0.006299370062993701,
            "greater": 0.9938006199380062,
        }

        # Two sided permutation test.
        _, p_2sided = paired_permutation_test(
            ones_like(self.treatment),
            array(self.treatment),
            array(self.control),
            metric=mean_estimator,
            alternative="two-sided",
            n_iterations=10000,
        )
        assert_almost_equal(p_2sided, p_mlxtend["two-sided"], decimal=3)

        # One sided smaller permutation test.
        _, p_less = paired_permutation_test(
            ones_like(self.treatment),
            array(self.treatment),
            array(self.control),
            metric=mean_estimator,
            alternative="less",
            n_iterations=10000,
        )
        assert_almost_equal(p_less, p_mlxtend["less"], decimal=3)

        # One sided greater permutation test.
        _, p_greater = paired_permutation_test(
            ones_like(self.treatment),
            array(self.treatment),
            array(self.control),
            metric=mean_estimator,
            alternative="greater",
            n_iterations=10000,
        )
        assert_almost_equal(p_greater, p_mlxtend["greater"], decimal=3)
