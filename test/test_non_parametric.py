from unittest import TestCase
from numpy import ones_like, sqrt

from numpy import array, fromiter
from numpy.testing import assert_almost_equal

from statkit.non_parametric import (
    bootstrap_score,
    paired_permutation_test,
    unpaired_permutation_test,
)


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

    def test_unpaired_permutation_test(self):
        """Test permutation test on exactly solvable combinatorial problem."""
        # Split [0, 1, .., 20] in two groups:
        # [0, 1] + [2, .., 20]
        # which is significant for 1/21 = 5 %.
        group1 = array([0, 1])
        group2 = fromiter(range(2, 21), dtype=int)
        n_total = group1.size + group2.size

        value, p_less = unpaired_permutation_test(
            y_true_1=group1,
            y_pred_1=group1,
            y_true_2=group2,
            y_pred_2=group2,
            metric=mean_estimator,
            alternative="less",
            n_iterations=10000,
        )

        self.assertEqual(value, -10.5)
        # Out of all n(n-1) permutations (diagonals terms are skipped because
        # the number of unique items in y_true is 1), the following pairs are
        # counted:
        # [0, 1] + [1, 0]
        assert_almost_equal(p_less, 2 / (n_total * (n_total - 1)), decimal=3)

        _, p_greater = unpaired_permutation_test(
            y_true_1=group1,
            y_pred_1=group1,
            y_true_2=group2,
            y_pred_2=group2,
            metric=mean_estimator,
            alternative="greater",
            n_iterations=10000,
        )
        # There are no pairs with mean smaller than [0, 1], because [0, 0] is
        # disallowed (unique items y_true_1 = 1 is forbidden).
        self.assertEqual(p_greater, 1)

        # The two-sided test is like the one-sided test, but now the following
        # pairs are counted:
        # [0, 1] + [1, 0] + [19, 20] + [20, 19]
        _, p_symmetr = unpaired_permutation_test(
            y_true_1=group1,
            y_pred_1=group1,
            y_true_2=group2,
            y_pred_2=group2,
            metric=mean_estimator,
            alternative="two-sided",
            n_iterations=10000,
        )
        assert_almost_equal(p_symmetr, 4 / (n_total * (n_total - 1)), decimal=3)

    def test_95_confidence_interval(self):
        """Test bootstrapping on exactly solvable combinatorial problem."""
        n = 60
        p = 0.5
        # Make a dataset with probability `p` of 1 and `n` items in total.
        group = array([0] * int((1 - p) * n) + [1] * int(p * n))

        # Output is bionomially distributed:
        # E[x] = np
        # Var[x] = np(1-p)
        estimate = bootstrap_score(
            y_true=group, y_pred=group, metric=mean_estimator, n_iterations=10000
        )
        assert_almost_equal(estimate["point"], p, decimal=1)
        std = sqrt(p * (1 - p) / n)
        # Gaussian: 95 % confidence interval is roughly 1.96 standard deviations.
        assert_almost_equal(estimate["lower"], p - 1.96 * std, decimal=1)
        assert_almost_equal(estimate["point"], p + 1.96 * std, decimal=1)
