from unittest import TestCase
from numpy import ones_like, sqrt, zeros

from numpy import array, arange, concatenate, stack
from numpy.testing import assert_almost_equal
from pandas import Series, concat
from sklearn.metrics import accuracy_score

from statkit.non_parametric import (
    bootstrap_score,
    paired_permutation_test,
    unpaired_permutation_test,
)


def mean_estimator(_, y_pred) -> float:
    """Compute mean of `y_pred` and discard `y_test`."""
    return y_pred.mean(axis=0)


class TestBootstrap(TestCase):
    """Check bootstrap method with exact solutions."""

    def setUp(self):
        """Generate normal distributed data."""
        self.treatment = array(
            [
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
        )
        self.control = array(
            [
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
        )

    def test_paired_permutation_test(self):
        """Test a mean metric so that it coincides with conventional bootstrap."""
        # Compare with values computed using mlxtend.
        p_mlxtend = {
            "two-sided": 0.011898810118988102,
            "less": 0.006299370062993701,
            "greater": 0.9938006199380062,
        }

        # Make treatment and control matrices (where we invert the roles between the 1st
        # and 2nd column).
        x_treatment = stack([self.treatment, self.control], axis=1)
        x_control = stack([self.control, self.treatment], axis=1)

        # Two sided permutation test.
        p_2sided = paired_permutation_test(
            ones_like(self.treatment),
            x_treatment,
            x_control,
            metric=mean_estimator,
            alternative="two-sided",
            n_iterations=10000,
            random_state=1234,
        )
        # Since the test is two sided, the values should be the same.
        assert_almost_equal(p_2sided, p_mlxtend["two-sided"], decimal=3)

        # Do the same analysis but with a column instead of matrix.
        p_2sided_scalar = paired_permutation_test(
            ones_like(self.treatment),
            x_treatment[:, 0],
            x_control[:, 0],
            metric=mean_estimator,
            alternative="two-sided",
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(p_2sided_scalar, p_mlxtend["two-sided"], decimal=3)

        # One sided smaller permutation test.
        p_less = paired_permutation_test(
            ones_like(self.treatment),
            x_treatment,
            x_control,
            metric=mean_estimator,
            alternative="less",
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(
            p_less, [p_mlxtend["less"], p_mlxtend["greater"]], decimal=3
        )

        # One sided greater permutation test.
        p_greater = paired_permutation_test(
            ones_like(self.treatment),
            x_treatment,
            x_control,
            metric=mean_estimator,
            alternative="greater",
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(
            p_greater, [p_mlxtend["greater"], p_mlxtend["less"]], decimal=3
        )

    def test_unpaired_permutation_test(self):
        """Test permutation test on exactly solvable combinatorial problem."""
        # Split [0, 1, .., 20] in two groups.
        # In the 1st column, the groups are:
        # [0, 1] + [2, .., 20]
        # In the 2nd column:
        # [0, .., 19] + [19, 20]
        # which is significant for 1/21 = 5 %.
        group1 = array([[0, 1], [19, 20]]).transpose()
        group2 = stack([arange(2, 21), arange(0, 19)], axis=-1)
        n_total = group1.shape[0] + group2.shape[0]

        p_less = unpaired_permutation_test(
            y_true_1=group1,
            y_pred_1=group1,
            y_true_2=group2,
            y_pred_2=group2,
            metric=mean_estimator,
            alternative="less",
            n_iterations=10000,
            random_state=1234,
        )

        # Out of all n(n-1) permutations (diagonals terms are skipped because
        # the number of unique items in y_true is 1), the following pairs are
        # counted:
        # [0, 1] + [1, 0]
        assert_almost_equal(p_less[0], 2 / (n_total * (n_total - 1)), decimal=3)
        # All pairs have a mean smaller or equal to [19, 20], because [20, 20] is
        # disallowed (unique items y_true_1 = 1 is forbidden).
        assert_almost_equal(p_less[1], 1)

        p_greater = unpaired_permutation_test(
            y_true_1=group1,
            y_pred_1=group1,
            y_true_2=group2,
            y_pred_2=group2,
            metric=mean_estimator,
            alternative="greater",
            n_iterations=10000,
            random_state=1234,
        )
        # There are no pairs with mean smaller than [0, 1], because [0, 0] is
        # disallowed (unique items y_true_1 = 1 is forbidden).
        self.assertEqual(p_greater[0], 1)
        assert_almost_equal(p_greater[1], 2 / (n_total * (n_total - 1)), decimal=3)

        # The two-sided test is like the one-sided test, but now the following
        # pairs are counted:
        # [0, 1] + [1, 0] + [19, 20] + [20, 19]
        p_symmetr = unpaired_permutation_test(
            y_true_1=group1,
            y_pred_1=group1,
            y_true_2=group2,
            y_pred_2=group2,
            metric=mean_estimator,
            alternative="two-sided",
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(p_symmetr, 4 / (n_total * (n_total - 1)), decimal=3)

        # 2) Test implementation with vector instead of matrix.
        p_symmetr_scalar = unpaired_permutation_test(
            y_true_1=group1[:, 0],
            y_pred_1=group1[:, 0],
            y_true_2=group2[:, 0],
            y_pred_2=group2[:, 0],
            metric=mean_estimator,
            alternative="two-sided",
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(p_symmetr_scalar, 4 / (n_total * (n_total - 1)), decimal=3)

    def test_95_confidence_interval(self):
        """Test bootstrapping on exactly solvable combinatorial problem."""
        n = 120
        p = array([0.5, 0.75])
        std = sqrt(p * (1 - p) / n)

        # Make a dataset with probability `p` of 1 and `n` items in total.
        group = zeros((n, p.shape[0]))
        mask = arange(n).reshape(-1, 1) < p * n
        group[mask] = 1

        # Output is bionomially distributed:
        # E[x] = p
        # Var[x] = p(1-p)/n

        # 1) Test matrix implementation.
        estimate = bootstrap_score(
            y_true=group,
            y_pred=group,
            metric=mean_estimator,
            n_iterations=10000,
            random_state=1234,
        )

        self.assertEqual(estimate.point.shape, p.shape)
        self.assertEqual(estimate.lower.shape, p.shape)
        self.assertEqual(estimate.upper.shape, p.shape)

        assert_almost_equal(estimate.point, p, decimal=1)
        # Gaussian: 95 % confidence interval is roughly 1.96 standard deviations.
        assert_almost_equal(estimate.lower, p - 1.96 * std, decimal=1)
        assert_almost_equal(estimate.point, p + 1.96 * std, decimal=1)

        # 2) Test with vector instead of matrix.
        estimate = bootstrap_score(
            y_true=group[:, 0],
            y_pred=group[:, 0],
            metric=mean_estimator,
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(estimate.point, p[0], decimal=1)
        # Gaussian: 95 % confidence interval is roughly 1.96 standard deviations.
        assert_almost_equal(estimate.lower, p[0] - 1.96 * std[0], decimal=1)
        assert_almost_equal(estimate.point, p[0] + 1.96 * std[0], decimal=1)

    def test_bootstrap_no_variability(self):
        """Test confidence interval for `y_true` and `y_pred` identical."""
        y_true = y_pred = array([0, 0, 0, 0])
        estimate = bootstrap_score(y_true, y_pred, accuracy_score, n_iterations=100)
        self.assertEqual(estimate.point, 1.0)
        self.assertEqual(estimate.lower, 1.0)
        self.assertEqual(estimate.upper, 1.0)

    def test_bootstrap_positive_class(self):
        """Test that we can specify a positive class."""
        n = 60
        p = 0.5
        # Make a dataset with probability `p` of 1 and `n` items in total.
        y_np = array([0] * int((1 - p) * n) + [1] * int(p * n))

        # Output is bionomially distributed, like the test above. But now we invert the
        # positive class so that:
        # E[x] = (1-p)
        # Var[x] = p(1-p)/n

        # First test as numpy array.
        estimate_np = bootstrap_score(
            y_true=y_np,
            y_pred=y_np,
            metric=mean_estimator,
            pos_label=0,
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(estimate_np.point, 1 - p, decimal=1)
        std = sqrt(p * (1 - p) / n)
        # Gaussian: 95 % confidence interval is roughly 1.96 standard deviations.
        assert_almost_equal(estimate_np.lower, p - 1.96 * std, decimal=1)
        assert_almost_equal(estimate_np.point, p + 1.96 * std, decimal=1)

        # Secondly, test as pandas Series, but now we don't invert the positive class,
        # E[x] = p
        # but make the positive label ordered as follows:
        # positive class = "a" < "b" = negative class
        y_pd = Series(y_np).replace({0: "b", 1: "a"})
        estimate_pd = bootstrap_score(
            y_true=y_pd,
            y_pred=y_np,
            metric=mean_estimator,
            pos_label="a",
            n_iterations=10000,
            random_state=1234,
        )
        assert_almost_equal(estimate_pd.point, p, decimal=1)
        std = sqrt(p * (1 - p) / n)
        # Gaussian: 95 % confidence interval is roughly 1.96 standard deviations.
        assert_almost_equal(estimate_pd.lower, p - 1.96 * std, decimal=1)
        assert_almost_equal(estimate_pd.point, p + 1.96 * std, decimal=1)

        # Verify that an assertion is raised when label not a binary class while
        # pos_label is specified.
        with self.assertRaises(ValueError):
            bootstrap_score(
                # Add two extra labels -> no longer binary classification.
                y_true=concat([y_pd, y_pd.replace({"a": "d", "b": "c"})]),
                y_pred=concatenate([y_np, y_np]),
                metric=mean_estimator,
                pos_label="a",
                n_iterations=100,
                random_state=1234,
            )
