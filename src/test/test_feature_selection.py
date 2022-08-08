from unittest import TestCase

from numpy import array
from numpy.testing import assert_array_equal
from statkit.feature_selection import StatisticalTestFilter
from scipy.stats import ranksums


class TestStatisticalTestFilter(TestCase):
    """Test StatisticalTestFilter on exactly solvable problems."""

    def setUp(self):
        """Generate exactly solvable data."""
        self.X = array(
            [
                # Odd integers for y=0, even integers for y = 1.
                [1, 3, 5, 7, 8, 10, 12, 14],
                # As much smaller as larger numbers between y=0 and y = 0.
                # ==> (Mann-Whitney U = 0).
                [1, 3, 5, 7, 0, 2, 6, 8],
            ]
        ).transpose()
        self.y = array([0, 0, 0, 0, 1, 1, 1, 1])

    def test_mann_whitney_u_bonferroni(self):
        """Test Mann-Whitney U StatisticalTestFilter with bonferroni correction."""

        # Verify that we set up the data correctly.
        U_x2 = ranksums(self.X[self.y == 0, 1], self.X[self.y == 1, 1])
        self.assertEqual(U_x2.statistic, 0)

        filter = StatisticalTestFilter(
            statistical_test="mann-whitney-u",
            multiple_testing="bonferroni",
            p_value=0.1,
        ).fit(self.X, self.y)
        X_transformed = filter.transform(self.X)

        # Select significantly different column (= index 0)
        self.assertEqual(X_transformed.shape, (8, 1))
        assert_array_equal(X_transformed, self.X[:, [0]])

        filter_inverted = StatisticalTestFilter(
            statistical_test="mann-whitney-u",
            multiple_testing="bonferroni",
            p_value=0.1,
            invert=True,
        ).fit(self.X, self.y)
        X_transformed = filter_inverted.transform(self.X)

        # Select non-significantly different column (= index 1)
        self.assertEqual(X_transformed.shape, (8, 1))
        assert_array_equal(X_transformed, self.X[:, [1]])
