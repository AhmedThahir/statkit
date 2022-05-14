from unittest import TestCase

from numpy.testing import assert_almost_equal

from statkit.metrics import (
    false_positive_rate,
    sensitivity,
    specificity,
    true_positive_rate,
    youden_j,
)


class TestMetrics(TestCase):
    def setUp(self):
        """Make up data where we know the answer."""
        # Lets construct data with:
        # - true positives: 2
        # - true negatives: 3
        # - false negatives: 4
        # - false positives: 5.
        self.y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.y_pred = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def test_sensitivity_specificity(self):
        """Test computation of sensitivity and specificity."""
        self.assertEqual(sensitivity(self.y_true, self.y_pred), 2 / (2 + 4))
        self.assertEqual(specificity(self.y_true, self.y_pred), 3 / (3 + 5))

    def test_youden_j(self):
        """Test expression with equivalent formulations."""
        # Youden J: true postives / (true positive + false negatives) +
        # true negatives / (true negatives + false positives) -1.

        assert_almost_equal(
            youden_j(self.y_true, self.y_pred),
            sensitivity(self.y_true, self.y_pred)
            + specificity(self.y_true, self.y_pred)
            - 1,
        )
        assert_almost_equal(
            youden_j(self.y_true, self.y_pred),
            true_positive_rate(self.y_true, self.y_pred)
            - false_positive_rate(self.y_true, self.y_pred),
        )
