from unittest import TestCase

from numpy import array, linspace
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from statkit.decision import net_benefit, net_benefit_oracle, overall_net_benefit


class TestNetBenefit(TestCase):
    def test_net_benefit(self):
        """Test bounds of net benefit curve.

        References:
        [1]: Rousson-Zumbrunn. "Decision curve analysis revisited: overall net
        benefit, relationships to ROC curve analysis, and application to case-control
        studies." BMC medical informatics and decision making 11.1 (2011): 1–9.
        """
        centers = [[0, 0], [0, 1]]

        X, y = make_blobs(
            n_samples=10_000,
            n_features=2,
            centers=centers,
            cluster_std=5,
            random_state=5,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.002)
        clf = LogisticRegression().fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]

        # Test that the net benefit does not exceed maximum net benefit pi: prevalance.
        thresholds_benefit, benefit = net_benefit(y_test, y_pred)
        self.assertTrue(all(benefit <= y_test.mean()))

        self.assertIn(0.0, thresholds_benefit)
        self.assertIn(1.0, thresholds_benefit)

    def test_oracle(self):
        """Test oracle curve with actual perfect predictor."""
        # Make perfect predictions.
        y_pred = array([0.0] * 5 + [1.0] * 6)
        y_true = array([0] * 5 + [1] * 6)
        y_pred, y_true = shuffle(y_pred, y_true, random_state=0)

        _, benefit = net_benefit(y_true, y_pred)
        assert_almost_equal(benefit, net_benefit_oracle(y_true))

        _, benefit = net_benefit(y_true, y_pred, action=False)
        assert_almost_equal(benefit, net_benefit_oracle(y_true, action=False))

        _, overall_benefit = overall_net_benefit(y_true, y_pred)
        assert_almost_equal(overall_benefit, 1.0)

    def test_net_benefit_action(self):
        """Test that action argument is equivalent to y: 0<-->1 permutation."""
        y_pred = linspace(0, 1, 11)
        y_true = array([0] * 5 + [1] * 6)
        y_pred, y_true = shuffle(y_pred, y_true, random_state=0)

        thresholds, benefit_action = net_benefit(y_true, y_pred, action=True)
        _, benefit_no_action = net_benefit(
            1 - y_true, 1.0 - y_pred, thresholds, action=False
        )

        assert_almost_equal(benefit_action, benefit_no_action[::-1])

    def test_overall_net_benefit(self):
        """Test that overall net benefit is symmetric under y: 0<--> 1 permutation."""
        y_pred = linspace(0, 1, 11)
        y_true = array([0] * 5 + [1] * 6)
        y_pred, y_true = shuffle(y_pred, y_true, random_state=0)

        _, overall_benefit = overall_net_benefit(y_true, y_pred)
        _, overall_benefit_permuted = overall_net_benefit(1 - y_true, 1.0 - y_pred)
        assert_almost_equal(overall_benefit, overall_benefit_permuted[::-1])

        # Test that the overall net benefit does not exceed 1.
        self.assertTrue(all(overall_benefit <= 1))
