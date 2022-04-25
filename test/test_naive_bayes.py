from unittest import TestCase

from numpy import (
    array,
    concatenate,
    exp,
    inf,
    log,
    nan,
    ones,
    ones_like,
    sqrt,
    zeros,
    zeros_like,
)
from numpy.random import randint
from numpy.testing import assert_array_almost_equal
from pandas import DataFrame, Series
import pomegranate as pg
from pomegranate import NaiveBayes, NormalDistribution
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from statkit.naive_bayes import _BaseNaiveBayes, NaiveBayesClassifier
from statkit.distributions import (
    Gaussian,
    InflatedPinnedLogNormal,
    LogNormal,
    ZeroInflatedGaussian,
)


class TestBaseNaiveBayes(TestCase):
    def setUp(self):
        X, y = make_blobs(n_samples=100, n_features=1, centers=2, random_state=1234)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_pomegranate_equivalence(self):
        """Test that no initialisation is equivalent to pomegranate NaiveBayes."""
        p_kwargs = {
            Gaussian: {
                0: {"mean": 0, "std": 1},
                1: {"mean": 0, "std": 1},
            }
        }
        m_pred = _BaseNaiveBayes(
            [Gaussian],
            distribution_kwargs=p_kwargs,
        ).fit(self.X_train, self.y_train)
        m_reference = pg.NaiveBayes.from_samples(
            [NormalDistribution, NormalDistribution], self.X_train, self.y_train
        )

        assert_array_almost_equal(
            m_pred.predict_proba(self.X_test), m_reference.predict_proba(self.X_test)
        )

    def test_pseudo_count_equivalence(self):
        """Test with class dependent pseudo counts."""
        # First initialise distributions with sufficient statistics of first 50
        # examples.
        X_lower_train, y_lower_train = self.X_train[:50], self.y_train[:50]
        X_upper_train, y_upper_train = self.X_train[50:], self.y_train[50:]

        is_y0 = y_lower_train == 0
        n_y0 = is_y0.astype(int).sum()
        n_y1 = (~is_y0).astype(int).sum()

        p_args = {
            Gaussian: {
                0: {
                    "mean": X_lower_train[is_y0].mean(),
                    "std": X_lower_train[is_y0].std(),
                    "pseudo_count": n_y0,
                },
                1: {
                    "mean": X_lower_train[~is_y0].mean(),
                    "std": X_lower_train[~is_y0].std(),
                    "pseudo_count": n_y1,
                },
            }
        }
        # Train on remaining 50 examples.
        distributions = [Gaussian]
        m_pred = _BaseNaiveBayes(
            distributions,
            distribution_kwargs=p_args,
        ).fit(X_upper_train, y_upper_train)

        # Compare with Pomegranate model that trains on all data in one go.
        m_reference = pg.NaiveBayes.from_samples(
            NormalDistribution, self.X_train, self.y_train
        )

        # Check class 0 Gaussian.
        assert_array_almost_equal(
            m_pred.model_.distributions[0].distributions[0].parameters,
            m_reference.distributions[0].parameters,
        )
        # Check class 1 Gaussian.
        assert_array_almost_equal(
            m_pred.model_.distributions[1].distributions[0].parameters,
            m_reference.distributions[1].parameters,
        )

    def test_fit_with_prior(self):
        """Test model training with class-dependent priors."""
        x_c0 = array([-1, -1, -1, nan, -1, 0, 0, 0, 0])
        x_c1 = array([0, -0.5, 0.5, nan])
        X = concatenate([x_c0, x_c1]).reshape(-1, 1)
        y = concatenate([zeros_like(x_c0), ones_like(x_c1)])

        c0_n_pseudo = 4
        c0_prior_pi = 1 / 4
        c0_prior_mu = 1
        c0_prior_sigma = 2

        c1_n_pseudo = 2
        c1_prior_pi = 1
        c1_prior_mu = -1
        c1_prior_sigma = 1
        p_args = {
            ZeroInflatedGaussian: {
                0: {
                    "pi": c0_prior_pi,
                    "mean": c0_prior_mu,
                    "std": c0_prior_sigma,
                    "pseudo_count": c0_n_pseudo,
                },
                1: {
                    "pi": c1_prior_pi,
                    "mean": c1_prior_mu,
                    "std": c1_prior_sigma,
                    "pseudo_count": c1_n_pseudo,
                },
            }
        }
        m_pred = _BaseNaiveBayes(
            [ZeroInflatedGaussian],
            distribution_kwargs=p_args,
        ).fit(X, y)

        # Compute new coefficients of distributions after training, taking into
        # account priors.
        # 1) c=0.
        pi_true = (1 / 3) * c0_prior_pi + (2 / 3) * 0.5
        mu_true = (1 / 2) * c0_prior_mu + (1 / 2) * -1
        sigma_true = sqrt(
            (1 / 2) * (c0_prior_sigma ** 2 + c0_prior_mu ** 2)
            + (1 / 2) * (0 + (-1) ** 2)
            - 0
        )
        zig_c0 = m_pred.model_.distributions[0].distributions[0]
        assert_array_almost_equal(zig_c0.parameters, [pi_true, mu_true, sigma_true])

        # 2) c=1.
        pi_true = (2 / 5) * c1_prior_pi + (3 / 5) * 1 / 3
        mu_true = (1 / 2) * c1_prior_mu + (1 / 2) * 0
        sigma_true = sqrt(
            (1 / 2) * (c1_prior_sigma ** 2 + c1_prior_mu ** 2)
            + (1 / 2) * (0.5 ** 2 + (0) ** 2)
            - mu_true ** 2
        )
        zig_c1 = m_pred.model_.distributions[1].distributions[0]
        assert_array_almost_equal(zig_c1.parameters, [pi_true, mu_true, sigma_true])


class TestNaiveBayesClassifier(TestCase):
    """Test sklearn-wrapped Pomegranate model."""

    def test_pomegranate_equivalence(self):
        """Test that the sklearn wrapped model gives same predictions."""
        X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=1234)
        X = DataFrame(X)
        y = Series(y)

        # Randomly corrupt some 20 % of the data.
        mask = randint(low=0, high=5, size=[100, 2]) == 0
        X[mask] = nan
        X_train, X_test, y_train, _ = train_test_split(X, y)

        m = NaiveBayesClassifier(distributions=Gaussian, pseudo_count=0).fit(
            X_train, y_train
        )
        m_reference = pg.NaiveBayes.from_samples(
            NormalDistribution, X_train.to_numpy(), y_train.to_numpy()
        )
        assert_array_almost_equal(
            m.predict_proba(X_test),
            m_reference.predict_proba(X_test.to_numpy()),
        )

    def test_feature_importance(self):
        """Test that sum over feature importance is just prediction prob."""
        X, y = make_blobs(n_samples=100, n_features=1, centers=2, random_state=1234)
        X = DataFrame(X)
        y = Series(y)

        # Randomly corrupt some 20 % of the data.
        mask = randint(low=0, high=5, size=[100, 1]) == 0
        X[mask] = nan
        X_train, X_test, y_train, _ = train_test_split(X, y)

        m = NaiveBayesClassifier(distributions=Gaussian).fit(X_train, y_train)
        importance = m.feature_importance(X_test).sum(axis="columns")

        # Note that:
        # ln p(y=1|x) - ln p(y=0|x)
        # = ln [p(y=1|x) p(x)] - ln [p(y=0|x) p(x)] = ln p(y=1,x) - ln p(y=0,x).
        logp_y_cond_x = log(m.predict_proba(X_test))
        assert_array_almost_equal(
            logp_y_cond_x.iloc[:, 1] - logp_y_cond_x.iloc[:, 0],
            importance,
        )

    def test_weights(self):
        """Test that the weighing of samples is correctly accounted for."""
        X = array([[1, 3, 2, 4]]).T
        y = zeros(X.size)

        # Similarly weighted, with pseudo counts.
        m_no_weights = NaiveBayesClassifier(
            distributions=lambda **kwargs: Gaussian(1, 1, **kwargs),
            pseudo_count=2,
        ).fit(X, y, ones(X.size))
        p_gauss = m_no_weights.model_.distributions[0].distributions[0]
        self.assertEqual(p_gauss.parameters[0], (1 / 3) * 1 + (1 / 3) * 2 + (1 / 3) * 3)

        # Weighted, no pseudo counts.
        weights = array([1, 1, 2, 2])
        m_weighted = NaiveBayesClassifier(
            distributions=lambda **kwargs: Gaussian(1, 1, **kwargs),
            pseudo_count=0,
        ).fit(X, y, weights)
        p_gauss = m_weighted.model_.distributions[0].distributions[0]
        self.assertEqual(p_gauss.parameters[0], (1 / 3) * 2 + (2 / 3) * 3)

        # Weighted, with psuedo counts.
        weights = array([1, 1, 2, 2])
        m_weighted = NaiveBayesClassifier(
            distributions=lambda **kwargs: Gaussian(1, 1, **kwargs),
            pseudo_count=2,
        ).fit(X, y, weights)
        p_gauss = m_weighted.model_.distributions[0].distributions[0]
        self.assertEqual(p_gauss.parameters[0], (1 / 4) * 1 + (1 / 4) * 2 + (2 / 4) * 3)

    def test_pseudo_count_propagation(self):
        """Test propagation of `pseudo_count` parameter to distributions."""
        x_c0 = array([-1, 3, -1, 3, nan])  # mean: 1; std: 2.
        x_c1 = array([-3, 1, nan, -3, 1])  # mean: -1; std: 2.
        x1 = concatenate([x_c0, x_c1])
        # Create dataset where x1 and x2 are each others class opposite.
        x2 = concatenate([x_c1, x_c0])
        X = array([x1, x2]).T

        y = concatenate([zeros_like(x_c0), ones_like(x_c1)])
        m_pred = NaiveBayesClassifier(distributions=Gaussian, pseudo_count=2.0).fit(
            X, y
        )

        # Gaussians of class 0.
        g1_c0 = m_pred.model_.distributions[0].distributions[0]
        g2_c0 = m_pred.model_.distributions[0].distributions[1]
        # Gaussians of class 1.
        g1_c1 = m_pred.model_.distributions[1].distributions[0]
        g2_c1 = m_pred.model_.distributions[1].distributions[1]
        # We swapped the class-dependent features between x1 and x2.
        assert_array_almost_equal(
            g1_c0.parameters,
            g2_c1.parameters,
        )
        assert_array_almost_equal(
            g2_c0.parameters,
            g1_c1.parameters,
        )

        # Check means.
        mu = 2 / 3 * array([1, -1])
        assert_array_almost_equal(g1_c0.parameters[0], mu[0])
        assert_array_almost_equal(g1_c1.parameters[0], mu[1])
        var = 1 / 3 * 1 + 2 / 3 * (4 + array([1, -1]) ** 2) - mu ** 2
        sigma = sqrt(var)
        # Check variances.
        assert_array_almost_equal(g1_c0.parameters[1], sigma[0])
        assert_array_almost_equal(g1_c1.parameters[1], sigma[1])

    def test_distribution_specific_pseudo_counts(self):
        """Test propagation of different pseudo counts per distribution."""
        x_c0 = array([-1, 3, -1, 3, nan])  # mean: 1; std: 2.
        x_c1 = array([-3, 1, nan, -3, 1])  # mean: -1; std: 2.
        x1 = concatenate([x_c0, x_c1])
        # Create dataset where `x1` is for a normal distribution and x2 for a
        # log normal distribution.
        x2 = concatenate([x_c1, x_c0])
        X = array([x1, exp(x2)]).T

        y = concatenate([zeros_like(x_c0), ones_like(x_c1)])
        m_pred = NaiveBayesClassifier(
            distributions=[Gaussian, LogNormal],
            pseudo_count={Gaussian: 2.0, LogNormal: 1.0},
        ).fit(X, y)

        # Normal and log normal of class 0.
        g1_c0 = m_pred.model_.distributions[0].distributions[0]
        lg2_c0 = m_pred.model_.distributions[0].distributions[1]
        # Normal and log normal of class 1.
        g1_c1 = m_pred.model_.distributions[1].distributions[0]
        lg2_c1 = m_pred.model_.distributions[1].distributions[1]

        # 2 pseudo observations for normal and 1 pseudo observation for log
        # normal + 4 data observations.
        pseudo_weights = array([4 / 6, 4 / 5])
        mu_prior = array([0, 0])
        mu_c0_data = array([1, -1])
        mu_c1_data = array([-1, 1])

        mu_c0_true = (1 - pseudo_weights) * mu_prior + pseudo_weights * mu_c0_data
        mu_c1_true = (1 - pseudo_weights) * mu_prior + pseudo_weights * mu_c1_data

        # Verify computed means.
        # i) c=0.
        assert_array_almost_equal(g1_c0.parameters[0], mu_c0_true[0])
        assert_array_almost_equal(lg2_c0.parameters[0], mu_c0_true[1])
        # ii) c=1.
        assert_array_almost_equal(g1_c1.parameters[0], mu_c1_true[0])
        assert_array_almost_equal(lg2_c1.parameters[0], mu_c1_true[1])

    def test_zero_probability_detection(self):
        """Test that the model detects distribution with zero probability."""
        # Log normal with inflation points: 0, 1, inf == log ==> -inf, 0, inf.
        # Generate a dataset without inflation point 1 (log 1 = 0) =>
        # probability is zero. This should trigger an exception without pseudo
        # counts.

        # Non inflated points: mean = 1; std = 2.
        x_c0 = array([-1, 3, -1, 3, nan, -inf, inf])
        # Non inflated points: mean: -1; std: 2.
        x_c1 = array([-3, 1, nan, -3, 1, -inf, inf])

        # Exponentiate features for log normal distribution.
        x1 = concatenate([x_c0, x_c1])
        x2 = concatenate([x_c1, x_c0])
        X = array([exp(x1), exp(x2)]).T

        y = concatenate([zeros_like(x_c0), ones_like(x_c1)])

        with self.assertRaises(AssertionError):
            NaiveBayesClassifier(
                distributions=[InflatedPinnedLogNormal, InflatedPinnedLogNormal],
                pseudo_count=0,
            ).fit(X, y)

    def test_schema_skew(self):
        """Test that schema skew raises an exception."""
        X, y = make_blobs(n_features=2)
        X = DataFrame({"a": X[:, 0], "b": X[:, 1]})
        model = NaiveBayesClassifier(distributions=Gaussian).fit(X, y)

        # Check that incongruent schema raises exceptions.
        X_permuted = X[["b", "a"]]
        with self.assertRaises(KeyError):
            model.predict(X_permuted)

        with self.assertRaises(KeyError):
            model.predict_proba(X_permuted)

        with self.assertRaises(KeyError):
            model.feature_importance(X_permuted)

        with self.assertRaises(KeyError):
            model.score(X_permuted, y)

        # Check that duplicated columns raise exceptions.
        X_dupl = X[["a", "a", "b"]]
        with self.assertRaises(KeyError):
            model.predict(X_dupl)

        with self.assertRaises(KeyError):
            model.predict_proba(X_dupl)

        with self.assertRaises(KeyError):
            model.feature_importance(X_dupl)

        with self.assertRaises(KeyError):
            model.score(X_dupl, y)