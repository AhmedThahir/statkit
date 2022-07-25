from unittest import TestCase

import numpy as np
from numpy import (
    array,
    concatenate,
    exp,
    inf,
    log,
    nan,
    ones_like,
    random,
    sqrt,
    zeros_like,
)
from numpy.random import lognormal, normal, randint, seed
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pandas import Series
from pandas.testing import assert_series_equal
from pomegranate import NaiveBayes

from statkit.distributions import (
    Bernoulli,
    InflatedGamma,
    InflatedGaussian,
    InflatedLogNormal,
    InflatedPinnedLogNormal,
    LogNormal,
    Discrete,
    Exponential,
    Gamma,
    Gaussian,
    PinnedLogNormal,
    Poisson,
    ZeroInflatedGaussian,
)


class TestGaussian(TestCase):
    """Test pseudo-count support of Gaussian distribution."""

    def test_mean_std(self):
        """Test mean computation with pseudo-count."""
        seed(1234)
        x = normal(size=10)
        n = 5
        gauss_pseudo = Gaussian(mu=x[:n].mean(), sigma=x[:n].std(), pseudo_count=n)
        gauss_pseudo.fit(x[n:])

        self.assertEqual(gauss_pseudo.parameters[0], x.mean())
        self.assertEqual(gauss_pseudo.parameters[1], x.std())


class TestPoisson(TestCase):
    """Test pseudo-count support of Poisson distribution."""

    def test_mean(self):
        """Test mean computation with pseudo-count."""
        seed(1234)
        x = np.fromiter(range(10), dtype=int)[::-1]
        n = 5
        poisson_pseudo = Poisson(l=x[:n].mean(), pseudo_count=n)
        poisson_pseudo.fit(x[n:])

        self.assertEqual(poisson_pseudo.parameters[0], x.mean())


class TestLogNormal(TestCase):
    """Test pseudo-count support of log-Normal distribution."""

    def test_mean_std(self):
        """Test mean computation with pseudo-count."""
        seed(1234)
        x = lognormal(size=10)
        logx = log(x)
        n = 5
        loggauss_pseudo = LogNormal(
            mu=logx[:n].mean(), sigma=logx[:n].std(), pseudo_count=n
        )
        loggauss_pseudo.fit(x[n:])

        assert_almost_equal(loggauss_pseudo.parameters[0], logx.mean())
        assert_almost_equal(loggauss_pseudo.parameters[1], logx.std())

    def test_pinned_log_normal(self):
        """Check that standard deviation is reset."""

        seed(1234)
        x = lognormal(size=10)
        logx = log(x)
        n = 5
        gauss_pseudo = PinnedLogNormal(
            mu=logx[:n].mean(), sigma=logx[:n].std(), pseudo_count=n
        )
        gauss_pseudo.fit(x[n:])

        # The mean is inferred from the data.
        assert_almost_equal(gauss_pseudo.parameters[0], logx.mean())
        # But the standard deviation is not.
        assert_almost_equal(gauss_pseudo.parameters[1], logx[:n].std())


class TestBernoulli(TestCase):
    """Test pseudo-count support of Bernoulli distribution."""

    def test_p(self):
        """Test mean computation with pseudo-count."""
        seed(1234)
        x = (randint(low=0, high=3, size=10) > 0).astype(int)
        n = 5
        binary_pseudo = Bernoulli(x[:n].mean(), pseudo_count=n)
        binary_pseudo.fit(x[n:])

        self.assertEqual(binary_pseudo.parameters[0], x.mean())


class TestDiscrete(TestCase):
    """Test pseudo-count support of Bernoulli distribution."""

    def test_p(self):
        """Test mean computation with pseudo-count."""
        seed(1234)
        n = 5
        cat_pseudo = Discrete({0: 2 / 5, 1: 2 / 5, 2: 1 / 5}, pseudo_count=n)
        # Data set with exact opposite of priors.
        x = array([0, 1, 1, 2, 2, 0, 0, 1, 2, 2])
        cat_pseudo.fit(x)

        p_fit = Series(cat_pseudo.parameters[0]).loc[[0, 1, 2]]
        p_true = Series({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
        assert_series_equal(p_fit, p_true)

    def test_log_likelihood(self):
        """Test vectorised implementation of log_probability."""
        p_in = {0: 2 / 5, 1: 2 / 5, 2: 1 / 5}
        x_test = array([2, nan, 1, 0])
        logp_true = array([log(1) - log(5), log(1), log(2) - log(5), log(2) - log(5)])

        logp = Discrete(p_in).log_probability(x_test)
        assert_array_almost_equal(logp, logp_true)


class TestExponential(TestCase):
    """Test pseudo-count support of Bernoulli distribution."""

    def test_p(self):
        """Test mean computation with pseudo-count."""
        seed(1234)
        x = random.exponential(scale=10, size=10)
        n = 5
        gauss_pseudo = Exponential(1 / x[:n].mean(), pseudo_count=n)
        gauss_pseudo.fit(x[n:])

        assert_almost_equal(gauss_pseudo.parameters[0], 1 / x.mean(), decimal=1e-6)


class TestGamma(TestCase):
    def test_p(self):
        """Test computation of rate and shape pseudo-count."""
        seed(1234)
        a_prior = 2
        b_prior = 3
        gamma_pseudo = Gamma(alpha=a_prior, beta=b_prior, pseudo_count=2.0)
        # Dataset with mean 1, variance 1.
        x = array([1, 3])
        gamma_pseudo.fit(x)
        a_fit, b_fit = gamma_pseudo.parameters
        # We expect both means to be weight equally (both two counts).
        assert_almost_equal(
            a_fit / b_fit, 0.5 * (a_prior / b_prior) + 0.5 * (x.mean()), decimal=1e-6
        )


class TestInflatedGaussian(TestCase):
    """Test multi-value inflated Gaussian."""

    def test_fit_zero_inflated_distributions(self):
        """The zero-inflated distribution is a special case of the multi-valued one."""
        X = array([2, 0, -2])
        p = InflatedGaussian(special_values=(0.0,)).fit(X)
        p_actual = Series(p.pi_.parameters[0]).sort_values()
        p_true = Series({0.0: 1 / 3, "complement": 2 / 3}).sort_values()
        assert_series_equal(p_actual, p_true)
        assert_array_almost_equal(p.p_complement_.parameters, [0, 2])

    def test_fit_multi_inflated_distributions(self):
        """Distribution with two values: -1 and 1."""
        X = array([2, -1, 1, 1, 1, -2])
        p = InflatedGaussian(special_values=(-1, 1)).fit(X)
        p_actual = Series(p.pi_.parameters[0]).sort_values()
        p_true = Series({1.0: 1 / 2, -1: 1 / 6, "complement": 1 / 3}).sort_values()
        assert_series_equal(p_actual, p_true)
        assert_array_almost_equal(p.p_complement_.parameters, [0, 2])

    def test_with_nan(self):
        """Test training and inference of distribution with NaN."""
        # 1) Test training with NaN.
        X = array([2, -1, nan, inf, inf, nan, inf, -2])
        p = InflatedGaussian(special_values=(-1, inf)).fit(X)
        p_actual = Series(p.pi_.parameters[0]).sort_values()
        p_true = Series({inf: 1 / 2, -1: 1 / 6, "complement": 1 / 3}).sort_values()
        assert_series_equal(p_actual, p_true)
        assert_array_almost_equal(p.p_complement_.parameters, [0, 2])

        # 2) Test inference with NaN.
        log_p = p.log_probability([0.0, nan, inf, -1])
        log_p_true = [log(1 / 3) - log(sqrt(2 * np.pi * 4)), 0, -log(2), -log(6)]
        assert_array_almost_equal(log_p, log_p_true)

    def test_pseudo_count(self):
        """Test training with pseudo counts."""
        n_pseudo = 4
        mu, sigma = 1, 2
        p_0 = 1 / 4

        # New samples to train on.
        x_samples = array([-1, -1, -1, nan, -1, 0, 0, 0, 0, 2, 2])
        p_prior = InflatedGaussian(
            special_values={0: p_0, 2: 0, "complement": 1 - p_0},
            mu=mu,
            sigma=sigma,
            pseudo_count=n_pseudo,
        ).fit(x_samples)

        # Check computed sufficient statistics with analytically calculated
        # values.
        pi_true = Series(
            {
                0: (4 / 14) * p_0 + (10 / 14) * (4 / 10),
                2: 0 + (10 / 14) * (2 / 10),
                "complement": (4 / 14) * (3 / 4) + (10 / 14) * (4 / 10),
            }
        )
        pi_fit = Series(p_prior.parameters[0])
        assert_series_equal(pi_fit, pi_true)

        mu_true = (1 / 2) * mu + (1 / 2) * -1
        assert_almost_equal(p_prior.parameters[1], mu_true)

        sigma_true = sqrt(
            (1 / 2) * (sigma**2 + mu**2) + (1 / 2) * (0 + (-1) ** 2) - 0
        )
        assert_almost_equal(p_prior.parameters[2], sigma_true)

    def test_samples(self):
        """Test that the samples coincide with the distribution."""
        seed(1234)
        p_true = InflatedGaussian(special_values=(-1, 1), mu=3, sigma=2)
        x_samples = p_true.sample(50000)
        p_fit = InflatedGaussian(special_values=(-1, 1)).fit(x_samples)

        # 10^4 samples -> we expect 3 digits (2 decimals) accuracy.
        # 1) Check distribution of inflated points.
        index_order = ["complement", -1, 1]
        pi_fit = Series(p_fit.parameters[0])
        pi_true = Series(p_true.parameters[0])
        assert_series_equal(
            pi_fit.loc[index_order], pi_true.loc[index_order], rtol=1e-2
        )

        # 2) Check parameters of complementary distribution.
        assert_array_almost_equal(
            p_true.parameters[1:], p_fit.parameters[1:], decimal=2
        )


class TestInflatedGamma(TestCase):
    def test_integer_input(self):
        """Test that integer input is correctly cast to float."""
        p = InflatedGamma(special_values=(0, 1, inf))
        p.parameters = [
            {
                "complement": 0.47838229931650567,
                0: 0.029142221834831655,
                1: 0.45616239123185354,
                inf: 0.03631308761680917,
            },
            2.4318927880348906,
            2.1306337496288323,
        ]
        assert_almost_equal(p.log_probability(1), log(0.45616239123185354))
        assert_almost_equal(p.probability(1), 0.45616239123185354)
        assert_almost_equal(p.probability(inf), 0.03631308761680917)
        assert_almost_equal(p.log_probability(inf), log(0.03631308761680917))

        x_int = array([0, 1], dtype=int)
        assert_array_almost_equal(
            p.probability(x_int), [0.029142221834831655, 0.45616239123185354]
        )
        assert_array_almost_equal(
            p.log_probability(x_int), log([0.029142221834831655, 0.45616239123185354])
        )


class TestInflatedLogNormal(TestCase):
    def test_fit_individual_distributions(self):
        """Check sufficient statistics of embedded distributions."""
        X = exp(array([2, 0, -2]))
        p = InflatedLogNormal(special_values=(1,)).fit(X)
        pi = p.parameters[0]
        assert_array_almost_equal(pi[1], [1 / 3])
        assert_array_almost_equal(p.parameters[1:], [0, 2])

    def test_fit_pinned_log_normal(self):
        """Check sufficient statistics of embedded distributions."""
        X = exp(array([2, 0, -2]))
        p = InflatedPinnedLogNormal(special_values=(1,)).fit(X)
        pi = p.parameters[0]
        assert_array_almost_equal(pi[1], [1 / 3])
        assert_array_almost_equal(p.parameters[1:], [0, 1])

    def test_with_nan_inf(self):
        """Test training and inference of distribution with NaN."""
        # 1) Test training with NaN (doesn't affect sufficient statistics).
        X = exp(array([3, nan, 0, nan, -1, inf]))
        p = InflatedLogNormal(special_values=(1, inf)).fit(X)
        pi = p.parameters[0]
        assert_almost_equal(pi[1], 1 / 4)
        assert_almost_equal(pi[inf], 1 / 4)
        assert_array_almost_equal(p.parameters[1:], [1, 2])

        # 2) Test inference with NaN.
        log_p = p.log_probability(exp([0.0, nan, 1]))
        log_p_true = [-log(4), 0, log(2 / 4) - log(sqrt(2 * np.pi * 4)) - 1]
        assert_array_almost_equal(log_p, log_p_true)


class TestZeroInflatedGaussian(TestCase):
    def test_fit_individual_distributions(self):
        """Check sufficient statistics of embedded distributions."""
        X = array([2, 0, -2])
        p = InflatedGaussian(special_values=(0,)).fit(X)
        pi = p.parameters[0]
        assert_array_almost_equal(pi[0], [1 / 3])
        assert_array_almost_equal(p.parameters[1:], [0, 2])

    def test_with_nan(self):
        """Test training and inference of distribution with NaN."""
        # 1) Test training with NaN (doesn't affect sufficient statistics).
        X = array([3, nan, 0, nan, -1])
        p = InflatedGaussian(special_values=(0,)).fit(X)
        pi = p.parameters[0]
        assert_almost_equal(pi[0], 1 / 3)
        assert_array_almost_equal(p.parameters[1:], [1, 2])

        # 2) Test inference with NaN.
        log_p = p.log_probability([0.0, nan, 1])
        log_p_true = [-log(3), 0, log(2 / 3) - log(sqrt(2 * np.pi * 4))]
        assert_array_almost_equal(log_p, log_p_true)

    def test_pseudo_count(self):
        """Test training with pseudo counts."""
        n_pseudo = 4
        mu, sigma = 1, 2
        pi = 1 / 4

        # New samples to train on.
        x_samples = array([-1, -1, -1, nan, -1, 0, 0, 0, 0])
        p_prior = ZeroInflatedGaussian(pi, mu, sigma, pseudo_count=n_pseudo).fit(
            x_samples
        )

        # Check computed sufficient statistics with analytically calculated
        # values.
        pi_true = (1 / 3) * pi + (2 / 3) * 0.5
        assert_almost_equal(p_prior.parameters[0], pi_true)

        mu_true = (1 / 2) * mu + (1 / 2) * -1
        assert_almost_equal(p_prior.parameters[1], mu_true)

        sigma_true = sqrt(
            (1 / 2) * (sigma**2 + mu**2) + (1 / 2) * (0 + (-1) ** 2) - 0
        )
        assert_almost_equal(p_prior.parameters[2], sigma_true)

    def test_samples(self):
        """Test that the samples coincide with the distribution."""
        seed(1234)
        p_true = ZeroInflatedGaussian(0.75, 3, 2)
        x_samples = p_true.sample(50000)
        p_fit = ZeroInflatedGaussian.from_samples(x_samples)
        # 10^4 samples -> we expect 3 digits (2 decimals) accuracy.
        assert_array_almost_equal(p_true.parameters, p_fit.parameters, decimal=2)

    def test_get_set_parameters(self):
        """Test `parameters` get- and setter methods."""
        X = array([2, 0, -2])
        # 1) test __get__ method by training.
        p = ZeroInflatedGaussian()
        p.fit(X)
        assert_array_almost_equal(p.parameters, [1 / 3, 0, 2])

        # 2) Set parameters and verify propagation of parameters to
        #    corresponding distribution.
        p.parameters = [2 / 3, 2, 4]
        assert_array_almost_equal(p.pi_.parameters, [2 / 3])
        assert_array_almost_equal(p.p_complement_.parameters, [2, 4])

    def test_propagation_sufficient_statistics(self):
        """Test item propagation of NaiveBayes to ZeroInflatedGaussian."""
        x_c0 = array([-3, 1, -3, nan, 1, 0, 0, 0, 0])
        x_c1 = array([3, -1, 3, -1, nan, 0, 0])
        X = concatenate([x_c0, x_c1]).reshape(-1, 1)
        y = concatenate([zeros_like(x_c0), ones_like(x_c1)])

        model = NaiveBayes.from_samples(ZeroInflatedGaussian, X, y)

        zig_c0 = model.distributions[0]
        assert_array_almost_equal(zig_c0.parameters, [1 / 2, -1, 2])

        zig_c1 = model.distributions[1]
        assert_array_almost_equal(zig_c1.parameters, [1 / 3, 1, 2])
