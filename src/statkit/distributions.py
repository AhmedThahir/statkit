"""
Extension of pomegranate distributions to support pseudo counts and value inflation.
"""
from abc import ABC, abstractmethod
from typing import Union

from numpy import (
    array,
    fromiter,
    full_like,
    inf,
    isin,
    isnan,
    log,
    ndarray,
    ones_like,
    zeros,
    zeros_like,
)
from numpy.testing import assert_almost_equal
import pomegranate as pg
from pomegranate.distributions import (
    Distribution,
)
from scipy.special import digamma


class Gaussian(pg.NormalDistribution):
    r"""Extension of Pomegranate Gaussian supporting pseudo_counts.

    $$
    p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
    $$
    """

    name = "NormalDistribution"

    def __init__(
        self, mu: float = 0.0, sigma: float = 1.0, pseudo_count: float = 0.0, **kwargs
    ):
        r"""
        Args:
            mu: Mean, \( \mu \), or location of Gaussian.
            sigma: Standard deviation, \( \sigma \), measuring the width/scale of the
                Gaussian.
        """
        super().__init__(mu, sigma, **kwargs)
        self.pseudo_count = pseudo_count

        # Pseudo-count hack: compute summaries so that the next `fit` will weigh
        # the prior mean and variance as if they were coming from `pseudo_count`
        # observations.
        if self.pseudo_count > 0:
            x2 = sigma**2 + mu**2
            self.summaries = [
                self.pseudo_count,
                mu * self.pseudo_count,
                x2 * self.pseudo_count,
            ]


class LogNormal(pg.LogNormalDistribution):
    r"""Extension of Pomegranate log-normal distribution supporting pseudo_counts.

    $$
    p(x) = \frac{1}{\sqrt{2\pi \sigma^2 x^2}}
        \exp \left( \frac{(\ln x - \mu)^2}{2\sigma^2} \right).
    $$
    """

    name = "LogNormalDistribution"

    def __init__(
        self, mu: float = 0.0, sigma: float = 1.0, pseudo_count: float = 0.0, **kwargs
    ):
        r"""
        Args:
            mu: Expected value of variable's log ( \( \mu \) ).
            sigma: Standard deviation of variable log ( \( \sigma \) ).
        """
        super().__init__(mu, sigma, **kwargs)
        self.mu_ = mu
        self.sigma_ = sigma
        self.pseudo_count = pseudo_count

        # Pseudo-count hack: compute summaries so that the next `fit` will weigh
        # the prior mean and variance as if they were coming from `pseudo_count`
        # observations.
        if self.pseudo_count > 0:
            x2 = self.sigma_**2 + self.mu_**2
            self.summaries = [
                self.pseudo_count,
                self.mu_ * self.pseudo_count,
                x2 * self.pseudo_count,
            ]


class PinnedLogNormal(LogNormal):
    r"""Log normal distribution where the standard deviation is held fixed to \( \sigma\).

    Only the parameter \( \mu \) is infered from the data, the parameter \( \sigma \)
    is fixed after distribution initialisation.

    $$
    p(x) = \frac{1}{\sqrt{2\pi \sigma^2 x^2}}
        \exp \left( \frac{(\ln x - \mu)^2}{2\sigma^2} \right).
    $$
    """

    name = "PinnedLogNormalDistribution"

    def from_summaries(self, inertia: float = 0.0):
        """Reset pinned standard deviation."""
        super().from_summaries(inertia)
        self.parameters: list = [self.parameters[0], self.sigma_]
        return self


class Bernoulli(pg.BernoulliDistribution):
    """Extension of Pomegranate Bernoulli distribution supporting pseudo_counts."""

    name = "BernoulliDistribution"

    def __new__(cls, p: float = 0.5, pseudo_count: float = 0.0, **kwargs):
        """Peel off `pseudo_count` argument for pomegranate __cinit__."""
        return super().__new__(cls, p, **kwargs)

    def __init__(self, p: float = 0.5, pseudo_count: float = 0.0, **kwargs):
        self.pseudo_count = pseudo_count

        # Pseudo-count hack: compute summaries so that the next `fit` will weigh
        # the prior probability `p` as if it were coming from `pseudo_count`
        # observations.
        if self.pseudo_count > 0:
            self.summaries = [
                self.pseudo_count,
                p * self.pseudo_count,
            ]


class Exponential(pg.ExponentialDistribution):
    r"""Extension of Pomegranate exponential distribution supporting pseudo_counts.

    $$
    p(x) = \lambda \exp(-\lambda x).
    $$
    """

    name = "ExponentialDistribution"

    def __init__(self, rate: float = 1.0, pseudo_count: float = 0.0, **kwargs):
        r"""
        Args:
            rate: Decay rate \( \lambda \).
        """
        super().__init__(rate, **kwargs)
        self.pseudo_count = pseudo_count

        # Pseudo-count hack: compute summaries so that the next `fit` will weigh
        # the prior probability `p` as if it were coming from `pseudo_count`
        # observations.
        if self.pseudo_count > 0:
            self.summaries = [
                self.pseudo_count,
                self.pseudo_count / rate,
            ]


class Gamma(pg.GammaDistribution):
    r"""Extension of Pomegranate Gamma distribution supporting pseudo_counts.

    $$
    p(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} \exp[-\beta x].
    $$
    """

    name = "GammaDistribution"

    def __init__(
        self, alpha: float = 1.0, beta: float = 1.0, pseudo_count: float = 0.0, **kwargs
    ):
        r"""
        Args:
            alpha: Shape \( \alpha \) of the distribution.
            beta:  Rate \( \beta \), or inverse scale, of the distribution.
        """
        super().__init__(alpha, beta, **kwargs)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("Shape `alpha` and rate `beta` must be positive definite!")

        self.pseudo_count = pseudo_count

        if self.pseudo_count > 0:
            self.summaries = [
                # sum_i x_i * w_i,
                alpha / beta * self.pseudo_count,
                # sum_i log(x_i) * w_i, E[ln(X)] = ψ(α) − ln(β).
                self.pseudo_count * (digamma(alpha) - log(beta)),
                # sum_i w_i.
                self.pseudo_count,
            ]


class Discrete(pg.DiscreteDistribution):
    """Extension of Pomegranate discrete distribution supporting pseudo_counts."""

    name = "DiscreteDistribution"

    def __new__(cls, characters: dict, pseudo_count: float = 0.0, **kwargs):
        """Peel off `pseudo_count` argument for pomegranate __cinit__."""
        return super().__new__(cls, characters, **kwargs)

    def __init__(self, characters: dict, pseudo_count: float = 0.0, **kwargs):
        self.pseudo_count = pseudo_count

        if self.pseudo_count > 0:
            self.summaries: list = [
                {
                    # p(x=key) * n where `n` is the pseudo count.
                    key: characters[key] * self.pseudo_count
                    for key in self.summaries[0].keys()
                },
                self.pseudo_count,
            ]

    def log_probability(self, X):
        """Vectorised implementation of DiscreteDistribution log prob."""
        if isinstance(X, ndarray):
            if len(X) == 1:
                return super().log_probability(float(X))

            logp_iter = map(super().log_probability, X)
            return fromiter(logp_iter, dtype=float)

        return super().log_probability(X)


class Poisson(pg.PoissonDistribution):
    r"""Extension of Pomegranate Poisson supporting pseudo_counts.

    $$
    p(x) = \frac{x^l e^{-l}}{x!}
    $$
    """

    name = "PoissonDistribution"

    def __init__(self, l: float = 1.0, pseudo_count: float = 0.0, **kwargs):
        """
        Args:
            l: Rate (and therefore, the mean) of the distribution.
        """
        super().__init__(l, **kwargs)
        self.pseudo_count = pseudo_count

        # Pseudo-count hack: compute summaries so that the next `fit` will weigh
        # the prior mean and variance as if they were coming from `pseudo_count`
        # observations.
        if self.pseudo_count > 0:
            self.summaries = [
                # Nota bene: Here the order of the summaries is swapped compared to
                # other distributions.
                l * self.pseudo_count,
                self.pseudo_count,
            ]


class _AbstractInflated(ABC, Distribution):
    r"""
    Probability density where some values \( \{x_1,..,x_n\} \) have finite probability.

    That is,
    $$
        p(x) = \left \{ \begin{matrix}
            \pi_{x^\prime} \delta(x-x^\prime) & x^\prime \in \{x_1,\dots, x_n\}, \\
            \left(1 - \sum_{x^\prime} \pi_{x^\prime} \right) p_c(x)  & x \notin \{x_1, \dots ,x_n\},
        \end{matrix}
        \right.
    $$
    where \(\pi_x \) is a categorical distribution and \( p_c(x) \) is the
    complementary distribution.
    """

    @property
    @abstractmethod
    def ComplementaryDistribution(self):
        """Distribution class for modelling x!=0 values."""

    @property
    def parameters(self) -> list:
        return self.pi_.parameters + self.p_complement_.parameters

    @parameters.setter
    def parameters(self, parameters: list):
        self.pi_.parameters = [parameters[0]]
        self.p_complement_.parameters = parameters[1:]

    def from_summaries(self, inertia: float = 0.0):
        """Compute sufficient statistics from summaries."""
        self.pi_.from_summaries(inertia)
        self.p_complement_.from_summaries(inertia)

    def fit(self, *args, **kwargs):
        """Add sklearn compatible `self` return to pomegranate implementation."""
        super().fit(*args, **kwargs)

        return self

    @classmethod
    def blank(cls):
        return cls()


class _BaseInflated(_AbstractInflated):
    r"""Probability density where some values \( \{x_1,\dots ,x_n\} \) have finite probability.

    That is,
    $$
        p(x) = \left \{ \begin{matrix}
            \pi_{x^\prime} \delta(x-x^\prime) & x^\prime \in \{x_1,\dots, x_n\}, \\
            \left(1 - \sum_{x^\prime} \pi_{x^\prime} \right) p_c(x)  & x \notin \{x_1, \dots ,x_n\},
        \end{matrix}
        \right.
    $$
    where `pi` is a categorical distribution and \( p_c(x) \) is the complementary
    distribution."""

    name = "InflatedDistribution"

    def __init__(
        self,
        special_values: Union[tuple, dict],
        *complement_args,
        pseudo_count: float = 0.0
    ):
        """
        Args:
            special_values: Values to inflate (tuple), or the values (=key) and
                corresponding probabilities (=value) when a dict.
            *complement_args: Arguments to pass to the original distribution (the
                probability density).
        """
        self.pseudo_count = pseudo_count

        if isinstance(special_values, dict):
            assert "complement" in special_values, "Missing complement key!"
            # Check that the probabilities are normalised.
            assert_almost_equal(sum(special_values.values()), 1)

            # Elements to inflate.
            self.inflated_keys = tuple(
                key for key in special_values.keys() if key != "complement"
            )
            p_keys = special_values

        else:
            # Elements to inflate.
            self.inflated_keys = tuple(
                key for key in special_values if key != "complement"
            )

            # No probabilities provided, equal a priori probability.
            p_flat = 1 / (len(self.inflated_keys) + 1)

            # Initialise probability of non-inflated element.
            p_keys = {"complement": p_flat}

            # Initialise probability of inflated elements.
            p_keys.update({character: p_flat for character in special_values})

        self.pi_ = Discrete(p_keys, pseudo_count=self.pseudo_count)
        self.p_complement_ = self.ComplementaryDistribution(
            *complement_args, pseudo_count=pseudo_count
        )

    def __reduce__(self) -> tuple:
        """Serialize distribution for pickling."""
        return self.__class__, tuple(self.parameters)

    def summarize(self, X, weights=None, column_index: int = 0):
        """Reduce samples to summary statistics."""
        # First summarize Bernoulli distribution.
        if len(X.shape) > 1:
            x_i = X[:, column_index]
        else:
            x_i = X

        if weights is None:
            weights = ones_like(x_i)

        x_pi = x_i.copy().astype(object)
        is_inflated_x = isin(x_i, test_elements=self.inflated_keys)
        is_complement = (~isnan(x_i)) & (~is_inflated_x)
        x_pi[is_inflated_x] = x_i[is_inflated_x]
        x_pi[is_complement] = "complement"

        self.pi_.summarize(x_pi, weights)
        self.p_complement_.summarize(x_i[is_complement], weights[is_complement])

    def log_probability(self, X):
        """Log probability of zero inflated distribution."""
        if not isinstance(X, ndarray):
            X = array(X, dtype=float)
        elif X.dtype != float:
            X = X.astype(float)

        # NaN have probability= 1 -> log[probability] = 0.
        log_p = zeros_like(X, dtype=float)

        # p(x=0) = pi.
        is_inflated_x = isin(X, test_elements=self.inflated_keys)
        # Categorical distribution doesn't support direct element-wise
        # operation, so use map instead..
        log_p[is_inflated_x] = fromiter(
            map(self.pi_.log_probability, X[is_inflated_x]), dtype=float
        )

        is_complement = (~isnan(X)) & (~is_inflated_x)
        log_pi_compl = full_like(
            X[is_complement], self.pi_.log_probability("complement")
        )
        log_p_complement = self.p_complement_.log_probability(X[is_complement])
        log_p[is_complement] = log_pi_compl + log_p_complement

        return log_p

    def sample(self, size: int = 1):
        """Generate random variate from distribution."""
        x_samples = self.pi_.sample(size).astype(object)
        is_complement = x_samples == "complement"
        # Convert 'text' samples back to floats.
        x_samples[~is_complement] = x_samples[~is_complement].astype(float)
        n_complement = is_complement.sum()
        x_samples[is_complement] = self.p_complement_.sample(n_complement)
        return x_samples.astype(float)


class _BaseZeroInflated(_AbstractInflated):
    """Zero-inflated flated density where p(x=0) has finite probability.

    That is,
        p(x) = { pi             x=0,
               { (1-pi) p_c(x)  x =/= 0,
    where `pi` is a Bernoulli distribution and p_c(x) is the complementary
    distribution.
    """

    name = "ZeroInflatedDistribution"

    def __init__(self, pi: float = 0.5, *complement_args, pseudo_count: float = 0.0):
        self.pseudo_count = pseudo_count
        self.pi = pi
        self.complement_args = complement_args
        self.pi_ = Bernoulli(self.pi, pseudo_count=self.pseudo_count)
        self.p_complement_ = self.ComplementaryDistribution(
            *self.complement_args, pseudo_count=self.pseudo_count
        )

    def fit(self, X, y=None):
        """Initialise models and fit distributions."""
        super().fit(X, y)
        return self

    def sample(self, size: int = 1):
        """Generate random variate from distribution."""
        x_samples = zeros(size)
        # Zeros are correspond to 1 in self.pi.
        non_zero = self.pi_.sample(size) == 0
        n_non_zero = non_zero.sum()
        x_samples[non_zero] = self.p_complement_.sample(n_non_zero)
        return x_samples

    def log_probability(self, X):
        """Log probability of zero inflated distribution."""
        if not isinstance(X, ndarray):
            X = array(X, dtype=float)
        elif X.dtype != float:
            X = X.astype(float)

        # NaN have probability= 1 -> log[probability] = 0.
        log_p = zeros_like(X, dtype=float)

        # p(x=0) = pi.
        is_zero = X == 0.0
        log_p[is_zero] = self.pi_.log_probability(ones_like(X[is_zero]))

        is_non_zero = (~isnan(X)) & (~is_zero)
        log_one_min_pi = self.pi_.log_probability(zeros_like(X[is_non_zero]))
        log_p_complement = self.p_complement_.log_probability(X[is_non_zero])
        log_p[is_non_zero] = log_one_min_pi + log_p_complement

        return log_p

    def summarize(self, X, weights=None, column_index: int = 0):
        """Reduce samples to summary statistics."""
        # First summarize Bernoulli distribution.
        if len(X.shape) > 1:
            x_i = X[:, column_index]
        else:
            x_i = X

        if weights is None:
            weights = ones_like(x_i)

        x_pi = x_i.copy()
        is_zero = x_i == 0.0
        is_non_zero = (~isnan(x_i)) & (~is_zero)
        x_pi[is_zero] = 1
        x_pi[is_non_zero] = 0

        self.pi_.summarize(x_pi, weights)
        self.p_complement_.summarize(x_i[is_non_zero], weights[is_non_zero])


class ZeroInflatedGaussian(_BaseZeroInflated):
    """Model p(x=0) with finite probability, and the remainder as Gaussian."""

    ComplementaryDistribution = Gaussian
    name = "ZeroInflatedGaussian"

    def __init__(
        self,
        pi: float = 0.5,
        mu: float = 0.0,
        sigma: float = 1.0,
        pseudo_count: float = 0.0,
    ):
        super().__init__(pi, mu, sigma, pseudo_count=pseudo_count)


class InflatedGaussian(_BaseInflated):
    """Gaussian with finite probability of observing `special_values`."""

    ComplementaryDistribution = Gaussian
    name = "InflatedGaussian"

    def __init__(
        self,
        special_values: Union[dict, tuple] = (0, 1, inf),
        mu: float = 0.0,
        sigma: float = 1.0,
        pseudo_count: float = 0.0,
    ):
        super().__init__(special_values, mu, sigma, pseudo_count=pseudo_count)


class InflatedLogNormal(_BaseInflated):
    r"""Log normal with finite probability of observing `special_values`.

    That is,
    $$
        p(x) = \left \{ \begin{matrix}
            \pi_{x^\prime} \delta(x-x^\prime) & x^\prime \in \{x_1,\dots, x_n\}, \\
            \left(1 - \sum_{x^\prime} \pi_{x^\prime} \right) \frac{1}{\sqrt{2\pi \sigma^2 x^2}}
        \exp \left( \frac{(\ln x - \mu)^2}{2\sigma^2} \right)  & x \notin \{x_1, \dots ,x_n\},
        \end{matrix}
        \right.
    $$
    where \(\pi_x \) is a categorical distribution.
    """

    ComplementaryDistribution = LogNormal
    name = "InflatedLogNormal"

    def __init__(
        self,
        special_values: Union[dict, tuple] = (0, 1, inf),
        mu: float = 0.0,
        sigma: float = 1,
        pseudo_count: float = 0.0,
    ):
        r"""
        Args:
            special_values: Set of values \( \{x_1,\dots, x_n\} \) with finite
                probability.
            mu: Expected value of variable's log ( \( \mu \) ).
            sigma: Standard deviation of variable log ( \( \sigma \) ).
        """
        super().__init__(special_values, mu, sigma, pseudo_count=pseudo_count)


class InflatedPinnedLogNormal(_BaseInflated):
    """Log normal with finite probability of observing `special_values`."""

    ComplementaryDistribution = PinnedLogNormal
    name = "InflatedPinnedLogNormal"

    def __init__(
        self,
        special_values: Union[dict, tuple] = (0, 1, inf),
        mu: float = 0.0,
        sigma: float = 1,
        pseudo_count: float = 0.0,
    ):
        super().__init__(special_values, mu, sigma, pseudo_count=pseudo_count)


class InflatedGamma(_BaseInflated):
    """Gamma distribution with finite prob. of observing `special_values`."""

    ComplementaryDistribution = Gamma
    name = "InflatedGamma"

    def __init__(
        self,
        special_values: Union[dict, tuple] = (0, 1, inf),
        alpha: float = 1.0,
        beta: float = 1.0,
        pseudo_count: float = 0.0,
    ):
        super().__init__(special_values, alpha, beta, pseudo_count=pseudo_count)


class InflatedExponential(_BaseInflated):
    """Exponential distribution with finite prob. of observing `special_values`."""

    ComplementaryDistribution = Exponential
    name = "InflatedExponential"

    def __init__(
        self,
        special_values: Union[dict, tuple] = (0, 1, inf),
        rate: float = 1.0,
        pseudo_count: float = 0.0,
    ):
        super().__init__(special_values, rate, pseudo_count=pseudo_count)


class ZeroInflatedExponential(_BaseZeroInflated):
    r"""Exponential distribution with finite probability of value zero.

    That is,
    $$
        p(x) = \left \{ \begin{matrix}
            \pi_0 \delta(x) & x = 0, \\
            \left(1 - \pi_0 \right) \lambda  e^{-\lambda x}  & x \neq 0.
        \end{matrix}
        \right.
    $$
    """

    ComplementaryDistribution = Exponential
    name = "ZeroInflatedExponential"

    def __init__(
        self,
        pi: float = 0.5,
        rate: float = 1.0,
        pseudo_count: float = 0.0,
    ):
        r"""Initialise distribution with default values.

        Args:
            pi: Probability of observing the value zero ( \( \pi_0 \) ).
            rate: Inverse length scale \( \lambda \).
            pseudo_count: Act like `pi` and `rate` were obtained from this many
                observations.
        """
        super().__init__(pi, rate, pseudo_count=pseudo_count)


class ZeroInflatedGamma(_BaseZeroInflated):
    r"""Gamma distribution with finite probability of observering zero.

    That is,
    $$
        p(x) = \left \{ \begin{matrix}
            \pi_0 \delta(x) & x = 0, \\
            \left(1 - \pi_0 \right)
            \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x} & x \neq 0.
        \end{matrix}
        \right.
    $$
    """

    ComplementaryDistribution = Gamma
    name = "ZeroInflatedGamma"

    def __init__(
        self,
        pi: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        pseudo_count: float = 0.0,
    ):
        super().__init__(pi, alpha, beta, pseudo_count=pseudo_count)
