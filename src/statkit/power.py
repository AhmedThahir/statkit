"""Estimate population size needed to reject null hypothesis for a given metric."""

from numpy import sqrt
from scipy.stats import norm


def _Q(theta: float) -> tuple[float, float]:
    """Eq. (2) Hanley-McNeil. Radiology 143.1 (1982): 29-36."""
    Q1 = theta / (2.0 - theta)
    Q2 = 2 * theta**2 / (1 + theta)
    return Q1, Q2


def _V(theta: float) -> float:
    """
    Args:
        theta: Area under the ROC curve."""
    return _Q(theta)[0] + _Q(theta)[1] - 2 * theta**2


def sample_size_roc_auc(
    area_1: float,
    area_2: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative="one-sided",
) -> tuple[float, float]:
    r"""Estimate required population needed to detect significantly different ROC areas.

    Assumes balanced class labels (i.e., equal number of normals and abnormals).
    Null hypothesis ( \( H_0 \) ) is both areas under the receiver operating characteristic (ROC)
    curve are drawn from the same population.


    Based on the paper:
    Hanley-McNeil. Radiology 143.1 (1982): 29-36.

    Args:
        area_1: Area under the ROC curve of population 1.
        area_2: Area under the ROC curve of population 2.
        alpha: Probabiliy of type I error.
        power: Probability of a true positive ( \( 1 - \beta \) ).

    Returns:
        Pair of normal and abnormals samples required.
    """
    # One sided test for 5 % significance.
    # Z_a = 1.645
    if alternative == "one-sided":
        Z_a = norm.ppf(1 - alpha)
    elif alternative == "two-sided":
        Z_a = norm.ppf(1 - alpha / 2)
    else:
        raise ValueError(f"Unknown alternative {alternative}.")

    # Amount of power: 0.84 -> 80%.
    # Z_b = 0.84
    Z_b = norm.ppf(power)
    delta = area_2 - area_1

    V_1 = _V(area_1)
    V_2 = _V(area_2)

    n = (Z_a * sqrt(2 * V_1) + Z_b * sqrt(V_1 + V_2)) ** 2 / delta**2
    return n, n
