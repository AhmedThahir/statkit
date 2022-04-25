from typing import Callable, Literal
from grpc import Call

from numpy import array, mean, percentile, random, unique, where
from pandas import Series, concat
from sklearn.utils import resample
from sklearn.utils import shuffle


def bootstrap_score(
    y_true, y_pred, metric: Callable, n_iterations: int = 1000, random_state=1234
) -> dict:
    """Compute 95 % confidence interval for `metric` using bootstrapping.

    Args:
        y_true: Ground truth labels.
        y_pred: Labels predicted by the classifier.
        metric: Performance metric that takes the true and predicted labels and
            returns a score.
        n_iterations: Resample the data (with replacement) this many times.

    """
    statistics = []
    for i in range(n_iterations):
        y_true_rnd, y_pred_rnd = resample(y_true, y_pred, random_state=random_state + i)
        # Reject sample if all class labels are the same.
        if len(unique(y_true_rnd)) == 1:
            continue
        score = metric(y_true_rnd, y_pred_rnd)
        statistics.append(score)

    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, percentile(statistics, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, percentile(statistics, p))
    point_estimate = metric(y_true, y_pred)
    return {"point": point_estimate, "lower": lower, "upper": upper}


def unpaired_permutation_test(
    y_true: Series,
    y_pred_1: Series,
    y_pred_2: Series,
    metric: Callable,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_iterations: int = 1000,
    random_state=1234,
) -> tuple:
    """Unpaired permutation test if `metric` estimate of `y_pred_1` is different from `y_pred_2`.

    Null hypothesis (H_0): metric is not different.

    Args:
        y_true: Ground truth labels.
        y_pred_1, y_pred_2: Predicted labels to compare.
        metric: Performance metric that takes the true and predicted labels and
            returns a score.
        n_iterations: Resample the data (with replacement) this many times.
    """
    score1 = metric(y_true.loc[y_pred_1.index], y_pred_1)
    score2 = metric(y_true.loc[y_pred_2.index], y_pred_2)
    observed_difference = score1 - score2

    n_1 = len(y_pred_1)
    permuted_diff = []
    for i in range(n_iterations):
        # Pool slices and randomly split into groups of size n_1 and n_2.
        y_H0 = shuffle(concat([y_pred_1, y_pred_2]), random_state=random_state + i)
        y1_H0 = y_H0.iloc[:n_1]
        y2_H0 = y_H0.iloc[n_1:]

        # Pair y_test with corresponding splits.
        y1_true = y_true.loc[y1_H0.index]
        y2_true = y_true.loc[y2_H0.index]
        if len(unique(y1_true)) == 1 or len(unique(y2_true)) == 1:
            continue

        permuted_score1 = metric(y1_true, y1_H0)
        permuted_score2 = metric(y2_true, y2_H0)
        permuted_diff.append(permuted_score1 - permuted_score2)

    permuted_diff = array(permuted_diff)
    if alternative == "greater":
        p_value = mean(permuted_diff >= observed_difference)
    elif alternative == "less":
        p_value = mean(permuted_diff <= observed_difference)
    elif alternative == "two-sided":
        p_value = mean(abs(permuted_diff) >= abs(observed_difference))

    return observed_difference, p_value


def paired_permutation_test(
    y_true,
    y_pred_1,
    y_pred_2,
    metric: Callable,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_iterations: int = 1000,
    random_state=1234,
) -> tuple:
    """Paired permutation test if `y_pred_1` metrics are significantly different from `y_pred_2`.

    Args:
        y_true: Ground truth labels.
        y_pred_1, y_pred_2: Predicted labels to compare.
        metric: Performance metric that takes the true and predicted labels and
            returns a score.
        n_iterations: Resample the data (with replacement) this many times.

    Returns:
        Estimate and corresponding p-value."""

    random.seed(random_state)
    score1 = metric(y_true, y_pred_1)
    score2 = metric(y_true, y_pred_2)
    observed_difference = score1 - score2

    m = len(y_true)
    permuted_diff = []
    for _ in range(n_iterations):
        mask = random.randint(2, size=m)
        p1 = where(mask, y_pred_1, y_pred_2)
        p2 = where(mask, y_pred_2, y_pred_1)

        permuted_score1 = metric(y_true, p1)
        permuted_score2 = metric(y_true, p2)
        permuted_diff.append(permuted_score1 - permuted_score2)

    permuted_diff = array(permuted_diff)

    if alternative == "greater":
        p_value = mean(permuted_diff >= observed_difference)
    elif alternative == "less":
        p_value = mean(permuted_diff <= observed_difference)
    elif alternative == "two-sided":
        p_value = mean(abs(permuted_diff) >= abs(observed_difference))

    return observed_difference, p_value
