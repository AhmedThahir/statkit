r"""Confidence intervals and \(p\)-values of a model's (test) score.

This module contains a set of non-parametric (i.e., without assuming any
specific distribution) and exact methods of computing 95 % confidence intervals
and \(p\)-values of your sci-kit learn model's predictions.
"""
from typing import Callable, Literal, Optional, Union

from numpy import (
    array,
    stack,
    concatenate,
    mean,
    ones_like,
    quantile,
    unique,
    where,
    zeros_like,
)
from pandas import Series
from sklearn.utils import check_random_state, resample, shuffle

from statkit.types import Estimate


def bootstrap_score(
    y_true,
    y_pred,
    metric: Callable,
    n_iterations: int = 1000,
    random_state=None,
    pos_label: Optional[Union[str, int]] = None,
    quantile_range: tuple[float, float] = (0.025, 0.975),
    metrics_kwargs: dict = {},
) -> Estimate:
    """Estimate 95 % confidence interval for `metric` by bootstrapping.

    Example:
        Estimate 95 % confidence interval of area under the receiver operating
        characteristic curve (ROC AUC) on the test set of a binary classifier:
        ```python
        y_pred = model.predict_proba(X_test)[:, 1]
        bootstrap_score(y_test, y_pred, metric=roc_auc_score)
        ```

    Args:
        y_true: Ground truth labels.
        y_pred: Labels predicted by the classifier (or label probabilities,
            depending on the metric).
        metric: Performance metric that takes the true and predicted labels and
            returns a score.
        n_iterations: Resample the data (with replacement) this many times.
        pos_label: When `y_true` is a binary classification, the positive class.
        quantile_range: Confidence interval range.
        metric_kwargs: Pass additional keyword arguments to `metric`.

    Returns:
        The point estimate (see `statkit.types.Estimate`) with 95 % confidence interval
        of `metric` distribution.
    """
    if pos_label is not None:
        labels = unique(y_true)
        if len(labels) != 2:
            raise ValueError("Must be binary labels when `pos_label` is specified")
        _y_true = where(y_true == pos_label, ones_like(y_true), zeros_like(y_true))
    else:
        _y_true = y_true

    random_state = check_random_state(random_state)

    statistics = []
    for _ in range(n_iterations):
        y_true_rnd, y_pred_rnd = resample(_y_true, y_pred, random_state=random_state)
        # Reject sample if all class labels are the same.
        if len(unique(y_true_rnd)) == 1:
            continue
        score = metric(y_true_rnd, y_pred_rnd, **metrics_kwargs)
        statistics.append(score)

    point_estimate = metric(_y_true, y_pred, **metrics_kwargs)
    # If all samples were rejected, there is no variability in the dataset to bootstrap.
    if len(statistics) == 0:
        return Estimate(
            point=point_estimate, lower=point_estimate, upper=point_estimate
        )

    # Estimate confidence intervals.
    lower = quantile(statistics, quantile_range[0], axis=0)
    upper = quantile(statistics, quantile_range[1], axis=0)
    return Estimate(point=point_estimate, lower=lower, upper=upper)


def unpaired_permutation_test(
    y_true_1: Series,
    y_pred_1: Series,
    y_true_2: Series,
    y_pred_2: Series,
    metric: Callable,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_iterations: int = 1000,
    random_state=None,
    metrics_kwargs: dict = {},
) -> float:
    r"""Unpaired permutation test comparing scores `y_pred_1` with `y_pred_2`.

    Null hypothesis, \( H_0 \): metric is not different.

    Example:
        ```python
        unpaired_permutation_test(
            # Ground truth - prediction pair first sample set.
            y_test_1,
            y_pred_1,
            # Ground truth - prediction pair second sample set.
            y_test_2,
            y_pred_2,
            metric=roc_auc_score,
        )
        ```

    Args:
        y_true_1, y_true_2: Ground truth labels of unpaired groups.
        y_pred_1, y_pred_2: Predicted labels (or label probabilities,
            depending on the metric) of corresponding groups.
        metric: Performance metric that takes the true and predicted labels and
            returns a score.
        n_iterations: Resample the data (with replacement) this many times.
        metric_kwargs: Pass additional keyword arguments to `metric`.

    Returns:
        The p-value for observering the difference given \( H_0 \).
    """
    random_state = check_random_state(random_state)

    score1 = metric(y_true_1, y_pred_1, **metrics_kwargs)
    score2 = metric(y_true_2, y_pred_2, **metrics_kwargs)
    observed_difference = score1 - score2

    n_1 = len(y_pred_1)
    score_diff = []
    for i in range(n_iterations):
        # Pool slices and randomly split into groups of size n_1 and n_2.
        Y_1 = stack([y_true_1, y_pred_1], axis=-1)
        Y_2 = stack([y_true_2, y_pred_2], axis=-1)
        y_H0 = shuffle(concatenate([Y_1, Y_2]), random_state=random_state)

        y1_true = y_H0[:n_1, ..., 0]
        y2_true = y_H0[n_1:, ..., 0]
        y1_pred_H0 = y_H0[:n_1, ..., 1]
        y2_pred_H0 = y_H0[n_1:, ..., 1]

        if len(unique(y1_true)) == 1 or len(unique(y2_true)) == 1:
            continue

        permuted_score1 = metric(y1_true, y1_pred_H0, **metrics_kwargs)
        permuted_score2 = metric(y2_true, y2_pred_H0, **metrics_kwargs)
        score_diff.append(permuted_score1 - permuted_score2)

    permuted_diff = array(score_diff)
    if alternative == "greater":
        p_value = mean(permuted_diff >= observed_difference, axis=0)
    elif alternative == "less":
        p_value = mean(permuted_diff <= observed_difference, axis=0)
    elif alternative == "two-sided":
        p_value = mean(abs(permuted_diff) >= abs(observed_difference), axis=0)

    return p_value


def paired_permutation_test(
    y_true,
    y_pred_1,
    y_pred_2,
    metric: Callable,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_iterations: int = 1000,
    random_state=None,
    metrics_kwargs: dict = {},
) -> float:
    r"""Paired permutation test comparing scores from `y_pred_1` with `y_pred_2`.

    Non-parametric head-to-head comparison of two predictions. Test if
    `y_pred_1` is statistically different from `y_pred_2` for a given `metric`.


    \( H_0 \): metric scores of `y_pred_1` and `y_pred_2` come from the same population
    (i.e., invariant under group permutation 1 <--> 2).

    Example:
        Test if the area under the receiver operating characteristic curve
        (ROC AUC) of model 1 statistically significantly better than model 2:
        ```python
        y_pred_1 = model_1.predict_proba(X_test)[:, 1]
        y_pred_2 = model_2.predict_proba(X_test)[:, 1]
        paired_permutation_test(
            y_test,
            y_pred_1,
            y_pred_2,
            metric=roc_auc_score,
        )
        ```

    Args:
        y_true: Ground truth labels.
        y_pred_1, y_pred_2: Predicted labels to compare (or label probabilities,
            depending on the metric).
        metric: Performance metric that takes the true and predicted labels and
            returns a score.
        n_iterations: Resample the data (with replacement) this many times.
        metric_kwargs: Pass additional keyword arguments to `metric`.

    Returns:
        The p-value for observering the difference given \( H_0 \).
    """
    random_state = check_random_state(random_state)

    score1 = metric(y_true, y_pred_1, **metrics_kwargs)
    score2 = metric(y_true, y_pred_2, **metrics_kwargs)
    observed_difference = score1 - score2

    # Broadcast mask to shape of y_pred_1.
    mask_shape = (-1,) + len(y_pred_1.shape[1:]) * (1,)

    m = len(y_true)
    score_diff = []
    for _ in range(n_iterations):
        mask = random_state.randint(2, size=m).reshape(mask_shape)
        p1 = where(mask, y_pred_1, y_pred_2)
        p2 = where(mask, y_pred_2, y_pred_1)

        permuted_score1 = metric(y_true, p1, **metrics_kwargs)
        permuted_score2 = metric(y_true, p2, **metrics_kwargs)
        score_diff.append(permuted_score1 - permuted_score2)

    permuted_diff = array(score_diff)

    if alternative == "greater":
        p_value = mean(permuted_diff >= observed_difference, axis=0)
    elif alternative == "less":
        p_value = mean(permuted_diff <= observed_difference, axis=0)
    elif alternative == "two-sided":
        p_value = mean(abs(permuted_diff) >= abs(observed_difference), axis=0)

    return p_value
