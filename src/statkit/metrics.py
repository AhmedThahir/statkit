"""Classification metrics not part of sci-kit learn."""
from numpy import array, ndarray
from pandas import DataFrame, Series
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from statkit.non_parametric import bootstrap_score


def youden_j_threshold(y_true, y_pred) -> float:
    """Classification threshold with highest Youden's J.

    Args:
        y_true: Ground truth labels.
        y_pred: Labels predicted by the classifier.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


def youden_j(y_true, y_pred) -> float:
    r"""Classifier informedness as a balance between true and false postivies.

    Youden's J statistic is defined as:
    $$
    J = r_{\mathrm{tp}} - r_{\mathrm{fp}}.
    $$

    Args:
        y_true: Ground truth labels.
        y_pred: Labels predicted by the classifier.
    """
    return sensitivity(y_true, y_pred) + specificity(y_true, y_pred) - 1


def true_positive_rate(y_true, y_prob, threshold: float = 0.5) -> float:
    r"""The number of true positive out of all positives (recall).

    Aliases:
        - Sensitivity,
        - Recall,
        - Hit rate.

    $$r_{\mathrm{tp}} = \frac{t_p}{t_p + f_n} = \frac{t_p}{p}$$

    Args:
        y_true: Ground truth label (binarised).
        y_prob: Probability of positive class.
        threshold: Classify as positive when probability exceeds threshold.
    """
    if not isinstance(y_true, (ndarray, Series)):
        y_true = array(y_true)
    if not isinstance(y_prob, (ndarray, Series)):
        y_prob = array(y_prob)

    y_pred = y_prob >= threshold
    positives = sum(y_true)
    true_positives = sum(y_true.astype(bool) & y_pred)
    return true_positives / positives


def false_positive_rate(y_true, y_prob, threshold: float = 0.5) -> float:
    r"""The number of false positive out of all negatives.

    Also called the fall out rate.
    $$r_{\mathrm{fp}} = \frac{f_p}{t_p + f_n} = \frac{f_p}{p}$$

    Args:
        y_true: Ground truth label (binarised).
        y_prob: Probability of positive class.
        threshold: Classify as positive when probability exceeds threshold.
    """
    if not isinstance(y_true, (ndarray, Series)):
        y_true = array(y_true)
    if not isinstance(y_prob, (ndarray, Series)):
        y_prob = array(y_prob)

    y_pred = y_prob > threshold
    negatives = y_true.size - sum(y_true)
    # Actual negative, but classified as positive.
    false_positives = sum((~y_true.astype(bool)) & y_pred)
    return false_positives / negatives


def sensitivity(y_true, y_prob, threshold: float = 0.5) -> float:
    r"""The number of true positive out of all positives.

    Aliases:
        - True positive rate,
        - Recall,
        - Hit rate.

    $$r_{\mathrm{tp}} = \frac{t_p}{t_p + f_n} = \frac{t_p}{p}$$

    Args:
        y_true: Ground truth label (binarised).
        y_prob: Probability of positive class.
        threshold: Classify as positive when probability exceeds threshold.
    """
    return true_positive_rate(y_true, y_prob, threshold)


def specificity(y_true, y_prob, threshold: float = 0.5) -> float:
    r"""The number of true negatives out of all negatives.

    $$r_{\mathrm{tn}} = \frac{t_n}{t_n + f_p} = \frac{t_n}{n} = 1 - r_{\mathrm{fp}}$$

    Aliases:
        - True negative rate,
        - Selectivity.

    Args:
        y_true: Ground truth label (binarised).
        y_prob: Probability of positive class.
        threshold: Classify as positive when probability exceeds threshold.
    """
    return 1 - false_positive_rate(y_true, y_prob, threshold)


def binary_classification_report(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
    n_iterations: int = 1000,
    quantile_range: tuple = (0.025, 0.975),
    random_state=None,
) -> DataFrame:
    r"""Compile performance metrics of a binary classifier.

    Evaluates the model according to the following metrics:

    - Accuracy,
    - Average precision (area under the precision-recall curve),
    - \( F_1 \),
    - Area under the receiver operating characteristic curve (ROC AUC),
    - Sensitivity (or, true positive rate, see `sensitivity`),
    - Specificity (or, true negative rate, see `specificity`).

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probability of the positive class.
        threshold: Dichotomise probabilities greater or equal to this threshold as
            positive.
        quantile_range: Confidence interval quantiles.
        n_iterations: Number of bootstrap permutations.

    Returns:
        A dataframe with the estimated classification metrics (`point`) and (by default
        95 %) confidence interval (from `lower` to `upper`).
    """
    scores = DataFrame(
        index=[
            "Accuracy",
            "Average precision",
            "$F_1$",
            "ROC AUC",
            "Sensitivity",
            "Specificity",
        ],
        columns=["point", "lower", "upper"],
    )

    kwargs = {
        "n_iterations": n_iterations,
        "random_state": random_state,
        "quantile_range": quantile_range,
    }

    rank_scorers = {
        "ROC AUC": roc_auc_score,
        "Average precision": average_precision_score,
    }
    for name, scorer in rank_scorers.items():
        score = bootstrap_score(y_true, y_pred_proba, metric=scorer, **kwargs)
        scores.loc[name] = dict(score)

    # Metrics that require the probability to be dichotomised.
    class_scorers = {
        "$F_1$": f1_score,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy_score,
    }
    for name, scorer in class_scorers.items():
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = bootstrap_score(y_true, y_pred, metric=scorer, **kwargs)
        scores.loc[name] = dict(score)

    return scores
