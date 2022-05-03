from sklearn.metrics import roc_curve


def youden_j(y_true, y_pred) -> float:
    """Compute threshold correspoding to Youden's J.

    Args:
        y_true: Ground truth labels.
        y_pred: Labels predicted by the classifier.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


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
