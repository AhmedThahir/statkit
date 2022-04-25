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
