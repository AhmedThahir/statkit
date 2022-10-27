"""Evaluate models using decision curve analysis."""
from typing import Literal, Optional, Union

from matplotlib import pyplot as plt
from numpy import array, divide, linspace, ones_like, r_, zeros_like
from numpy.typing import NDArray
from pandas import Series
from sklearn.metrics import confusion_matrix
from sklearn.utils import column_or_1d


def _binary_classification_thresholds(y_true, y_proba, thresholds):
    """Compute false and true positives for given probability thresholds."""
    y_true = column_or_1d(y_true)
    y_proba = column_or_1d(y_proba)
    matrices = []
    for t in thresholds:
        if t < 1:
            y_pred = (y_proba > t).astype(int)
        else:
            y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        matrices.append(cm)
    matrices = array(matrices)
    tns = matrices[:, 0, 0]
    fps = matrices[:, 0, 1]
    fns = matrices[:, 1, 0]
    tps = matrices[:, 1, 1]
    return tns, fps, fns, tps


def net_benefit(
    y_true: Union[Series, NDArray],
    y_pred: Union[Series, NDArray],
    thresholds=100,
    action: bool = True,
):
    """Net benefit of taking an action using a model's predictions.

    Args:
        y_true: Binary ground truth label (1: positive, 0: negative class).
        y_pred: Probability of positive class label.
        thresholds: When an array, evaluate net benefit at these coordinates (the
            probability thresholds). When an int, the number of x coordinates.
        action: When `True` (`False`), estimate net benefit of taking (not taking) an
            action/intervention/treatment.

    Returns:
        thresholds: Probability threshold of prediction a positive class.
        benefit: The net benefit corresponding to the thresholds.

    References:
        [1]: Vickers-Elkin. "Decision curve analysis: a novel method for evaluating
        prediction models." Medical Decision Making 26.6 (2006): 565-574.
        [2]: Rousson-Zumbrunn. "Decision curve analysis revisited: overall net
        benefit, relationships to ROC curve analysis, and application to case-control
        studies." BMC medical informatics and decision making 11.1 (2011): 1–9.
    """
    if set(y_true.astype(int)) != {0, 1}:
        raise ValueError(
            "Decision curve analysis only supports binary classification (with labels 1 and 0)."
        )

    if isinstance(thresholds, int):
        thresholds = linspace(0, 1, num=thresholds)

    N = len(y_true)
    tns, fps, fns, tps = _binary_classification_thresholds(y_true, y_pred, thresholds)
    if action:
        loss_over_profit = divide(
            thresholds, 1 - thresholds, where=thresholds < 1, out=zeros_like(thresholds)
        )

        benefit = tps / N - fps / N * loss_over_profit
    else:
        # Invert 0<-->1 so that true positives are true negatives, and false positives
        # are false negatives.
        profit_over_loss = divide(
            1 - thresholds,
            thresholds,
            where=thresholds != 0,
            out=zeros_like(thresholds),
        )
        benefit = tns / N - fns / N * profit_over_loss
    return thresholds, benefit


def net_benefit_oracle(y_true, action: bool = True) -> float:
    """Net benefit of omniscient strategy, i.e., a hypothetical perfect predictor."""
    if action:
        return y_true.mean()
    return 1 - y_true.mean()


def net_benefit_action(y_true, thresholds, action: bool = True):
    """Net benefit of always doing an action/intervention/treatment.

    Args:
        action: When `False`, invert positive label in `y_true`.
    """
    if action:
        loss_over_profit = divide(
            thresholds,
            1 - thresholds,
            where=thresholds < 1,
            out=1e9 * ones_like(thresholds),
        )
        return y_true.mean() - (1 - y_true.mean()) * loss_over_profit
    profit_over_loss = divide(
        1 - thresholds,
        thresholds,
        where=thresholds != 0,
        out=1e9 * ones_like(thresholds),
    )
    return 1 - y_true.mean() - y_true.mean() * profit_over_loss


def overall_net_benefit(y_true, y_pred, n_thresholds: int = 100):
    """Net benefit combining both taking and not-taking action."""
    thresholds_action, benefit_action = net_benefit(
        y_true, y_pred, n_thresholds, action=True
    )
    _, benefit_no_action = net_benefit(y_true, y_pred, thresholds_action, action=False)
    return thresholds_action, benefit_action + benefit_no_action


class NetBenefitDisplay:
    """Net benefit decision curve analysis visualisation.

    Args:
        threshold_probability: Probability to dichotomise the predicted probability
            of the model.
        net_benefit: Net benefit of taking an action as a function of
            `threshold_probability`.
        oracle: The (constant) net benefit of a perfect predictor.
        benefit_type: Literal["action", "noop", "overall"] = "action",

    """

    def __init__(
        self,
        threshold_probability,
        net_benefit,
        net_benefit_action=None,
        net_benefit_noop=None,
        benefit_type: Literal["action", "noop", "overall"] = "action",
        oracle: Optional[float] = None,
        estimator_name: Optional[str] = None,
    ):
        self.threshold_probability = threshold_probability
        self.net_benefit = net_benefit
        self.net_benefit_action = net_benefit_action
        self.net_benefit_noop = net_benefit_noop
        self.benefit_type = benefit_type
        self.estimator_name = estimator_name
        self.oracle = oracle

    def plot(self, show_references: bool = True, ax=None):
        """
        Args:
            show_references: Show oracle (requires prevalence) and no
                action/treatment/intervention reference curves.
            ax: Optional axes object to plot on. If `None`, a new figure and axes is
                created.
        """
        if ax is None:
            _, ax = plt.subplots()
        self.ax_ = ax
        self.figure_ = ax.figure

        ax.plot(
            self.threshold_probability, self.net_benefit, "-", label=self.estimator_name
        )
        if show_references:
            ax.plot(
                self.threshold_probability,
                self.net_benefit_action,
                ":",
                label="Always act",
            )
            ax.plot(
                self.threshold_probability,
                self.net_benefit_noop,
                "-.",
                label="Never act",
            )

            if self.oracle is not None:
                ax.plot([0, 1], [self.oracle, self.oracle], "--", label="Oracle")

        ylabel = "Net benefit"
        if self.benefit_type == "action":
            ylabel = "Net benefit (action)"
        if self.benefit_type == "noop":
            ylabel = "Net benefit (no-action)"
        if self.benefit_type == "overall":
            ylabel = "Overall net benefit"

        ax.set(xlabel="Threshold probability", ylabel=ylabel)
        margin = 0.05
        ax.set_ylim([-margin, 1 + margin])
        ax.legend(loc="upper right", frameon=False)

        return self

    @classmethod
    def from_predictions(
        cls,
        y_true,
        y_pred,
        benefit_type: Literal["action", "noop", "overall"] = "action",
        thresholds: int = 100,
        name: Optional[str] = None,
        show_references: bool = True,
        ax=None,
    ):
        """Make a net benefit plot from true and predicted labels.

        Args:
            y_true: Binary ground truth label (1: positive, 0: negative class).
            y_pred: Predicted class labels.
            benefit_type: Type of net benefit curve. `"action"`: net benefit of
                treatment/intervention/action; `"noop`: net benefit of no
                treatment/intervention/action; `"overall"`: overall net benefit (see
                `overall_net_benefit`).
            show_references: Show oracle (requires prevalence) and no
                action/treatment/intervention reference curves.
            thresholds: When an array, evaluate net benefit at these coordinates (the
                probability thresholds). When an int, the number of x coordinates.
            ax: Optional axes object to plot on. If `None`, a new figure and axes is
                created.

        Example:
            ```python
            from sklearn.datasets import make_blobs
            from sklearn.linear_model import LogisticRegression
            from statkit.decision import NetBenefitDisplay

            centers = [[0, 0], [1, 1]]
            X_train, y_train = make_blobs(
                centers=centers, cluster_std=1, n_samples=20, random_state=5
            )
            X_test, y_test = make_blobs(
                centers=centers, cluster_std=1, n_samples=20, random_state=1005
            )

            clf = LogisticRegression(random_state=5).fit(X_train, y_train)
            y_pred_base = clf.predict_proba(X_test)[:, 1]
            NetBenefitDisplay.from_predictions(y_test, y_pred_base, name='Logistic Regression')
            ```
        """
        if benefit_type == "action":
            oracle = net_benefit_oracle(y_true, action=True)
            thresholds, benefit = net_benefit(y_true, y_pred, thresholds)
            benefit_action = net_benefit_action(y_true, thresholds, action=True)
            benefit_noop = zeros_like(benefit)

        elif benefit_type == "noop":
            oracle = net_benefit_oracle(y_true, action=False)
            thresholds, benefit = net_benefit(y_true, y_pred, thresholds, action=False)
            benefit_noop = net_benefit_action(y_true, thresholds, action=False)
            benefit_action = zeros_like(benefit)

        elif benefit_type == "overall":
            oracle = 1.0
            thresholds, benefit = overall_net_benefit(y_true, y_pred, thresholds)
            benefit_action = net_benefit_action(y_true, thresholds, action=True)
            benefit_noop = net_benefit_action(y_true, thresholds, action=False)

        return cls(
            thresholds,
            benefit,
            benefit_action,
            benefit_noop,
            benefit_type,
            oracle,
            estimator_name=name,
        ).plot(ax=ax, show_references=show_references)
