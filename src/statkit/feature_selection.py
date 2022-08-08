"""Select features using statistical hypothesis testing."""
from typing import Literal

from numpy import linalg, nan
from numpy.typing import NDArray

from pandas import DataFrame
from scipy.stats import (
    epps_singleton_2samp as epps_singleton,
    ks_2samp as kolmogorov_smirnov,
    mannwhitneyu as mann_whitney_u,
)
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_X_y
from statsmodels.stats.multitest import fdrcorrection, multipletests


class StatisticalTestFilter(BaseEstimator, SelectorMixin):
    """Select columns with significant difference between labels.

    Test which features the distribution of the postive class is stastistically
    different from the negative class, using multiple testing correction. Keep only the
    features that passed the statistical test.
    """

    def _apply_test(
        self,
        X_pos: NDArray,
        X_neg: NDArray,
        multiple_testing: Literal[
            "benjamini-hochberg", "bonferroni"
        ] = "benjamini-hochberg",
    ) -> DataFrame:
        """Column-wise test between positive and negative group."""
        columns = tuple(range(X_pos.shape[1]))
        if hasattr(self, "feature_names_in_"):
            columns = self.feature_names_in_

        result = DataFrame(columns=["statistic", "pvalue"], index=columns)

        # Perform test for each feature.
        for i, col in enumerate(columns):
            try:
                statistic, p_value = self.test_(
                    X_pos[:, i], X_neg[:, i], **self.test_kwargs_
                )
            except (linalg.LinAlgError, ValueError):
                statistic, p_value = nan, nan
            result.loc[col] = [statistic, p_value]

        # Apply multiple-testing correction.
        if multiple_testing == "benjamini-hochberg":
            reject, pvalue_corrected = fdrcorrection(result.pvalue, alpha=self.p_value)
        elif multiple_testing == "bonferroni":
            reject, pvalue_corrected, _, _ = multipletests(
                result.pvalue, alpha=self.p_value, method="bonferroni"
            )

        result["pvalue-corrected"] = pvalue_corrected
        result["reject"] = reject

        return result

    def __init__(
        self,
        statistical_test: Literal[
            "kolmogorov-smirnov", "mann-whitney-u", "epps-singleton"
        ] = "kolmogorov-smirnov",
        p_value: float = 0.05,
        multiple_testing: Literal[
            "benjamini-hochberg", "bonferroni"
        ] = "benjamini-hochberg",
        invert: bool = False,
        **kwargs,
    ):
        """
        Args:
            statistical_test: Test for difference in feature distributions
                between labels.
            p_value: The null hypothesis rejection probability (including
                `correction`).
            multiple_testing: What type of correction strategy to apply to account for
                multiple testing.
            invert: Invert selection, by keeping only the non-significant (instead of
                significant) columns.
        """
        super().__init__(**kwargs)
        self.statistical_test = statistical_test
        self.p_value = p_value
        self.multiple_testing = multiple_testing
        self.invert = invert

    def _get_support_mask(self):
        """Compute support mask of features."""
        if self.invert:
            return ~self.scores_["reject"]
        return self.scores_["reject"]

    def fit(self, X, y):
        """Perform column-wise statistical test."""
        check_X_y(X, y)
        self._check_feature_names(X, reset=True)

        self.test_kwargs_ = {}
        statistical_functions = {
            "mann-whitney-u": mann_whitney_u,
            "kolmogorov-smirnov": kolmogorov_smirnov,
            "epps-singleton": epps_singleton,
        }

        if self.statistical_test not in statistical_functions.keys():
            raise KeyError(f"Unknown statistical method {self.statistical_test}.")

        self.test_ = statistical_functions[self.statistical_test]

        # Only allow two classes right now.
        self.classes_ = unique_labels(y)
        assert len(self.classes_) == 2
        X_neg = X[y == self.classes_[0]]
        X_pos = X[y == self.classes_[1]]

        if isinstance(X, DataFrame):
            X_neg = X_neg.to_numpy()
            X_pos = X_pos.to_numpy()

        self.scores_ = self._apply_test(
            X_pos, X_neg, multiple_testing=self.multiple_testing
        )

        return self
