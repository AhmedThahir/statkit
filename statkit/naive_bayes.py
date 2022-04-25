from typing import Union

from pomegranate.bayes import BayesModel
from pomegranate.distributions import (
    Distribution,
    IndependentComponentsDistribution,
)
from pandas import DataFrame, Series
from numpy import fromiter, testing, unique, vectorize, zeros
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class _BaseNaiveBayes(ClassifierMixin, BaseEstimator):
    """Pomegranate NaiveBayes extension with distribution kwargs and pseudo counts."""

    def __init__(
        self,
        distributions: list,
        pseudo_count: Union[float, dict[Distribution, float]] = 0.0,
        distribution_kwargs: dict[dict] = {},
    ):
        """
        Args:
            pseudo_count: Pseudo count for all distributions (when float) or distribution
                specific (by key) pseudo count.
            distribution_kwargs: Initialisation of arguments, per distribution (=key), and
                per class (key of value). These arguments are positional (in stead of
                keyword) because pomegranate is Cython compiled.
                Example:
                    {Gaussian: {0: (0, 1), 1: (0, 1)}}
        """
        self.distributions_ = self.distributions = distributions
        self.distribution_kwargs = distribution_kwargs
        self.pseudo_count = pseudo_count

    def fit(self, X, y, weights=None):
        """Initialise a BayesModel."""
        self.classes_ = unique_labels(y)

        if len(self.classes_) == 1:
            if set(self.classes_).issubset([0, 1]):
                self.classes_ = [0, 1]
            else:
                raise ValueError(
                    f"Training data contains single class {self.classes_}."
                )

        # For each class, generate a `IndependentComponentsDistribution`.
        distributions = []
        for c in self.classes_:
            components = []
            # Each component is initialised using the class specific keyword
            # arguments.
            for i, p_i in enumerate(self.distributions_):
                if isinstance(self.pseudo_count, (float, int)):
                    kwargs = {"pseudo_count": self.pseudo_count}
                elif isinstance(self.pseudo_count, dict):
                    # Use distribution specific pseudo count.
                    kwargs = {"pseudo_count": self.pseudo_count[p_i]}

                kwargs.update(self.distribution_kwargs.get(p_i, {}).get(c, {}))
                p_i_given_c = self.distributions_[i](**kwargs)
                components.append(p_i_given_c)
                icd = IndependentComponentsDistribution(components)
                # IMPORTANT: By default, cython=1 in
                # `IndependentComponentsDistribution` causing it to call
                # `_summarize` from its children instead of `summarize`. Since
                # the former is not implemented, toggle cython to 0 so that
                # `IndependentComponentsDistribution` falls back on `summarize`
                # (which we did implement for, e.g., ZeroInflatedGaussian).
                icd.cython = 0

            distributions.append(icd)

        self.model_ = BayesModel(distributions)
        self.model_.fit(X, y, weights=weights)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class NaiveBayesClassifier(_BaseNaiveBayes):
    """
    scikit-learn wrapper around Pomegranate NaiveBayes classifier.
    """

    def __init__(
        self,
        distributions: Union[list, dict, Distribution],
        pseudo_count: Union[float, dict[Distribution, float]] = 1.0,
        distribution_kwargs: dict[dict] = {},
    ):
        super().__init__(distributions, pseudo_count, distribution_kwargs)

    def _clean(self, X, y=None):
        """
        Turn into numpy array.
        """
        Xnp = X
        if isinstance(X, DataFrame):
            Xnp = X.to_numpy()

        if y is None:
            return Xnp

        ynp = self.map_label_(y)
        return Xnp, ynp

    def check_schema(self, X):
        """Check schema of `X` with X_train."""
        if isinstance(X, DataFrame):
            if len(unique(X.columns)) != len(X.columns):
                raise KeyError("Duplicate columns!")

            # Python guarantees order of dicts.
            if list(X.columns) != list(self.column_map_.keys()):
                raise KeyError("Schema skew!")

    def check_smooth_distributions(self):
        """Verify that all discrete distributions are non-zero."""
        for c in range(len(self.classes_)):
            distributions_c = self.model_.distributions[c].distributions
            for i in range(len(distributions_c)):
                distr_params = distributions_c[i].parameters
                # When the first element is a dict, it is probably an inflated
                # distribution.
                if isinstance(distr_params[0], dict):
                    inflated_probabilities = fromiter(
                        distr_params[0].values(), dtype=float
                    )
                    testing.assert_almost_equal(
                        inflated_probabilities.sum(),
                        1,
                        err_msg=(
                            f"Probabilties of inflated points ({inflated_probabilities})"
                            " are not normalised."
                        ),
                    )
                    assert all(
                        inflated_probabilities != 0
                    ), "Some inflated points have zero probability."

    def fit(self, X, y, weights=None):
        """Fit pomegranate NaiveBayes model from BernoulliDistribution's."""
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.class_map_ = {k: i for i, k in enumerate(self.classes_)}
        self.class_map_inverse_ = {i: k for i, k in enumerate(self.classes_)}
        self.map_label_ = vectorize(lambda x: self.class_map_[x])
        self.map_label_inverse_ = vectorize(lambda x: self.class_map_inverse_[x])

        # Make a map of the columns.
        if isinstance(X, DataFrame):
            assert len(unique(X.columns)) == len(X.columns), "Duplicate columns!"
            self.column_map_ = {col: i for i, col in enumerate(X.columns)}
        else:
            # Identity map in case of NumPy matrix.
            self.column_map_ = {i: i for i in range(X.shape[1])}

        self.distributions_ = self.distributions
        if isinstance(self.distributions, dict):
            if not isinstance(X, DataFrame):
                raise ValueError(
                    "Expected dataframe when passing distributions as dict."
                )
            self.distributions_ = [self.distributions[c] for c in X.columns]
        elif callable(self.distributions):
            self.distributions_ = [self.distributions] * X.shape[1]

        X, y = self._clean(X, y)
        super().fit(X, y, weights)
        self.check_smooth_distributions()

        self.is_fitted_ = True
        return self

    def inspect_distribution(self, column, y=None):
        """
        Inspect sufficient statistics of given variable.
        """
        check_is_fitted(self)

        if y is None:
            ys = self.classes_
        else:
            ys = [y]

        distributions = {}
        for yi in ys:
            y_index = self.class_map_[yi]
            variable_index = self.column_map_[column]
            dist_i = self.model_.distributions[y_index].distributions[variable_index]
            distributions[yi] = dist_i

        if len(distributions) == 1:
            return distributions[y]
        return distributions

    def predict(self, X):
        """
        Predict labels using pomegranate.
        """
        # Check is fit had been called
        check_is_fitted(self)
        self.check_schema(X)
        X_np = self._clean(X)
        y = self.model_.predict(X_np)
        y = self.map_label_inverse_(y)
        if isinstance(X, DataFrame):
            return Series(y, index=X.index, name=X.index.name)
        return y

    def feature_importance(self, X):
        """Compute feature importance for given samples.

        Importance is defined as:
            [ln p(x_1|c=1) - ln p(x_1|c=0)
             ,..,
             ln p(x_n|c=1) - ln p(x_n|c=0),
             ln p(c=1) - ln p(c=0) ]
        """
        check_is_fitted(self)
        self.check_schema(X)
        X_np = self._clean(X)

        m_rows, n_features = X_np.shape
        n_classes = len(self.classes_)

        assert n_classes == 2, "Importance only defined for binary classes."

        # Log probability per class, per feature (i.e., ln p(x_i|c)) and the
        # class itself ln p(c).
        logp_yx = zeros(shape=[m_rows, n_classes, n_features + 1])

        # Compute ln p(x,y) = lnp(x|y) + ln p(y).
        for y_index in range(n_classes):
            for feature_index in range(n_features):
                dist_i = self.model_.distributions[y_index].distributions[feature_index]
                logp_yx[:, y_index, feature_index] = dist_i.log_probability(
                    X_np[:, feature_index]
                )
            # Last element is reserved for class itself: ln[p(c)].
            logp_yx[:, y_index, -1] = self.model_.weights[y_index]

        importance = logp_yx[:, 1] - logp_yx[:, 0]

        if isinstance(X, DataFrame):
            columns = list(X.columns) + ["label"]
            return DataFrame(importance, index=X.index, columns=columns)
        return importance

    def predict_proba(self, X):
        """
        Make probability prediction.
        """
        # Check is fit had been called
        check_is_fitted(self)
        self.check_schema(X)
        X_np = self._clean(X)
        y_pred = self.model_.predict_proba(X_np)
        if isinstance(X, DataFrame):
            return DataFrame(
                y_pred, index=X.index, columns=self.map_label_inverse_([0, 1])
            )
        return y_pred

    def decision_function(self, X) -> ndarray:
        """The probability of class 1 is the decision function."""
        y_prob = self.predict_proba(X)
        if isinstance(y_prob, DataFrame):
            return y_prob.to_numpy()[:, 1]
        return y_prob[:, 1]

    def score(self, X, y):
        """Calculate accuracy."""
        check_is_fitted(self)
        self.check_schema(X)
        X, y = self._clean(X, y)
        return self.model_.score(X, y)
