from typing import Optional, Union
import re

from pandas import Series


def format_confidence_interval(
    estimate: Union[Series, dict[str, float]], latex: bool = True
) -> str:
    """Format 95 % confidence interval dictionary into string.

    Args:
        estimate: Dictionary like that returned by
            `statkit.non_parametric.bootstrap_score`,  with keys "point" (the estimate),
            "lower" [lower limit of 95 % confidence interval(CI)], and "upper"
            (upper limit of 95 % CI).
        latex: Format string as LaTeX math.

    Example:
        Compute 95 % confidence interval and format string.
        ```python
            from sklearn.metrics import roc_auc_score
            from statkit.non_parametric import bootstrap_score
            from statkit.views import format_confidence_interval

            y_prob = model.predict(X_test)
            auc_95ci = bootstrap_score(y_test, y_prob, metric=roc_auc_score)
            print(
                'Area under the ROC curve:',
                format_confidence_interval(auc_95ci, latex=False),
            )
        ```
    """
    value, lower, upper = estimate["point"], estimate["lower"], estimate["upper"]
    label_args = (
        value,
        upper - value,
        lower - value,
    )
    if latex:
        return "{:.2f}$^{{+{:.2f}}}_{{{:.2f}}}$".format(*label_args)
    return f"{value:.2f} (95 % CI: {lower:.2f}-{upper:.2f})"


def format_p_value(
    number: float, latex: bool = True, symbol: Optional[str] = None
) -> str:
    r"""Format p-value with two significant digits as string except when ≥ 0.1.

    Args:
        number: Floating point number to format.
        latex: Format string as LaTeX math (with enclosing $ characters).
        symbol: When not `None` but, e.g., "p" it prints "p = number".

    Returns:
        A string representation of the number.

    Example:
        ```python
            >>> print(format_p_value(0.0012, symbol='p'))
            $p = 1.2 \cdot 10^{-3}$
        ```
    """
    if number < 0.1:
        number_str = "{:.1E}".format(number)
        if latex:
            number_str = re.sub(
                r"([0-9]+\.[0-9])E-0([0-9]+)", r"\1 \\cdot 10^{-\2}", number_str
            )
    else:
        number_str = f"{number:.2f}"

    if symbol:
        number_str = f"{symbol} = {number_str}"

    if latex:
        return "$" + number_str + "$"
    return number_str
