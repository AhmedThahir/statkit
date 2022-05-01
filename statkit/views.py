from typing import Union

from pandas import Series


def confidence_interval_label(score: Union[Series, dict], latex: bool = True) -> str:
    """Turn value with confidence interval into text.

    Args:
        latex: Format string as LaTeX math.
    """
    value, lower, upper = score["point"], score["lower"], score["upper"]
    label_args = (
        value,
        upper - value,
        lower - value,
    )
    if latex:
        return "{:.2f}$^{{+{:.2f}}}_{{{:.2f}}}$".format(*label_args)
    return f"{value:.2f} (95 % CI: {lower:.2f}-{upper:.2f})"
