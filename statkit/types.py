import re
from typing import Optional


def _parse_scientific_notation(input: float) -> Optional[tuple[float, int]]:
    """Convert number 0.00001234 into 1.234 and -6"""
    input_str = "{:e}".format(input)
    scientific_number = r"([-+]?[\d]+\.?[\d]*)[Ee]((?:[-+]?[\d]+)?)"
    if (match := re.search(string=input_str, pattern=scientific_number)) is not None:
        number = float(match.group(1))
        exponent = int(match.group(2))
        return number, exponent

    return None


def _get_number_of_significant_digits(value: float, error: float) -> int:
    """Determine how many significant digits to use."""
    _, exponent_estimate = _parse_scientific_notation(value)
    _, exponent_error = _parse_scientific_notation(error)
    return max(exponent_estimate - exponent_error + 1, 2)


def _format_float(value: float, n_digits: int):
    """Format string upto given number of digits."""
    number, exponent = _parse_scientific_notation(value)

    # No need to add the e+00 part.
    if exponent == 0:
        format_template = "{:." + str(n_digits - 1) + "f}"
        return format_template.format(number)
    elif exponent == -1:
        format_template = "{:." + str(n_digits) + "f}"
        return format_template.format(value)

    format_template = "{:." + str(n_digits - 1) + "e}"
    return format_template.format(value)


class Estimate:
    """Estimate with range of uncertainty (95 % confidence interval).

    Example:
        ```python
        mean = 1.0
        standard_deviation = 0.2
        estimate = Estimate(
            point=mean,
            lower=mean - 1.96*standard_deviation,
            upper=mean + 1.96*standard_deviation,
            )
        # Print estimate in LaTeX format.
        print(estimate.latex())
        ```
    """

    def __init__(self, point: float, lower: float, upper: float):
        """
        Args:
            point: Point estimate.
            lower: Lower limit of 95 % confidence interval (CI).
            upper: Upper limit of 95 % CI.
        """
        self.point = point
        self.lower = lower
        self.upper = upper

    def _significant_digits(self) -> int:
        """Number of significant digits"""
        upper_error = self.upper - self.point
        lower_error = self.point - self.lower
        error = max(upper_error, lower_error)
        return _get_number_of_significant_digits(self.point, error)

    def latex(self) -> str:
        """Format 95 % confidence interval as LaTeX."""
        n_digits = self._significant_digits()
        number, exponent = _parse_scientific_notation(self.point)
        if exponent == 0:
            label_args = (
                number,
                (self.upper - self.point),
                (self.lower - self.point),
            )
            return (
                "{:."
                + str(n_digits - 1)
                + "f}$^{{+{:."
                + str(n_digits - 1)
                + "f}}}_{{{:."
                + str(n_digits - 1)
                + "f}}}$"
            ).format(*label_args)

        elif exponent == -1:
            label_args = (
                self.point,
                (self.upper - self.point),
                (self.lower - self.point),
            )
            return (
                "{:."
                + str(n_digits)
                + "f}$^{{+{:."
                + str(n_digits)
                + "f}}}_{{{:."
                + str(n_digits)
                + "f}}}$"
            ).format(*label_args)

        label_args = (
            number,
            (self.upper - self.point) / 10**exponent,
            (self.lower - self.point) / 10**exponent,
            exponent,
        )

        return (
            "{:."
            + str(n_digits - 1)
            + r"f}$^{{+{:."
            + str(n_digits - 1)
            + r"f}}}_{{{:."
            + str(n_digits - 1)
            + r"f}}} \cdot 10^{{{:}}}$"
        ).format(*label_args)

    def __str__(self) -> str:
        """String representation of estimate with 95 % confidence interval."""
        n_digits = self._significant_digits()
        point_str = _format_float(self.point, n_digits)
        lower_str = _format_float(self.lower, n_digits)
        upper_str = _format_float(self.upper, n_digits)
        return f"{point_str} (95 % CI: {lower_str}-{upper_str})"

    def __eq__(self, other) -> bool:
        """Estimates are equal if all their values are equal."""
        # Perhaps we should make a more lenient definition, that estimate is not
        # significantly different?
        return (
            (self.point == other.point)
            and (self.upper == other.upper)
            and (self.lower == other.lower)
        )

    def __repr__(self) -> str:
        """Machine friendly representation."""
        return f"Estimate({self.point}, lower={self.lower}, upper={self.upper})"
