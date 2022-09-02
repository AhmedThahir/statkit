from typing import Literal, Optional
import re

from statkit.types import (
    _parse_scientific_notation,
)


def _format_scientific(number: float, latex: bool) -> str:
    """Represent value with two significant digits, unless >= 0.1;"""
    if number < 0.1:
        number_str = "{:.1E}".format(number)
        if latex:
            number_str = re.sub(
                r"([0-9]+\.[0-9])E-0([0-9]+)", r"\1 \\cdot 10^{-\2}", number_str
            )
    else:
        number_str = f"{number:.2f}"

    if latex:
        return "$" + number_str + "$"
    return number_str


def _format_compact(value: float, latex: bool) -> str:
    """Format value (floating point or scientific notation) with the least number of characters (including the minus
    sign.)

    Example:
        0.0013  -> 0.0013
        0.00013 -> 1.3 * 10^{-4}
    """
    _, exponent = _parse_scientific_notation(value)
    if exponent <= -4:
        return _format_scientific(value, latex)

    str_format = "{:." + str(abs(exponent) + 1) + "f}"
    value_str = str_format.format(value)
    if latex:
        return "$" + value_str + "$"
    return value_str


def format_p_value(
    number: float,
    latex: bool = True,
    symbol: Optional[str] = None,
    format: Literal["scientific", "compact"] = "scientific",
) -> str:
    r"""Format p-value with two significant digits as string except when ≥ 0.1.

    Args:
        number: Floating point number to format.
        latex: Format string as LaTeX math (with enclosing $ characters).
        symbol: When not `None` but, e.g., "p" it prints "p = number".
        format: `scientific`, represent p-value with two significant digits,
            unless >= 0.1; `compact`, uses the representation (floating point or
            scientific notation) with the least number of decimals including the minus
            sign.

    Returns:
        A string representation of the number.

    Example:
        ```python
            >>> print(format_p_value(0.0012, symbol='p'))
            $p = 1.2 \cdot 10^{-3}$
        ```
    """
    if format == "scientific":
        number_str = _format_scientific(number, latex)
    elif format == "compact":
        number_str = _format_compact(number, latex)

    if symbol:
        if latex:
            return r"${} = {}$".format(
                symbol,
                number_str.removeprefix("$").removesuffix("$"),
            )
        return f"{symbol} = {number_str}"

    return number_str
