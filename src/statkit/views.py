from typing import Optional
import re


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
