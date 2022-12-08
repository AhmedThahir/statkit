from unittest import TestCase

from statkit.types import (
    Estimate,
    _format_float,
    _get_number_of_significant_digits,
    _parse_scientific_notation,
)


class TestEstimate(TestCase):
    def test_utils(self):
        """Test helper functions of `Estimate`."""
        number, exponent = _parse_scientific_notation(0.000001234)
        self.assertEqual(number, 1.234)
        self.assertEqual(exponent, -6)

        self.assertEqual(_get_number_of_significant_digits(value=12.3, error=0.1), 3)
        self.assertEqual(
            _get_number_of_significant_digits(value=12.300, error=0.001), 5
        )
        self.assertEqual(_get_number_of_significant_digits(value=0.123, error=0.001), 3)

        self.assertEqual(_format_float(1.234, n_digits=2), "1.2")
        self.assertEqual(_format_float(1.234, n_digits=4), "1.234")
        self.assertEqual(_format_float(123.4, n_digits=2), "1.2e+02")

    def test_format_str(self):
        """Test string representation of estimate with uncertainty."""
        estimate1 = Estimate(
            point=2.5150,
            lower=2.5150 - 0.0201,
            upper=2.5150 + 0.0301,
        )
        str_representation = str(estimate1)
        self.assertEqual(str_representation, "2.52 (95 % CI: 2.49-2.55)")

        estimate2 = Estimate(
            point=25.160,
            lower=25.160 - 0.201,
            upper=25.160 + 0.301,
        )
        str_representation = str(estimate2)
        self.assertEqual(str_representation, "2.52e+01 (95 % CI: 2.50e+01-2.55e+01)")

        estimate3 = Estimate(
            point=0.25150,
            lower=0.25150 - 0.00201,
            upper=0.25150 + 0.00301,
        )
        str_representation = str(estimate3)
        self.assertEqual(str_representation, "0.252 (95 % CI: 0.249-0.255)")

        # When the estimate is of the same order of magnitude as the error, than use two digits.
        estimate4 = Estimate(
            point=0.711,
            lower=0.711 - 0.112,
            upper=0.711 + 0.191,
        )
        str_representation = str(estimate4)
        self.assertEqual(str_representation, "0.71 (95 % CI: 0.60-0.90)")

    def test_format_latex(self):
        """Test LaTeX representation of estimate with uncertainty."""
        estimate1 = Estimate(
            point=2.5150,
            lower=2.5150 - 0.0201,
            upper=2.5150 + 0.0301,
        )

        self.assertEqual(estimate1.latex(), "2.52$^{+0.03}_{-0.02}$")

        estimate2 = Estimate(
            point=25.150,
            lower=25.150 - 0.201,
            upper=25.150 + 0.301,
        )
        self.assertEqual(estimate2.latex(), r"2.52$^{+0.03}_{-0.02} \cdot 10^{1}$")

        estimate3 = Estimate(
            point=0.025150,
            lower=0.025150 - 0.000201,
            upper=0.025150 + 0.000301,
        )
        self.assertEqual(estimate3.latex(), r"2.52$^{+0.03}_{-0.02} \cdot 10^{-2}$")

        estimate4 = Estimate(
            point=0.25150,
            lower=0.25150 - 0.00201,
            upper=0.25150 + 0.00301,
        )
        self.assertEqual(estimate4.latex(), "0.252$^{+0.003}_{-0.002}$")

        # When the estimate is of the same order of magnitude as the error, than use two digits.
        estimate5 = Estimate(
            point=0.711,
            lower=0.711 - 0.112,
            upper=0.711 + 0.191,
        )
        self.assertEqual(estimate5.latex(), "0.71$^{+0.19}_{-0.11}$")

    def test_representation(self):
        """Test that machine representation is actual python code."""
        estimate = Estimate(
            point=0.711,
            lower=0.711 - 0.112,
            upper=0.711 + 0.191,
        )
        # Verify that the representation can be used to instantiate new object.
        representation = estimate.__repr__()
        self.assertEqual(estimate, eval(representation))

    def test_dict(self):
        """Test that object can be converted to dict."""
        estimate = Estimate(
            point=0.711,
            lower=0.7,
            upper=0.81234,
        )
        self.assertDictEqual(
            dict(estimate), {"point": 0.711, "lower": 0.7, "upper": 0.81234}
        )
