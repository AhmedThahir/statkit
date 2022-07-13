from unittest import TestCase

from statkit.views import format_p_value


class TestViews(TestCase):
    def test_format_p_value(self):
        """Test formatting of p-value."""
        # When >= 0.10, don't use 10 to power stuff.
        self.assertEqual(format_p_value(0.10, latex=False), "0.10")
        self.assertEqual(format_p_value(0.10, latex=True), "$0.10$")
        self.assertEqual(format_p_value(0.10, symbol="p", latex=False), "p = 0.10")
        self.assertEqual(format_p_value(0.10, latex=True, symbol="p"), "$p = 0.10$")
        self.assertEqual(format_p_value(0.0512, latex=False), "5.1E-02")
        self.assertEqual(
            format_p_value(0.0512, latex=True, symbol="q"), r"$q = 5.1 \cdot 10^{-2}$"
        )
