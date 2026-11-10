"""Deterministic cash-return tests, no market data/network required."""
import unittest

import pandas as pd

from qqq_a2.cash_yield import cash_overlay


class CashCarryTests(unittest.TestCase):
    def test_cash_receives_calendar_weekend_interest(self):
        c = pd.DataFrame({"Date": ["2024-01-05", "2024-01-08"],
                          "Equity": [100.0, 100.0], "Invested": [False, False]})
        v = cash_overlay(c, 3.65)
        self.assertAlmostEqual(v.iloc[-1], 100 * (1 + 0.0365 * 3 / 365))

    def test_no_interest_when_invested_previous_close(self):
        c = pd.DataFrame({"Date": ["2024-01-05", "2024-01-08"],
                          "Equity": [100.0, 95.0], "Invested": [True, False]})
        self.assertAlmostEqual(cash_overlay(c, 5).iloc[-1], 95.0)

    def test_overnight_cash_on_next_open_buy(self):
        c = pd.DataFrame({"Date": ["2024-01-05", "2024-01-08"],
                          "Equity": [100.0, 102.0], "Invested": [False, True]})
        self.assertAlmostEqual(cash_overlay(c, 3.65).iloc[-1],
                               102 * (1 + .0365 * 3 / 365))

    def test_zero_cash_yield_preserves_baseline_every_day(self):
        c = pd.DataFrame({"Date": ["2024-01-04", "2024-01-05", "2024-01-08"],
                          "Equity": [100.0, 98.0, 101.0],
                          "Invested": [True, False, True]})
        self.assertTrue((cash_overlay(c, 0).round(8) == c.Equity.round(8)).all())

    def test_prior_known_variable_rate_not_todays_rate(self):
        c = pd.DataFrame({"Date": ["2024-01-04", "2024-01-05", "2024-01-08"],
                          "Equity": [100.0] * 3, "Invested": [False] * 3})
        v = cash_overlay(c, pd.Series([1.0, 2.0, 20.0]))
        expected = 100 * (1 + .01 / 365) * (1 + .02 * 3 / 365)
        self.assertAlmostEqual(v.iloc[-1], expected)


if __name__ == "__main__":
    unittest.main()
