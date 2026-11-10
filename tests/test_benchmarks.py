"""Synthetic checks of the separately defined QQQ buy/hold comparator."""
import unittest

import pandas as pd

from qqq_a2.benchmarks import buy_and_hold
from qqq_a2.engine import Config


class BuyHoldTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({
            "Date": pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]),
            "adj_open": [100., 85., 40.],
            "adj_close": [100., 40., 105.],
        })

    def test_first_open_fee_and_terminal_mark(self):
        a = buy_and_hold(self.prices, Config(fee_per_side=.0005))
        expect = 100_000 / 1.0005 * 1.05
        self.assertAlmostEqual(a["stats"]["end_equity"], expect)
        self.assertAlmostEqual(a["stats"]["end_equity_if_sold_at_last_close_and_paid_exit_fee"],
                               expect * .9995)
        self.assertFalse(a["stats"]["exit_fee_assessed"])

    def test_daily_mdd_recovery_dates(self):
        a = buy_and_hold(self.prices, Config(fee_per_side=0))
        self.assertAlmostEqual(a["stats"]["mdd"], -.60)
        self.assertEqual(a["stats"]["mdd_peak_date"], "2000-01-03")
        self.assertEqual(a["stats"]["mdd_trough_date"], "2000-01-04")
        self.assertEqual(a["stats"]["mdd_recovery_date"], "2000-01-05")
        self.assertAlmostEqual(a["stats"]["mdd_recovery_return_needed"], 1.5)

    def test_reject_missing_start(self):
        with self.assertRaisesRegex(ValueError, "First full session"):
            buy_and_hold(self.prices.iloc[1:], Config())


if __name__ == "__main__":
    unittest.main()
