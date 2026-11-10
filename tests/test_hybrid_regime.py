"""Hybrid regime experiment invariants; no production baseline changes."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/experiments"))

from a2_hybrid_regime import REGIMES, completed_regime, hybrid_backtest
from qqq_a2.engine import Config


class HybridRegimeTests(unittest.TestCase):
    @staticmethod
    def synthetic():
        close = ([100 - .15 * i for i in range(120)] +
                 [82 + .36 * i for i in range(150)] +
                 [136 - .30 * i for i in range(120)] +
                 [100 + .32 * i for i in range(140)])
        return pd.DataFrame({"Date": pd.bdate_range("2022-01-03", periods=len(close)),
                             "adj_open": close, "adj_close": close})

    def test_no_current_week_or_month_lookahead(self):
        d = self.synthetic()
        for name in REGIMES:
            first = completed_regime(d, name)
            changed = d.copy()
            changed.loc[100, "adj_close"] *= 9
            updated = completed_regime(changed, name)
            # The changed daily close cannot change that period's classification.
            bucket = d.Date.dt.to_period(REGIMES[name][0])
            in_same_period = bucket == bucket.iloc[100]
            self.assertTrue(first.loc[in_same_period].equals(updated.loc[in_same_period]), name)

    def test_last_row_cannot_change_prior_signals_or_equity(self):
        d = self.synthetic()
        for name in REGIMES:
            original = completed_regime(d, name)
            changed = d.copy()
            changed.loc[len(d)-1, "adj_close"] *= 6
            altered = completed_regime(changed, name)
            pd.testing.assert_series_equal(original, altered)

    def test_locked_positions_and_fast_exit_gate(self):
        d = self.synthetic()
        for name in REGIMES:
            for mode in ("locked", "bear_fast"):
                r = hybrid_backtest(d, name, mode, Config(start_date="2022-01-03"))
                self.assertTrue(r["equity"].Equity.gt(0).all())
                self.assertTrue(r["trades"].entry_family.isin(["SMA", "EMA"]).all())
                self.assertTrue((r["trades"].hold_days >= 1).all())
                if mode == "locked":
                    self.assertEqual(r["stats"]["bear_fast_exits"], 0)
                self.assertEqual(r["stats"]["ema_entries"] + r["stats"]["sma_entries"],
                                 r["stats"]["completed_trades"])


if __name__ == "__main__":
    unittest.main()
