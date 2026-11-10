"""Synthetic tests for the independent EMA always-on loss-stop experiment."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from a2_ema_hard_stop8 import replay_guarded, stop_trigger_price
from a2_ema_comparison import ema_signals, replay_signals
from qqq_a2.engine import Config


def fixture(opens, closes, lows, *, protect_at=None):
    n = len(opens)
    return pd.DataFrame({
        "Date": pd.bdate_range("2024-01-02", periods=n),
        "adj_open": opens, "adj_close": closes, "adj_low": lows,
        "gold": [True] + [False] * (n - 1),
        "death": [False] * n,
        "protect_cross": [i == protect_at for i in range(n)],
    })


class EmaHardStop8Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(start_date="2024-01-02")

    def test_disabled_stop_reproduces_original_ema_without_missing_reentries(self):
        close = ([100 - .22 * i for i in range(90)] +
                 [80 + .39 * i for i in range(130)] +
                 [130 - .44 * i for i in range(85)] +
                 [93 + .23 * i for i in range(145)])
        frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=len(close)),
                              "adj_open": close, "adj_low": close, "adj_close": close})
        signal = ema_signals(frame)
        base = replay_signals(signal, self.cfg)
        for mode in ("close_next_open", "intraday_gap_aware"):
            test = replay_guarded(signal, config=self.cfg, mode=mode, stop_loss=.999)
            pd.testing.assert_frame_equal(
                test["trades"][list(base["trades"].columns)], base["trades"])
            pd.testing.assert_series_equal(test["equity"].Equity, base["equity"].Equity)

    def test_touching_stop_intraday_sells_at_net_eight_percent(self):
        f = fixture([100, 100, 100, 100], [100, 100, 99, 100], [100, 99, 90, 99])
        t = replay_guarded(f, config=self.cfg, mode="intraday_gap_aware")["trades"]
        self.assertEqual(len(t), 1)
        self.assertEqual(t.iloc[0].exit_date, "2024-01-04")
        self.assertAlmostEqual(t.iloc[0].ret_net, -.08)
        self.assertFalse(t.iloc[0].gap_below_stop)

    def test_stop_is_armed_on_entry_session_without_n20_wait(self):
        f = fixture([100, 100, 100, 100], [100, 99, 100, 100],
                    [100, 90, 99, 99])
        t = replay_guarded(f, config=self.cfg, mode="intraday_gap_aware")["trades"]
        self.assertEqual(t.iloc[0].entry_date, "2024-01-03")
        self.assertEqual(t.iloc[0].exit_date, "2024-01-03")
        self.assertEqual(t.iloc[0].hold_days, 0)
        self.assertAlmostEqual(t.iloc[0].ret_net, -.08)

    def test_gap_stop_uses_worse_open_not_fictional_stop_price(self):
        f = fixture([100, 100, 85, 100], [100, 100, 95, 100], [100, 99, 84, 99])
        t = replay_guarded(f, config=self.cfg, mode="intraday_gap_aware")["trades"]
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t.iloc[0].exit_price, 85.0)
        self.assertLess(t.iloc[0].ret_net, -.14)
        self.assertTrue(t.iloc[0].gap_below_stop)

    def test_close_stop_waits_until_next_open_and_can_overshoot(self):
        f = fixture([100, 100, 100, 85, 85],
                    [100, 100, 90, 85, 85], [100, 99, 89, 84, 84])
        t = replay_guarded(f, config=self.cfg, mode="close_next_open")["trades"]
        self.assertEqual(t.iloc[0].exit_signal_date, "2024-01-04")
        self.assertEqual(t.iloc[0].exit_date, "2024-01-05")
        self.assertAlmostEqual(t.iloc[0].exit_price, 85.0)
        self.assertLess(t.iloc[0].ret_net, -.14)

    def test_n20_first_cancels_later_eight_percent_stop(self):
        f = fixture([100, 100, 100, 100, 100],
                    [100, 100, 99, 90, 100],
                    [100, 99, 98, 89, 99], protect_at=2)
        t = replay_guarded(f, config=self.cfg, mode="close_next_open")["trades"]
        self.assertEqual(t.iloc[0].exit_reason, "N20 protection")
        self.assertEqual(t.iloc[0].exit_date, "2024-01-05")

    def test_stop_threshold_accounts_for_both_fees(self):
        price = stop_trigger_price(100, .0005, .08)
        self.assertAlmostEqual(price * .9995 / (100 * 1.0005) - 1, -.08)


if __name__ == "__main__":
    unittest.main()
