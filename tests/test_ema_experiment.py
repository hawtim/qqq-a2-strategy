"""Independent synthetic invariants for the EMA experiment branch."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/experiments"))

from a2_ema_comparison import ema_signals, replay_signals, validate_sma_against_frozen
from qqq_a2.engine import Config


class EmaExperimentTests(unittest.TestCase):
    def test_ema_is_recursive_and_requires_full_period(self):
        close = [100.0] * 10 + [110.0, 90.0] + [100.0] * 210
        frame = pd.DataFrame({"Date": pd.bdate_range("2023-01-02", periods=len(close)),
                              "adj_open": close, "adj_close": close})
        actual = ema_signals(frame)
        self.assertTrue(actual.ma10.iloc[:9].isna().all())
        self.assertAlmostEqual(actual.ma10.iloc[9], 100.0)
        self.assertAlmostEqual(actual.ma10.iloc[10], 100 + 10 * 2 / 11)
        self.assertTrue(actual.ma60.iloc[:59].isna().all())
        self.assertTrue(actual.ma200.iloc[:199].isna().all())

    def test_signals_never_depend_on_future_closes(self):
        close = [100 - 0.21 * i for i in range(100)] + [79 + 0.33 * i for i in range(120)]
        frame = pd.DataFrame({"Date": pd.bdate_range("2023-01-02", periods=len(close)),
                              "adj_open": close, "adj_close": close})
        original = ema_signals(frame)
        changed = frame.copy()
        changed.loc[len(frame) - 1, "adj_close"] *= 2
        updated = ema_signals(changed)
        pd.testing.assert_frame_equal(original.iloc[:-1], updated.iloc[:-1])

    def test_replay_matches_frozen_execution_engine_on_synthetic_history(self):
        close = ([100 - 0.20 * i for i in range(95)] +
                 [81 + 0.39 * i for i in range(110)] +
                 [124 - 0.55 * i for i in range(75)] +
                 [83 + 0.45 * i for i in range(120)])
        frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-02", periods=len(close)),
                              "adj_open": close, "adj_close": close})
        config = Config(start_date="2024-01-02")
        baseline = validate_sma_against_frozen(frame, config)
        self.assertGreater(len(baseline["trades"]), 0)
        alternative = replay_signals(ema_signals(frame), config)
        self.assertTrue(alternative["equity"].Equity.gt(0).all())
        self.assertGreaterEqual(len(alternative["trades"]), 1)

    def test_next_open_fee_and_no_same_day_exit(self):
        dates = pd.bdate_range("2024-01-01", periods=8)
        frame = pd.DataFrame({"Date": dates, "adj_open": [100, 110, 100, 100, 100, 120, 100, 100],
                              "adj_close": [100] * 8,
                              "gold": [True] + [False] * 7,
                              "death": [False] * 4 + [True] + [False] * 3,
                              "protect_cross": [False] * 8})
        r = replay_signals(frame, Config(start_date="2024-01-01"))
        trade = r["trades"].iloc[0]
        self.assertEqual(trade.entry_date, "2024-01-02")
        self.assertEqual(trade.exit_date, "2024-01-08")
        self.assertAlmostEqual(trade.ret_net, 120 * .9995 / (110 * 1.0005) - 1)


if __name__ == "__main__":
    unittest.main()
