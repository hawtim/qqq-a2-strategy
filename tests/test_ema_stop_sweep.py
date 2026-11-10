"""Fixed-grid stop parameter tests; none use future data or actual QQQ bars."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from a2_ema_hard_stop8 import replay_guarded
from a2_ema_stop_threshold_sweep import THRESHOLDS, affected_trades, make_summary, pct_label
from qqq_a2.engine import Config


def signals() -> pd.DataFrame:
    """Gold cross, shallow crash, then a later ordinary death-cross exit."""
    return pd.DataFrame({
        "Date": pd.bdate_range("2024-01-02", periods=6),
        "adj_open": [100, 100, 100, 100, 100, 100],
        "adj_close": [100, 100, 92.8, 100, 100, 100],
        "adj_low": [100, 100, 92.0, 99, 99, 99],
        "gold": [True, False, False, False, False, False],
        "death": [False, False, False, True, False, False],
        "protect_cross": [False] * 6,
    })


class StopSweepTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(start_date="2024-01-02")

    def test_candidates_are_predeclared_and_labels_stable(self):
        self.assertEqual(THRESHOLDS, (.05, .06, .07, .075, .08, .09))
        self.assertEqual([pct_label(t) for t in THRESHOLDS],
                         ["5", "6", "7", "7.5", "8", "9"])

    def test_close_stop_uses_actual_threshold_in_reason_and_summary(self):
        frame = signals()
        r5 = replay_guarded(frame, config=self.cfg, stop_loss=.05)
        r8 = replay_guarded(frame, config=self.cfg, stop_loss=.08)
        self.assertEqual(r5["trades"].iloc[0].exit_reason, "5% close stop")
        self.assertEqual(r5["stats"]["hard_stop_exits"], 1)
        self.assertEqual(r8["trades"].iloc[0].exit_reason, "MA20/60 death cross")
        self.assertEqual(r8["stats"]["hard_stop_exits"], 0)
        table = make_summary({"EMA_close_next_open_stop5": r5,
                              "EMA_close_next_open_stop8": r8}, "2024-01-09")
        self.assertEqual(table.stop_threshold.to_list(), [.05, .08])
        self.assertEqual(table.hard_stop_exits.to_list(), [1, 0])

    def test_intraday_stop_uses_touched_price_and_threshold_label(self):
        r7 = replay_guarded(signals(), config=self.cfg, stop_loss=.07,
                            mode="intraday_gap_aware")
        self.assertEqual(r7["trades"].iloc[0].exit_reason, "7% intraday stop")
        self.assertAlmostEqual(r7["trades"].iloc[0].ret_net, -.07)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            replay_guarded(signals(), config=self.cfg, stop_loss=0)

    def test_affected_trade_join_uses_entry_date(self):
        frame = signals()
        # Both start on the same day; the 5% model gets out two sessions earlier.
        r5 = replay_guarded(frame, config=self.cfg, stop_loss=.05)
        r8 = replay_guarded(frame, config=self.cfg, stop_loss=.08)
        audit = affected_trades({"EMA_original": r8,
                                 "EMA_close_next_open_stop5": r5})
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit.iloc[0].entry_date, "2024-01-03")
        self.assertNotEqual(audit.iloc[0].original_exit, audit.iloc[0].variant_exit)


if __name__ == "__main__":
    unittest.main()
