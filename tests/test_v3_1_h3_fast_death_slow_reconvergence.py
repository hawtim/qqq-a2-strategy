"""Profitable FAST-death then near-slow-gap reentry: synthetic causal state tests."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS
from a2_month_ema5_slope_regime import replay_gated
from a2_v3_1_h3_fast_death_slow_reconvergence import ARMS, replay
from qqq_a2.engine import Config
from test_v3_1_h3_trend_spread_quick_reentry import sample

CFG = Config(start_date="2024-01-02")


class FastDeathNarrowSpreadTests(unittest.TestCase):
    def test_frozen_h3_unmodified(self):
        bars = sample()
        a = replay(bars, "H3_FROZEN", CFG)
        b = replay_gated(
            bars, ENTRY_GATE, config=CFG,
            **V3_PARAMETERS, defer_rejected_fast=True,
            fast_negative_month_mode="accelerating")
        pd.testing.assert_frame_equal(a["trades"], b["trades"])
        pd.testing.assert_frame_equal(a["equity"], b["equity"])
        self.assertTrue(a["fast_profit_events"].empty)
        self.assertEqual(len(ARMS), 15)

    def test_profit_fast_death_exits_next_open_with_real_fill_fee(self):
        bars = sample()
        result = replay(bars, "PROFIT_FAST_DEATH_EXIT_ONLY", CFG)
        self.assertEqual(len(result["trades"]), 1)
        t = result["trades"].iloc[0]
        self.assertEqual(t.exit_signal_date, str(bars.Date.iloc[21].date()))
        self.assertEqual(t.exit_date, str(bars.Date.iloc[22].date()))
        self.assertEqual(t.exit_reason, "research profitable FAST death")
        self.assertGreater(t.ret_net, 0.)
        event = result["fast_profit_events"].iloc[0]
        self.assertEqual(event.action, "EXIT")
        self.assertEqual(event.signal_date, t.exit_signal_date)
        self.assertEqual(event.fill_date, t.exit_date)
        self.assertAlmostEqual(event.slow_gap_pct, 20.)

    def test_narrow_then_fresh_fast_gold_reenters_next_open(self):
        bars = sample()
        bars.loc[22:, "ma60"] = 119.
        yes = replay(bars, "PROFIT_FAST_DEATH_RE_FAST_NEAR_1", CFG)
        events = yes["fast_profit_events"]
        self.assertEqual(events.action.tolist(), ["EXIT", "REENTRY"])
        self.assertEqual(events.iloc[1].signal_date, str(bars.Date.iloc[27].date()))
        self.assertEqual(events.iloc[1].fill_date, str(bars.Date.iloc[28].date()))
        self.assertEqual(yes["open_position"]["entry_date"],
                         str(bars.Date.iloc[28].date()))
        self.assertEqual(yes["open_position"]["entry_kind"],
                         "FAST_EMA10_20_MONTH_EMA5")
        self.assertEqual(len(yes["trades"]), 1)
        no = replay(bars, "PROFIT_FAST_DEATH_RE_FAST_NEAR_0_5", CFG)
        self.assertEqual(no["fast_profit_events"].action.tolist(), ["EXIT"])
        self.assertFalse(no["stats"]["active_position_at_end"])

    def test_touch_then_gold_is_not_using_future_information(self):
        bars = sample()
        bars.loc[22:25, "ma60"] = 119.
        # Slow spread expands again before day27 FAST gold.
        current = replay_gated(
            bars, ENTRY_GATE, config=CFG,
            **V3_PARAMETERS, defer_rejected_fast=True,
            fast_negative_month_mode="accelerating",
            fast_profit_exit_peak_min=0.,
            fast_profit_reentry_mode="current",
            fast_profit_reentry_gap_pct=1.)
        seen = replay_gated(
            bars, ENTRY_GATE, config=CFG,
            **V3_PARAMETERS, defer_rejected_fast=True,
            fast_negative_month_mode="accelerating",
            fast_profit_exit_peak_min=0.,
            fast_profit_reentry_mode="seen",
            fast_profit_reentry_gap_pct=1.)
        self.assertEqual(current["fast_profit_events"].action.tolist(),
                         ["EXIT"])
        self.assertEqual(seen["fast_profit_events"].action.tolist(),
                         ["EXIT", "REENTRY"])
        self.assertEqual(seen["fast_profit_events"].iloc[1].fill_date,
                         str(bars.Date.iloc[28].date()))

    def test_slow_death_invalidates_persistent_reentry_state(self):
        bars = sample()
        bars.loc[22:, "ma60"] = 119.
        bars.loc[25, "death"] = True
        result = replay(bars, "PROFIT_FAST_DEATH_RE_FAST_NEAR_1", CFG)
        self.assertEqual(result["fast_profit_events"].action.tolist(), ["EXIT"])
        self.assertFalse(result["stats"]["active_position_at_end"])

    def test_no_profit_at_fast_death_cannot_trigger_exit(self):
        bars = sample()
        bars.loc[21, ["adj_close", "adj_low"]] = [99., 98.]
        result = replay(bars, "PROFIT_FAST_DEATH_RE_FAST_ANY", CFG)
        self.assertTrue(result["fast_profit_events"].empty)

    def test_hard_stop_overrides_early_profit_exit(self):
        bars = sample()
        bars.loc[21, ["adj_close", "adj_low"]] = [89., 87.]
        result = replay(bars, "PROFIT_FAST_DEATH_EXIT_ONLY", CFG)
        self.assertEqual(len(result["trades"]), 1)
        self.assertIn("close stop", result["trades"].iloc[0].exit_reason)
        self.assertTrue(result["fast_profit_events"].empty)

    def test_invalid_reentry_state(self):
        bars = sample()
        with self.assertRaises(ValueError):
            replay_gated(bars, ENTRY_GATE, config=CFG,
                         fast_profit_reentry_mode="current")
        with self.assertRaises(ValueError):
            replay_gated(bars, ENTRY_GATE, config=CFG,
                         fast_profit_exit_peak_min=-.1)
        with self.assertRaises(ValueError):
            replay_gated(bars, ENTRY_GATE, config=CFG,
                         fast_profit_exit_peak_min=0.,
                         fast_profit_reentry_mode="oracle")
        with self.assertRaises(ValueError):
            replay_gated(bars, ENTRY_GATE, config=CFG,
                         fast_profit_exit_peak_min=0.,
                         fast_profit_reentry_gap_pct=-.1)
        with self.assertRaises(ValueError):
            replay_gated(bars, ENTRY_GATE, config=CFG,
                         fast_profit_exit_peak_min=0.,
                         fast_profit_exit_slow_gap_min_pct=-1.)


if __name__ == "__main__":
    unittest.main()
