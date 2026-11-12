"""Partial H3 execution tests: original entry rules, full-exit cash, next-open fills."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_v3_1_h3_partial_position import ARMS, replay
from a2_month_ema5_slope_regime import replay_gated
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS
from qqq_a2.engine import Config
from test_v3_1_h3_trend_spread_quick_reentry import sample

CFG = Config(start_date="2024-01-02")


class PartialH3Tests(unittest.TestCase):
    def test_frozen_h3_unchanged_and_no_partial_orders(self):
        s = sample()
        baseline = replay(s, "H3_FROZEN", CFG)
        reference = replay_gated(
            s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
            defer_rejected_fast=True,
            fast_negative_month_mode="accelerating")
        pd.testing.assert_frame_equal(baseline["trades"], reference["trades"])
        pd.testing.assert_frame_equal(baseline["equity"], reference["equity"])
        self.assertTrue(baseline["rebalance_events"].empty)
        self.assertNotIn("PositionWeight", baseline["equity"].columns)

    def test_reduce_then_restore_at_next_open_preserves_original_entry(self):
        s = sample()
        r = replay(s, "CURRENT_1_5_TRIM50_RE5", CFG)
        events = r["rebalance_events"]
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events.iloc[0].action, "trim")
        self.assertEqual(events.iloc[0].signal_date, str(s.Date.iloc[21].date()))
        self.assertEqual(events.iloc[0].fill_date, str(s.Date.iloc[22].date()))
        self.assertAlmostEqual(events.iloc[0].fraction_of_shares, .5)
        self.assertEqual(events.iloc[1].action, "restore")
        # The synthetic data has a 5/10 gold cross after the trim.
        self.assertEqual(events.iloc[1].signal_date, str(s.Date.iloc[27].date()))
        self.assertEqual(events.iloc[1].fill_date, str(s.Date.iloc[28].date()))
        self.assertEqual(r["open_position"]["entry_date"], str(s.Date.iloc[1].date()))
        self.assertEqual(r["open_position"]["entry_index"], 1)
        self.assertEqual(r["open_position"]["restore_count"], 1)
        self.assertNotIn("N20 reset", events.to_string())
        self.assertIn("PositionWeight", r["equity"].columns)
        row = r["equity"].set_index("Date")
        partial = row.at[str(s.Date.iloc[22].date()), "PositionWeight"]
        self.assertGreater(partial, .4)
        self.assertLess(partial, .6)
        self.assertAlmostEqual(
            row.at[str(s.Date.iloc[28].date()), "PositionWeight"], 1.,
            places=9)

    def test_fee_debits_at_every_participating_order(self):
        s = sample()
        r = replay(s, "CURRENT_1_5_TRIM50_RE5", CFG)
        events = r["rebalance_events"]
        self.assertTrue((events.fee > 0).all())
        self.assertTrue((events.price > 0).all())
        self.assertEqual(len(r["trades"]), 0)
        self.assertGreater(r["equity"].Equity.min(), 0.)

    def test_original_stop_still_clears_partially_held_position(self):
        s = sample()
        # After trim at day22 OPEN, force the entry-relative -7% guard on
        # day23 CLOSE. The remaining position sells day24 OPEN.
        s.loc[23, ["adj_close", "adj_low"]] = [92., 91.]
        r = replay(s, "CURRENT_1_5_TRIM50_RE5", CFG)
        self.assertGreaterEqual(len(r["trades"]), 1)
        t = r["trades"].iloc[0]
        self.assertIn("close stop", t.exit_reason)
        self.assertEqual(t.exit_signal_date, str(s.Date.iloc[23].date()))
        self.assertEqual(t.exit_date, str(s.Date.iloc[24].date()))
        self.assertEqual(r["rebalance_events"].iloc[0].action, "trim")
        # Includes proceeds already retained after the partial trim,
        # not just the 50% shares that were still invested.
        self.assertGreater(t.equity_after_trade, 0.)
        self.assertEqual(r["rebalance_events"].iloc[0].entry_date, t.entry_date)

    def test_threshold_gating_prevents_partial_action(self):
        s = sample()
        r = replay(s, "PEAK40_2_5_TRIM50_RE5", CFG)
        self.assertEqual(len(ARMS), 10)
        # This 45-session synthetic example may meet a 2.5 ATR distance;
        # a much larger predetermined threshold must never trigger.
        no = replay_gated(
            s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
            defer_rejected_fast=True, fast_negative_month_mode="accelerating",
            trend_spread_atr_min=1000.,
            trend_spread_trim_fraction=.5)
        self.assertTrue(no["rebalance_events"].empty)
        self.assertTrue(no["stats"]["active_position_at_end"])

    def test_invalid_partial_fraction_and_missing_filter(self):
        s = sample()
        for fraction in (-.5, 0, 1, 1.5):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                replay_gated(s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                             trend_spread_atr_min=1.5,
                             trend_spread_trim_fraction=fraction)
        with self.assertRaises(ValueError):
            replay_gated(s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         trend_spread_trim_fraction=.5)


if __name__ == "__main__":
    unittest.main()
