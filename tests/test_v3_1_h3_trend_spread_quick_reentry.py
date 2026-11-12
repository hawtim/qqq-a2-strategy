"""H3 quick-reentry research: the new flags are opt-in and strictly causal."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_month_ema5_slope_regime import replay_gated
from a2_v3_1_h3_trend_spread_quick_reentry import ARMS, replay
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS
from qqq_a2.engine import Config

CFG = Config(start_date="2024-01-02")


def sample():
    days = pd.bdate_range("2024-01-02", periods=45)
    c = [100.] * 45
    for k in range(12, 22):
        c[k] = 112. + (k-12)*1.5
    c[18] = 125.
    c[19] = 124.
    c[20] = 120.
    c[21] = 118.
    c[22] = 118.
    c[23] = 119.
    c[24] = 121.
    c[25] = 122.
    c[26] = 120.
    c[27] = 123.
    for k in range(28,45):
        c[k] = 125.
    ma10 = [122.] * 45
    ma10[21:27] = [119.] * 6
    ma10[27:] = [122.] * (45-27)
    return pd.DataFrame({
        "Date": days, "adj_open": [100.] * 45,
        "adj_high": [v+2 for v in c],
        "adj_low": [v-2 for v in c],
        "adj_close": c,
        "ma10": ma10,
        "ma20": [120.] * 45,
        "ma60": [100.] * 45,
        "gold": [True]+[False]*44,
        "death": [False]*45,
        "protect_cross": [i==21 for i in range(45)],
        "ema10_20_gold": [i==27 for i in range(45)],
        "gate_up_slow_veto_1_abs": [True]*45,
        "prev_completed_month_ema5_up": [False]*45,
        "known_slope_abs": [-1.]*45,
        "known_prev_abs": [-2.]*45,
    })


class H3TrendSpreadQuickReentryTests(unittest.TestCase):
    def test_frozen_replay_unchanged(self):
        x = sample()
        baseline = replay(x, "H3_FROZEN", CFG)
        ref = replay_gated(
            x, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
            defer_rejected_fast=True, fast_negative_month_mode="accelerating")
        pd.testing.assert_frame_equal(baseline["trades"], ref["trades"])
        pd.testing.assert_frame_equal(baseline["equity"], ref["equity"])
        self.assertEqual(len(ARMS), 8)

    def test_profit_exit_waits_until_full_close_and_next_open(self):
        x = sample()
        original = replay(x, "H3_FROZEN", CFG)
        self.assertEqual(len(original["trades"]), 0)
        research = replay(x, "EXIT_FAST_DEATH_NO_RE", CFG)
        tr = research["trades"]
        self.assertEqual(len(tr), 1)
        self.assertIn("research trend spread", tr.iloc[0].exit_reason)
        self.assertEqual(tr.iloc[0].exit_signal_date, str(x.Date.iloc[21].date()))
        self.assertEqual(tr.iloc[0].exit_date, str(x.Date.iloc[22].date()))
        self.assertGreater(float(tr.iloc[0].ret_net), 0.)

    def test_reentry_only_after_research_exit(self):
        x = sample()
        no_re = replay(x, "EXIT_FAST_DEATH_NO_RE", CFG)
        yes_re = replay(x, "FAST_DEATH_RE_EMA10_20", CFG)
        self.assertFalse(no_re["stats"]["active_position_at_end"])
        self.assertTrue(yes_re["stats"]["active_position_at_end"])
        self.assertEqual(
            yes_re["open_position"]["entry_date"], str(x.Date.iloc[28].date()))
        self.assertEqual(len(yes_re["trades"]), 1)

    def test_no_signal_when_spread_is_not_extended(self):
        x = sample()
        rule = replay(x, "FAST_DEATH_RE_EMA5_10_D2_5", CFG)
        self.assertFalse(rule["stats"]["active_position_at_end"] is False)
        self.assertEqual(len(rule["trades"]), 0)

    def test_no_trade_activated_without_prior_10pct_profit(self):
        x = sample()
        x.loc[:, "adj_close"] = 100.
        x.loc[:, "adj_low"] = 98.
        x.loc[:, "adj_high"] = 102.
        y = replay(x, "FAST_DEATH_RE_EMA10_20", CFG)
        self.assertEqual(len(y["trades"]), 0)

    def test_reject_invalid_reentry_or_distance(self):
        x = sample()
        with self.assertRaises(ValueError):
            replay_gated(x, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         trend_spread_reentry="ema10_20_gold")
        with self.assertRaises(ValueError):
            replay_gated(x, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         trend_spread_atr_min=-1)
        with self.assertRaises(ValueError):
            replay_gated(x, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         trend_spread_atr_min=1.5,
                         trend_spread_exit="future_oracle")


if __name__ == "__main__":
    unittest.main()
