"""H1/H2 cross-asset V3.1 research: signals, gate, reversals and controls."""
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_v3_1_regime_research import replay_variant, VARIANTS
from a2_v3_1_cross_asset import replay_v3_1
from a2_v3_selected import replay_selected_v3, ENTRY_GATE, V3_PARAMETERS
from a2_month_ema5_slope_regime import replay_gated
from qqq_a2.engine import Config


def frame(*, slope=-1.0, previous=-2.0, recovering=False,
          slow_bull=True, death=False):
    days = pd.bdate_range("2024-01-02", periods=9)
    monthly_up = [True, True, True, False, False,
                  bool(recovering)] + [bool(recovering)] * 3
    m10 = [12, 12, 8, 12, 12, 12, 12, 12, 12]
    m20 = [10.] * 9
    m60 = [11.] * 9
    if slow_bull:
        m60[5:] = [9.] * 4
    flags = {
        "Date": days,
        "adj_open": [100.] * 9, "adj_close": [100.] * 9,
        "adj_low": [100.] * 9,
        "ma10": m10, "ma20": m20, "ma60": m60,
        "gold": [True] + [False] * 8,
        "death": [False, False, True] + [False] * 6,
        "protect_cross": [False] * 9,
        "ema10_20_gold": [False, False, False, True] + [False] * 5,
        "prev_completed_month_ema5_up": monthly_up,
        "gate_up_slow_veto_1_abs": [True] * 9,
        "known_slope_abs": [slope] * 9,
        "known_prev_abs": [previous] * 9,
    }
    if death:
        flags["protect_cross"][4] = True
        flags["ma10"][4] = 8.
    return pd.DataFrame(flags)


CFG = Config(start_date="2024-01-02")


class RegimeHypothesisTests(unittest.TestCase):
    def test_freeze_originals(self):
        s = frame(recovering=True)
        for name, wrapper in (("V3_FROZEN", replay_selected_v3),
                              ("V3_1_A_FROZEN", replay_v3_1)):
            outcome = replay_variant(s, name, CFG)
            ref = wrapper(s, CFG)
            pd.testing.assert_frame_equal(outcome["trades"], ref["trades"])
            pd.testing.assert_frame_equal(outcome["equity"], ref["equity"])
        self.assertEqual(len(VARIANTS), 6)

    def test_h1_negative_decelerating_entry_next_open(self):
        s = frame(slope=-1., previous=-2., recovering=False)
        baseline = replay_variant(s, "V3_1_A_FROZEN", CFG)
        h1 = replay_variant(s, "H1_DOWN_DECELERATING", CFG)
        control = replay_variant(s, "H1_DOWN_ACCELERATING_CONTROL", CFG)
        self.assertFalse(baseline["stats"]["active_position_at_end"])
        self.assertFalse(control["stats"]["active_position_at_end"])
        self.assertTrue(h1["stats"]["active_position_at_end"])
        self.assertEqual(h1["open_position"]["entry_signal_date"],
                         str(s.Date.iloc[3].date()))
        self.assertEqual(h1["open_position"]["entry_date"],
                         str(s.Date.iloc[4].date()))

    def test_h1_accelerating_control_not_conflated_with_decelerating(self):
        s = frame(slope=-3., previous=-2.)
        positive = replay_variant(s, "H1_DOWN_DECELERATING", CFG)
        negative_control = replay_variant(s, "H1_DOWN_ACCELERATING_CONTROL", CFG)
        self.assertFalse(positive["stats"]["active_position_at_end"])
        self.assertTrue(negative_control["stats"]["active_position_at_end"])

    def test_h2_deferred_only_slow_bull(self):
        s = frame(slope=-1., previous=-2., recovering=True, slow_bull=True)
        bull = replay_variant(s, "H2_DEFERRED_FAST_SLOW_BULL", CFG)
        bear = replay_variant(s, "H2_DEFERRED_FAST_SLOW_BEAR", CFG)
        self.assertTrue(bull["stats"]["active_position_at_end"])
        self.assertFalse(bear["stats"]["active_position_at_end"])
        self.assertEqual(bull["open_position"]["entry_date"],
                         str(s.Date.iloc[6].date()))

    def test_h2_deferred_only_slow_bear(self):
        s = frame(slope=-1., previous=-2., recovering=True, slow_bull=False)
        bull = replay_variant(s, "H2_DEFERRED_FAST_SLOW_BULL", CFG)
        bear = replay_variant(s, "H2_DEFERRED_FAST_SLOW_BEAR", CFG)
        self.assertFalse(bull["stats"]["active_position_at_end"])
        self.assertTrue(bear["stats"]["active_position_at_end"])

    def test_h1_candidate_cannot_reuse_fast_cross_after_death(self):
        s = frame(slope=-3., previous=-2., recovering=False, death=True)
        result = replay_variant(s, "H1_DOWN_DECELERATING", CFG)
        self.assertFalse(result["stats"]["active_position_at_end"])

    def test_negative_month_regime_requires_deferred_opt_in(self):
        with self.assertRaises(ValueError):
            replay_gated(frame(), ENTRY_GATE, config=CFG,
                         **V3_PARAMETERS, fast_negative_month_mode="decelerating")
        with self.assertRaises(ValueError):
            replay_gated(frame(), ENTRY_GATE, config=CFG,
                         **V3_PARAMETERS, deferred_fast_slow_state="bull")
        with self.assertRaises(ValueError):
            replay_gated(frame(), ENTRY_GATE, config=CFG,
                         **V3_PARAMETERS, defer_rejected_fast=True,
                         fast_negative_month_mode="invented")


if __name__ == "__main__":
    unittest.main()
