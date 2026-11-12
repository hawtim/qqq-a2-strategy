"""Regression tests for research-only V3.1-A deferred FAST qualification."""
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_v3_1_cross_asset import VERSION, HYPOTHESIS, replay_v3_1
from a2_v3_selected import replay_selected_v3, ENTRY_GATE, V3_PARAMETERS
from a2_month_ema5_slope_regime import replay_gated
from qqq_a2.engine import Config


class DeferredFastCrossAssetTests(unittest.TestCase):
    @staticmethod
    def signals(blocked="gate", cancel=False):
        dates = pd.bdate_range("2024-01-02", periods=8)
        # A regular buy, then slow death and an actual sale; only AFTER
        # the sale does EMA10/20 cross up. Its monthly condition is blocked.
        ma10 = [12, 12, 8, 10, 8 if cancel else 10, 10, 10, 10]
        ma20 = [11, 11, 9, 9, 9, 9, 9, 9]
        ma60 = [10] * 8
        gates = ([True, True, True, False, False, True, True, True]
                 if blocked == "gate" else [True] * 8)
        monthly = ([True] * 8 if blocked == "gate"
                   else [True, True, True, False, False, True, True, True])
        return pd.DataFrame({
            "Date": dates, "adj_open": [100.] * 8,
            "adj_close": [100.] * 8,
            "adj_low": [100.] * 8,
            "ma10": [float(x) for x in ma10],
            "ma20": [float(x) for x in ma20],
            "ma60": [float(x) for x in ma60],
            "gold": [True] + [False] * 7,
            "death": [False, False, True] + [False] * 5,
            "protect_cross": ([False, False, False, False, True,
                               False, False, False] if cancel else
                              [False] * 8),
            "ema10_20_gold": [False, False, False, True] + [False] * 4,
            "prev_completed_month_ema5_up": monthly,
            "gate_up_slow_veto_1_abs": gates,
        })

    def test_rules_are_isolated_from_selected_v3(self):
        self.assertEqual(VERSION, "EMA_V3_1_DEFERRED_FAST_CROSS_ASSET")
        self.assertIn("EMA10>EMA20", HYPOTHESIS)
        self.assertEqual(V3_PARAMETERS["entry_day_close_guard"], .03)
        self.assertEqual(V3_PARAMETERS["stop_loss"], .07)
        s = self.signals(blocked="gate")
        cfg = Config(start_date="2024-01-02")
        chosen = replay_selected_v3(s, cfg)
        manual = replay_gated(s, ENTRY_GATE, config=cfg,
                              **V3_PARAMETERS, defer_rejected_fast=False)
        pd.testing.assert_frame_equal(chosen["trades"], manual["trades"])
        pd.testing.assert_frame_equal(chosen["equity"], manual["equity"])

    def test_gate_recovery_buys_from_deferred_fast_next_open(self):
        s = self.signals(blocked="gate")
        cfg = Config(start_date="2024-01-02")
        base = replay_selected_v3(s, cfg)
        updated = replay_v3_1(s, cfg)
        self.assertFalse(base["stats"]["active_position_at_end"])
        self.assertTrue(updated["stats"]["active_position_at_end"])
        self.assertEqual(updated["open_position"]["entry_kind"],
                         "FAST_EMA10_20_MONTH_EMA5")
        self.assertEqual(updated["open_position"]["entry_signal_date"],
                         str(s.Date.iloc[5].date()))
        self.assertEqual(updated["open_position"]["entry_date"],
                         str(s.Date.iloc[6].date()))

    def test_monthly_ema5_up_recovery_also_allows_deferred_entry(self):
        s = self.signals(blocked="monthly_down")
        cfg = Config(start_date="2024-01-02")
        base = replay_selected_v3(s, cfg)
        updated = replay_v3_1(s, cfg)
        self.assertFalse(base["stats"]["active_position_at_end"])
        self.assertEqual(updated["open_position"]["entry_date"],
                         str(s.Date.iloc[6].date()))

    def test_fast_death_invalidates_deferred_qualification(self):
        s = self.signals(blocked="gate", cancel=True)
        cfg = Config(start_date="2024-01-02")
        updated = replay_v3_1(s, cfg)
        self.assertFalse(updated["stats"]["active_position_at_end"])
        self.assertEqual(updated["stats"]["completed_trades"], 1)

    def test_pre_exit_fast_cross_cannot_be_reused(self):
        s = self.signals(blocked="gate")
        s["ema10_20_gold"] = [False, False, True] + [False] * 5
        s["adj_close"] = [100., 100., 100., 89., 89., 89., 89., 89.]
        s["adj_open"] = [100., 100., 100., 100., 89., 89., 89., 89.]
        s["gold"] = [True] + [False] * 7
        s["death"] = [False] * 8
        s["gate_up_slow_veto_1_abs"] = [True] * 8
        s["prev_completed_month_ema5_up"] = [True] * 8
        s["ma10"] = [12., 12., 12., 12., 12., 12., 12., 12.]
        s["ma20"] = [11.] * 8
        cfg = Config(start_date="2024-01-02")
        updated = replay_v3_1(s, cfg)
        self.assertEqual(updated["trades"].iloc[0].exit_date,
                         str(s.Date.iloc[4].date()))
        self.assertFalse(updated["stats"]["active_position_at_end"])

    def test_no_deferred_cross_never_buys_on_level_alone(self):
        s = self.signals(blocked="gate")
        s["ema10_20_gold"] = [False] * 8
        cfg = Config(start_date="2024-01-02")
        base = replay_selected_v3(s, cfg)
        updated = replay_v3_1(s, cfg)
        pd.testing.assert_frame_equal(base["equity"], updated["equity"])


if __name__ == "__main__":
    unittest.main()
