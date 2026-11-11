"""Regression tests for experimental month EMA slope entry gates."""
import math
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals, replay_dynamic
from a2_month_ema5_slope_regime import with_month_slope_features, replay_gated
from qqq_a2.engine import Config


class MonthSlopeGateTests(unittest.TestCase):
    def test_baseline_gate_is_exact_fourth_strategy(self):
        dates = pd.bdate_range("2020-01-02", periods=1100)
        prices = [100 + .014*i + 9 * math.sin(i / 23) for i in range(len(dates))]
        data = pd.DataFrame({"Date": dates, "adj_open": prices,
                             "adj_close": prices, "adj_low": prices})
        s = with_month_slope_features(completed_month_signals(data))
        cfg = Config(start_date="2020-01-02")
        expected = replay_dynamic(s, config=cfg)
        actual = replay_gated(s, "baseline", config=cfg)
        pd.testing.assert_frame_equal(expected["trades"], actual["trades"])
        pd.testing.assert_frame_equal(expected["equity"], actual["equity"])

    def test_completed_month_signals_ignore_mutation_within_current_month(self):
        dates = pd.bdate_range("2023-01-02", "2024-05-23")
        prices = [90 + i * .03 + 2 * math.sin(i / 10) for i in range(len(dates))]
        original = pd.DataFrame({"Date": dates, "adj_open": prices,
                                 "adj_close": prices, "adj_low": prices})
        changed = original.copy()
        current = original.Date.dt.to_period("M") == pd.Period("2024-05")
        changed.loc[current, "adj_close"] = 3000.
        a = with_month_slope_features(completed_month_signals(original))
        b = with_month_slope_features(completed_month_signals(changed))
        for col in ("known_month_ema5", "known_slope_abs", "known_slope_pct",
                    "gate_down_relief_2_abs", "gate_up_slow_veto_1_abs"):
            pd.testing.assert_series_equal(a.loc[current, col], b.loc[current, col])
        self.assertFalse(a.loc[current, "mtd_slope_abs"].equals(
            b.loc[current, "mtd_slope_abs"]))

    def test_deferred_gold_enters_next_open_after_gate_recovers(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        s = pd.DataFrame({
            "Date": dates, "adj_open": [100.] * 5,
            "adj_close": [100.] * 5,
            "ma10": [12.] * 5, "ma20": [11.] * 5, "ma60": [10.] * 5,
            "gold": [True, False, False, False, False],
            "death": [False] * 5, "protect_cross": [False] * 5,
            "ema10_20_gold": [False] * 5,
            "prev_completed_month_ema5_up": [False] * 5,
            "gate_up_only": [False, True, True, True, True],
        })
        cfg = Config(start_date="2024-01-02")
        missed = replay_gated(s, "up_only", config=cfg)
        wait = replay_gated(s, "up_only", config=cfg, defer_rejected_gold=True)
        self.assertFalse(missed["stats"]["active_position_at_end"])
        self.assertTrue(wait["stats"]["active_position_at_end"])
        self.assertEqual(wait["open_position"]["entry_signal_date"], "2024-01-03")
        self.assertEqual(wait["open_position"]["entry_date"], "2024-01-04")

    def test_when_gate_rejects_new_gold_in_position_confirmation_still_occurs(self):
        dates = pd.bdate_range("2024-01-02", periods=43)
        rows = []
        for i, dt in enumerate(dates):
            bull = i in {0, 1, 2, 16} or i >= 17
            rows.append({
                "Date": dt, "adj_open": 100., "adj_close": 99. if i == 31 else 100.,
                "ma10": 9. if i == 31 else 11., "ma20": 10.,
                "ma60": 9. if bull else 11.,
                "gold": i in {0, 16}, "death": i == 3,
                "protect_cross": i == 31, "ema10_20_gold": i == 6,
                "prev_completed_month_ema5_up": True,
                "gate_up_only": i != 16,
            })
        s = pd.DataFrame(rows)
        cfg = Config(start_date="2024-01-02")
        original = replay_dynamic(s, config=cfg)
        gated = replay_gated(s, "up_only", config=cfg)
        pd.testing.assert_frame_equal(original["trades"], gated["trades"])
        self.assertEqual(gated["trades"].iloc[-1].trend_confirm_date,
                         str(dates[16].date()))

    def test_same_close_stop_is_only_an_oracle_and_skips_next_open_gap(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        s = pd.DataFrame({
            "Date": dates,
            "adj_open": [100., 100., 100., 82., 82.],
            "adj_close": [100., 100., 90., 80., 80.],
            "ma10": [12.] * 5, "ma20": [11.] * 5,
            "ma60": [10.] * 5,
            "gold": [True, False, False, False, False],
            "death": [False] * 5, "protect_cross": [False] * 5,
            "ema10_20_gold": [False] * 5,
            "prev_completed_month_ema5_up": [False] * 5,
            "gate_baseline": [True] * 5,
        })
        cfg = Config(start_date="2024-01-02")
        nxt = replay_gated(s, "baseline", config=cfg)
        close = replay_gated(s, "baseline", config=cfg,
                             hard_stop_fill="same_close_oracle")
        slipped = replay_gated(s, "baseline", config=cfg,
                               hard_stop_fill="same_close_oracle",
                               close_fill_adverse_slippage=.01)
        self.assertEqual(nxt["trades"].iloc[0].exit_date, str(dates[3].date()))
        self.assertEqual(close["trades"].iloc[0].exit_date, str(dates[2].date()))
        self.assertAlmostEqual(float(nxt["trades"].iloc[0].exit_price), 82.)
        self.assertAlmostEqual(float(close["trades"].iloc[0].exit_price), 90.)
        self.assertAlmostEqual(float(slipped["trades"].iloc[0].exit_price), 89.1)

    def test_n20_day0_regular_only_exits_next_open_on_losing_fresh_death(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        s = pd.DataFrame({
            "Date": dates, "adj_open": [100., 100., 98., 98., 98.],
            "adj_close": [100., 99., 98., 98., 98.],
            "ma10": [12., 9., 9., 9., 9.], "ma20": [11., 10., 10., 10., 10.],
            "ma60": [10., 9., 9., 9., 9.],
            "gold": [True, False, False, False, False],
            "death": [False] * 5,
            "protect_cross": [False, True, False, False, False],
            "ema10_20_gold": [False] * 5,
            "prev_completed_month_ema5_up": [False] * 5,
            "gate_baseline": [True] * 5,
        })
        cfg = Config(start_date="2024-01-02")
        original = replay_gated(s, "baseline", config=cfg)
        fast_only = replay_gated(s, "baseline", config=cfg, n20_day0="fast")
        regular = replay_gated(s, "baseline", config=cfg, n20_day0="regular")
        both = replay_gated(s, "baseline", config=cfg, n20_day0="all")
        self.assertEqual(original["stats"]["completed_trades"], 0)
        self.assertEqual(fast_only["stats"]["completed_trades"], 0)
        for run in (regular, both):
            self.assertEqual(run["stats"]["completed_trades"], 1)
            trade = run["trades"].iloc[0]
            self.assertEqual(trade.exit_signal_date, str(dates[1].date()))
            self.assertEqual(trade.exit_date, str(dates[2].date()))
            self.assertEqual(trade.exit_reason, "N20 protection")

    def test_n20_day0_fast_exits_next_open_not_buy_open(self):
        dates = pd.bdate_range("2024-01-02", periods=9)
        s = pd.DataFrame({
            "Date": dates,
            "adj_open": [100., 100., 100., 100., 100., 100., 98., 98., 98.],
            "adj_close": [100., 100., 100., 100., 100., 99., 98., 98., 98.],
            "ma10": [12., 12., 12., 12., 12., 9., 9., 9., 9.],
            "ma20": [11., 11., 11., 11., 11., 10., 10., 10., 10.],
            "ma60": [10.] * 9,
            "gold": [True] + [False] * 8,
            "death": [False, False, True] + [False] * 6,
            "protect_cross": [False] * 5 + [True] + [False] * 3,
            "ema10_20_gold": [False] * 4 + [True] + [False] * 4,
            "prev_completed_month_ema5_up": [True] * 9,
            "gate_baseline": [True] * 9,
        })
        cfg = Config(start_date="2024-01-02")
        original = replay_gated(s, "baseline", config=cfg)
        regular_only = replay_gated(s, "baseline", config=cfg,
                                    n20_day0="regular")
        fast_only = replay_gated(s, "baseline", config=cfg, n20_day0="fast")
        both = replay_gated(s, "baseline", config=cfg, n20_day0="all")
        pd.testing.assert_frame_equal(original["trades"], regular_only["trades"])
        for run in (fast_only, both):
            trade = run["trades"].iloc[-1]
            self.assertEqual(trade.entry_kind, "FAST_EMA10_20_MONTH_EMA5")
            self.assertEqual(trade.entry_date, str(dates[5].date()))
            self.assertEqual(trade.exit_signal_date, str(dates[5].date()))
            self.assertEqual(trade.exit_date, str(dates[6].date()))
            self.assertEqual(trade.exit_reason, "N20 extended 1d")

    def test_n20_day0_does_not_exit_on_same_day_profit(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        s = pd.DataFrame({
            "Date": dates, "adj_open": [100.] * 4,
            "adj_close": [100., 101., 101., 101.],
            "ma10": [12.] * 4, "ma20": [11.] * 4, "ma60": [10.] * 4,
            "gold": [True, False, False, False],
            "death": [False] * 4,
            "protect_cross": [False, True, False, False],
            "ema10_20_gold": [False] * 4,
            "prev_completed_month_ema5_up": [False] * 4,
            "gate_baseline": [True] * 4,
        })
        cfg = Config(start_date="2024-01-02")
        original = replay_gated(s, "baseline", config=cfg)
        all_day0 = replay_gated(s, "baseline", config=cfg, n20_day0="all")
        pd.testing.assert_frame_equal(original["trades"], all_day0["trades"])
        pd.testing.assert_frame_equal(original["equity"], all_day0["equity"])

    def test_invalid_day0_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_gated(pd.DataFrame(), "baseline", n20_day0="next_week")


if __name__ == "__main__":
    unittest.main()
