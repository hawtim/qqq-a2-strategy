"""Execution-timing tests for the independent N20 price-reclaim experiment."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "experiments"))
from a2_n20_price_reclaim_study import Rule, simulate


def bars():
    dates = pd.bdate_range("2000-01-03", periods=8)
    return pd.DataFrame({
        "Date": dates,
        "adj_open": [10, 10, 10, 9, 9.1, 9.1, 8.8, 8.8],
        "adj_close": [10, 10, 8.9, 9.2, 9.2, 8.8, 8.8, 8.8],
        "gold": [True, False, False, False, False, False, False, False],
        "death": [False] * 8,
        "protect_cross": [False, False, True, False, False, True, False, False],
        "ma10": [11, 11, 9, 9, 9, 9, 9, 9],
        "ma20": [10] * 8,
        "ma60": [9] * 8,
    })


class PriceReclaimTests(unittest.TestCase):
    def test_reclaim_decision_uses_stop_fill_day_close_and_next_open(self):
        t, _, actions, _ = simulate(bars(), Rule("reclaim", "fill_open"))
        self.assertEqual(list(actions.date), ["2000-01-04", "2000-01-06",
                                              "2000-01-07", "2000-01-11"])
        self.assertEqual(list(t.exit_reason), ["N20 protection", "N20 protection"])
        self.assertEqual(t.entry_origin.iloc[1], "PRICE_RECLAIM")
        self.assertEqual(t.reclaim_wait.iloc[1], 1)
        # Once-per-cycle: after reentry's second N20 exit, no third buy.
        self.assertEqual(sum(actions.action == "buy"), 2)

    def test_wait_one_full_close_defers_reentry(self):
        _, _, actions, _ = simulate(bars(), Rule(
            "reclaim_after_wait", "fill_open", min_signal_wait=1))
        # The stop-fill day's close cannot cause a same-evening signal.
        self.assertEqual(actions.date.iloc[2], "2000-01-10")
        self.assertEqual(actions.origin.iloc[2], "PRICE_RECLAIM")

    def test_original_baseline_has_no_reentries(self):
        t, _, actions, _ = simulate(bars(), Rule("original"))
        self.assertEqual(len(t), 1)
        self.assertEqual(sum(actions.action == "buy"), 1)

    def test_second_guard_is_state_based_and_only_applies_to_reentry(self):
        sample = bars()
        sample["ma30"] = [11.0] * len(sample)
        t, _, actions, _ = simulate(sample, Rule(
            "second_guard", "fill_open", second_guard_ma=30))
        self.assertEqual(list(t.exit_reason), [
            "N20 protection", "RECLAIM_MA20_30_BELOW"])
        self.assertEqual(list(t.exit_date), ["2000-01-06", "2000-01-10"])
        self.assertEqual(sum(actions.action == "buy"), 2)

    def test_second_guard_can_exit_when_ma20_breaks_below_ma50_later(self):
        sample = bars()
        sample["ma50"] = [9.0] * len(sample)
        sample.loc[5:, "ma50"] = 11.0
        sample.loc[5, "protect_cross"] = False
        t, _, _, _ = simulate(sample, Rule(
            "second_guard_50", "fill_open", second_guard_ma=50))
        self.assertEqual(list(t.exit_reason), [
            "N20 protection", "RECLAIM_MA20_50_BELOW"])
        self.assertEqual(t.exit_date.iloc[-1], "2000-01-11")

    def test_five_day_reclaim_window_has_inclusive_close_boundary(self):
        sample = bars()
        sample = pd.concat([sample, sample.iloc[-4:].copy()], ignore_index=True)
        sample["Date"] = pd.bdate_range("2000-01-03", periods=len(sample))
        sample.loc[3:, "adj_close"] = 8.5
        sample.loc[8, "adj_close"] = 9.3
        sample.loc[9:, "adj_close"] = 9.3
        sample.loc[5:, "protect_cross"] = False
        _, _, actions_5, _ = simulate(sample, Rule(
            "five_days", "fill_open", wait_days=5))
        _, _, actions_4, _ = simulate(sample, Rule(
            "four_days", "fill_open", wait_days=4))
        self.assertEqual(sum(actions_5.origin == "PRICE_RECLAIM"), 1)
        self.assertEqual(sum(actions_4.origin == "PRICE_RECLAIM"), 0)


if __name__ == "__main__":
    unittest.main()
