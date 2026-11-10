"""Fourth strategy state machine and reproducibility tests.

Fast synthetic fixtures intentionally expose signal dates independently of
future prices. No licensed market-history file is required for these tests.
"""
from pathlib import Path
import math
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/experiments"))

from a2_ema_dynamic_n20 import completed_month_signals, replay_dynamic
from a2_ema_comparison import ema_signals
from a2_ema_hard_stop8 import replay_guarded
from a2_fourway_current import compare_four, NAME, ORDER, changed_trade_audit
from qqq_a2.engine import Config


class DynamicN20Tests(unittest.TestCase):
    def test_no_fast_entries_exact_original_ema_baseline(self):
        dates = pd.bdate_range("2023-01-02", periods=490)
        prices = [100 + .025*i + 10*math.sin(i/17) for i in range(len(dates))]
        data = pd.DataFrame({"Date": dates, "adj_open": [x*1.0002 for x in prices],
                             "adj_close": prices, "adj_low": [x*.99 for x in prices]})
        config = Config(start_date="2023-04-03")
        frozen = replay_guarded(ema_signals(data), config=config,
                                stop_loss=.07, mode="close_next_open")
        dynamic_off = replay_dynamic(completed_month_signals(data), config=config,
                                     fast_reentry=False)
        pd.testing.assert_series_equal(frozen["equity"].Equity,
                                       dynamic_off["equity"].Equity,
                                       check_names=False, rtol=1e-12, atol=1e-8)
        pd.testing.assert_series_equal(frozen["trades"].ret_net,
                                       dynamic_off["trades"].ret_net,
                                       check_names=False, rtol=1e-12, atol=1e-8)

    def test_month_ema5_uses_previous_completed_month(self):
        dates = pd.bdate_range("2023-01-02", "2024-05-23")
        prices = [80 + i*.16 for i in range(len(dates))]
        original = pd.DataFrame({"Date": dates, "adj_open": prices,
                                 "adj_close": prices, "adj_low": prices})
        extreme = original.copy()
        may = extreme.Date.dt.to_period("M") == pd.Period("2024-05")
        extreme.loc[may, "adj_close"] = 1.0
        old = completed_month_signals(original)
        new = completed_month_signals(extreme)
        # The May closing levels differ enormously, but NO May trading day's
        # monthly filter can observe the May final bar.
        self.assertEqual(old.loc[may, "prev_completed_month_ema5_up"].tolist(),
                         new.loc[may, "prev_completed_month_ema5_up"].tolist())
        self.assertTrue(old.loc[may, "prev_completed_month_ema5_up"].all())

    @staticmethod
    def _manual_fixture() -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=43)
        rows = []
        for i, dt in enumerate(dates):
            slow_bull = i in {0,1,2,16} or 17 <= i <= 42
            rows.append({
                "Date": dt, "adj_open": 100.,
                "adj_close": 99. if i == 31 else 100.,
                "ma10": 9. if i == 31 else 11., "ma20": 10.,
                "ma60": 9. if slow_bull else 11.,
                "gold": i in {0,16}, "death": i == 3,
                "protect_cross": i == 31,
                "ema10_20_gold": i == 6,
                "prev_completed_month_ema5_up": True,
            })
        return pd.DataFrame(rows)

    def test_early_entry_resets_guard_on_later_slow_confirmation(self):
        s = self._manual_fixture()
        config = Config(start_date="2024-01-02")
        old = replay_dynamic(s, config=config, reset_n20_at_confirmation=False)
        new = replay_dynamic(s, config=config, reset_n20_at_confirmation=True)
        self.assertEqual(len(old["trades"]), 1)
        self.assertEqual(len(new["trades"]), 2)
        fast = new["trades"].iloc[-1]
        self.assertEqual(fast.entry_kind, "FAST_EMA10_20_MONTH_EMA5")
        self.assertEqual(fast.entry_date, str(s.iloc[7].Date.date()))
        self.assertEqual(fast.trend_confirm_date, str(s.iloc[16].Date.date()))
        self.assertEqual(fast.exit_signal_date, str(s.iloc[31].Date.date()))
        self.assertEqual(fast.exit_date, str(s.iloc[32].Date.date()))
        self.assertEqual(fast.exit_reason, "N20 extended 1d")

    def test_profitable_ema10_20_death_cross_does_not_stop(self):
        s = self._manual_fixture()
        # A fresh 10/20 death cross in a profitable trade does not meet N20;
        # later oscillation while below the entry is not another new cross.
        s.loc[31, "adj_close"] = 101.
        config = Config(start_date="2024-01-02")
        result = replay_dynamic(s, config=config)
        self.assertEqual(len(result["trades"]), 1)
        self.assertTrue(result["stats"]["active_position_at_end"])

    def test_four_comparison_keeps_three_baselines_and_all_dates(self):
        dates = pd.bdate_range("2023-01-02", periods=490)
        prices = [100 + .03*i + 13*math.sin(i/18) for i in range(len(dates))]
        data = pd.DataFrame({"Date": dates, "adj_open": [x*1.001 for x in prices],
                             "adj_close": prices, "adj_low": [x*.985 for x in prices]})
        config = Config(start_date="2023-04-03")
        out = compare_four(data, end=str(dates[-1].date()), config=config)
        self.assertEqual(out["summary"].strategy.tolist(), ORDER)
        self.assertEqual(len(out["equity"]), int((dates >= pd.Timestamp(config.start_date)).sum()))
        self.assertEqual(set(out["periods"].strategy), set(ORDER))
        self.assertEqual(out["runs"][NAME]["stats"]["end_equity"],
                         float(out["summary"].set_index("strategy").loc[NAME,"end_equity"]))

    def test_audit_counts_only_real_executed_exit_date_changes(self):
        before = pd.DataFrame({"entry_date": ["2011-07-01","2012-01-01"],
                               "exit_date": ["2011-08-09","2012-03-01"],
                               "exit_reason": ["7% close stop","EMA20/60 death cross"],
                               "ret_net": [-.1013,.2]})
        after = before.copy()
        after.loc[0, "exit_date"] = "2011-08-04"
        after.loc[0, "ret_net"] = -.0199
        audit = changed_trade_audit(before, after)
        self.assertEqual(audit.entry_date.tolist(), ["2011-07-01"])
        self.assertAlmostEqual(float(audit.iloc[0].saved_loss_percentage_points), 8.14)


if __name__ == "__main__":
    unittest.main()
