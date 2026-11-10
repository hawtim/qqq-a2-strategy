"""Synthetic unit tests: no restricted raw market data needed."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

import qqq_a2.engine as engine


def fake_bars(*, count=8, gold=(0,), death=(), fast_death=(),
              close_overrides=None, open_overrides=None):
    closes, opens = [100.0] * count, [100.0] * count
    for day, price in (close_overrides or {}).items():
        closes[day] = price
    for day, price in (open_overrides or {}).items():
        opens[day] = price
    frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=count),
                          "adj_open": opens, "adj_close": closes,
                          "gold": [i in gold for i in range(count)],
                          "death": [i in death for i in range(count)],
                          "protect_cross": [i in fast_death for i in range(count)]})
    for k in (10, 20, 60, 200):
        frame[f"ma{k}"] = 100.0
    return frame


def run_fake(frame, n20_days=20):
    with patch.object(engine, "compute_signals", lambda data: data.copy()):
        return engine.run_backtest(frame, engine.Config(start_date="2024-01-01", n20_days=n20_days))


class EngineTests(unittest.TestCase):
    def test_next_open_and_both_fees(self):
        frame = fake_bars(gold=(0,), death=(4,), open_overrides={1: 110, 5: 120})
        trade = run_fake(frame)["trades"].iloc[0]
        self.assertEqual(trade.entry_signal_date, "2024-01-01")
        self.assertEqual(trade.entry_date, "2024-01-02")
        self.assertEqual(trade.exit_signal_date, "2024-01-05")
        self.assertEqual(trade.exit_date, "2024-01-08")
        self.assertEqual(trade.exit_reason, "MA20/60 death cross")
        self.assertAlmostEqual(trade.ret_net, 120 * 0.9995 / (110 * 1.0005) - 1)

    def test_n20_needs_fresh_cross_and_loss(self):
        frame = fake_bars(count=8, gold=(0,), fast_death=(3,),
                          close_overrides={3: 102, 4: 97})
        # Profitable on crossing day, lower the next day without a new cross.
        result = run_fake(frame)
        self.assertEqual(result["stats"]["completed_trades"], 0)
        self.assertTrue(result["stats"]["active_position_at_end"])

    def test_day20_included_day21_excluded(self):
        frame = fake_bars(count=27, gold=(0,), fast_death=(21,),
                          close_overrides={21: 94}, open_overrides={22: 89})
        result = run_fake(frame)
        self.assertEqual(result["trades"].iloc[0].exit_reason, "N20 protection")
        self.assertEqual(result["trades"].iloc[0].exit_date, str(frame.Date.iloc[22].date()))

        frame = fake_bars(count=28, gold=(0,), fast_death=(22,), close_overrides={22: 94})
        self.assertEqual(run_fake(frame)["stats"]["completed_trades"], 0)

    def test_ma20_60_death_takes_priority(self):
        frame = fake_bars(gold=(0,), fast_death=(3,), death=(3,), close_overrides={3: 90})
        self.assertEqual(run_fake(frame)["trades"].iloc[0].exit_reason, "MA20/60 death cross")

    def test_no_ma10_20_fast_reentry(self):
        frame = fake_bars(count=10, gold=(0,), fast_death=(3,), close_overrides={3: 90})
        result = run_fake(frame)
        self.assertEqual(len(result["trades"]), 1)
        self.assertFalse(result["stats"]["active_position_at_end"])
        self.assertIsNone(result["pending_order"])

    def test_new_gold_reentry_only_on_fresh_cross(self):
        frame = fake_bars(count=12, gold=(0, 7), fast_death=(3,), close_overrides={3: 90})
        result = run_fake(frame)
        self.assertEqual(len(result["trades"]), 1)
        self.assertTrue(result["stats"]["active_position_at_end"])
        self.assertEqual(result["open_position"]["entry_date"], str(frame.Date.iloc[8].date()))

    def test_sma_gold_is_cross_not_status(self):
        closing = [100 - i * 0.3 for i in range(110)] + [67 + i * 0.5 for i in range(110)]
        frame = pd.DataFrame({"Date": pd.bdate_range("2023-01-02", periods=len(closing)),
                              "adj_close": closing, "adj_open": closing})
        data = engine.compute_signals(frame)
        gold_indices = data.index[data.gold].tolist()
        self.assertEqual(len(gold_indices), 1)
        idx = gold_indices[0]
        self.assertGreater(data.loc[idx, "ma20"], data.loc[idx, "ma60"])
        self.assertLessEqual(data.loc[idx - 1, "ma20"], data.loc[idx - 1, "ma60"])
        self.assertFalse(bool(data.loc[idx + 1, "gold"]))

    def test_reject_partial_daily_bar(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "incomplete.csv"
            pd.DataFrame({"Date": ["2024-01-01"], "Open": [100], "High": [105],
                          "Low": [99], "Close": [None], "Adj Close": [None]}).to_csv(file, index=False)
            with self.assertRaisesRegex(ValueError, "Missing OHLC"):
                engine.load_adjusted_csv(file)


if __name__ == "__main__":
    unittest.main()
