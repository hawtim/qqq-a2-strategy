"""Three-way strategy release checks, independent of licensed raw QQQ bars."""
from pathlib import Path
import math
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_threeway_current import compare
from qqq_a2.engine import Config


class ThreeWayTests(unittest.TestCase):
    def test_common_dates_fees_and_trade_only_metrics(self):
        dates = pd.bdate_range("2023-01-02", periods=360)
        c = [100 + i * .04 + 12 * math.sin(i / 15) for i in range(len(dates))]
        data = pd.DataFrame({
            "Date": dates, "adj_open": [x * 1.0004 for x in c],
            "adj_close": c, "adj_low": [x * .985 for x in c],
        })
        config = Config(start_date="2023-04-03")
        result = compare(data, end=str(dates[-1].date()), config=config)
        summary = result["summary"].set_index("strategy")
        curve = result["equity"]
        self.assertEqual(len(curve), int((dates >= pd.Timestamp(config.start_date)).sum()))
        self.assertEqual(summary.index.to_list(),
                         ["SMA_A2_N20", "EMA_A2_N20_close7", "QQQ_buy_hold"])
        self.assertGreater(int(summary.loc["SMA_A2_N20", "completed_trades"]), 0)
        self.assertGreater(int(summary.loc["EMA_A2_N20_close7", "completed_trades"]), 0)
        self.assertTrue(pd.isna(summary.loc["QQQ_buy_hold", "completed_trades"]))
        self.assertEqual(float(summary.loc["QQQ_buy_hold", "market_exposure"]), 1.0)
        self.assertTrue(curve.Date.is_unique)
        self.assertEqual(set(result["periods"].strategy), set(summary.index))
        self.assertFalse(result["periods"].period_return.isna().any())
        for strategy in summary.index:
            marked = curve[f"Equity_{strategy}"]
            self.assertAlmostEqual(float(marked.iloc[-1]),
                                   float(summary.loc[strategy, "end_equity"]), places=6)

    def test_missing_common_start_is_not_silently_shifted(self):
        dates = pd.bdate_range("2024-01-03", periods=80)
        data = pd.DataFrame({"Date": dates,
                             "adj_open": [100.] * len(dates),
                             "adj_close": [100.] * len(dates),
                             "adj_low": [99.] * len(dates)})
        with self.assertRaisesRegex(ValueError, "Missing expected first"):
            compare(data, end=str(dates[-1].date()),
                    config=Config(start_date="2024-01-02"))


if __name__ == "__main__":
    unittest.main()
