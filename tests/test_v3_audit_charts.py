"""Regression tests for the fixed-cutoff, trade-ledger-driven V3 charts."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from scripts.render_v3_audit_charts import (
    ROOT, _read_csv, _svg_geometry, _validate_trades,
    render_trade_map, render_equity_chart,
)


FOLDER = ROOT / "research/results/a2_v3_selected_2026_11_09"
NS = {"s": "http://www.w3.org/2000/svg"}


class SelectedV3ChartTests(unittest.TestCase):
    def setUp(self):
        self.trades = FOLDER / "selected_v3_trades.csv"
        self.equity = FOLDER / "selected_v3_daily_equity.csv"
        self.template = ROOT / "assets/a2_ema_dynamic_n20_trade_map_2000_2026.svg"
        self.stats = next(r for r in _read_csv(FOLDER / "summary.csv")
                          if r["version"] == "EMA_V3_ADAPTIVE_N20_DAY0_CLOSE3")

    def test_seven_weekly_panels_use_published_vector_only(self):
        panels = _svg_geometry(self.template)
        self.assertEqual([p.year for p in panels], [2000,2004,2008,2012,2016,2020,2024])
        self.assertGreater(sum(p.candle_count for p in panels), 1_300)
        for p in panels:
            self.assertLess(p.x(f"{p.year}-01-03"), p.x(f"{p.year}-12-31"))
            self.assertLess(p.y(100), p.y(50))

    def test_frozen_v3_trade_map_exact_execution_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "v3.svg"
            render_trade_map(self.template, self.trades, dest, self.stats)
            root = ET.parse(dest).getroot()
            self.assertEqual(len(root.findall(".//s:g[@class='v3-panel']", NS)), 7)
            self.assertEqual(len(root.findall(".//s:g[@class='v3-buy']", NS)), 52)
            self.assertEqual(len(root.findall(".//s:g[@class='v3-sell']", NS)), 52)
            self.assertEqual(len(root.findall(".//s:rect[@class='v3-ledger-row']", NS)), 52)
            content = dest.read_text(encoding="utf-8")
            self.assertIn("N20 to day30", content)
            self.assertIn("Day0 -3%", content)
            self.assertIn("2011-11-21", content)
            self.assertIn("2000-06-23", content)
            self.assertNotIn("59 closed trades", content)
            self.assertNotIn("Dynamic EMA N20 | Seven weekly", content)

    def test_selected_v3_only_not_v2_or_4pct_negative_control(self):
        rows = _read_csv(self.trades)
        self.assertEqual(len(rows), 52)
        _validate_trades(rows)
        self.assertEqual(rows[0]["exit_reason"], "entry-day close <= -3%")
        self.assertEqual(rows[21]["exit_reason"], "N20 conditional extension")
        self.assertAlmostEqual(float(rows[-1]["equity_after_trade"]),
                               float(self.stats["end_equity"]), places=4)
        self.assertEqual({r["entry_kind"] for r in rows},
                         {"REGULAR_EMA20_60", "FAST_EMA10_20_MONTH_EMA5"})

    def test_equity_curve_end_and_mdd_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "equity.svg"
            render_equity_chart(self.equity, dest, self.stats)
            root = ET.parse(dest).getroot()
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            content = dest.read_text(encoding="utf-8")
            self.assertIn("$878,933", content)
            self.assertIn("-19.68%", content)
            self.assertIn("2000-06-23", content)
            self.assertIn("2011-11-21", content)


if __name__ == "__main__":
    unittest.main()
