"""Audit the committed research-only H3 SVG against its derived trade ledger.

Does not download or redistribute market OHLCV. The plotted weekly backdrop is
visual context; daily-close signals and next-open fills come from the study.
"""
from __future__ import annotations

import csv
from pathlib import Path
from xml.etree import ElementTree as ET
import unittest


ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "assets" / "a2_ema_v3_1_h3_trade_map_2000_2026_07.svg"
TRADES = ROOT / "research" / "results" / "v3_1_h3_2026_07_13" / "h3_trades.csv"
SUMMARY = ROOT / "research" / "results" / "v3_1_h3_2026_07_13" / "summary.csv"
NS = {"s": "http://www.w3.org/2000/svg"}


class H3TradeMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ET.parse(SVG)
        cls.root = cls.tree.getroot()
        with TRADES.open(newline="", encoding="utf-8") as file:
            cls.trades = list(csv.DictReader(file))
        with SUMMARY.open(newline="", encoding="utf-8") as file:
            cls.summary = list(csv.DictReader(file))[0]

    def test_svg_is_complete_and_end_date_is_candid(self):
        self.assertEqual(self.root.attrib["viewBox"], "0 0 1440 3245")
        title = self.root.find("s:title", NS).text
        self.assertIn("QQQ", title)
        self.assertEqual(self.summary["end"], "2026-07-13")
        self.assertEqual(len(self.trades), 58)
        self.assertEqual(
            [int(row["id"]) for row in self.trades], list(range(1, 59)))
        self.assertEqual(int(self.summary["completed_trades"]), 58)
        self.assertEqual(int(self.summary["wins"]), 25)
        self.assertEqual(int(self.summary["losses"]), 33)
        self.assertAlmostEqual(float(self.summary["end_equity"]),
                               1323429.0451020633, places=2)
        self.assertAlmostEqual(float(self.summary["mdd"]),
                               -0.2429991081991566, places=7)

    def test_seven_weekly_panels_and_correct_marker_counts(self):
        groups = self.root.findall(".//s:g", NS)
        panels = [g for g in groups if g.get("class") == "h3-panel"]
        buys = [g for g in groups if
                (g.get("class") or "").split(" ")[0] == "h3-buy"]
        sells = [g for g in groups if
                 (g.get("class") or "").split(" ")[0] == "h3-sell"]
        self.assertEqual(len(panels), 7)
        self.assertEqual(len(buys), 59)  # 58 closed + last open
        self.assertEqual(len(sells), 58)
        self.assertEqual(
            {int(g.get("data-trade")) for g in sells}, set(range(1, 59)))
        self.assertTrue(any(g.get("data-trade") == "59" for g in buys))
        self.assertTrue(all(int(g.get("data-weekly-candles")) >= 130
                            for g in panels))

    def test_h3_entry_coloring_and_ledger_are_consistent(self):
        rectangles = self.root.findall(".//s:rect", NS)
        rows = [r for r in rectangles if
                (r.get("class") or "").startswith("h3-ledger-row")]
        self.assertEqual(len(rows), 58)
        h3_rows = [r for r in rows if "h3-accelerated" in r.get("class")]
        h3_csv = [r for r in self.trades if r["entry_source"] == "H3_ACCEL"]
        self.assertEqual(len(h3_rows), 21)
        self.assertEqual(len(h3_csv), 21)
        self.assertEqual(
            {r.get("data-trade") for r in h3_rows},
            {r["id"] for r in h3_csv})
        self.assertEqual(
            len([r for r in rectangles if r.get("class") == "h3-open-position"]),
            1)

    def test_no_future_trades_or_overlapping_positions(self):
        prev_exit = ""
        for row in self.trades:
            self.assertGreater(row["entry_date"], prev_exit)
            self.assertLessEqual(row["entry_date"], row["exit_date"])
            self.assertLessEqual(row["exit_date"], "2026-07-13")
            prev_exit = row["exit_date"]
        self.assertLess(prev_exit, "2026-04-10")  # open H3 FAST
        self.assertEqual(self.summary["open_position_entry"], "2026-04-10")


if __name__ == "__main__":
    unittest.main()
