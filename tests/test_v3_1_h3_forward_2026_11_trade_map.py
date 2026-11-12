"""Contract tests for H3 Nov 2026 extension; old July result remains frozen.

Derived-only data: no raw daily OHLCV is committed to this repository.
Signals are close-confirmed, fills are next completed session's OPEN.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import unittest

ROOT = Path(__file__).resolve().parents[1]
JUL = ROOT / "research/results/v3_1_h3_2026_07_13"
NOV = ROOT / "research/results/v3_1_h3_2026_11_11"
SVG_CANONICAL = ROOT / "assets/a2_ema_v3_1_h3_trade_map_2000_2026_11.svg"
SVG_LEGACY = ROOT / "assets/a2_ema_v3_1_h3_trade_map_2000_2026_07.svg"
SVG_FROZEN = ROOT / "assets/a2_ema_v3_1_h3_trade_map_2000_2026_07_frozen.svg"
NS = {"s": "http://www.w3.org/2000/svg"}


def csv_rows(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class H3NovemberForwardMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trades = csv_rows(NOV / "h3_trades.csv")
        cls.july = csv_rows(JUL / "h3_trades.csv")
        cls.risk = csv_rows(NOV / "h3_trade_risk.csv")
        cls.events = csv_rows(NOV / "post_july_signal_audit.csv")
        cls.summary = json.loads((NOV / "summary.json").read_text(encoding="utf-8"))
        cls.root = ET.parse(SVG_CANONICAL).getroot()

    def test_no_stale_or_partial_2026_rows(self):
        s = self.summary
        self.assertEqual(s["end"], "2026-11-11")
        self.assertEqual(s["new_sessions"], 86)
        self.assertEqual(s["total_sessions"], 6963)
        self.assertEqual(s["position_status"], "FLAT")
        self.assertEqual(s["pending_order"], "NONE")
        self.assertEqual(s["completed_trades"], 60)
        self.assertEqual((s["wins"], s["losses"]), (26, 34))
        self.assertAlmostEqual(s["ending_equity"], 1257933.89348445, delta=.02)
        self.assertAlmostEqual(s["max_drawdown"], -.2429991081991566, delta=1e-8)
        self.assertEqual(s["adjustment"]["ex_dividend_date"], "2026-09-21")

    def test_frozen_58_trade_schedule_and_net_returns_unchanged(self):
        self.assertEqual(len(self.july), 58)
        self.assertEqual(len(self.trades), 60)
        for prior, current in zip(self.july, self.trades):
            for name in ("id", "entry_signal_date", "entry_date", "exit_signal_date",
                         "exit_date", "entry_kind", "exit_reason"):
                self.assertEqual(current[name], prior[name])
            self.assertAlmostEqual(float(current["ret_net"]), float(prior["ret_net"]),
                                   places=9)
            self.assertAlmostEqual(float(current["entry_price"]) /
                                   float(prior["entry_price"]),
                                   .9989609457128912, delta=.00001)

    def test_causal_august_veto_and_september_deferred_regular_buy(self):
        events = {row["date"]: row for row in self.events}
        self.assertEqual(set(events), {"2026-08-03", "2026-08-06", "2026-09-01"})
        self.assertEqual(events["2026-08-03"]["type"], "EMA20/60_GOLD")
        self.assertEqual(events["2026-08-06"]["type"], "EMA10/20_GOLD")
        self.assertEqual(events["2026-08-03"]["monthGate"], "false")
        self.assertEqual(events["2026-08-06"]["monthGate"], "false")
        self.assertEqual(events["2026-09-01"]["monthGate"], "true")
        self.assertEqual(events["2026-09-01"]["entry_signal"], "true")
        self.assertEqual(events["2026-09-01"]["entry_kind"], "REGULAR_EMA20_60")

        trade59, trade60 = self.trades[-2:]
        self.assertEqual((trade59["entry_date"], trade59["exit_signal_date"],
                          trade59["exit_date"]),
                         ("2026-04-10", "2026-07-31", "2026-08-03"))
        self.assertEqual(trade59["exit_reason"], "MA20/60 death cross")
        self.assertAlmostEqual(float(trade59["ret_net"]), .12507786318,
                               places=9)
        self.assertEqual((trade60["entry_signal_date"], trade60["entry_date"],
                          trade60["exit_signal_date"], trade60["exit_date"]),
                         ("2026-09-01", "2026-09-02", "2026-10-30", "2026-11-02"))
        self.assertEqual(trade60["entry_kind"], "REGULAR_EMA20_60")
        self.assertEqual(trade60["exit_reason"], "MA20/60 death cross")
        self.assertAlmostEqual(float(trade60["ret_net"]), -.01662773141,
                               places=9)

    def test_november_risk_covers_all_and_saves_peak_dd(self):
        self.assertEqual(len(self.risk), 60)
        for trade, risk in zip(self.trades, self.risk):
            self.assertEqual(trade["id"], risk["trade_id"])
            self.assertEqual(trade["entry_date"], risk["entry_date"])
            self.assertEqual(trade["exit_date"], risk["exit_date"])
            self.assertAlmostEqual(
                float(risk["mfe_close_pct"]), float(trade["mfe_close"]) * 100,
                delta=.00001,
            )
            self.assertAlmostEqual(
                float(risk["mae_close_pct"]),
                min(0, float(trade["mae_close"]) * 100),
                delta=.00001,
            )
        self.assertAlmostEqual(float(self.risk[-2]["peak_dd_pct"]),
                               -11.2178042465, delta=.0001)
        self.assertAlmostEqual(float(self.risk[-1]["peak_dd_pct"]),
                               -9.1053892531, delta=.0001)

    def test_svg_updated_at_original_url_plus_canonical_and_archive(self):
        self.assertEqual(SVG_CANONICAL.read_bytes(), SVG_LEGACY.read_bytes())
        self.assertNotEqual(SVG_FROZEN.read_bytes(), SVG_LEGACY.read_bytes())
        self.assertEqual(self.root.attrib["viewBox"], "0 0 1440 3655")
        source = SVG_CANONICAL.read_text(encoding="utf-8")
        for snippet in (
            'data-asof="2026-11-11"',
            "2026-09-01 deferred REGULAR entry accepted",
            "2026-08-03 EMA20/60 golden cross rejected",
            "2026-08-06 EMA10/20 golden cross rejected",
            "Status: FLAT as of 2026-11-11",
        ):
            self.assertIn(snippet, source)
        all_groups = self.root.findall(".//s:g", NS)
        panels = [g for g in all_groups if g.attrib.get("class") == "h3-panel"]
        buy = [g for g in all_groups if
               (g.attrib.get("class") or "").split(" ")[0] == "h3-buy"]
        sell = [g for g in all_groups if
                (g.attrib.get("class") or "").split(" ")[0] == "h3-sell"]
        self.assertEqual(len(panels), 7)
        self.assertEqual(len(buy), 60)
        self.assertEqual(len(sell), 60)
        self.assertEqual(len([g for g in all_groups if
                              g.attrib.get("class") == "h3-daily-zoom"]), 1)
        rects = self.root.findall(".//s:rect", NS)
        self.assertEqual(len([r for r in rects if
                              "h3-ledger-row" in (r.get("class") or "")]), 60)
        self.assertEqual(len([r for r in rects if
                              "h3-post-july" in (r.get("class") or "")]), 2)


if __name__ == "__main__":
    unittest.main()
