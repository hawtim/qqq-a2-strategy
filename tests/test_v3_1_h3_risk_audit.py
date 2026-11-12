"""Guard against conflating realized P/L, intraday MAE and profit giveback."""
from pathlib import Path
from xml.etree import ElementTree as ET
import csv
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_v3_1_h3_trade_risk_audit import audit_trade

DIRECTORY = ROOT / "research/results/v3_1_h3_2026_07_13"
SVG = ROOT / "assets/a2_ema_v3_1_h3_trade_map_2000_2026_07.svg"
NS = {"s": "http://www.w3.org/2000/svg"}


def read_csv(name):
    with (DIRECTORY / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


class H3RiskAuditTests(unittest.TestCase):
    def test_no_exit_day_future_low_and_intraday_mae(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-02", periods=4),
            "adj_open": [100., 102., 98., 90.],
            "adj_low": [96., 94., 91., 5.],  # 5 on SELL day must be ignored
            "adj_close": [101., 105., 93., 20.],
        })
        tr = {
            "id": "1",
            "entry_date": "2024-01-02",
            "exit_date": "2024-01-05",
            "entry_price": "100",
            "exit_price": "90",
            "hold_days": "3",
            "entry_source": "H3_ACCEL",
            "entry_kind": "FAST_EMA10_20_MONTH_EMA5",
            "ret_net": "-0.101",
            "mae_close": "-0.07",
        }
        a = audit_trade(df, tr, cutoff="2024-01-05")
        self.assertAlmostEqual(a["max_close_floating_loss_pct"], -7)
        self.assertAlmostEqual(a["max_intraday_floating_loss_pct"], -9)
        self.assertEqual(a["worst_intraday_date"], "2024-01-04")
        self.assertAlmostEqual(a["max_adverse_including_exit_open_pct"], -10)
        self.assertAlmostEqual(
            a["max_peak_close_to_later_close_dd_pct"],
            round((90. / 105. - 1) * 100, 4),
        )
        self.assertTrue(a["exit_open_was_worse_than_held_intraday_low"])

    def test_mae_close_ledger_reconciles_and_intra_is_never_better(self):
        trades = read_csv("h3_trades.csv")
        risk = read_csv("h3_trade_risk_audit.csv")
        self.assertEqual(len(trades), 58)
        self.assertEqual(len(risk), 59)
        self.assertEqual([int(x["id"]) for x in risk], list(range(1, 60)))
        for t, r in zip(trades, risk):
            self.assertEqual(t["entry_date"], r["entry_date"])
            self.assertEqual(t["exit_date"], r["exit_date"])
            self.assertEqual(t["entry_source"], r["entry_source"])
            expected = min(0., float(t["mae_close"]) * 100)
            self.assertAlmostEqual(float(r["max_close_floating_loss_pct"]),
                                   expected, delta=.0001)
            self.assertLessEqual(
                float(r["max_intraday_floating_loss_pct"]),
                float(r["max_close_floating_loss_pct"]) + .0001)
            self.assertLessEqual(
                float(r["max_adverse_including_exit_open_pct"]),
                float(r["max_intraday_floating_loss_pct"]) + .0001)
            self.assertLessEqual(
                float(r["max_peak_close_to_later_close_dd_pct"]), 0.)
        self.assertEqual(risk[-1]["status"], "OPEN")
        self.assertEqual(risk[-1]["entry_date"], "2026-04-10")
        self.assertAlmostEqual(
            float(risk[-1]["max_intraday_floating_loss_pct"]),
            -.6096, delta=.0001)

    def test_worst_and_subgroup_risks(self):
        risks = read_csv("h3_trade_risk_audit.csv")[:-1]
        self.assertEqual(len([r for r in risks
                              if float(r["max_intraday_floating_loss_pct"]) <= -5]),
                         10)
        h3 = [r for r in risks if r["entry_source"] == "H3_ACCEL"]
        self.assertEqual(len(h3), 21)
        self.assertEqual(len([r for r in h3
                              if float(r["max_intraday_floating_loss_pct"]) <= -5]),
                         4)
        worst = min(risks, key=lambda r: float(r["max_intraday_floating_loss_pct"]))
        self.assertEqual(worst["entry_date"], "2001-04-20")
        self.assertAlmostEqual(float(worst["max_intraday_floating_loss_pct"]),
                               -10.6476, places=4)
        biggest_giveback = min(
            risks, key=lambda r: float(r["max_peak_close_to_later_close_dd_pct"]))
        self.assertEqual(biggest_giveback["entry_date"], "2019-10-02")
        self.assertAlmostEqual(
            float(biggest_giveback["max_peak_close_to_later_close_dd_pct"]),
            -18.3180, places=4)

    def test_svg_ledger_now_shows_both_risk_columns(self):
        root = ET.parse(SVG).getroot()
        body = "".join("".join(n.itertext())
                       for n in root.findall(".//s:text", NS))
        self.assertIn("MAE low", body)
        self.assertIn("Peak DD", body)
        self.assertIn("-10.65%", body)
        self.assertIn("-18.32%", body)
        self.assertEqual(
            len([n for n in root.findall(".//s:rect", NS)
                 if (n.get("class") or "").startswith("h3-ledger-row")]), 58)


if __name__ == "__main__":
    unittest.main()
