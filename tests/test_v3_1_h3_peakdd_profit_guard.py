"""H3 profit protection must be causal, next-open, default-off, and auditable."""
import csv
from pathlib import Path
import sys
import unittest
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_month_ema5_slope_regime import replay_gated
from a2_v3_1_h3_peakdd_profit_guard import ARMS, replay
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS
from a2_v3_1_h3_trade_risk_audit import audit_trade
from qqq_a2.engine import Config
from test_v3_1_regime_hypotheses import frame

DIR = ROOT / "research/results/v3_1_h3_2026_07_13"
CFG = Config(start_date="2024-01-02")


def rising_then_falling():
    s = frame(slope=-3., previous=-2., recovering=True)
    # Original synthetic data: regular gold on day0, enter next open.
    # Day2 closes +20%; day3 falls 9.17% from the known day2 peak.
    s["adj_close"] = [100., 100., 120., 109., 109., 109., 109., 109., 109.]
    s["adj_open"] = [100.] * len(s)
    s["adj_high"] = [100., 100., 122., 112., 110., 110., 110., 110., 110.]
    s["adj_low"] = [100.] * len(s)
    s["death"] = [False] * len(s)
    s["protect_cross"] = [False] * len(s)
    return s


class H3PeakDdTests(unittest.TestCase):
    def test_default_h3_matches_explicitly_disabled_guard(self):
        s = rising_then_falling()
        ref = replay(s, "H3_FROZEN", CFG)
        x = replay_gated(s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         defer_rejected_fast=True,
                         fast_negative_month_mode="accelerating",
                         profit_trail_pct=None)
        pd.testing.assert_frame_equal(ref["trades"], x["trades"])
        pd.testing.assert_frame_equal(ref["equity"], x["equity"])

    def test_confirmed_peak_drawdown_triggers_next_open_not_same_day(self):
        s = rising_then_falling()
        out = replay(s, "TRAIL_8_AFTER_10", CFG)
        tr = out["trades"]
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.iloc[0].entry_date, "2024-01-03")
        self.assertEqual(tr.iloc[0].exit_signal_date, "2024-01-05")
        self.assertEqual(tr.iloc[0].exit_date, "2024-01-08")
        self.assertIn("research profit giveback", tr.iloc[0].exit_reason)
        self.assertEqual(len(replay(s, "TRAIL_12_AFTER_10", CFG)["trades"]), 0)

    def test_intraday_high_not_known_for_profit_activation(self):
        s = rising_then_falling()
        # Only intraday HIGH was above +10%, never a prior closing gain.
        s["adj_close"] = [100.] * len(s)
        s.loc[2, "adj_high"] = 150.
        s.loc[3, "adj_close"] = 96.
        self.assertEqual(len(replay(s, "TRAIL_8_AFTER_10", CFG)["trades"]), 0)

    def test_fast_death_requires_new_death_event(self):
        s = rising_then_falling()
        out = replay(s, "TRAIL_8_FAST_DEATH", CFG)
        self.assertEqual(len(out["trades"]), 0)
        s.loc[3, "protect_cross"] = True
        # N20 regular may independently sell if underwater, but 109 > entry,
        # so this event only qualifies profit protection.
        out = replay(s, "TRAIL_8_FAST_DEATH", CFG)
        self.assertEqual(len(out["trades"]), 1)
        self.assertIn("fast_death", out["trades"].iloc[0].exit_reason)

    def test_bad_guard_validation(self):
        s = rising_then_falling()
        with self.assertRaises(ValueError):
            replay_gated(s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         profit_trail_pct=.08, profit_atr_mult=3.)
        with self.assertRaises(ValueError):
            replay_gated(s, ENTRY_GATE, config=CFG, **V3_PARAMETERS,
                         profit_trail_pct=.08, profit_confirmation="oracle")

    def test_mfe_high_ledger_dates_and_svg(self):
        with (DIR / "h3_trades.csv").open(newline="") as fh:
            trades = list(csv.DictReader(fh))
        with (DIR / "h3_trade_risk_audit.csv").open(newline="") as fh:
            risk = list(csv.DictReader(fh))
        self.assertEqual(len(risk), 59)
        for t, r in zip(trades, risk):
            self.assertEqual(t["entry_date"], r["entry_date"])
            self.assertAlmostEqual(float(r["max_close_floating_gain_pct"]),
                                   round(100*float(t["mfe_close"]), 4),
                                   delta=.0001)
            self.assertGreaterEqual(
                float(r["max_intraday_floating_gain_pct"]),
                float(r["max_close_floating_gain_pct"])-.0001)
            self.assertGreaterEqual(float(r["max_intraday_floating_gain_pct"]), 0.)
            self.assertTrue(r["best_intraday_date"])
        self.assertEqual(risk[-1]["status"], "OPEN")
        root = ET.parse(ROOT / "assets/a2_ema_v3_1_h3_trade_map_2000_2026_07.svg").getroot()
        ns={"s": "http://www.w3.org/2000/svg"}
        strs="".join("".join(z.itertext()) for z in root.findall(".//s:text",ns))
        self.assertIn("MFE high",strs)
        self.assertIn("+92.54%",strs)
        self.assertEqual(len([n for n in root.findall(".//s:rect",ns)
                              if (n.get("class") or "").startswith("h3-ledger-row")]),58)


if __name__ == "__main__":
    unittest.main()
