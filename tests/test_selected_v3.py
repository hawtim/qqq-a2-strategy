"""The selected V3 wrapper must never drift from its frozen rule set."""
import math
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features, replay_gated
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS, VERSION, replay_selected_v3
from qqq_a2.engine import Config


class SelectedV3Tests(unittest.TestCase):
    def test_selected_v3_parameters_are_fixed_and_exclude_stop4(self):
        self.assertEqual(VERSION, "EMA_V3_ADAPTIVE_N20_DAY0_CLOSE3")
        self.assertEqual(ENTRY_GATE, "up_slow_veto_1_abs")
        self.assertEqual(V3_PARAMETERS["stop_loss"], .07)
        self.assertEqual(V3_PARAMETERS["entry_day_close_guard"], .03)
        self.assertEqual(V3_PARAMETERS["regular_n20_extension"],
                         "underwater_2of5_spread_down")
        self.assertEqual(V3_PARAMETERS["extended_n20_days"], 30)
        self.assertEqual(V3_PARAMETERS["n20_day0"], "all")
        self.assertTrue(V3_PARAMETERS["defer_rejected_gold"])

    def test_selected_wrapper_is_identical_to_explicit_frozen_configuration(self):
        dates = pd.bdate_range("2020-01-02", periods=1050)
        prices = [100 + .018 * i + 9 * math.sin(i / 21) for i in range(len(dates))]
        adjusted = pd.DataFrame({
            "Date": dates, "adj_open": prices,
            "adj_close": prices, "adj_low": prices,
        })
        signals = with_month_slope_features(completed_month_signals(adjusted))
        cfg = Config(start_date="2020-01-02")
        selected = replay_selected_v3(signals, config=cfg)
        manual = replay_gated(signals, ENTRY_GATE, config=cfg, **V3_PARAMETERS)
        pd.testing.assert_frame_equal(selected["trades"], manual["trades"])
        pd.testing.assert_frame_equal(selected["equity"], manual["equity"])
        self.assertEqual(selected["stats"], manual["stats"])


if __name__ == "__main__":
    unittest.main()
