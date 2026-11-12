"""Tests that forward-event returns never enter the trade signal itself."""
import unittest
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_v3_1_fast_cross_event_study import (
    classify_monthly_slope, event_outcomes, summarize_events,
)


class FastCrossEventStudyTests(unittest.TestCase):
    def test_completed_month_classification(self):
        self.assertEqual(classify_monthly_slope(-1., -2.),
                         "negative_decelerating")
        self.assertEqual(classify_monthly_slope(-3., -2.),
                         "negative_accelerating")
        self.assertEqual(classify_monthly_slope(2., 3.),
                         "positive_decelerating")

    def test_forward_marks_begin_after_signal_and_use_next_open(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-02", periods=5),
            "adj_open": [99., 100., 102., 103., 104.],
            "adj_close": [100., 100., 103., 104., 105.],
            "ema10_20_gold": [False, True, False, False, False],
            "ma20": [9.] * 5, "ma60": [10.] * 5,
            "known_slope_abs": [-2.] * 5,
            "known_prev_abs": [-3.] * 5,
        })
        outcome = event_outcomes(df, start="2024-01-02", horizons=(2,))
        self.assertEqual(len(outcome), 1)
        r = outcome.iloc[0]
        self.assertEqual(r.completed_month_bucket, "negative_decelerating")
        self.assertEqual(r.entry_date, str(df.Date.iloc[2].date()))
        self.assertAlmostEqual(r.forward_return_open_to_close, 104. / 102. - 1.)
        self.assertAlmostEqual(r.min_close_mark_return, 103. / 102. - 1.)
        summary = summarize_events(outcome)
        self.assertEqual(int(summary.iloc[0]["count"]), 1)

    def test_no_future_bar_means_no_event_label(self):
        df = pd.DataFrame({
            "Date": pd.bdate_range("2024-01-02", periods=4),
            "adj_open": [100.] * 4,
            "adj_close": [100.] * 4,
            "ema10_20_gold": [False, False, True, False],
            "ma20": [9.] * 4, "ma60": [10.] * 4,
            "known_slope_abs": [-1.] * 4,
            "known_prev_abs": [-2.] * 4,
        })
        self.assertTrue(event_outcomes(df, start="2024-01-02",
                                       horizons=(5,)).empty)


if __name__ == "__main__":
    unittest.main()
