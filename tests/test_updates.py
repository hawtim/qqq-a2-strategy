"""Synthetic source-change tests; no copyrighted market bars in repository."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from qqq_a2.updates import extend_adjusted_history


class ForwardUpdateTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.bdate_range("2026-07-01", periods=11)
        self.frozen = pd.DataFrame({
            "Date": self.days[:9], "Close": [100.0] * 9,
            "adj_open": [100.0] * 9, "adj_high": [101.0] * 9,
            "adj_low": [99.0] * 9, "adj_close": [100.0] * 9,
            "Volume": [100_000] * 9,
        })
        self.cutoff, self.end = self.days[8].strftime("%Y-%m-%d"), self.days[10].strftime("%Y-%m-%d")

    def _write_update(self, path: Path, *, unstable: bool = False):
        df = pd.DataFrame({"Date": self.days, "Open": [100.0] * 11,
                           "High": [101.0] * 11, "Low": [99.0] * 11,
                           "Close": [100.0] * 11,
                           "Adj Close": [99.0] * 9 + [100.0] * 2,
                           "Volume": [100_000] * 11})
        if unstable:
            df.loc[6, "Adj Close"] = 95.0
        df.to_csv(path, index=False)

    def test_all_old_ohlc_scaled_and_no_phantom_return(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "latest.csv"
            self._write_update(f)
            merged, meta = extend_adjusted_history(
                self.frozen, str(f), frozen_cutoff=self.cutoff,
                completed_cutoff=self.end, overlap_days=40)
        self.assertEqual(meta["additional_sessions"], 2)
        self.assertAlmostEqual(meta["rebase_ratio"], 0.99)
        self.assertAlmostEqual(float(merged.adj_open.iloc[0]), 99.0)
        self.assertAlmostEqual(float(merged.adj_close.iloc[8]), 99.0)
        self.assertAlmostEqual(float(merged.adj_close.iloc[9]), 100.0)

    def test_bad_adjustment_overlap_rejected(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "bad.csv"
            self._write_update(f, unstable=True)
            with self.assertRaisesRegex(ValueError, "ratio inconsistent"):
                extend_adjusted_history(self.frozen, str(f), frozen_cutoff=self.cutoff,
                                        completed_cutoff=self.end, overlap_days=40)

    def test_missing_last_daily_bar_rejected(self):
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "stale.csv"
            self._write_update(f)
            with self.assertRaisesRegex(ValueError, "does not reach"):
                extend_adjusted_history(self.frozen, str(f), frozen_cutoff=self.cutoff,
                                        completed_cutoff="2026-07-20", overlap_days=40)


if __name__ == "__main__":
    unittest.main()
