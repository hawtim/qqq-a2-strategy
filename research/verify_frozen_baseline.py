"""Compare a self-supplied historical data pair against the frozen trade ledger.

Usage:
  python research/verify_frozen_baseline.py \\
    --old sources/qqq_jtang.csv --new sources/qqq_james.csv

Requires: pip install -e .
This script never downloads or writes raw market data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qqq_a2.engine import Config, combine_rebased_sources, run_backtest


EXPECTED_STATS = {
    "completed_trades": 58,
    "wins": 24,
    "losses": 34,
    "win_rate": 24 / 58,
    "payoff_ratio": 4.0168235353073145,
    "profit_factor": 2.8354048484522214,
    "mdd": -0.2028819139814283,
    "end_equity": 450595.8505036286,
    "cagr": 0.058397033165,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the strict historical A2 N20 ledger")
    parser.add_argument("--old", required=True, help="Old historical CSV; locally obtained")
    parser.add_argument("--new", required=True, help="Newer overlapping CSV; locally obtained")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    archived = pd.read_csv(root / "research/results/A2_strict_58_trades.csv")
    data = combine_rebased_sources(args.old, args.new, end_date="2026-07-13")
    result = run_backtest(data, Config())
    actual = result["trades"]
    assert len(archived) == len(actual) == 58, (len(archived), len(actual))

    date_and_reason = ("entry_date", "exit_signal_date", "exit_date", "exit_reason")
    for field in date_and_reason:
        mismatch = archived[field].astype(str) != actual[field].astype(str)
        assert not mismatch.any(), f"{field}: mismatch in trades {mismatch[mismatch].index.tolist()}"

    for field in ("entry_price", "exit_price", "ret_net", "mfe_close", "mae_close"):
        diff = (archived[field] - actual[field]).abs()
        assert float(diff.max()) < 1e-7, f"{field}: maximum absolute difference {diff.max()}"

    for field, expected in EXPECTED_STATS.items():
        value = result["stats"][field]
        assert abs(value - expected) < 1e-6, f"{field}: expected {expected}, observed {value}"
    assert result["stats"]["active_position_at_end"]
    print("PASS: 58 historical trades, 24 wins / 34 losses, prices, reasons, and A2 metrics match frozen ledger")


if __name__ == "__main__":
    main()
