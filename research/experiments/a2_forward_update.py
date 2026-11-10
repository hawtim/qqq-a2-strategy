"""Replay frozen A2 against newer *user-supplied* daily QQQ observations.

No remote downloads and no re-optimization. This script writes ONLY derived
evidence to research/results and a merged, vendor-controlled price CSV to
sources/ (the latter must stay gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qqq_a2.engine import Config, combine_rebased_sources, compute_signals, run_backtest
from qqq_a2.updates import adjusted_prices_to_local_csv, extend_adjusted_history


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 forward OHLC extension, audited against frozen data")
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--update", required=True, help="New daily OHLCV + Adj Close; local only")
    parser.add_argument("--asof", default="2026-11-09", help="Last completed US session")
    parser.add_argument("--frozen", default="2026-07-13")
    parser.add_argument("--outdir", type=Path, default=Path("research/results"))
    parser.add_argument("--merged", type=Path, default=Path("sources/qqq_merged_updated_local.csv"))
    args = parser.parse_args()
    frozen = combine_rebased_sources(args.old, args.new, end_date=args.frozen)
    updated, info = extend_adjusted_history(frozen, args.update, frozen_cutoff=args.frozen,
                                            completed_cutoff=args.asof)
    # Make sure ONLY the data horizon moves: no historical executed trade or
    # daily account equity may silently change before the original cutoff.
    before = run_backtest(frozen, Config())
    after = run_backtest(updated, Config())
    old_trades = before["trades"].reset_index(drop=True)
    prefix = after["trades"].iloc[:len(old_trades)].reset_index(drop=True)
    for col in ("entry_signal_date", "entry_date", "exit_signal_date", "exit_date",
                "exit_reason", "hold_days"):
        if not old_trades[col].equals(prefix[col]):
            raise AssertionError(f"Frozen trade events changed in column {col}")
    for col in ("ret_net", "mfe_close", "mae_close", "equity_after_trade"):
        np.testing.assert_allclose(old_trades[col], prefix[col], rtol=1e-11, atol=1e-7)
    for col in ("entry_price", "exit_price"):
        np.testing.assert_allclose(old_trades[col] * info["rebase_ratio"],
                                   prefix[col], rtol=1e-11, atol=1e-7)
    earlier_equity = after["equity"].iloc[:len(before["equity"])].Equity
    np.testing.assert_allclose(before["equity"].Equity, earlier_equity, rtol=1e-11, atol=1e-6)

    # Explicit simple coverage check: weekdays missing from the new source
    # must be explained by an exchange holiday before claiming completeness.
    f, t = pd.Timestamp(args.frozen), pd.Timestamp(args.asof)
    observed = set(updated.loc[updated.Date > f, "Date"])
    excluded = set(pd.to_datetime(["2026-09-07"]))  # Labor Day; NYSE/Nasdaq closed.
    missing = sorted(set(pd.bdate_range(f + pd.Timedelta(days=1), t)) - observed - excluded)
    if missing:
        raise ValueError("Unexplained weekdays in updated source: " + ", ".join(map(str, missing)))

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.merged.parent.mkdir(parents=True, exist_ok=True)
    adjusted_prices_to_local_csv(updated).to_csv(args.merged, index=False)

    trades = after["trades"]
    trades.to_csv(args.outdir / "A2_current_trades_to_2026_11_09.csv", index=False)
    newest_trades = trades[pd.to_datetime(trades.exit_date) > f].copy()
    newest_trades.to_csv(args.outdir / "A2_newly_completed_trades_2026_07_14_to_11_09.csv", index=False)
    e = after["equity"]
    e.loc[pd.to_datetime(e.Date) > f].to_csv(
        args.outdir / "A2_forward_daily_equity_2026_07_14_to_11_09.csv", index=False)
    signals = compute_signals(updated)
    candidates = signals[signals.Date > f].copy()
    candidates["actual_pending"] = e.loc[pd.to_datetime(e.Date) > f, "Pending"].to_numpy()
    candidates["invested_at_close"] = e.loc[pd.to_datetime(e.Date) > f, "Invested"].to_numpy()
    events = candidates.loc[candidates.gold | candidates.death | candidates.protect_cross |
                            (candidates.actual_pending != ""),
                            ["Date", "gold", "death", "protect_cross", "actual_pending",
                             "invested_at_close", "ma10", "ma20", "ma60", "ma200"]]
    events.to_csv(args.outdir / "A2_forward_signal_events_2026_07_14_to_11_09.csv", index=False)

    last_close = float(updated.adj_close.iloc[-1])
    last_raw_close = float(updated.Close.iloc[-1])
    last_old_day = e[pd.to_datetime(e.Date) == f].iloc[-1]
    def stable_summary(result: dict) -> dict:
        return {"stats": result["stats"], "last_completed_date": result["latest_date"],
                "open_position": result["open_position"],
                "pending_order_at_close": result["pending_order"],
                "last_indicators": result["latest_indicators"]}

    report = {
        "purpose": "A2-N20 unchanged rule, new daily observations only",
        "frozen": stable_summary(before),
        "current": stable_summary(after),
        "source_update": {**info, "source_sha256": digest(args.update),
                          "vendor": "StockQuote.io (Yahoo-style adjusted OHLCV, crosschecked with StockAnalysis/S&P Global)",
                          "as_of_us_eastern": args.asof,
                          "new_source_bars_have_no_unexplained_weekday_gaps": True,
                          "prior_trades_and_equity_exactly_preserved": True},
        "forward_interval": {
            "first_session": str(updated.loc[updated.Date > f, "Date"].iloc[0].date()),
            "final_session": str(t.date()),
            "added_session_count": info["additional_sessions"],
            "completed_trades_after_frozen_cutoff": len(newest_trades),
            "completed_trade_ids": newest_trades.id.tolist(),
            "equity_at_frozen_cutoff": float(last_old_day.Equity),
            "equity_at_current_cutoff": float(e.Equity.iloc[-1]),
            "equity_change_pct": float(e.Equity.iloc[-1] / last_old_day.Equity - 1),
            "unadjusted_latest_qqq_close": last_raw_close,
            "adjusted_latest_qqq_close": last_close,
        }
    }
    dst = args.outdir / "A2_forward_update_to_2026_11_09.json"
    dst.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
