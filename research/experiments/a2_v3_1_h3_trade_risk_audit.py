"""Rebuild H3 per-trade maximum floating loss from *locally supplied* QQQ bars.

The script never exports OHLCV. No unobservable intraday order is inferred:
* entry-anchored MAE low = minimum adjusted daily LOW while shares were held;
* closing MAE = minimum adjusted CLOSE while shares were held;
* peak-to-close giveback = maximal drawdown from earlier closing peaks, including
  the exit-day OPEN (a realized terminal mark), never the exit-day LOW.
The BUY is a next-session opening fill; after an exit-day opening sale, prices
later that day must NOT be attributed to the completed trade.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from qqq_a2.engine import combine_rebased_sources


FIELDS = [
    "id", "entry_date", "exit_date", "entry_source", "entry_kind",
    "status", "hold_sessions", "trade_net_return_pct",
    "cutoff_unrealized_price_return_pct", "max_close_floating_loss_pct",
    "max_close_floating_gain_pct", "max_intraday_floating_gain_pct",
    "best_close_date", "best_intraday_date",
    "max_intraday_floating_loss_pct", "max_adverse_including_exit_open_pct",
    "worst_close_date", "worst_intraday_date",
    "max_peak_close_to_later_close_dd_pct", "peak_close_date_before_dd",
    "trough_close_date_for_dd", "exit_open_was_worse_than_held_intraday_low",
]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pct(v: float) -> float:
    return round(v * 100, 4)


def audit_trade(prices: pd.DataFrame, trade: dict, *, cutoff: str,
                is_open: bool = False) -> dict:
    """Use adjusted daily marks during the actual ownership interval only."""
    by_date = prices.set_index(prices.Date.dt.strftime("%Y-%m-%d"),
                               verify_integrity=True)
    entry_date = trade["entry_date"]
    exit_date = cutoff if is_open else trade["exit_date"]
    if entry_date not in by_date.index or exit_date not in by_date.index:
        raise ValueError(f"Missing trade boundary: {entry_date}, {exit_date}")
    start_price = float(trade.get("entry_price") or
                        by_date.at[entry_date, "adj_open"])
    observed_buy_open = float(by_date.at[entry_date, "adj_open"])
    if abs(start_price - observed_buy_open) > max(1e-5, start_price * 1e-6):
        raise ValueError(f"Wrong input adjustment at entry {entry_date}")

    interval = by_date.loc[entry_date:exit_date].copy()
    if not is_open:
        interval = interval.iloc[:-1]  # already sold at exit date OPEN
    if interval.empty:
        raise ValueError("No holding-day observations")

    low = interval.adj_low.astype(float)
    closes = interval.adj_close.astype(float)
    lowest_low = float(low.min())
    lowest_close = float(closes.min())
    lowest_low_date = str(low.idxmin())
    lowest_close_date = str(closes.idxmin())
    highs = interval.adj_high.astype(float)
    highest_high = float(highs.max())
    highest_close_observed = float(closes.max())
    highest_high_date = str(highs.idxmax())
    highest_close_date = str(closes.idxmax())
    if highest_high + 1e-8 < highest_close_observed:
        raise AssertionError("Intraday high cannot be below day close")
    # Do not clamp the closing maximum: the frozen ledger's 'mfe_close'
    # can be negative when all completed holding closes are below entry.
    if not is_open:
        original_mfe = float(trade["mfe_close"])
        if abs(highest_close_observed / start_price - 1 - original_mfe) > 1e-8:
            raise ValueError(f"Original ledger closing MFE mismatch: {entry_date}")
    if lowest_low > lowest_close + 1e-8:
        raise AssertionError("Intraday low cannot exceed day close")
    if not is_open:
        original_mae = float(trade["mae_close"])
        if abs(lowest_close / start_price - 1 - original_mae) > 1e-8:
            raise ValueError(f"Original ledger closing MAE mismatch: {entry_date}")

    highest_close = start_price
    peak_date = entry_date
    max_peak_dd = 0.
    worst_peak_date = ""
    worst_trough_date = ""
    for day, r in interval.iterrows():
        mark = float(r.adj_close)
        dd = mark / highest_close - 1
        if dd < max_peak_dd:
            max_peak_dd = dd
            worst_peak_date = peak_date
            worst_trough_date = str(day)
        if mark > highest_close:
            highest_close = mark
            peak_date = str(day)

    terminal_open = float("inf")
    if not is_open:
        terminal_open = float(by_date.at[exit_date, "adj_open"])
        if abs(float(trade["exit_price"]) - terminal_open) > max(
                1e-5, terminal_open * 1e-6):
            raise ValueError(f"Wrong input adjustment at exit {exit_date}")
        dd = terminal_open / highest_close - 1
        if dd < max_peak_dd:
            max_peak_dd = dd
            worst_peak_date = peak_date
            worst_trough_date = exit_date

    result = {
        "id": int(trade["id"]),
        "entry_date": entry_date,
        "exit_date": (f"OPEN ({cutoff})" if is_open else exit_date),
        "entry_source": trade["entry_source"],
        "entry_kind": trade["entry_kind"],
        "status": "OPEN" if is_open else "CLOSED",
        "hold_sessions": (len(interval) if is_open else int(trade["hold_days"])),
        "trade_net_return_pct": ("" if is_open
                                 else _pct(float(trade["ret_net"]))),
        "cutoff_unrealized_price_return_pct": (
            _pct(float(by_date.at[cutoff, "adj_close"]) / start_price - 1)
            if is_open else ""),
        "max_close_floating_loss_pct":
            _pct(min(0., lowest_close / start_price - 1)),
        # MFE high is the maximum profit a trader could have *seen* on
        # an intraday mark. It is NOT a fillable backtest price.
        "max_close_floating_gain_pct":
            _pct(highest_close_observed / start_price - 1),
        "max_intraday_floating_gain_pct":
            _pct(max(0., highest_high / start_price - 1)),
        "best_close_date": highest_close_date,
        "best_intraday_date": highest_high_date,
        "max_intraday_floating_loss_pct":
            _pct(min(0., lowest_low / start_price - 1)),
        "max_adverse_including_exit_open_pct":
            _pct(min(0., min(lowest_low, terminal_open) / start_price - 1)),
        "worst_close_date": lowest_close_date,
        "worst_intraday_date": lowest_low_date,
        "max_peak_close_to_later_close_dd_pct": _pct(max_peak_dd),
        "peak_close_date_before_dd": worst_peak_date,
        "trough_close_date_for_dd": worst_trough_date,
        "exit_open_was_worse_than_held_intraday_low": (
            "" if is_open else terminal_open < lowest_low),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--trades", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--end", default="2026-07-13")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    full = combine_rebased_sources(args.old, args.new, end_date=args.end)
    if full.empty or str(full.Date.iloc[-1].date()) != args.end:
        raise ValueError("Cutoff last trading bar does not match requested date")
    ledger = read_rows(args.trades)
    summary = read_rows(args.summary)
    if len(summary) != 1:
        raise ValueError("One summary row expected")
    if str(summary[0]["end"]) != args.end:
        raise ValueError("Summary and bars cutoff mismatch")
    rows = [audit_trade(full, item, cutoff=args.end) for item in ledger]
    last = summary[0].get("open_position_entry")
    if last:
        extra = {
            "id": str(len(rows) + 1),
            "entry_date": last,
            "entry_source": "H3_ACCEL",  # frozen H3 2026-04-10 candidate
            "entry_kind": "FAST_EMA10_20_MONTH_EMA5",
        }
        rows.append(audit_trade(full, extra, cutoff=args.end, is_open=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(args.out), "completed_trades": len(ledger),
        "risk_rows": len(rows), "cutoff": args.end,
        "no_raw_daily_ohlcv_exported": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
