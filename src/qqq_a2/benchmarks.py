"""Independent QQQ buy-and-hold benchmark on already adjusted user-owned bars.

This is a gross-dividend total-return proxy, *before* personal taxes or
dividend withholding. It never reads future prices or modifies A2 signals.
"""
from __future__ import annotations

import pandas as pd

from qqq_a2.engine import Config


def buy_and_hold(data: pd.DataFrame, config: Config = Config()) -> dict:
    """Buy at the first simulation session's adjusted OPEN, mark each CLOSE.

    Entry pays the same 0.05% per-side trading fee as A2. There is no exit
    during the sample, so the final equity is the unsold marked-to-market
    position and excludes the *hypothetical* terminal sale fee and all taxes.
    """
    required = {"Date", "adj_open", "adj_close"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing adjusted benchmark columns: {sorted(required-data.columns)}")
    if config.initial_cash <= 0 or not 0 <= config.fee_per_side < 1:
        raise ValueError("Invalid benchmark initial cash or fee")
    start = pd.Timestamp(config.start_date)
    d = data.loc[pd.to_datetime(data.Date) >= start].copy()
    d["Date"] = pd.to_datetime(d.Date)
    d = d.sort_values("Date").reset_index(drop=True)
    if d.empty or d.Date.iloc[0] != start:
        raise ValueError("First full session is missing; cannot silently shift buy date")
    if d.Date.duplicated().any():
        raise ValueError("Duplicate benchmark sessions")
    if (d[["adj_open", "adj_close"]] <= 0).any().any() or d[["adj_open", "adj_close"]].isna().any().any():
        raise ValueError("Invalid adjusted OHLC")

    entry = float(d.adj_open.iloc[0])
    units = config.initial_cash / (entry * (1 + config.fee_per_side))
    equity = d.adj_close.astype(float) * units
    peak = equity.cummax()
    drawdown = equity / peak - 1
    low_i = int(drawdown.idxmin())
    high_i = int(equity.iloc[:low_i+1].idxmax())
    peak_value = float(equity.iloc[high_i])
    recovered = d.loc[(d.index > low_i) & (equity >= peak_value), "Date"]
    recovery_date = str(recovered.iloc[0].date()) if not recovered.empty else None
    years = (d.Date.iloc[-1] - start).days / 365.25
    if years <= 0:
        raise ValueError("Benchmark needs at least two distinct sessions")
    terminal = float(equity.iloc[-1])
    curve = pd.DataFrame({
        "Date": d.Date,
        "Equity": equity,
        "Peak": peak,
        "Drawdown": drawdown,
    })
    return {
        "equity": curve,
        "stats": {
            "start_date": str(start.date()),
            "end_date": str(d.Date.iloc[-1].date()),
            "initial_cash": config.initial_cash,
            "entry_adjusted_open": entry,
            "entry_fee_per_side": config.fee_per_side,
            "exit_fee_assessed": False,
            "end_equity": terminal,
            "total_return": terminal / config.initial_cash - 1,
            "cagr": (terminal / config.initial_cash) ** (1 / years) - 1,
            "mdd": float(drawdown.min()),
            "mdd_peak_date": str(d.Date.iloc[high_i].date()),
            "mdd_trough_date": str(d.Date.iloc[low_i].date()),
            "mdd_recovery_date": recovery_date,
            "mdd_underwater_years": (
                float((recovered.iloc[0] - d.Date.iloc[high_i]).days / 365.25)
                if not recovered.empty else None
            ),
            "mdd_recovery_return_needed": 1 / (1 + float(drawdown.min())) - 1,
            "end_equity_if_sold_at_last_close_and_paid_exit_fee": terminal * (1-config.fee_per_side),
            "taxes": "not modeled; adjusted close is gross-dividend total-return proxy",
        },
    }
