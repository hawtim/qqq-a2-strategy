"""Independent, read-only cash carry overlay for an existing A2 equity curve.

Trading dates and positions NEVER change: no optimization, no additional entries
or exits. All figures are USD before personal taxes and FX conversion.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def read_yield_csv(path: str) -> pd.DataFrame:
    """Accept FRED DGS3MO CSV (observation_date,DGS3MO), '-' as missing."""
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df else "DATE"
    if date_col not in df or "DGS3MO" not in df:
        raise ValueError("Expected FRED CSV columns observation_date/DATE and DGS3MO")
    out = pd.DataFrame({"Date": pd.to_datetime(df[date_col]),
                        "RatePct": pd.to_numeric(df["DGS3MO"], errors="coerce")})
    out = out.dropna(subset=["RatePct"]).sort_values("Date").reset_index(drop=True)
    if out.Date.duplicated().any() or out.empty:
        raise ValueError("Rate history must be nonempty with unique dates")
    if ((out.RatePct < -10) | (out.RatePct > 30)).any():
        raise ValueError("Unexpected DGS3MO yield; check units (percent not decimal)")
    return out


def align_previous_quote(curve: pd.DataFrame, yields: pd.DataFrame) -> pd.Series:
    """Yield known on prior completed QQQ session; no future rate used."""
    dates = pd.DataFrame({"Date": pd.to_datetime(curve.Date)}).reset_index(drop=True)
    right = yields.rename(columns={"Date": "YieldDate"}).sort_values("YieldDate")
    joined = pd.merge_asof(dates, right, left_on="Date", right_on="YieldDate",
                           direction="backward")
    age = (joined.Date - joined.YieldDate).dt.days
    if joined.RatePct.isna().any() or age.max() > 10:
        raise ValueError("DGS3MO data missing/stale (>10 calendar days); do not silently assume a rate")
    return joined.RatePct.astype(float)


def cash_overlay(curve: pd.DataFrame, annual_pct: float | pd.Series) -> pd.Series:
    """Apply interest across calendar days when the PREVIOUS close was flat.

    On a next-open buy day, previous-close cash earns overnight carry; on a
    next-open sell day, prior held shares do not retroactively earn carry.
    Each session's stock return and fee are the frozen original A2 return.
    """
    curve = curve.reset_index(drop=True)
    dates = pd.to_datetime(curve.Date)
    equity = curve.Equity.astype(float)
    if equity.isna().any() or (equity <= 0).any():
        raise ValueError("Equity must be fully observed and positive")
    base_factor = equity.div(equity.shift(1)).fillna(1.0)
    gaps = dates.diff().dt.days.fillna(0).astype(float)
    previous_was_cash = ~curve.Invested.shift(1, fill_value=False).astype(bool)
    rate = (pd.Series(float(annual_pct), index=curve.index)
            if isinstance(annual_pct, (int, float)) else pd.Series(annual_pct).reset_index(drop=True))
    if len(rate) != len(curve) or rate.isna().any():
        raise ValueError("Rate must have one observation per equity date")
    # The rate for calendar period (t-1,t] is the quote known at t-1.
    prior_rate = rate.shift(1).fillna(0).astype(float)
    interest_factor = 1.0 + previous_was_cash * prior_rate / 100.0 * gaps / 365.0
    if (interest_factor <= 0).any():
        raise ValueError("Invalid interest factor")
    result = float(equity.iloc[0]) * (base_factor * interest_factor).cumprod()
    return pd.Series(result, index=curve.index, name="EquityWithCashCarry")


def equity_stats(values: pd.Series, dates: pd.Series, start_date: str = "2000-01-03") -> dict:
    v = pd.Series(values, dtype=float)
    years = (pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(start_date)).days / 365.25
    drawdown = v / v.cummax() - 1
    return {"end_equity": float(v.iloc[-1]),
            "cagr": float((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1),
            "mdd": float(drawdown.min())}
