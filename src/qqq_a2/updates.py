"""Append newly obtained, independently audited daily QQQ bars.

This module does not fetch or redistribute any market history. Both original
and updated vendor exports must be supplied locally by the operator.
"""
from __future__ import annotations

import pandas as pd

from qqq_a2.engine import load_adjusted_csv

ADJUSTED_COLS = ("adj_open", "adj_high", "adj_low", "adj_close")


def extend_adjusted_history(
    frozen: pd.DataFrame,
    latest_vendor_file: str,
    *,
    frozen_cutoff: str = "2026-07-13",
    completed_cutoff: str = "2026-11-09",
    overlap_days: int = 15,
    min_overlap: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Rebase old adjusted OHLC to the updated vendor's dividend basis.

    Raw overlapping closes must match. The latest vendor's *adjusted* close is
    divided by the old adjusted close across the last N sessions before the
    frozen boundary. A stable ratio indicates newly reported dividends after
    the frozen snapshot. Old pre-cutoff adjusted OHLC are multiplied uniformly
    by that ratio; new post-cutoff adjusted OHLC are appended unchanged.

    The frozen trading strategy and pre-cutoff returns remain invariant under
    the scale change. The newest session must be a fully completed daily bar.
    """
    cutoff, asof = pd.Timestamp(frozen_cutoff), pd.Timestamp(completed_cutoff)
    if cutoff >= asof:
        raise ValueError("Forward update end must be after frozen cutoff")
    old = frozen.copy().sort_values("Date").reset_index(drop=True)
    old["Date"] = pd.to_datetime(old.Date)
    if old.Date.iloc[-1] != cutoff or old.Date.duplicated().any():
        raise ValueError("Frozen history does not end exactly on frozen cutoff")
    update = load_adjusted_csv(latest_vendor_file)
    if update.Date.iloc[-1] < asof:
        raise ValueError("Latest vendor snapshot does not reach requested cutoff")
    update = update[update.Date <= asof].copy()
    if update.Date.iloc[-1] != asof:
        raise ValueError("Completed cutoff has no fully formed daily bar")

    recent_overlap = old.loc[old.Date >= cutoff - pd.Timedelta(days=overlap_days)]
    overlap = recent_overlap.merge(update, on="Date", suffixes=("_old", "_update"))
    if len(overlap) < min_overlap:
        raise ValueError(f"Need >={min_overlap} overlapping completed sessions")
    raw_diff = (overlap.Close_old - overlap.Close_update).abs()
    if float(raw_diff.max()) > 0.05:
        raise ValueError("Raw closes disagree across frozen/latest sources")
    scale_samples = overlap.adj_close_update / overlap.adj_close_old
    scale = float(scale_samples.median())
    spread = float((scale_samples - scale).abs().max())
    if spread > 0.0001:
        raise ValueError("Dividend rebase ratio inconsistent across overlap")
    if not 0.8 < scale < 1.2:
        raise ValueError("Unexpected corporate action adjustment; manual review needed")

    result_old = old.copy()
    for field in ADJUSTED_COLS:
        result_old[field] *= scale
    extra = update[update.Date > cutoff].copy()
    if extra.empty:
        raise ValueError("No new fully completed session")
    merged = pd.concat([result_old, extra], ignore_index=True).sort_values("Date")
    if not merged.Date.is_unique or merged.Date.iloc[-1] != asof:
        raise ValueError("Invalid merged date coverage")
    # Recheck adjusted OHLC are finite and respect a valid candlestick range.
    if merged[list(ADJUSTED_COLS)].isna().any().any():
        raise ValueError("Missing adjusted OHLC")
    if (merged.adj_high + 0.05 < merged[list(ADJUSTED_COLS)].max(axis=1)).any():
        raise ValueError("Adjusted high violated")
    if (merged.adj_low - 0.05 > merged[list(ADJUSTED_COLS)].min(axis=1)).any():
        raise ValueError("Adjusted low violated")
    meta = {
        "frozen_cutoff": str(cutoff.date()), "completed_cutoff": str(asof.date()),
        "old_rows": len(old), "additional_sessions": len(extra),
        "total_sessions": len(merged), "overlap_sessions": len(overlap),
        "rebase_ratio": scale, "rebase_max_deviation": spread,
        "overlap_max_raw_close_difference": float(raw_diff.max()),
        "latest_close_unadjusted": float(extra.Close.iloc[-1]),
    }
    return merged.reset_index(drop=True), meta


def adjusted_prices_to_local_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a *local-only* CSV that will not be committed/distributed."""
    return pd.DataFrame({
        "Date": pd.to_datetime(df.Date).dt.strftime("%Y-%m-%d"),
        "Open": df.adj_open, "High": df.adj_high, "Low": df.adj_low,
        "Close": df.adj_close, "Adj Close": df.adj_close,
        "Volume": df.Volume,
    })
