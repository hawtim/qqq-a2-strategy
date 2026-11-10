"""Reproducible A2-N20 baseline analysis; no strategy optimization.

Run: python research/a2_n20.py
Inputs: sources/qqq_jtang.csv and sources/qqq_james.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "results"
START = pd.Timestamp("2000-01-03")
END = pd.Timestamp("2026-07-13")
FEE = 0.0005


def prices() -> tuple[pd.DataFrame, dict]:
    a = pd.read_csv(ROOT / "sources/qqq_jtang.csv", parse_dates=["Date"])
    b = pd.read_csv(ROOT / "sources/qqq_james.csv", parse_dates=["Date"])
    for name, source in (("jtang", a), ("james", b)):
        assert source["Date"].is_monotonic_increasing and source["Date"].is_unique, name
    overlap = a.merge(b, on="Date", suffixes=("_a", "_b"))
    assert (overlap["Close_a"] - overlap["Close_b"]).abs().max() < 1e-8
    ratio = overlap["Adj Close_b"] / overlap["Adj Close_a"]
    scale = float(ratio.median())
    assert float(ratio.std()) < 0.00001

    # Historical split-adjusted OHLC is *further* adjusted for distributions.
    # Source A's Adj Close as of 2022 must be rebased to Source B's 2026 basis.
    cutoff = b["Date"].min()
    a = a[a.Date < cutoff].copy()
    b = b[(b.Date >= cutoff) & (b.Date <= END)].copy()
    assert b[["Open", "High", "Low", "Close", "Adj Close"]].notna().all().all()
    a["origin"] = "jtang-rebased"
    b["origin"] = "james"
    for frame, rebasing in ((a, scale), (b, 1.0)):
        f = frame["Adj Close"] / frame["Close"] * rebasing
        for col in ("Open", "High", "Low", "Close"):
            frame["adj_" + col.lower()] = frame[col] * f
    df = pd.concat((a, b)).sort_values("Date").reset_index(drop=True)
    assert df.Date.is_unique and df.Date.is_monotonic_increasing
    for k in (10, 20, 60, 200):
        df[f"ma{k}"] = df.adj_close.rolling(k, min_periods=k).mean()
    df["gold"] = (df.ma20 > df.ma60) & (df.ma20.shift(1) <= df.ma60.shift(1))
    df["death"] = (df.ma20 < df.ma60) & (df.ma20.shift(1) >= df.ma60.shift(1))
    df["protect_cross"] = (df.ma10 < df.ma20) & (df.ma10.shift(1) >= df.ma20.shift(1))
    meta = {
        "source_a_rows": 5915,
        "source_b_completed_rows": len(b),
        "overlap_rows": len(overlap),
        "overlap_close_max_abs_diff": float((overlap.Close_a - overlap.Close_b).abs().max()),
        "adj_close_scale_a_to_b": scale,
        "adj_close_overlap_ratio_std": float(ratio.std()),
        "sample_start": str(df.Date.iloc[0].date()),
        "sample_end": str(df.Date.iloc[-1].date()),
        "sample_rows": len(df),
        "historical_join_date": str(cutoff.date()),
        "strategy_start": str(START.date()),
    }
    return df, meta


def run(df: pd.DataFrame, protection_window: int = 20,
        early_stop: bool = True, warm_start: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict | None]:
    """A2 pending fills. Day 1 is the first trading day AFTER purchase.

    warm_start=True reproduces the prior engine's synthetic long state if the
    pre-simulation day is bullish. Not a naturally occurring fresh gold cross.
    """
    cash = 100000.0
    shares = 0.0
    open_trade = None
    pending = None
    trades = []
    curve = []
    for i, bar in df.iterrows():
        date = bar.Date
        if date < START:
            continue
        if warm_start and date == START and df.iloc[i-1].ma20 > df.iloc[i-1].ma60:
            pending = ("buy", i - 1, "warm-start")
        if pending is not None:
            if pending[0] == "buy":
                assert shares == 0
                shares = cash / (bar.adj_open * (1 + FEE))
                cash = 0.0
                signal_idx = pending[1]
                signal = df.iloc[signal_idx]
                prior = df.iloc[signal_idx - 1]
                def rret(k):
                    if signal_idx - k < 0:
                        return np.nan
                    return float(signal.adj_close / df.iloc[signal_idx - k].adj_close - 1)
                recent = df.iloc[max(0, signal_idx - 60):signal_idx + 1]
                open_trade = {
                    "entry_origin": "WARM_START (not a fresh cross)" if len(pending)>2 else "FRESH_GOLD_CROSS",
                    "signal_date": str(signal.Date.date()),
                    "entry_date": str(date.date()),
                    "signal_idx": signal_idx,
                    "entry_idx": i,
                    "entry_price": float(bar.adj_open),
                    "entry_ma200_distance": float(signal.adj_close / signal.ma200 - 1),
                    "entry_ma20_slope_10d": float(signal.ma20 / df.iloc[signal_idx - 10].ma20 - 1),
                    "entry_ma60_slope_10d": float(signal.ma60 / df.iloc[signal_idx - 10].ma60 - 1),
                    "entry_ma200_slope_20d": float(signal.ma200 / df.iloc[signal_idx - 20].ma200 - 1),
                    "entry_ma20_ma60_spread": float(signal.ma20 / signal.ma60 - 1),
                    "entry_60d_return": rret(60),
                    "entry_20d_return": rret(20),
                    "entry_5d_return": rret(5),
                    "entry_60d_max_drawdown_from_close_peak": float(signal.adj_close / recent.adj_close.max() - 1),
                    "entry_20d_vol_ann": float(df.iloc[signal_idx - 19:signal_idx + 1].adj_close.pct_change().std() * np.sqrt(252)),
                    "entry_close_over_ma200": bool(signal.adj_close > signal.ma200),
                    "entry_ma200_rising": bool(signal.ma200 > df.iloc[signal_idx - 20].ma200),
                    "entry_ma60_rising": bool(signal.ma60 > df.iloc[signal_idx - 10].ma60),
                    "entry_signal_price": float(signal.adj_close),
                    "prior_close": float(prior.adj_close),
                    "max_close": float(bar.adj_close),
                    "min_close": float(bar.adj_close),
                    "entry_equity": float(shares * bar.adj_open),
                }
            else:
                assert shares > 0 and open_trade is not None
                proceeds = shares * bar.adj_open * (1 - FEE)
                entry_debit = shares * open_trade["entry_price"] * (1 + FEE)
                ret = proceeds / entry_debit - 1
                trade = dict(open_trade)
                trade.update({
                    "id": len(trades) + 1,
                    "exit_signal_date": str(df.iloc[pending[1]].Date.date()),
                    "exit_date": str(date.date()),
                    "exit_price": float(bar.adj_open),
                    "exit_reason": pending[2],
                    "hold_days": i - open_trade["entry_idx"],
                    "ret_net": float(ret),
                    "ret_gross": float(bar.adj_open / trade["entry_price"] - 1),
                    "mfe_close": float(open_trade["max_close"] / trade["entry_price"] - 1),
                    "mae_close": float(open_trade["min_close"] / trade["entry_price"] - 1),
                    "equity_after_trade": float(proceeds),
                })
                trades.append(trade)
                cash = proceeds
                shares = 0.0
                open_trade = None
            pending = None
        if shares > 0:
            open_trade["max_close"] = max(open_trade["max_close"], float(bar.adj_close))
            open_trade["min_close"] = min(open_trade["min_close"], float(bar.adj_close))
            day_number = i - open_trade["entry_idx"]
            if bool(bar.death):
                pending = ("sell", i, "MA20/60 death cross")
            elif early_stop and 1 <= day_number <= protection_window and bool(bar.protect_cross) and bar.adj_close < open_trade["entry_price"]:
                pending = ("sell", i, "N20 protection")
        elif bool(bar.gold):
            pending = ("buy", i)
        # closing-value equity, following actual fill; cash gets no interest
        curve.append({"Date": str(date.date()), "Equity": cash + shares * bar.adj_close,
                      "Invested": bool(shares > 0), "Pending": pending[0] if pending else ""})

    t = pd.DataFrame(trades)
    e = pd.DataFrame(curve)
    e["Peak"] = e.Equity.cummax()
    e["Drawdown"] = e.Equity / e.Peak - 1
    nwin = int((t.ret_net > 0).sum())
    nloss = int((t.ret_net < 0).sum())
    avgw = float(t.loc[t.ret_net > 0, "ret_net"].mean())
    avgl = float(t.loc[t.ret_net < 0, "ret_net"].mean())
    cash_run = 0
    longest = 0
    for invested in e.Invested:
        cash_run = 0 if invested else cash_run + 1
        longest = max(longest, cash_run)
    last = e.iloc[-1]
    n_years = (pd.Timestamp(last.Date) - START).days / 365.25
    stats = {
        "completed_trades": len(t), "wins": nwin, "losses": nloss,
        "win_rate": nwin / len(t), "avg_win": avgw, "avg_loss": avgl,
        "payoff_ratio": avgw / abs(avgl),
        "profit_factor": float(t.loc[t.ret_net > 0, "ret_net"].sum() / -t.loc[t.ret_net < 0, "ret_net"].sum()),
        "mdd": float(e.Drawdown.min()), "end_equity": float(last.Equity),
        "cagr": float((last.Equity / 100000) ** (1 / n_years) - 1),
        "invested_fraction": float(e.Invested.mean()), "longest_cash_days": longest,
        "max_single_loss": float(t.ret_net.min()),
        "max_single_gain": float(t.ret_net.max()),
        "mean_hold_days": float(t.hold_days.mean()),
        "median_hold_days": float(t.hold_days.median()),
        "active_position_at_end": open_trade is not None,
        "pending_order_at_end": pending[0] if pending else None,
    }
    if open_trade is not None:
        open_trade = dict(open_trade)
        open_trade.update({
            "id": len(trades) + 1,
            "status": "OPEN (NOT COMPLETED)",
            "mark_date": str(df.Date.iloc[-1].date()),
            "mark_close": float(df.adj_close.iloc[-1]),
            "hold_days_to_mark": int(len(df) - 1 - open_trade["entry_idx"]),
            "return_mark_to_market_before_exit_fee": float(df.adj_close.iloc[-1] / (open_trade["entry_price"] * (1 + FEE)) - 1),
            "return_mark_to_market_with_hypothetical_exit_fee": float(df.adj_close.iloc[-1] * (1 - FEE) / (open_trade["entry_price"] * (1 + FEE)) - 1),
        })
    return t, e, stats, open_trade


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df, meta = prices()
    t, e, stats, active = run(df)
    legacy_t, legacy_e, legacy_stats, legacy_active = run(df,warm_start=True)
    df.to_csv(OUT / "price_series_rebased.csv", index=False)
    t.to_csv(OUT / "trades_complete.csv", index=False)
    legacy_t.to_csv(OUT / "legacy_trades_complete.csv", index=False)
    legacy_e.to_csv(OUT / "legacy_equity_curve.csv", index=False)
    if active:
        pd.DataFrame([active]).to_csv(OUT / "open_position_at_cutoff.csv", index=False)
    if legacy_active:
        pd.DataFrame([legacy_active]).to_csv(OUT / "legacy_open_position_at_cutoff.csv", index=False)
    e.to_csv(OUT / "equity_curve.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({"input": meta, "baseline": stats, "legacy_warm_start": legacy_stats}, indent=2))
    print(json.dumps({"input": meta, "baseline": stats, "legacy_warm_start": legacy_stats,
                      "legacy_initial_trade": legacy_t.iloc[0][["entry_origin","entry_date","exit_date","exit_reason","ret_net"]].to_dict()}, indent=2))
