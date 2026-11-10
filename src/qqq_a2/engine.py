"""Strategy v1.0: daily SMA20/60 fresh gold, N20 guard, fresh death exit.

No news, no leverage, no alternative exit rules, no same-close fills.
The research/a2_n20.py engine is retained separately as a historical reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Config:
    start_date: str = "2000-01-03"
    fee_per_side: float = 0.0005
    initial_cash: float = 100_000.0
    n20_days: int = 20


def load_adjusted_csv(path: str | Path) -> pd.DataFrame:
    """Read a USER-PROVIDED OHLCV CSV and normalize OHLC to adjusted basis.

    Required columns: Date, Open, High, Low, Close, Adj Close.
    If dividend adjustment isn't available, set Adj Close equal to Close
    explicitly, understanding that results won't match the frozen baseline.
    No remote data is downloaded or redistributed by this library.
    """
    frame = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "Open", "High", "Low", "Close", "Adj Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = frame.sort_values("Date").reset_index(drop=True)
    if not frame.Date.is_unique:
        raise ValueError("Dates must be unique")
    if frame[["Open", "High", "Low", "Close", "Adj Close"]].isna().any().any():
        raise ValueError("Missing OHLC / Adj Close; no partially completed bar is allowed")
    if (frame[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    adjustment = frame["Adj Close"] / frame["Close"]
    for field in ("Open", "High", "Low", "Close"):
        frame[f"adj_{field.lower()}"] = frame[field] * adjustment
    return frame


def combine_rebased_sources(old_path: str | Path, new_path: str | Path,
                            end_date: str | None = None) -> pd.DataFrame:
    """Rebase two Yahoo-style historical exports across a verified overlap.

    This is for private downloads of the original research inputs. We never
    bundle their unlicensed original bars in the public repository.
    """
    old = pd.read_csv(old_path, parse_dates=["Date"])
    new = pd.read_csv(new_path, parse_dates=["Date"])
    for source in (old, new):
        if not source.Date.is_unique:
            raise ValueError("Source dates are not unique")
        source.sort_values("Date", inplace=True)
    if end_date:
        new = new[new.Date <= pd.Timestamp(end_date)].copy()
    # Filter incomplete end-of-day rows (some public snapshots contain one).
    new = new.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"])
    overlap = old.merge(new, on="Date", suffixes=("_old", "_new"))
    if len(overlap) < 20:
        raise ValueError("Insufficient overlap to rebase adjustment factors")
    if (overlap.Close_old - overlap.Close_new).abs().max() > 1e-8:
        raise ValueError("Overlapping raw closes disagree")
    ratios = overlap["Adj Close_new"] / overlap["Adj Close_old"]
    if ratios.std() >= 1e-5:
        raise ValueError("Source adjustment ratios differ too much")
    scale = float(ratios.median())
    cutoff = new.Date.min()
    earlier = old[old.Date < cutoff].copy()
    later = new[new.Date >= cutoff].copy()
    for part, rebase in ((earlier, scale), (later, 1.0)):
        adjustment = part["Adj Close"] / part["Close"] * rebase
        for field in ("Open", "High", "Low", "Close"):
            part[f"adj_{field.lower()}"] = part[field] * adjustment
    return pd.concat([earlier, later]).sort_values("Date").reset_index(drop=True)


def compute_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Compute all signals only from known closes, never using future bars."""
    df = data.copy().sort_values("Date").reset_index(drop=True)
    if "adj_close" not in df or "adj_open" not in df:
        raise ValueError("Expected adj_open and adj_close columns")
    for k in (10, 20, 60, 200):
        df[f"ma{k}"] = df.adj_close.rolling(k, min_periods=k).mean()
    df["gold"] = (df.ma20 > df.ma60) & (df.ma20.shift(1) <= df.ma60.shift(1))
    df["death"] = (df.ma20 < df.ma60) & (df.ma20.shift(1) >= df.ma60.shift(1))
    df["protect_cross"] = (df.ma10 < df.ma20) & (df.ma10.shift(1) >= df.ma20.shift(1))
    return df


def run_backtest(data: pd.DataFrame, config: Config = Config()) -> dict:
    """Run V1.0 with signal-close / next-open fills and daily equity marking."""
    df = compute_signals(data)
    cash = float(config.initial_cash)
    shares = 0.0
    current: dict | None = None
    pending: dict | None = None
    trades: list[dict] = []
    curve: list[dict] = []

    for i, bar in df.iterrows():
        if bar.Date < pd.Timestamp(config.start_date):
            continue

        # Pending signal from previous complete session; now regular open.
        if pending:
            if pending["side"] == "buy":
                if shares:
                    raise AssertionError("Position already open on buy")
                shares = cash / (bar.adj_open * (1 + config.fee_per_side))
                cash = 0.0
                current = {"entry_signal_date": pending["signal_date"],
                           "entry_date": str(bar.Date.date()), "entry_index": int(i),
                           "entry_price": float(bar.adj_open),
                           "entry_cost": float(shares * bar.adj_open * (1 + config.fee_per_side)),
                           "max_close": float(bar.adj_close),
                           "min_close": float(bar.adj_close)}
            else:
                if not shares or current is None:
                    raise AssertionError("No position to sell")
                proceeds = shares * bar.adj_open * (1 - config.fee_per_side)
                trades.append({
                    "id": len(trades) + 1,
                    "entry_signal_date": current["entry_signal_date"],
                    "entry_date": current["entry_date"],
                    "entry_price": current["entry_price"],
                    "exit_signal_date": pending["signal_date"],
                    "exit_date": str(bar.Date.date()),
                    "exit_price": float(bar.adj_open),
                    "exit_reason": pending["reason"],
                    "hold_days": int(i - current["entry_index"]),
                    "ret_net": float(proceeds / current["entry_cost"] - 1),
                    "mfe_close": float(current["max_close"] / current["entry_price"] - 1),
                    "mae_close": float(current["min_close"] / current["entry_price"] - 1),
                    "equity_after_trade": float(proceeds),
                })
                cash, shares, current = float(proceeds), 0.0, None
            pending = None

        if shares:
            assert current is not None
            current["max_close"] = max(current["max_close"], float(bar.adj_close))
            current["min_close"] = min(current["min_close"], float(bar.adj_close))
            days = int(i - current["entry_index"])
            if bool(bar.death):
                pending = {"side": "sell", "signal_date": str(bar.Date.date()),
                           "reason": "MA20/60 death cross"}
            elif (1 <= days <= config.n20_days and bool(bar.protect_cross)
                  and bar.adj_close < current["entry_price"]):
                pending = {"side": "sell", "signal_date": str(bar.Date.date()),
                           "reason": "N20 protection"}
        elif bool(bar.gold):
            pending = {"side": "buy", "signal_date": str(bar.Date.date()),
                       "reason": "FRESH_GOLD_CROSS"}

        curve.append({"Date": str(bar.Date.date()), "Equity": cash + shares * float(bar.adj_close),
                      "Invested": bool(shares), "Pending": pending["side"] if pending else ""})

    t = pd.DataFrame(trades)
    e = pd.DataFrame(curve)
    if e.empty:
        raise ValueError("No trading sessions at or after start_date")
    e["Peak"] = e.Equity.cummax()
    e["Drawdown"] = e.Equity / e.Peak - 1
    winners = t[t.ret_net > 0] if not t.empty else t
    losers = t[t.ret_net < 0] if not t.empty else t
    yrs = (pd.Timestamp(e.Date.iloc[-1]) - pd.Timestamp(config.start_date)).days / 365.25
    stats = {
        "completed_trades": len(t), "wins": len(winners), "losses": len(losers),
        "win_rate": (len(winners) / len(t)) if len(t) else None,
        "payoff_ratio": (float(winners.ret_net.mean() / abs(losers.ret_net.mean()))
                         if len(winners) and len(losers) else None),
        "profit_factor": (float(winners.ret_net.sum() / -losers.ret_net.sum())
                          if len(winners) and len(losers) else None),
        "mdd": float(e.Drawdown.min()), "end_equity": float(e.Equity.iloc[-1]),
        "cagr": float((e.Equity.iloc[-1] / config.initial_cash) ** (1 / yrs) - 1) if yrs > 0 else None,
        "active_position_at_end": current is not None,
        "pending_order_at_end": pending,
    }
    open_position = dict(current) if current is not None else None
    return {"trades": t, "equity": e, "stats": stats, "open_position": open_position,
            "pending_order": pending, "latest_date": e.Date.iloc[-1],
            "latest_indicators": {k: float(df[k].iloc[-1]) for k in ("ma10", "ma20", "ma60", "ma200")}}
