"""Independent A2 Hybrid experiment: bearish SMA entries, bullish EMA entries.

Weekly/monthly regimes use PREVIOUSLY COMPLETED higher-timeframe bars only.
The selected MA family is locked for the lifetime of a trade. An optional,
separately measured downside-risk override exits at the next open when the
prior completed higher-timeframe trend is bearish and daily SMA10 < SMA20.

    python research/experiments/a2_hybrid_regime.py \
      --csv sources/qqq_merged_updated_local.csv --end 2026-11-09

Raw vendor market prices are never written to publicly reproducible outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from qqq_a2.engine import Config, compute_signals, load_adjusted_csv
from a2_ema_comparison import ema_signals, validate_sma_against_frozen, replay_signals


REGIMES = {
    "weekly_slope": ("W-FRI", 20, 4, False),
    "monthly_slope": ("M", 10, 1, False),
    "weekly_close_slope": ("W-FRI", 20, 4, True),
    "monthly_close_slope": ("M", 10, 1, True),
}


def completed_regime(daily: pd.DataFrame, kind: str) -> pd.Series:
    """Regime on date t is computed from completed periods strictly BEFORE t.

    For example, even on a Friday we consult the preceding Friday's complete
    week, so there is never an intraday or final-day higher-timeframe leak.
    """
    freq, lookback, slope_lag, require_close = REGIMES[kind]
    periods = daily.Date.dt.to_period(freq)
    weekly_or_monthly = daily.groupby(periods, sort=True).agg(
        last_close=("adj_close", "last"), n=("adj_close", "size")
    )
    ma = weekly_or_monthly.last_close.rolling(lookback, min_periods=lookback).mean()
    up = ma.gt(ma.shift(slope_lag))
    if require_close:
        up &= weekly_or_monthly.last_close.gt(ma)
    # Use last *finished* timeframe, never the bucket containing today's bar.
    previous = up.shift(1).eq(True)
    mapped = periods.map(previous.to_dict())
    return pd.Series(mapped, index=daily.index).fillna(False).astype(bool)


def hybrid_backtest(data: pd.DataFrame, regime: str, mode: str = "locked",
                    config: Config = Config()) -> dict:
    if mode not in {"locked", "bear_fast"}:
        raise ValueError(mode)
    s = compute_signals(data)
    e = ema_signals(data)
    up = completed_regime(s, regime)
    cash, shares, pending, position = float(config.initial_cash), 0.0, None, None
    trades, curve = [], []

    for i, bar in s.iterrows():
        if bar.Date < pd.Timestamp(config.start_date):
            continue
        date = str(bar.Date.date())
        if pending is not None:
            if pending["side"] == "buy":
                if shares:
                    raise AssertionError("Unexpected overlapping position")
                shares = cash / (float(bar.adj_open) * (1 + config.fee_per_side))
                position = {"entry_signal_date": pending["signal_date"],
                            "entry_date": date, "entry_price": float(bar.adj_open),
                            "entry_cost": cash, "entry_index": i,
                            "family": pending["family"],
                            "entry_regime_up": pending["regime_up"],
                            "max_close": float(bar.adj_close),
                            "min_close": float(bar.adj_close)}
                cash = 0.0
            else:
                if not shares or position is None:
                    raise AssertionError("Exit without entry")
                proceeds = shares * float(bar.adj_open) * (1 - config.fee_per_side)
                trades.append({
                    "id": len(trades) + 1,
                    "entry_signal_date": position["entry_signal_date"],
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "entry_family": position["family"],
                    "entry_regime_up": position["entry_regime_up"],
                    "exit_signal_date": pending["signal_date"],
                    "exit_date": date, "exit_price": float(bar.adj_open),
                    "exit_reason": pending["reason"],
                    "hold_days": i - position["entry_index"],
                    "ret_net": proceeds / position["entry_cost"] - 1,
                    "mfe_close": position["max_close"] / position["entry_price"] - 1,
                    "mae_close": position["min_close"] / position["entry_price"] - 1,
                    "equity_after_trade": float(proceeds),
                })
                cash, shares, position = float(proceeds), 0.0, None
            pending = None

        if position is not None:
            position["max_close"] = max(position["max_close"], float(bar.adj_close))
            position["min_close"] = min(position["min_close"], float(bar.adj_close))
            age = i - position["entry_index"]
            signals = s.iloc[i] if position["family"] == "SMA" else e.iloc[i]
            if bool(signals.death):
                pending = {"side": "sell", "signal_date": date, "reason": "slow_death"}
            elif mode == "bear_fast" and age >= 1 and not bool(up.iloc[i]) \
                    and pd.notna(s.iloc[i].ma10) and s.iloc[i].ma10 < s.iloc[i].ma20:
                pending = {"side": "sell", "signal_date": date, "reason": "bear_fast_SMA10_below_SMA20"}
            elif (1 <= age <= config.n20_days and bool(signals.protect_cross)
                  and float(bar.adj_close) < position["entry_price"]):
                pending = {"side": "sell", "signal_date": date, "reason": "N20_protect"}
        else:
            family = "EMA" if bool(up.iloc[i]) else "SMA"
            signals = e.iloc[i] if family == "EMA" else s.iloc[i]
            if bool(signals.gold):
                pending = {"side": "buy", "signal_date": date,
                           "family": family, "regime_up": bool(up.iloc[i])}

        curve.append({"Date": date, "Equity": cash + shares * float(bar.adj_close),
                      "Invested": bool(shares), "UpRegime": bool(up.iloc[i]),
                      "PositionFamily": position["family"] if position else "",
                      "Pending": pending["side"] if pending else ""})

    t, c = pd.DataFrame(trades), pd.DataFrame(curve)
    c["Drawdown"] = c.Equity / c.Equity.cummax() - 1
    years = (pd.Timestamp(c.Date.iloc[-1]) - pd.Timestamp(config.start_date)).days / 365.25
    winners, losers = (t[t.ret_net > 0], t[t.ret_net < 0]) if len(t) else (t, t)
    stats = {
        "completed_trades": len(t), "wins": len(winners), "losses": len(losers),
        "win_rate": len(winners) / len(t) if len(t) else None,
        "mean_win": float(winners.ret_net.mean()) if len(winners) else None,
        "mean_loss": float(losers.ret_net.mean()) if len(losers) else None,
        "payoff_ratio": float(winners.ret_net.mean() / -losers.ret_net.mean()) if len(winners) and len(losers) else None,
        "profit_factor": float(winners.ret_net.sum() / -losers.ret_net.sum()) if len(winners) and len(losers) else None,
        "end_equity": float(c.Equity.iloc[-1]),
        "cagr": (c.Equity.iloc[-1] / config.initial_cash) ** (1 / years) - 1,
        "mdd": float(c.Drawdown.min()), "exposure": float(c.Invested.mean()),
        "average_hold_days": float(t.hold_days.mean()) if len(t) else None,
        "ema_entries": int((t.entry_family == "EMA").sum()) if len(t) else 0,
        "sma_entries": int((t.entry_family == "SMA").sum()) if len(t) else 0,
        "n20_exits": int((t.exit_reason == "N20_protect").sum()) if len(t) else 0,
        "trend_exits": int((t.exit_reason == "slow_death").sum()) if len(t) else 0,
        "bear_fast_exits": int((t.exit_reason == "bear_fast_SMA10_below_SMA20").sum()) if len(t) else 0,
        "active_position_at_end": position is not None,
        "pending_order_at_end": pending,
    }
    return {"trades": t, "equity": c, "stats": stats,
            "open_position": position, "pending_order": pending,
            "regime": regime, "mode": mode}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "research/results/hybrid_2026_11_09")
    args = parser.parse_args()
    data = load_adjusted_csv(args.csv)
    data = data[data.Date <= pd.Timestamp(args.end)].copy().reset_index(drop=True)
    if data.empty or str(data.Date.iloc[-1].date()) != args.end:
        raise ValueError("Missing final requested complete trading day")
    sma = validate_sma_against_frozen(data, Config())
    ema = replay_signals(ema_signals(data))
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_results = [("SMA_baseline", sma), ("EMA_baseline", ema)]
    for regime in REGIMES:
        for mode in ("locked", "bear_fast"):
            name = f"Hybrid_{regime}_{mode}"
            r = hybrid_backtest(data, regime, mode)
            all_results.append((name, r))
            r["trades"].to_csv(args.outdir / f"{name}_trades.csv", index=False)
    summary = []
    for name, r in all_results:
        summary.append({"strategy": name, "data_asof": args.end,
                        **{k: v for k, v in r["stats"].items()
                           if k != "pending_order_at_end"}})
    pd.DataFrame(summary).to_csv(args.outdir / "summary.csv", index=False)
    frames = []
    for name, r in all_results:
        c = r["equity"].copy()
        c["strategy"] = name
        frames.append(c)
    monthly = pd.concat(frames, ignore_index=True).assign(
        month=lambda x: x.Date.str[:7]
    ).groupby(["strategy", "month"], sort=True).tail(1)[
        ["strategy", "Date", "Equity"]
    ].copy()
    monthly["Equity"] = monthly.Equity.round(2)
    monthly.pivot(index="Date", columns="strategy", values="Equity").reset_index().to_csv(
        args.outdir / "monthly_equity.csv", index=False)
    periods = {"dotcom_2000_2003": ("2000-01-03", "2003-12-31"),
               "gfc_2007_2009": ("2007-01-01", "2009-12-31"),
               "pandemic_2020_2021": ("2020-01-01", "2021-12-31"),
               "rate_bear_2022": ("2022-01-01", "2022-12-30"),
               "recent_2023_2026": ("2023-01-01", args.end)}
    slices = []
    for name, r in all_results:
        c = r["equity"]
        for pname, (start, end) in periods.items():
            subset = c[(c.Date >= start) & (c.Date <= end)]
            if len(subset) < 2:
                continue
            base_equity = float(subset.Equity.iloc[0])
            segment_dd = subset.Equity / subset.Equity.cummax() - 1
            slices.append({"strategy": name, "segment": pname,
                           "start": str(subset.Date.iloc[0]),
                           "end": str(subset.Date.iloc[-1]),
                           "return": float(subset.Equity.iloc[-1] / base_equity - 1),
                           "segment_mdd": float(segment_dd.min())})
    pd.DataFrame(slices).to_csv(args.outdir / "period_slices.csv", index=False)
    draw_svg(all_results, ROOT / "assets/a2_hybrid_equity_drawdown_2000_2026.svg")
    print(pd.DataFrame(summary).to_string(index=False))


def draw_svg(all_results: list[tuple[str, dict]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    selected = {"SMA_baseline", "EMA_baseline", "Hybrid_weekly_slope_locked",
                "Hybrid_weekly_slope_bear_fast", "Hybrid_monthly_close_slope_locked"}
    colors = {"SMA_baseline": "#2455aa", "EMA_baseline": "#da6b21",
              "Hybrid_weekly_slope_locked": "#7661bf",
              "Hybrid_weekly_slope_bear_fast": "#20856c",
              "Hybrid_monthly_close_slope_locked": "#b04378"}
    labels = {"SMA_baseline": "SMA only", "EMA_baseline": "EMA only",
              "Hybrid_weekly_slope_locked": "Hybrid: weekly slope / locked",
              "Hybrid_weekly_slope_bear_fast": "Hybrid: weekly slope / bear fast exit",
              "Hybrid_monthly_close_slope_locked": "Hybrid: monthly close+slope / locked"}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.5]})
    for name, r in all_results:
        if name not in selected:
            continue
        c = r["equity"]
        shown = c.groupby(pd.to_datetime(c.Date).dt.to_period("M")).tail(1)
        dates = pd.to_datetime(shown.Date)
        ax.plot(dates, shown.Equity, label=labels[name], color=colors[name], linewidth=1.4)
        ax2.plot(dates, shown.Drawdown * 100, color=colors[name], linewidth=1.1)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value (USD, log scale)")
    ax.set_title("QQQ A2 | SMA, EMA and regime-aware hybrid | 2000-2026")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.18)
    ax2.set_ylabel("Portfolio drawdown (%)")
    ax2.set_xlabel("Year")
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(alpha=.18)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Date": "2026-11-10"})
    plt.close(fig)


if __name__ == "__main__":
    main()
