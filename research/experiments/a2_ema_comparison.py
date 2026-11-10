"""Isolated EMA10/20/60/200 alternative to frozen SMA A2-N20 V1.0.

Use exactly the same adjusted OHLC, close-signal/next-open fills, N20 protection,
fees, all-in sizing and no-reentry rule. No changes to src/qqq_a2/engine.py.

    python research/experiments/a2_ema_comparison.py \
        --csv sources/qqq_merged_updated_local.csv --end 2026-11-09

Raw historical market bars are deliberately not emitted or redistributed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qqq_a2.engine import Config, compute_signals, load_adjusted_csv, run_backtest


def ema_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Standard recursive EMA, alpha=2/(span+1), seeded at first observed close.

    ``min_periods=span`` hides incomplete indicators. This is not the
    SMA-seeded EMA convention; both conventions are valid but different.
    """
    df = data.copy().sort_values("Date").reset_index(drop=True)
    for period in (10, 20, 60, 200):
        df[f"ma{period}"] = df.adj_close.ewm(
            span=period, adjust=False, min_periods=period
        ).mean()
    df["gold"] = (df.ma20 > df.ma60) & (df.ma20.shift(1) <= df.ma60.shift(1))
    df["death"] = (df.ma20 < df.ma60) & (df.ma20.shift(1) >= df.ma60.shift(1))
    df["protect_cross"] = (df.ma10 < df.ma20) & (df.ma10.shift(1) >= df.ma20.shift(1))
    return df


def replay_signals(df: pd.DataFrame, config: Config = Config()) -> dict:
    """Research-only event replay with the same mechanics as frozen A2.

    This deliberately takes precomputed signals rather than monkeypatching
    production globals. SMA replay is asserted equal to the frozen engine.
    """
    cash, shares = float(config.initial_cash), 0.0
    position, pending = None, None
    trades, curve = [], []

    for i, bar in df.iterrows():
        if bar.Date < pd.Timestamp(config.start_date):
            continue
        if pending is not None:
            if pending["side"] == "buy":
                if shares:
                    raise AssertionError("Buy into existing position")
                shares = cash / (float(bar.adj_open) * (1 + config.fee_per_side))
                position = {
                    "entry_signal_date": pending["signal_date"],
                    "entry_date": str(bar.Date.date()),
                    "entry_index": i,
                    "entry_price": float(bar.adj_open),
                    "entry_cost": cash,
                    "max_close": float(bar.adj_close),
                    "min_close": float(bar.adj_close),
                }
                cash = 0.0
            else:
                if position is None or not shares:
                    raise AssertionError("Sell without position")
                proceeds = shares * float(bar.adj_open) * (1 - config.fee_per_side)
                trades.append({
                    "id": len(trades) + 1,
                    "entry_signal_date": position["entry_signal_date"],
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "exit_signal_date": pending["signal_date"],
                    "exit_date": str(bar.Date.date()),
                    "exit_price": float(bar.adj_open),
                    "exit_reason": pending["reason"],
                    "hold_days": int(i - position["entry_index"]),
                    "ret_net": float(proceeds / position["entry_cost"] - 1),
                    "mfe_close": position["max_close"] / position["entry_price"] - 1,
                    "mae_close": position["min_close"] / position["entry_price"] - 1,
                    "equity_after_trade": float(proceeds),
                })
                cash, shares, position = float(proceeds), 0.0, None
            pending = None

        if shares:
            position["max_close"] = max(position["max_close"], float(bar.adj_close))
            position["min_close"] = min(position["min_close"], float(bar.adj_close))
            age = int(i - position["entry_index"])
            if bool(bar.death):
                pending = {"side": "sell", "signal_date": str(bar.Date.date()),
                           "reason": "MA20/60 death cross"}
            elif (1 <= age <= config.n20_days and bool(bar.protect_cross)
                  and float(bar.adj_close) < position["entry_price"]):
                pending = {"side": "sell", "signal_date": str(bar.Date.date()),
                           "reason": "N20 protection"}
        elif bool(bar.gold):
            pending = {"side": "buy", "signal_date": str(bar.Date.date()),
                       "reason": "FRESH_GOLD_CROSS"}

        curve.append({"Date": str(bar.Date.date()),
                      "Equity": cash + shares * float(bar.adj_close),
                      "Invested": bool(shares),
                      "Pending": pending["side"] if pending else ""})

    t, e = pd.DataFrame(trades), pd.DataFrame(curve)
    e["Drawdown"] = e.Equity / e.Equity.cummax() - 1
    years = (pd.Timestamp(e.Date.iloc[-1]) - pd.Timestamp(config.start_date)).days / 365.25
    winners, losers = t[t.ret_net > 0], t[t.ret_net < 0]
    stats = {
        "completed_trades": len(t), "wins": len(winners), "losses": len(losers),
        "win_rate": len(winners) / len(t) if len(t) else None,
        "mean_win": float(winners.ret_net.mean()) if len(winners) else None,
        "mean_loss": float(losers.ret_net.mean()) if len(losers) else None,
        "payoff_ratio": (float(winners.ret_net.mean() / abs(losers.ret_net.mean()))
                         if len(winners) and len(losers) else None),
        "profit_factor": (float(winners.ret_net.sum() / -losers.ret_net.sum())
                          if len(winners) and len(losers) else None),
        "mdd": float(e.Drawdown.min()), "end_equity": float(e.Equity.iloc[-1]),
        "cagr": float((e.Equity.iloc[-1] / config.initial_cash) ** (1 / years) - 1),
        "exposure": float(e.Invested.mean()),
        "average_hold_days": float(t.hold_days.mean()) if len(t) else None,
        "n20_exits": int((t.exit_reason == "N20 protection").sum()),
        "trend_exits": int((t.exit_reason == "MA20/60 death cross").sum()),
        "active_position_at_end": position is not None,
        "pending_order_at_end": pending,
    }
    return {"trades": t, "equity": e, "stats": stats,
            "open_position": position, "pending_order": pending}


def validate_sma_against_frozen(data: pd.DataFrame, config: Config) -> dict:
    import pandas.testing as pdt

    frozen = run_backtest(data, config)
    replay = replay_signals(compute_signals(data), config)
    pdt.assert_frame_equal(replay["trades"], frozen["trades"], check_exact=False,
                           rtol=1e-12, atol=1e-9)
    pdt.assert_series_equal(replay["equity"].Equity, frozen["equity"].Equity,
                            check_exact=False, rtol=1e-12, atol=1e-8)
    for k in ("completed_trades", "wins", "losses", "win_rate", "payoff_ratio",
              "profit_factor", "mdd", "end_equity", "cagr", "active_position_at_end"):
        a, b = replay["stats"][k], frozen["stats"][k]
        if a is None or b is None:
            assert a is b, k
            continue
        assert (a == b) if isinstance(a, (bool, int)) else abs(a - b) < 1e-10, k
    return replay


def draw_svg(sma: dict, ema: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.5]})
    colors = {"SMA A2-N20": "#275dad", "EMA A2-N20": "#da6b21"}
    for label, result in (("SMA A2-N20", sma), ("EMA A2-N20", ema)):
        eq = result["equity"]
        # Weekly last-session marks keep the GitHub-rendered SVG legible and
        # compact; max drawdown statistics still use every DAILY observation.
        weeks = pd.to_datetime(eq.Date).dt.to_period("W-FRI")
        shown = eq.groupby(weeks, sort=True).tail(1)
        dates = pd.to_datetime(shown.Date)
        ax.plot(dates, shown.Equity, label=label, color=colors[label], linewidth=1.7)
        ax2.plot(dates, shown.Drawdown * 100, label=label, color=colors[label], linewidth=1.15)
        trough = eq.loc[eq.Drawdown.idxmin()]
        ax2.scatter(pd.Timestamp(trough.Date), trough.Drawdown * 100,
                    color=colors[label], marker="x", s=45, zorder=5)
    ax.set_yscale("log")
    last_day = ema["equity"].Date.iloc[-1]
    ax.set_title(f"QQQ A2-N20  |  SMA vs EMA  |  2000-01-03 to {last_day}", fontsize=14)
    ax.set_ylabel("Portfolio value (USD, log scale)")
    ax.grid(alpha=.20)
    ax.legend(loc="upper left")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Calendar year")
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(pd.Timestamp("2000-01-03"), pd.Timestamp(last_day))
    ax2.grid(alpha=.20)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Date": "2026-11-10"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True,
                        help="User-provided, licensed QQQ daily price data")
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "research/results/ema_2026_11_09")
    parser.add_argument("--chart", type=Path,
                        default=ROOT / "assets/a2_ema_vs_sma_equity_2000_2026.svg")
    args = parser.parse_args()
    data = load_adjusted_csv(args.csv)
    data = data[data.Date <= pd.Timestamp(args.end)].copy().reset_index(drop=True)
    if not len(data) or data.Date.iloc[-1].date() != pd.Timestamp(args.end).date():
        raise ValueError("The requested final session is missing: no extrapolation")
    config = Config()
    sma = validate_sma_against_frozen(data, config)
    ema = replay_signals(ema_signals(data), config)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, result in (("SMA_A2_N20", sma), ("EMA_A2_N20", ema)):
        summary.append({"strategy": name, "data_asof": args.end,
                        **{k: v for k, v in result["stats"].items()
                           if k != "pending_order_at_end"}})
        result["trades"].to_csv(args.outdir / f"{name}_trades.csv", index=False)
    pd.DataFrame(summary).to_csv(args.outdir / "summary.csv", index=False)

    frames = []
    for name, result in (("SMA_A2_N20", sma), ("EMA_A2_N20", ema)):
        temp = result["equity"].copy()
        temp["strategy"] = name
        frames.append(temp)
    pd.concat(frames).assign(YearMonth=lambda x: x.Date.str[:7]).groupby(
        ["strategy", "YearMonth"], sort=True
    ).tail(1).to_csv(args.outdir / "monthly_equity.csv", index=False)

    draw_svg(sma, ema, args.chart)
    for row in summary:
        print(row)
    print(f"Source bars: {len(data)}, from {data.Date.iloc[0].date()} to {data.Date.iloc[-1].date()}")
    print(f"Saved results to {args.outdir} and chart to {args.chart}")


if __name__ == "__main__":
    main()
