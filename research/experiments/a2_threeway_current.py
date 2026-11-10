"""Reproduce three current QQQ references without changing frozen A2 V1.0.

  1. SMA A2-N20 V1.0, untouched src/qqq_a2/engine.py
  2. EMA A2-N20 with 7% ALWAYS-ON NET-LOSS CLOSE stop / next-open fill
  3. QQQ buy-and-hold from first adjusted open (unsold at the end)

Both active trading variants buy on fresh daily 20/60 golden crosses,
use the first-20-day 10/20 death cross below entry as N20 protection,
and exit on fresh daily 20/60 death crosses. Cash earns 0% in this grid;
each executed side pays 0.05%. All results are before tax and slippage.

The script reads already-authorized local adjusted OHLCV and writes ONLY
derived account and transaction results; no raw market history is exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_comparison import ema_signals
from a2_ema_hard_stop8 import replay_guarded
from qqq_a2.benchmarks import buy_and_hold
from qqq_a2.engine import Config, load_adjusted_csv, run_backtest


def compare(data: pd.DataFrame, *, end: str, config: Config = Config()) -> dict:
    d = data.loc[pd.to_datetime(data.Date) <= pd.Timestamp(end)].copy()
    d = d.sort_values("Date").reset_index(drop=True)
    if d.empty or str(d.Date.iloc[-1].date()) != end:
        raise ValueError(f"Missing completed reference cutoff: {end}")
    start_sessions = d.loc[d.Date >= pd.Timestamp(config.start_date)]
    if start_sessions.empty or str(start_sessions.Date.iloc[0].date()) != config.start_date:
        raise ValueError("Missing expected first complete session")

    runs = {
        "SMA_A2_N20": run_backtest(d, config),
        "EMA_A2_N20_close7": replay_guarded(
            ema_signals(d), config=config, stop_loss=.07, mode="close_next_open"),
        "QQQ_buy_hold": buy_and_hold(d, config),
    }
    curves = None
    stats_rows = []
    for name, run in runs.items():
        e = run["equity"].copy()
        e.Date = pd.to_datetime(e.Date)
        if not e.Date.is_unique:
            raise AssertionError(f"Duplicate daily marks in {name}")
        marks = e[["Date", "Equity", "Drawdown"]].rename(
            columns={"Equity": f"Equity_{name}", "Drawdown": f"Drawdown_{name}"})
        curves = marks if curves is None else curves.merge(
            marks, on="Date", how="inner", validate="one_to_one")

        info = run["stats"]
        trades = run.get("trades")
        completed = len(trades) if trades is not None else None
        exposure = (float(e.Invested.mean()) if "Invested" in e else 1.0)
        stats_rows.append({
            "strategy": name, "data_asof": end,
            "end_equity": float(info["end_equity"]),
            "total_return": float(info["end_equity"] / config.initial_cash - 1),
            "cagr": float(info["cagr"]), "mdd": float(info["mdd"]),
            "completed_trades": completed,
            "wins": int((trades.ret_net > 0).sum()) if trades is not None else None,
            "losses": int((trades.ret_net < 0).sum()) if trades is not None else None,
            "win_rate": info.get("win_rate"),
            "payoff_ratio": info.get("payoff_ratio"),
            "profit_factor": info.get("profit_factor"),
            "market_exposure": exposure,
            "worst_completed_trade": float(trades.ret_net.min()) if trades is not None else None,
            "position_at_end": bool(info.get("active_position_at_end", True)),
        })
    if curves is None or len(curves) != len(start_sessions):
        raise AssertionError("Mismatched trading dates between three strategies")

    out = pd.DataFrame(stats_rows)
    if end == "2026-11-09" and config == Config():
        expected = {
            "SMA_A2_N20": (436162.7196, .2029, 60),
            "EMA_A2_N20_close7": (658226.5727, .259975, 54),
            "QQQ_buy_hold": (838629., .8296, None),
        }
        for name, (terminal, abs_mdd, count) in expected.items():
            row = out.set_index("strategy").loc[name]
            assert abs(float(row.end_equity) - terminal) < 20, (name, "equity drift")
            assert abs(float(row.mdd) + abs_mdd) < .0002, (name, "MDD drift")
            if count is not None:
                assert int(row.completed_trades) == count, (name, "trade count drift")
    return {"summary": out, "equity": curves, "runs": runs,
            "periods": period_metrics(curves, config.initial_cash)}


def period_metrics(curves: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """Same-capital, same-calendar subperiod sensitivity; no new capital added."""
    eras = [
        ("2000-2003 dotcom", "2000-01-03", "2003-12-31"),
        ("2004-2009 GFC", "2004-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2021", "2016-01-01", "2021-12-31"),
        ("2022-2026", "2022-01-01", "2026-11-09"),
    ]
    rows = []
    for label, first, last in eras:
        earlier = curves.Date < pd.Timestamp(first)
        within = curves.loc[(curves.Date >= pd.Timestamp(first)) &
                            (curves.Date <= pd.Timestamp(last))]
        if within.empty:
            continue
        for name in ("SMA_A2_N20", "EMA_A2_N20_close7", "QQQ_buy_hold"):
            column = "Equity_" + name
            prior = float(curves.loc[earlier, column].iloc[-1]) if earlier.any() else initial_cash
            period = pd.concat([pd.Series([prior]), within[column].reset_index(drop=True)],
                               ignore_index=True)
            rows.append({"period": label, "strategy": name,
                         "first_date": str(within.Date.iloc[0].date()),
                         "last_date": str(within.Date.iloc[-1].date()),
                         "period_return": float(period.iloc[-1] / prior - 1),
                         "period_mdd": float((period / period.cummax() - 1).min())})
    return pd.DataFrame(rows)


def plot(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = {"SMA_A2_N20": "#3282a8", "EMA_A2_N20_close7": "#d77b38",
              "QQQ_buy_hold": "#8665a8"}
    labels = {"SMA_A2_N20": "SMA A2-N20 (frozen)",
              "EMA_A2_N20_close7": "EMA A2-N20 + 7% close stop (selected)",
              "QQQ_buy_hold": "QQQ Buy & Hold"}
    e = result["equity"]
    weekly = e.groupby(e.Date.dt.to_period("W-FRI")).tail(1)
    fig, (top, mid, bottom) = plt.subplots(3, 1, sharex=True, figsize=(14.5, 9.8),
                                           gridspec_kw={"height_ratios": [2.4, 1, .85]})
    fig.patch.set_facecolor("white")
    for ax in (top, mid, bottom):
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for name in labels:
        top.plot(weekly.Date, weekly[f"Equity_{name}"],
                 label=labels[name], color=colors[name], lw=1.8)
    top.set_yscale("log")
    top.set_ylabel("USD account equity (log)")
    top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y/1000:,.0f}k"))
    top.legend(loc="upper left", fontsize=9, frameon=False)
    for name in ("SMA_A2_N20", "EMA_A2_N20_close7"):
        mid.plot(weekly.Date, weekly[f"Drawdown_{name}"] * 100,
                 label=labels[name], color=colors[name], lw=1.3)
    mid.set_ylabel("Trend-system DD (%)")
    mid.legend(loc="lower left", fontsize=9, frameon=False)
    bottom.plot(weekly.Date, weekly.Drawdown_QQQ_buy_hold * 100,
                label="QQQ Buy & Hold", color=colors["QQQ_buy_hold"], lw=1.4)
    bottom.set_ylabel("Buy & Hold DD (%)")
    bottom.set_xlabel("Calendar year")
    bottom.legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("QQQ 2000-2026: SMA N20 vs EMA N20 + 7% loss stop vs Buy & Hold",
                 fontsize=17, y=.985)
    fig.text(.08, .012,
             "Weekly points for legibility; CAGR & MDD use every daily mark. $100k initial, fees 0.05%/side, cash 0%, pre-tax.",
             fontsize=9, color="#555")
    fig.tight_layout(rect=[.01, .035, 1, .975])
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, format="svg", metadata={"Title": "QQQ three-way trend strategy comparison",
                                               "Date": "2026-11-10"})
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--end", default="2026-11-09")
    p.add_argument("--outdir", type=Path,
                   default=ROOT / "research/results/threeway_current_2026_11_09")
    p.add_argument("--chart", type=Path,
                   default=ROOT / "assets/a2_threeway_sma_ema7_buyhold_2000_2026.svg")
    args = p.parse_args()
    data = load_adjusted_csv(args.csv)
    result = compare(data, end=args.end)
    args.outdir.mkdir(exist_ok=True, parents=True)
    result["summary"].to_csv(args.outdir / "summary.csv", index=False)
    e = result["equity"]
    month = e.groupby(e.Date.dt.to_period("M")).tail(1).copy()
    month.Date = month.Date.dt.strftime("%Y-%m-%d")
    month.to_csv(args.outdir / "monthly_equity.csv", index=False)
    result["periods"].to_csv(args.outdir / "period_metrics.csv", index=False)
    for name in ("SMA_A2_N20", "EMA_A2_N20_close7"):
        result["runs"][name]["trades"].to_csv(args.outdir / f"{name}_trades.csv", index=False)
    metadata = {
        "data_asof": args.end, "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "simulation_start": "2000-01-03", "initial_cash_usd": 100000,
        "single_side_fee": .0005, "cash_interest": 0,
        "mark_price": "same adjustment base OHLC, daily adjusted close",
        "fills": "signals at complete daily close, orders at next regular open",
        "stop7": "net round-trip return reaches -7% at daily close; sell next open; gaps can exceed 7%",
        "dividends": "adjusted total-return approximation, gross of withholding",
        "buy_hold": "first adjusted open, initial fee paid, no sale/exit fee at end",
        "taxes_and_slippage": "excluded", "lookahead": "none",
    }
    (args.outdir / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(result, args.chart)
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
