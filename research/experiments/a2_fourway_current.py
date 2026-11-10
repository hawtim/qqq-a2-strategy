"""Publish four comparable QQQ benchmarks, preserving the frozen three-way audit.

Run with legally acquired adjusted daily QQQ bars:
    python research/experiments/a2_fourway_current.py \
      --csv sources/qqq_merged_updated_local.csv --end 2026-11-09

Derived summaries, complete trade ledgers, change attribution and SVG charts
are publishable. Raw vendor OHLCV is neither copied nor emitted.
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
sys.path.insert(0, str(ROOT / "research/experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from a2_threeway_current import compare as compare_three
from a2_ema_dynamic_n20 import completed_month_signals, replay_dynamic
from qqq_a2.engine import Config, load_adjusted_csv

NAME = "EMA_A2_DynamicN20_monthEMA5_close7"
ORDER = ["SMA_A2_N20", "EMA_A2_N20_close7", NAME, "QQQ_buy_hold"]
PERIODS = [
    ("2000-2003 dotcom", "2000-01-03", "2003-12-31"),
    ("2004-2009 GFC", "2004-01-01", "2009-12-31"),
    ("2010-2015", "2010-01-01", "2015-12-31"),
    ("2016-2021", "2016-01-01", "2021-12-31"),
    ("2022-2026", "2022-01-01", "2026-11-09"),
]


def period_metrics_four(curves: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    rows = []
    for label, first, last in PERIODS:
        before = curves.Date < pd.Timestamp(first)
        within = curves.loc[curves.Date.between(first, last)]
        if within.empty:
            continue
        for name in ORDER:
            field = "Equity_" + name
            start = float(curves.loc[before, field].iloc[-1]) if before.any() else initial_cash
            values = pd.concat([pd.Series([start]), within[field].reset_index(drop=True)],
                               ignore_index=True)
            rows.append({
                "period": label, "strategy": name,
                "first_date": str(within.Date.iloc[0].date()),
                "last_date": str(within.Date.iloc[-1].date()),
                "period_return": float(values.iloc[-1] / start - 1),
                "period_mdd": float((values / values.cummax() - 1).min()),
            })
    return pd.DataFrame(rows)


def compare_four(data: pd.DataFrame, *, end: str, config: Config = Config()) -> dict:
    baseline = compare_three(data, end=end, config=config)
    d = data.loc[pd.to_datetime(data.Date) <= pd.Timestamp(end)].copy()
    signals = completed_month_signals(d)
    candidate = replay_dynamic(signals, config=config)
    controls = {
        "fast_entry_old_fixed_n20": replay_dynamic(
            signals, config=config, reset_n20_at_confirmation=False),
        "dynamic_n20_two_closes": replay_dynamic(
            signals, config=config, n20_confirm_closes=2),
        "dynamic_n20_three_closes": replay_dynamic(
            signals, config=config, n20_confirm_closes=3),
    }

    previous = baseline["summary"].copy()
    t, e, st = candidate["trades"], candidate["equity"].copy(), candidate["stats"]
    new_row = {
        "strategy": NAME, "data_asof": end,
        "end_equity": st["end_equity"],
        "total_return": st["end_equity"] / config.initial_cash - 1,
        "cagr": st["cagr"], "mdd": st["mdd"],
        "completed_trades": len(t),
        "wins": int((t.ret_net > 0).sum()),
        "losses": int((t.ret_net < 0).sum()),
        "win_rate": st["win_rate"], "payoff_ratio": st["payoff_ratio"],
        "profit_factor": st["profit_factor"],
        "market_exposure": st["exposure"],
        "worst_completed_trade": float(t.ret_net.min()) if len(t) else None,
        "position_at_end": st["active_position_at_end"],
    }
    summary = pd.concat([previous, pd.DataFrame([new_row])], ignore_index=True)
    summary = summary.set_index("strategy").loc[ORDER].reset_index()
    e.Date = pd.to_datetime(e.Date)
    marks = e[["Date", "Equity", "Drawdown"]].rename(
        columns={"Equity": "Equity_" + NAME, "Drawdown": "Drawdown_" + NAME})
    combined = baseline["equity"].merge(marks, on="Date", validate="one_to_one")
    if len(combined) != len(baseline["equity"]):
        raise AssertionError("Different daily marks")
    if end == "2026-11-09" and config == Config():
        if abs(st["end_equity"] - 742702.78) > 5 or abs(st["mdd"] + .259975) > .0001:
            raise AssertionError("Dynamic EMA N20 canonical result drift")
        if len(t) != 59:
            raise AssertionError("Dynamic EMA N20 trade-count drift")
        if abs(controls["fast_entry_old_fixed_n20"]["stats"]["end_equity"] - 681031.88) > 5:
            raise AssertionError("Pre-dynamic fast re-entry control drift")
    return {
        "summary": summary, "equity": combined,
        "runs": {**baseline["runs"], NAME: candidate},
        "periods": period_metrics_four(combined, config.initial_cash),
        "controls": controls,
    }


def changed_trade_audit(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Compare *execution events*, not changing account-equity compounding.

    A genuine rule difference requires a different entry or exit event date;
    identical trade returns still produce different dollar P&L when earlier
    trades preserve more equity. Do not call those separate signal changes.
    """
    names = ["entry_date", "exit_date", "exit_reason", "ret_net"]
    old = before[names].set_index("entry_date")
    new = after[names].set_index("entry_date")
    both = old.join(new, how="outer", lsuffix="_old", rsuffix="_new")
    changed = both.loc[both.exit_date_old.ne(both.exit_date_new)].copy()
    changed.index.name = "entry_date"
    changed["saved_loss_percentage_points"] = (
        (changed.ret_net_new - changed.ret_net_old) * 100)
    return changed.reset_index()


def chart_four(result: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = {
        "SMA_A2_N20": "SMA A2-N20 (frozen)",
        "EMA_A2_N20_close7": "EMA N20 + 7% (baseline)",
        NAME: "EMA early re-entry + monthly EMA5 + dynamic N20 + 7%",
        "QQQ_buy_hold": "QQQ Buy & Hold",
    }
    colors = {
        "SMA_A2_N20": "#287aa3", "EMA_A2_N20_close7": "#c36d34",
        NAME: "#188c72", "QQQ_buy_hold": "#8b66ad",
    }
    e = result["equity"]
    week = e.groupby(e.Date.dt.to_period("W-FRI")).tail(1)
    fig, (top, mid, bottom) = plt.subplots(3, 1, sharex=True, figsize=(15, 10),
                                            gridspec_kw={"height_ratios": [2.5, 1.2, .85]})
    for ax in (top, mid, bottom):
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for name in ORDER:
        top.plot(week.Date, week["Equity_" + name], lw=1.7,
                 color=colors[name], label=labels[name])
    top.set_yscale("log")
    top.set_ylabel("USD equity (log)")
    top.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y/1000:,.0f}k"))
    top.legend(fontsize=9, frameon=False)
    for name in ORDER[:-1]:
        mid.plot(week.Date, week["Drawdown_" + name] * 100, lw=1.25,
                 color=colors[name], label=labels[name])
    mid.set_ylabel("Trend strategies DD (%)")
    mid.legend(fontsize=8, frameon=False)
    bottom.plot(week.Date, week.Drawdown_QQQ_buy_hold * 100,
                color=colors["QQQ_buy_hold"], lw=1.2)
    bottom.set_ylabel("Buy & Hold DD (%)")
    bottom.set_xlabel("Year")
    fig.suptitle("QQQ 2000-2026 | Four reproducible strategies", fontsize=17, y=.985)
    fig.text(.08, .012,
             "Daily stats; weekly display only. $100k initial, 0.05% per side, cash 0%, before tax/slippage.",
             fontsize=9, color="#555")
    fig.tight_layout(rect=[.01, .035, 1, .975])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Title": "QQQ four-strategy comparison"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "research/results/fourway_current_2026_11_09")
    parser.add_argument("--chart", type=Path,
                        default=ROOT / "assets/a2_fourway_sma_ema7_dynamic_buyhold_2000_2026.svg")
    args = parser.parse_args()
    data = load_adjusted_csv(args.csv)
    result = compare_four(data, end=args.end)
    args.outdir.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(args.outdir / "summary.csv", index=False)
    result["periods"].to_csv(args.outdir / "period_metrics.csv", index=False)
    month = result["equity"].groupby(result["equity"].Date.dt.to_period("M")).tail(1).copy()
    month.Date = month.Date.dt.strftime("%Y-%m-%d")
    month.to_csv(args.outdir / "monthly_equity.csv", index=False)
    # Include copies of all already-derived trade ledgers in this four-way
    # snapshot; original three-way files remain unchanged.
    for name in ORDER[:-1]:
        result["runs"][name]["trades"].to_csv(args.outdir / f"{name}_trades.csv", index=False)
    dynamic = result["runs"][NAME]
    controls = result["controls"]
    audit = changed_trade_audit(controls["fast_entry_old_fixed_n20"]["trades"],
                                dynamic["trades"])
    audit.to_csv(args.outdir / "dynamic_n20_changed_trades.csv", index=False)
    control_rows = []
    for label, run in [("FAST_old_N20", controls["fast_entry_old_fixed_n20"]),
                       ("DYNAMIC_N20_1_close", dynamic),
                       ("DYNAMIC_N20_2_closes", controls["dynamic_n20_two_closes"]),
                       ("DYNAMIC_N20_3_closes", controls["dynamic_n20_three_closes"])]:
        st = run["stats"]
        control_rows.append({"variant": label, "end_equity": st["end_equity"],
                             "cagr": st["cagr"], "mdd": st["mdd"],
                             "completed_trades": st["completed_trades"],
                             "worst_trade": float(run["trades"].ret_net.min())})
    pd.DataFrame(control_rows).to_csv(args.outdir / "n20_confirmation_sensitivity.csv", index=False)
    metadata = {
        "data_asof": args.end,
        "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "comparison": ORDER,
        "simulation_start": "2000-01-03", "initial_cash_usd": 100000,
        "fee_per_side": .0005, "cash_interest": 0,
        "signal_timing": "completed daily close; monthly EMA5 previous completed month only",
        "fill": "next regular trading session adjusted open",
        "fast_entry": "fresh EMA10/20 gold after ANY exit, last completed month EMA5 slope > 0",
        "normal_entry": "fresh daily EMA20/60 gold, irrespective of monthly filter",
        "dynamic_n20": "for fast entries: EMA10/20 fresh death below buy open until EMA20/60 fresh gold, then 20 further trading days; regular entries original N20",
        "exit_priority": "7% net-loss daily close, EMA20/60 death, losing N20 cross; next open",
        "published_confirmation_closes": 1,
        "slippage_and_tax": "excluded", "cash_yield": "0%",
        "external_validation": "none; all results in-sample",
        "reentry_eligibility": "any completed exit incl. 7%, EMA20/60 and N20",
        "signal_audit_changed_exits": len(audit),
    }
    (args.outdir / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chart_four(result, args.chart)
    from render_ema_dynamic_trade_charts import render_trade_charts
    render_trade_charts(data, result, ROOT / "assets")
    print(result["summary"].to_string(index=False))
    print("Changed exit executions vs fast old N20:", len(audit))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
