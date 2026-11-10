"""Frozen A2-N20 + cash-carry scenarios, and README equity/drawdown SVG.

This is NOT a backtest of SGOV or of buying a specific Treasury bill. It
models frictionless, daily-liquid USD cash earning the DGS3MO constant-maturity
yield, with an optional flat annualized haircut, while preserving every A2
trade, position day, and fee. FRED DGS3MO is a historical benchmark quote,
not a guaranteed future account deposit rate.

Usage (after `pip install -e '.[charts]'`):
  python research/experiments/a2_cash_yield_backtest.py \
    --old sources/qqq_jtang.csv --new sources/qqq_james.csv \
    --yield-csv sources/dgs3mo.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from qqq_a2.engine import Config, combine_rebased_sources, load_adjusted_csv, run_backtest
from qqq_a2.cash_yield import (align_previous_quote, append_rate_quotes, cash_overlay,
                               equity_stats, read_yield_csv)

BASE_LABEL = "A2_0pct_cash"
NET_LABEL = "A2_DGS3MO_minus_15bp"


def simulation(data: pd.DataFrame, rate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result = run_backtest(data, Config())
    baseline = result["equity"].copy().reset_index(drop=True)
    quote = align_previous_quote(baseline, rate)
    curves = pd.DataFrame({"Date": pd.to_datetime(baseline.Date),
                           "Invested": baseline.Invested.astype(bool),
                           "DGS3MO_pct": quote,
                           BASE_LABEL: baseline.Equity.astype(float)})
    for name, y in (("A2_fixed_2pct", 2.0),
                    ("A2_fixed_3pct", 3.0),
                    ("A2_fixed_4pct", 4.0),
                    ("A2_DGS3MO_gross", quote),
                    (NET_LABEL, quote - 0.15)):
        curves[name] = cash_overlay(baseline, y)
    # Important regression: 0%-cash portfolio must exactly match frozen A2.
    np.testing.assert_allclose(cash_overlay(baseline, 0.0), baseline.Equity,
                               rtol=1e-12, atol=1e-5)

    metrics = []
    for name in (BASE_LABEL, "A2_fixed_2pct", "A2_fixed_3pct",
                 "A2_fixed_4pct", "A2_DGS3MO_gross", NET_LABEL):
        m = equity_stats(curves[name], curves.Date)
        m["scenario"] = name
        m["equity_uplift_vs_cash0"] = m["end_equity"] - float(curves[BASE_LABEL].iloc[-1])
        m["equity_uplift_pct"] = m["equity_uplift_vs_cash0"] / float(curves[BASE_LABEL].iloc[-1])
        metrics.append(m)
    mdf = pd.DataFrame(metrics)[["scenario", "end_equity", "equity_uplift_vs_cash0",
                                 "equity_uplift_pct", "cagr", "mdd"]]

    gaps = curves.Date.diff().dt.days.fillna(0)
    last_flat = ~curves.Invested.shift(1, fill_value=False)
    meta = {"trades": len(result["trades"]),
            "days_in_cash": int((gaps * last_flat).sum()),
            "days_total": int(gaps.sum()),
            "time_in_cash_pct": float((gaps * last_flat).sum() / gaps.sum()),
            "mean_dgs3mo_when_flat": float((quote.shift(1)[last_flat & (gaps > 0)]
                                            * gaps[last_flat & (gaps > 0)]).sum()
                                           / gaps[last_flat & (gaps > 0)].sum()),
            "start_date": str(curves.Date.iloc[0].date()),
            "end_date": str(curves.Date.iloc[-1].date()),
            "daily_quote_series": "FRED DGS3MO, percent per year, prior known date",
            "net_haircut_bps": 15}
    return curves, mdf, meta


def plot_equity(curves: pd.DataFrame, metrics: pd.DataFrame,
                out: Path, preview: Path | None = None) -> None:
    cjk_files = [p for p in font_manager.findSystemFonts()
                 if "NotoSansCJK" in Path(p).name or "NotoSansSC" in Path(p).name]
    font = "DejaVu Sans"
    if cjk_files:
        preferred = sorted(cjk_files, key=lambda p: ("Regular" not in p, p))[0]
        font_manager.fontManager.addfont(preferred)
        font = font_manager.FontProperties(fname=preferred).get_name()
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"],
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    # Monthly end-of-month points avoid an unreadably dense SVG while retaining
    # the exact DAILY account-level backtest for all summary statistics.
    months = curves.groupby(curves.Date.dt.to_period("M")).tail(1).copy()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(17, 8.8),
                                      gridspec_kw={"height_ratios": [2.45, 1.0]},
                                      sharex=True)
    fig.patch.set_facecolor("white")
    for ax in (top, bottom):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e5eaef", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d9e2ea")
        ax.tick_params(axis="both", labelsize=11, length=0, pad=9, colors="#34495d")
    base, net = BASE_LABEL, NET_LABEL
    top.plot(months.Date, months[base], color="#98a6b3", linewidth=2.3,
             label="A2｜空仓现金 0%", zorder=2)
    top.plot(months.Date, months[net], color="#167c9d", linewidth=2.8,
             label="A2｜空仓按动态 3个月美债收益率 −0.15%/年", zorder=3)
    top.set_ylabel("账户权益（USD）", fontsize=13, color="#34495d", labelpad=14)
    top.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))
    top.legend(loc="upper left", frameon=False, fontsize=12)
    for name, color, style in ((base, "#98a6b3", "--"), (net, "#167c9d", "-")):
        d = curves[name] / curves[name].cummax() - 1
        monthly_dd = d.groupby(curves.Date.dt.to_period("M")).last()
        bottom.plot(months.Date, monthly_dd.to_numpy() * 100,
                    color=color, linestyle=style, linewidth=1.7)
    bottom.set_ylabel("回撤（%）", fontsize=13, color="#34495d", labelpad=14)
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bottom.set_xlim(curves.Date.iloc[0], curves.Date.iloc[-1])

    a = metrics.set_index("scenario")
    b, n = a.loc[base], a.loc[net]
    fig.suptitle("A2-N20｜空仓期间的短债收益能改善多少？",
                 x=0.09, y=0.975, ha="left", fontsize=26,
                 fontweight="bold", color="#26384b")
    first_date = str(curves.Date.iloc[0].date())
    last_date = str(curves.Date.iloc[-1].date())
    fig.text(0.09, 0.915,
             f"{first_date} — {last_date}    初始 $100,000    |    0% 现金期末 ${b.end_equity:,.0f}    →    浮动短债净代理 ${n.end_equity:,.0f}",
             fontsize=14, color="#52687a")
    fig.text(0.09, 0.872,
             f"CAGR {b.cagr:.2%} → {n.cagr:.2%}    |    日度最大回撤 {b.mdd:.2%} → {n.mdd:.2%}    |    策略买卖信号完全不变",
             fontsize=13, color="#52687a")
    fig.text(0.09, 0.024,
             "图线按月末压缩；回测统计用每日权益。DGS3MO 是美债历史参考报价，非可保证成交利率；未模拟真实 ETF 的净值波动、价差、税收和汇率。",
             fontsize=11, color="#718294")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.115, hspace=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", metadata={
        "Title": "QQQ A2-N20 cash yield sensitivity",
        "Description": "Frozen A2-N20 with zero-cash vs historical FRED DGS3MO less 15bp annual haircut; daily simulation, monthly chart rendering.",
    })
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, format="png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 cash yield comparison with historical DGS3MO")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Locally authorized daily QQQ OHLCV CSV")
    source.add_argument("--old", help="Locally authorized older QQQ data")
    parser.add_argument("--new", help="Locally authorized newer QQQ data")
    parser.add_argument("--yield-csv", required=True, help="Locally supplied FRED DGS3MO daily CSV")
    parser.add_argument("--yield-csv-extra", help="Optional later, official DGS3MO quotes extending archived CSV")
    parser.add_argument("--end", default="2026-07-13", help="Inclusive cutoff date")
    parser.add_argument("--outdir", type=Path, default=Path("research/results"))
    parser.add_argument("--chart", type=Path, default=Path("assets/a2_cash_yield_equity.svg"))
    parser.add_argument("--preview", type=Path, help="Optional PNG for offline visual QA")
    args = parser.parse_args()
    if bool(args.old) != bool(args.new):
        parser.error("--old and --new required together")
    data = (load_adjusted_csv(args.csv) if args.csv else
            combine_rebased_sources(args.old, args.new, end_date=args.end))
    if args.csv:
        data = data[data.Date <= pd.Timestamp(args.end)].copy()
    yields = read_yield_csv(args.yield_csv)
    if args.yield_csv_extra:
        yields = append_rate_quotes(yields, read_yield_csv(args.yield_csv_extra))
    curve, metrics, meta = simulation(data, yields)
    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.outdir / "a2_cash_yield_scenarios.csv", index=False)
    # Only derived month-end account statistics, never underlying OHLCV rates.
    month_end = curve.groupby(curve.Date.dt.to_period("M")).tail(1).drop(columns=["Invested", "DGS3MO_pct"])
    month_end.to_csv(args.outdir / "a2_cash_yield_monthly_equity.csv", index=False)
    plot_equity(curve, metrics, args.chart, args.preview)
    print(meta)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
