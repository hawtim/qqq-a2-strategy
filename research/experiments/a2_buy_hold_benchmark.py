"""Independent buy-and-hold QQQ comparator for A2's frozen/updated windows.

The operator supplies locally authorized bars. Neither raw bars nor personal
tax assumptions are published; outputs are derived curves and audit metrics.

Example:
  python research/experiments/a2_buy_hold_benchmark.py \
    --old sources/qqq_jtang.csv --new sources/qqq_james.csv \
    --csv sources/qqq_merged_updated_local.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import dates as mdates, font_manager
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import pandas as pd

from qqq_a2.benchmarks import buy_and_hold
from qqq_a2.engine import Config, combine_rebased_sources, load_adjusted_csv, run_backtest


def compare(data: pd.DataFrame, *, end: str) -> tuple[pd.DataFrame, dict]:
    clipped = data.loc[pd.to_datetime(data.Date) <= pd.Timestamp(end)].copy()
    if clipped.empty or str(pd.Timestamp(clipped.Date.iloc[-1]).date()) != end:
        raise ValueError(f"Missing completed sample cutoff: {end}")
    a2, hold = run_backtest(clipped, Config()), buy_and_hold(clipped, Config())
    a = a2["equity"].copy()
    b = hold["equity"].copy()
    a["Date"] = pd.to_datetime(a.Date)
    joint = a[["Date", "Equity", "Drawdown"]].merge(
        b[["Date", "Equity", "Drawdown"]], on="Date",
        suffixes=("_a2", "_hold"), validate="one_to_one")
    if len(joint) != len(a) or len(joint) != len(b):
        raise ValueError("Benchmark date alignment failed")
    if end == "2026-07-13":
        assert a2["stats"]["completed_trades"] == 58, "Frozen A2 altered"
        assert abs(a2["stats"]["end_equity"] - 450595.8505) < 1, "Frozen A2 equity drift"
        assert abs(a2["stats"]["mdd"] + .2028819) < .0001, "Frozen A2 MDD drift"
    if end == "2026-11-09":
        assert a2["stats"]["completed_trades"] == 60, "Updated A2 altered"
        assert abs(a2["stats"]["end_equity"] - 436162.7196) < 1, "Updated A2 equity drift"
    a2stats = {
        "completed_trades": a2["stats"]["completed_trades"],
        "end_equity": a2["stats"]["end_equity"],
        "total_return": a2["stats"]["end_equity"] / Config().initial_cash - 1,
        "cagr": a2["stats"]["cagr"],
        "mdd": a2["stats"]["mdd"],
        "active_position_at_end": a2["stats"]["active_position_at_end"],
    }
    return joint, {
        "end_date": end,
        "sessions": len(joint),
        "a2": a2stats,
        "buy_and_hold": hold["stats"],
        "assumptions": {
            "simulation_start": Config().start_date,
            "first_entry": "first regular session adjusted open",
            "initial_cash": Config().initial_cash,
            "entry_fee_per_side": Config().fee_per_side,
            "mark_to_market": "daily adjusted close, no terminal sale",
            "adjusted_price": "dividends reinvested gross total-return approximation",
            "personal_capital_gains_tax": "excluded: residency-dependent",
            "dividend_withholding_tax": "excluded: nonresident US dividend tax may apply",
            "tax_deferred_compounding": "not included; selling after cutoff is separate",
        },
    }


def plot_comparison(curves: pd.DataFrame, stats: dict, out: Path,
                    preview: Path | None = None) -> None:
    cjk = [p for p in font_manager.findSystemFonts()
           if "NotoSansCJK" in Path(p).name or "NotoSansSC" in Path(p).name]
    font = "DejaVu Sans"
    if cjk:
        preferred = sorted(cjk, key=lambda p: ("Regular" not in p, p))[0]
        font_manager.fontManager.addfont(preferred)
        font = font_manager.FontProperties(fname=preferred).get_name()
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"],
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    monthly = curves.groupby(curves.Date.dt.to_period("M")).tail(1).copy()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(17, 9.2),
                                     gridspec_kw={"height_ratios": [2.3, 1.3]}, sharex=True)
    fig.patch.set_facecolor("white")
    for ax in (top, bottom):
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e6edf2", linewidth=.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d9e3ea")
        ax.tick_params(axis="both", labelsize=11, length=0, pad=8, colors="#3d5062")
    blue, amber = "#166e98", "#da8026"
    top.plot(monthly.Date, monthly.Equity_a2, color=blue, linewidth=2.7,
             label="A2-N20｜趋势持有，空仓现金 0%")
    top.plot(monthly.Date, monthly.Equity_hold, color=amber, linewidth=2.7,
             label="QQQ｜首日买入、全程持有、分红复投代理")
    top.set_ylabel("账户权益（美元）", fontsize=13, color="#32465a", labelpad=10)
    top.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1000:,.0f}k"))
    top.legend(loc="upper left", frameon=False, fontsize=12)
    bottom.plot(monthly.Date, monthly.Drawdown_a2*100, color=blue, linewidth=1.8,
                label="A2 日度历史峰值回撤")
    bottom.plot(monthly.Date, monthly.Drawdown_hold*100, color=amber, linewidth=1.8,
                label="QQQ 日度历史峰值回撤")
    bottom.set_ylabel("相对历史峰值回撤", fontsize=13, color="#32465a", labelpad=10)
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    bottom.set_ylim(-92, 7)
    bottom.set_xlim(curves.Date.iloc[0], curves.Date.iloc[-1])
    bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bottom.legend(loc="lower right", frameon=False, fontsize=11)
    top.annotate(f"${stats['buy_and_hold']['end_equity']:,.0f}",
                 xy=(monthly.Date.iloc[-1], monthly.Equity_hold.iloc[-1]),
                 xytext=(-12, 12), textcoords="offset points", color=amber,
                 fontsize=12, fontweight="bold", ha="right")
    top.annotate(f"${stats['a2']['end_equity']:,.0f}",
                 xy=(monthly.Date.iloc[-1], monthly.Equity_a2.iloc[-1]),
                 xytext=(-12, -21), textcoords="offset points", color=blue,
                 fontsize=12, fontweight="bold", ha="right")
    fig.suptitle("QQQ｜收益与回撤：A2-N20 vs 买入并持有",
                 x=.09, y=.975, ha="left", fontsize=25, fontweight="bold", color="#26384b")
    fig.text(.09, .920,
             f"2000-01-03 — {stats['end_date']}    初始 $100,000  |  "
             f"A2 CAGR {stats['a2']['cagr']:.2%}，MDD {stats['a2']['mdd']:.2%}  |  "
             f"持有 CAGR {stats['buy_and_hold']['cagr']:.2%}，MDD {stats['buy_and_hold']['mdd']:.2%}",
             fontsize=13, color="#52687a")
    fig.text(.09, .877,
             "高终值并不等于可承受回撤：QQQ 2000—2002 年最大回撤约 −83%，直到 2015 年才修复前高。",
             fontsize=13, color="#52687a")
    fig.text(.09, .025,
             "图线月末抽样，CAGR 与 MDD 均按完整日度曲线计算。复权价模拟税前分红再投资；不含分红预扣税、个人资本利得税与汇率。",
             fontsize=11, color="#64788a")
    fig.subplots_adjust(left=.09, right=.98, top=.81, bottom=.11, hspace=.13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", metadata={
        "Title": "QQQ A2-N20 versus buy-and-hold, equity and drawdown",
        "Description": "Both backtests share the 2000-01-03 start, daily adjusted marks, initial 100000 USD; monthly chart samples, daily metric calculation. Personal taxes excluded.",
    })
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True, help="Original locally authorized old source")
    p.add_argument("--new", required=True, help="Original locally authorized later source")
    p.add_argument("--csv", required=True, help="Audited updated local-only CSV")
    p.add_argument("--outdir", default="research/results")
    p.add_argument("--chart", default="assets/a2_vs_buy_hold_equity_drawdown.svg")
    p.add_argument("--preview", default=None)
    args = p.parse_args()
    frozen = combine_rebased_sources(args.old, args.new, end_date="2026-07-13")
    latest = load_adjusted_csv(args.csv)
    early_curve, early = compare(frozen, end="2026-07-13")
    recent_curve, recent = compare(latest, end="2026-11-09")
    # The same past market history must produce the same benchmark returns
    # even when dividend adjustments uniformly rebase the absolute price.
    overlap = recent_curve.loc[recent_curve.Date <= pd.Timestamp("2026-07-13")]
    assert len(overlap) == len(early_curve), "Historical session count drift"
    assert abs(early_curve.Equity_hold.iloc[-1]/early_curve.Equity_hold.iloc[0]
               - overlap.Equity_hold.iloc[-1]/overlap.Equity_hold.iloc[0]) < .0002
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"frozen_2026_07_13": early, "updated_2026_11_09": recent,
               "research_asof": "2026-11-10", "benchmark": "independent QQQ buy-and-hold"}
    (out / "a2_vs_buy_hold_2026_07_13_and_11_09.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m = recent_curve.groupby(recent_curve.Date.dt.to_period("M")).tail(1).copy()
    m.Date = m.Date.dt.strftime("%Y-%m-%d")
    m.to_csv(out / "a2_vs_buy_hold_monthly_equity.csv", index=False, float_format="%.6f")
    plot_comparison(recent_curve, recent, Path(args.chart),
                    Path(args.preview) if args.preview else None)
    for metric in (early, recent):
        print(json.dumps({"end": metric["end_date"], "a2": metric["a2"],
                          "buy_and_hold": metric["buy_and_hold"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
