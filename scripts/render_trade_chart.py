"""Render every A2-N20 buy and sell in one static SVG (no HTML).

Trading signals/fills remain DAILY. Candlesticks are monthly OHLC *for view
only* so that 26 years fit on one chart. Trade markers retain exact trading
dates and adjusted daily open prices. All inputs must be user-provided.

    python scripts/render_trade_chart.py --old sources/qqq_jtang.csv \
        --new sources/qqq_james.csv --end 2026-07-13
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd

from qqq_a2.engine import Config, combine_rebased_sources, load_adjusted_csv, run_backtest

COLORS = {
    "up": "#76a79c", "down": "#cf8882", "buy": "#1879b3",
    "guard": "#df942b", "trend": "#855dc0", "ink": "#26384b",
}


def month_coordinate(d: pd.Timestamp, first: pd.Timestamp) -> float:
    """Use actual fill DAY as a fractional x-coordinate inside each month."""
    month_number = (d.year - first.year) * 12 + d.month - first.month
    return month_number + (d.day - 0.5) / d.days_in_month - 0.5


def render(data: pd.DataFrame, output: Path, preview: Path | None = None) -> dict:
    # Backtest executes strictly on daily bars. Monthly bars NEVER create a signal.
    result = run_backtest(data, Config())
    trades = result["trades"]
    daily = data.loc[data.Date >= pd.Timestamp("2000-01-03")].set_index("Date").sort_index()
    if daily.empty:
        raise ValueError("No data since 2000-01-03")
    monthly = daily.resample("MS").agg({
        "adj_open": "first", "adj_high": "max",
        "adj_low": "min", "adj_close": "last",
    }).dropna()
    first = monthly.index[0]

    # Matplotlib's cache sometimes does not discover distro TTC fonts that
    # fontconfig knows about. Register a local CJK face explicitly if found.
    cjk_files = [p for p in font_manager.findSystemFonts()
                 if "NotoSansCJK" in Path(p).name or "NotoSansSC" in Path(p).name]
    cjk_name = "DejaVu Sans"
    if cjk_files:
        preferred = sorted(cjk_files, key=lambda p: ("Regular" not in p, p))[0]
        font_manager.fontManager.addfont(preferred)
        cjk_name = font_manager.FontProperties(fname=preferred).get_name()
    plt.rcParams.update({"font.family": [cjk_name, "DejaVu Sans"],
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(22, 8.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")

    for x, bar in enumerate(monthly.itertuples()):
        op, hi, lo, cl = map(float, (bar.adj_open, bar.adj_high,
                                     bar.adj_low, bar.adj_close))
        color = COLORS["up"] if cl >= op else COLORS["down"]
        ax.vlines(x, lo, hi, color=color, linewidth=1.2, zorder=2)
        ax.add_patch(Rectangle((x - 0.34, min(op, cl)), 0.68,
                               max(abs(cl - op), min(op, cl) * 0.003),
                               facecolor=color, edgecolor=color, linewidth=0.4, zorder=3))

    def markers(dates: pd.Series, prices: pd.Series, glyph: str,
                color: str, offset: float, size: int, z: int) -> None:
        xs = [month_coordinate(pd.Timestamp(d), first) for d in dates]
        ys = [float(p) * offset for p in prices]
        # A short stem connects offset markers to their exact execution prices.
        for x, p, y in zip(xs, prices, ys):
            ax.plot([x, x], [float(p), y], color=color, linewidth=0.85,
                    alpha=0.65, zorder=z)
        ax.scatter(xs, ys, marker=glyph, s=size, facecolor=color,
                   edgecolor="white", linewidths=0.65, zorder=z + 1)

    markers(trades.entry_date, trades.entry_price, "^", COLORS["buy"], 0.915, 65, 5)
    active = result["open_position"]
    if active:
        markers(pd.Series([active["entry_date"]]), pd.Series([active["entry_price"]]),
                "*", COLORS["buy"], 0.92, 210, 8)
    guard = trades[trades.exit_reason == "N20 protection"]
    trend = trades[trades.exit_reason == "MA20/60 death cross"]
    if len(guard) + len(trend) != len(trades):
        raise ValueError("Unexpected exit reason; check the legend and strategy")
    markers(guard.exit_date, guard.exit_price, "v", COLORS["guard"], 1.10, 72, 6)
    markers(trend.exit_date, trend.exit_price, "v", COLORS["trend"], 1.13, 72, 7)

    ax.set_yscale("log")
    lo = min(monthly.adj_low.min(), trades.entry_price.min()) * 0.76
    hi = max(monthly.adj_high.max(), trades.exit_price.max()) * 1.38
    ax.set_ylim(lo, hi)
    price_ticks = [v for v in (10, 20, 50, 100, 200, 500, 1000)
                   if lo <= v <= hi]
    ax.set_yticks(price_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlim(-1.5, len(monthly) + 0.5)
    years = list(range(2000, 2027, 2))
    year_x = [(y - first.year) * 12 - first.month + 1 for y in years]
    visible = [(x, y) for x, y in zip(year_x, years) if 0 <= x < len(monthly)]
    ax.set_xticks([x for x, _ in visible], [str(y) for _, y in visible])
    ax.tick_params(axis="x", labelsize=13, colors=COLORS["ink"], length=0, pad=14)
    ax.tick_params(axis="y", labelsize=12, colors=COLORS["ink"], length=0, pad=10)
    ax.set_ylabel("QQQ 复权价（USD，对数轴）", fontsize=14,
                  color=COLORS["ink"], labelpad=16)
    ax.grid(which="major", axis="y", color="#dfe7ed", linewidth=0.9, zorder=0)
    ax.grid(which="major", axis="x", color="#dfe7ed", linewidth=0.8,
            linestyle="--", zorder=0)
    ax.minorticks_off()
    for edge in ("left", "right", "top"):
        ax.spines[edge].set_visible(False)
    ax.spines["bottom"].set_color("#dfe7ed")

    buy_count = len(trades) + int(active is not None)
    first_day, last_day = daily.index.min().strftime("%Y-%m-%d"), daily.index.max().strftime("%Y-%m-%d")
    title = "QQQ A2-N20｜完整买卖点历史"
    fig.suptitle(title, x=0.075, y=0.981, ha="left", fontsize=29,
                 fontweight="bold", color=COLORS["ink"])
    fig.text(0.075, 0.885,
             f"{first_day} — {last_day}   |   {buy_count} 次买入 · {len(trades)} 次卖出 · {len(guard)} 次 N20 保护退出",
             fontsize=16, color="#586a78")
    legend = [
        Line2D([0], [0], marker="^", markersize=11, markerfacecolor=COLORS["buy"],
               markeredgecolor="white", linestyle="None", label=f"买入 {buy_count}"),
        Line2D([0], [0], marker="v", markersize=11, markerfacecolor=COLORS["guard"],
               markeredgecolor="white", linestyle="None", label=f"N20 保护卖出 {len(guard)}"),
        Line2D([0], [0], marker="v", markersize=11, markerfacecolor=COLORS["trend"],
               markeredgecolor="white", linestyle="None", label=f"MA20/60 趋势卖出 {len(trend)}"),
    ]
    if active:
        legend.append(Line2D([0], [0], marker="*", markersize=15,
                             markerfacecolor=COLORS["buy"], markeredgecolor="white",
                             linestyle="None", label="期末未平仓买入"))
    ax.legend(handles=legend, loc="upper left", ncol=4, bbox_to_anchor=(0, 1.035),
              frameon=False, fontsize=12, labelcolor=COLORS["ink"],
              handletextpad=0.3, columnspacing=1.5)
    fig.text(0.075, 0.038,
             "重要：底图为月度聚合 K 线，仅为容纳 26 年历史；全部交易信号仍按日线计算，下一交易日开盘成交。箭头已偏移以避免遮挡。",
             fontsize=13, color="#586a78")
    fig.text(0.075, 0.012,
             "数据与复现：docs/DATA_SOURCES.md  ·  全部成交明细：research/results/A2_strict_58_trades.csv  ·  历史样本内研究，非实时信号",
             fontsize=11.5, color="#748494")
    fig.subplots_adjust(left=0.075, right=0.983, top=0.825, bottom=0.132)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={
        "Title": title,
        "Description": f"Monthly OHLC background with exact daily A2-N20 fills. {buy_count} buys and {len(trades)} sells. {first_day} to {last_day}.",
    })
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, format="png", dpi=120)
    plt.close(fig)
    return {"candles": len(monthly), "buy_markers": buy_count,
            "n20_sells": len(guard), "trend_sells": len(trend),
            "open_at_cutoff": bool(active), "end_date": last_day}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render static A2-N20 README candlestick SVG")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path, help="User-authorized QQQ OHLCV CSV")
    inputs.add_argument("--old", type=Path, help="Older historical QQQ CSV")
    parser.add_argument("--new", type=Path, help="Newer historical QQQ CSV")
    parser.add_argument("--end", default="2026-07-13", help="Inclusive historical cutoff")
    parser.add_argument("--out", type=Path, default=Path("assets/a2_trade_map_2000_2026.svg"))
    parser.add_argument("--preview", type=Path, help="Optional PNG for local visual QA")
    args = parser.parse_args()
    if bool(args.old) != bool(args.new):
        parser.error("--old requires --new, and vice versa")
    if args.csv:
        data = load_adjusted_csv(args.csv)
        data = data[data.Date <= pd.Timestamp(args.end)].copy()
    else:
        data = combine_rebased_sources(args.old, args.new, end_date=args.end)
    print(f"Saved {args.out}: {render(data, args.out, args.preview)}")


if __name__ == "__main__":
    main()
