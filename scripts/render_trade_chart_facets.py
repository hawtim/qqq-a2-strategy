"""Seven weekly-candle panels AND the complete daily-fill ledger in one SVG.

Weekly OHLC is for DRAWING only; daily SMA triggers and next-day fills
come from the frozen A2 backtester. Unlike a 26-year single strip, each
panel has an independent price scale. The table below contains all 13
published per-trade fields, derived from the *same* result as the markers.

  python scripts/render_trade_chart_facets.py --old sources/qqq_jtang.csv \
      --new sources/qqq_james.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd

from qqq_a2.engine import Config, combine_rebased_sources, load_adjusted_csv, run_backtest

BLUE = "#1679b8"
AMBER = "#dc902d"
VIOLET = "#855abc"
INK = "#26384b"


def periods() -> list[tuple[str, str, str]]:
    return [(f"{year}-01-01", f"{min(year + 3, 2026)}-12-31",
             f"{year}—{min(year + 3, 2026)}") for year in (2000, 2004, 2008, 2012, 2016, 2020, 2024)]


def draw_trade_table(fig: plt.Figure, trades: pd.DataFrame, active: dict | None,
                     *, latest: pd.Timestamp, table_height: float, fig_height: float) -> int:
    """Append an exact 13-field executed-trades ledger using native SVG text.

    No embedded HTML / foreignObject, no independent CSV manually transcribed
    into the chart, and no hindsight-generated trade events. Native SVG text
    is rendered as font paths so GitHub's inline image works without fonts.
    """
    columns = ["#", "入场信号", "买入日", "买入价¹", "出场信号", "卖出日",
               "卖出价¹", "退出原因", "持有日", "净收益²", "MFE³", "MAE³", "卖后权益"]
    widths = [.038, .085, .081, .075, .085, .081, .075, .129,
              .049, .072, .065, .065, .101]

    def percent(value: float) -> str:
        return f"{float(value):+.2%}"

    rows: list[list[str]] = []
    for trade in trades.itertuples(index=False):
        reason = ("N20 保护" if trade.exit_reason == "N20 protection"
                  else "MA20/60 死叉" if trade.exit_reason == "MA20/60 death cross"
                  else str(trade.exit_reason))
        rows.append([
            str(int(trade.id)), str(trade.entry_signal_date), str(trade.entry_date),
            f"${float(trade.entry_price):,.2f}", str(trade.exit_signal_date),
            str(trade.exit_date), f"${float(trade.exit_price):,.2f}", reason,
            str(int(trade.hold_days)), percent(trade.ret_net),
            percent(trade.mfe_close), percent(trade.mae_close),
            f"${float(trade.equity_after_trade):,.0f}",
        ])
    if active:
        # No fictional sale or P&L for a position still open at the sample end.
        rows.append([
            str(len(rows)+1), str(active["entry_signal_date"]),
            str(active["entry_date"]), f"${float(active['entry_price']):,.2f}",
            "—", "—", "—", "期末仍持仓", "—", "—", "—", "—", "—",
        ])

    fig.text(.075, (table_height+.50)/fig_height,
             f"完整逐笔交易台账  |  {len(trades)} 笔已平仓"
             + (" + 1 笔期末未平仓" if active else "  |  期末空仓"),
             ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(.075, (table_height+.05)/fig_height,
             "信号均于收盘后确认，下一常规交易日开盘成交；下表数据与上方买卖点来自同一次回测。",
             fontsize=12, color="#607385")
    table_ax = fig.add_axes([.075, 1.05/fig_height, .905,
                             (table_height-1.55)/fig_height])
    table_ax.set_axis_off()
    table = table_ax.table(cellText=rows, colLabels=columns, colWidths=widths,
                           bbox=[0, 0, 1, 1], cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.7)
    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_edgecolor("#e1e8ef")
        cell.set_linewidth(.42)
        cell.PAD = .04
        if row_i == 0:
            cell.set_facecolor(INK)
            cell.set_edgecolor("#ffffff")
            cell.set_linewidth(.2)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(10.0)
        else:
            cell.set_facecolor("#f3f7fa" if row_i % 2 == 0 else "#ffffff")
            cell.get_text().set_color(INK)
            if col_i == 0:
                cell.get_text().set_fontweight("bold")
            if col_i == 7:
                kind = rows[row_i-1][7]
                cell.get_text().set_color(AMBER if kind == "N20 保护" else VIOLET)
                cell.get_text().set_fontweight("bold")
            if col_i == 9 and rows[row_i-1][9] != "—":
                profitable = float(trades.iloc[row_i-1].ret_net) > 0
                cell.set_facecolor("#e4f3ea" if profitable else "#fde9e6")
                cell.get_text().set_color("#196d49" if profitable else "#b4473b")
                cell.get_text().set_fontweight("bold")
    fig.text(.075, .40/fig_height,
             "¹ 美元复权开盘成交价（非当年未复权报价）。"
             "² 含买入/卖出各0.05%模拟费用的单笔净收益。"
             "³ MFE/MAE 为持仓期最高/最低复权收盘价相对买价的浮盈/浮亏，不扣费。"
             f"数据截止 {latest.date()}；所有历史账户权益均为模拟值。",
             fontsize=10.5, color="#687c8d")
    return len(rows)


def render(data: pd.DataFrame, output: Path, preview: Path | None = None) -> dict:
    result = run_backtest(data, Config())
    trades = result["trades"]
    daily = data.loc[data.Date >= pd.Timestamp("2000-01-03")].set_index("Date").sort_index()
    entries = pd.to_datetime(trades.entry_date)
    exits = pd.to_datetime(trades.exit_date)
    latest = pd.Timestamp(daily.index[-1])
    active = result["open_position"]

    cjk = [p for p in font_manager.findSystemFonts() if "NotoSansCJK" in Path(p).name]
    family = "DejaVu Sans"
    if cjk:
        chosen = sorted(cjk, key=lambda p: ("Regular" not in p, p))[0]
        font_manager.fontManager.addfont(chosen)
        family = font_manager.FontProperties(fname=chosen).get_name()
    plt.rcParams.update({"font.family": [family, "DejaVu Sans"],
                         "svg.fonttype": "path", "axes.unicode_minus": False})

    # Keep the original seven K panels approximately the same physical height;
    # grow the SVG downward rather than squeezing 60 rows over the candlesticks.
    table_height = .305 * (len(trades) + 1 + int(bool(active))) + 1.5
    fig_height = 24.6 + table_height
    fig, axes = plt.subplots(len(periods()), 1, figsize=(18, fig_height))
    fig.patch.set_facecolor("white")
    for ax, (left, right, title) in zip(axes, periods()):
        lo_dt, hi_dt = pd.Timestamp(left), min(pd.Timestamp(right), latest)
        subset = daily.loc[(daily.index >= lo_dt) & (daily.index <= hi_dt)]
        if subset.empty:
            ax.set_visible(False)
            continue
        weekly = subset.resample("W-FRI").agg({
            "adj_open": "first", "adj_high": "max",
            "adj_low": "min", "adj_close": "last",
        }).dropna()
        ax.set_facecolor("#fcfdfe")
        for d, row in weekly.iterrows():
            x = mdates.date2num(d)
            op, hi, lo, cl = map(float, row)
            color = "#74aaa0" if cl >= op else "#d9877e"
            ax.vlines(x, lo, hi, color=color, linewidth=0.9, zorder=2)
            ax.add_patch(Rectangle((x - 2.0, min(op, cl)), 4.0,
                                   max(abs(cl-op), max(min(op, cl)*0.0015, .01)),
                                   facecolor=color, edgecolor=color,
                                   linewidth=0.2, zorder=3))

        spread = float(weekly.adj_high.max() - weekly.adj_low.min())
        offset = max(spread * .047, float(weekly.adj_close.median()) * .005)
        count_buy, count_exit = 0, 0

        def draw(dates: pd.Series, prices: pd.Series, color: str,
                 marker: str, sign: int, zorder: int) -> int:
            n = 0
            for d, p in zip(dates, prices):
                date, price = pd.Timestamp(d), float(p)
                if not (lo_dt <= date <= hi_dt):
                    continue
                n += 1
                x, y = mdates.date2num(date), price + sign * offset
                ax.plot([x, x], [price, y], color=color, alpha=.62,
                        linewidth=0.8, zorder=zorder)
                ax.scatter([x], [y], marker=marker, s=57,
                           facecolor=color, edgecolor="white",
                           linewidths=.7, zorder=zorder+1)
            return n

        count_buy += draw(entries, trades.entry_price, BLUE, "^", -1, 5)
        if active:
            count_buy += draw(pd.Series([active["entry_date"]]),
                              pd.Series([active["entry_price"]]), BLUE, "*", -1, 8)
        g = trades.exit_reason == "N20 protection"
        t = trades.exit_reason == "MA20/60 death cross"
        count_exit += draw(exits[g], trades.loc[g, "exit_price"], AMBER, "v", +1, 6)
        count_exit += draw(exits[t], trades.loc[t, "exit_price"], VIOLET, "v", +1, 7)

        ax.set_xlim(lo_dt - pd.Timedelta(days=12), hi_dt + pd.Timedelta(days=10))
        ax.set_ylim(max(0, float(weekly.adj_low.min()) - offset * 2.1),
                    float(weekly.adj_high.max()) + offset * 2.2)
        ax.grid(axis="y", color="#e2eaf0", linewidth=.75)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#d7e0e8")
        ax.tick_params(axis="both", labelsize=12, length=0, pad=9, colors=INK)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_title(f"{title}  |  {count_buy} 买 / {count_exit} 卖",
                     loc="left", fontweight="bold", fontsize=18,
                     color=INK, pad=14)

    fig.suptitle("QQQ A2-N20｜七段 K 线 + 完整逐笔交易记录",
                 x=.075, y=1-.28/fig_height, ha="left", fontsize=27,
                 fontweight="bold", color=INK)
    buys = len(trades) + int(bool(active))
    sells = len(trades)
    fig.text(.075, 1-.92/fig_height,
             f"2000-01-03 — {latest.date()} · 七段周 K 仅用于清晰展示 · {buys} 次日线买入 / {sells} 次日线卖出"
             + (" · 期末1笔未平仓" if active else " · 期末空仓") + " · 每段纵轴独立",
             color="#607385", fontsize=13)
    keys = [
        Line2D([0], [0], marker="^", linestyle="None", markersize=11,
               markerfacecolor=BLUE, markeredgecolor="white", label="买入（日线金叉后次日开盘）"),
        Line2D([0], [0], marker="v", linestyle="None", markersize=11,
               markerfacecolor=AMBER, markeredgecolor="white", label="N20 保护退出"),
        Line2D([0], [0], marker="v", linestyle="None", markersize=11,
               markerfacecolor=VIOLET, markeredgecolor="white", label="MA20/60 趋势退出"),
    ]
    if active:
        keys.append(Line2D([0], [0], marker="*", linestyle="None", markersize=13,
                           markerfacecolor=BLUE, markeredgecolor="white", label="期末未平仓买入"))
    fig.legend(handles=keys, ncol=len(keys), frameon=False, fontsize=12,
               bbox_to_anchor=(.075, 1-1.40/fig_height), loc="upper left")
    # Exact physical spacing: the table occupies its own lower section.
    fig.subplots_adjust(left=.075, right=.98, top=1-2.35/fig_height,
                        bottom=(table_height+1.2)/fig_height, hspace=.61)
    table_rows = draw_trade_table(fig, trades, active, latest=latest,
                                  table_height=table_height, fig_height=fig_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={
        "Title": "QQQ A2-N20 seven K-line panels and all 13 fields for every executed trade",
        "Description": f"All {buys} daily entries, {sells} daily exits, and {table_rows} ledger rows, weekly OHLC for display only, cutoff {latest.date()}.",
    })
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, format="png", dpi=80)
    plt.close(fig)
    return {"buys": buys, "sells": sells, "table_rows": table_rows,
            "asof": str(latest.date()),
            "active_at_end": bool(active),
            "n20": int((trades.exit_reason == "N20 protection").sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate faceted A2-N20 static trade chart")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", type=Path, help="Authorized local QQQ daily history")
    inputs.add_argument("--old", type=Path, help="Original older QQQ source")
    parser.add_argument("--new", type=Path, help="Original newer QQQ source")
    parser.add_argument("--end", default="2026-07-13")
    parser.add_argument("--out", type=Path, default=Path("assets/a2_trade_map_facets_2000_2026.svg"))
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    if bool(args.old) != bool(args.new):
        parser.error("--old and --new required together")
    df = load_adjusted_csv(args.csv) if args.csv else combine_rebased_sources(args.old, args.new, end_date=args.end)
    if args.csv:
        df = df[df.Date <= pd.Timestamp(args.end)].copy()
    print(render(df, args.out, args.preview))


if __name__ == "__main__":
    main()
