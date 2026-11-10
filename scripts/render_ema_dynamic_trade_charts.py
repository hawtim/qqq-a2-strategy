"""SVG K-line audit charts for fourth benchmark, from daily executed fills.

The multi-panel graphic uses weekly aggregated OHLC solely as a legible
background; trade signals and markers remain precisely daily and next-open.
The 2011 close-up uses full DAILY candles and contrasts three executed paths.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import pandas as pd


INK = "#26384b"
BUY = "#1c83ac"
N20 = "#e49a37"
HARD = "#c65050"
TREND = "#8961af"


def _font() -> None:
    fonts = [p for p in font_manager.findSystemFonts()
             if "NotoSansCJK" in Path(p).name or "NotoSansSC" in Path(p).name]
    family = "DejaVu Sans"
    if fonts:
        preferred = sorted(fonts, key=lambda x: ("Regular" not in x, x))[0]
        font_manager.fontManager.addfont(preferred)
        family = font_manager.FontProperties(fname=preferred).get_name()
    plt.rcParams.update({"font.family": [family, "DejaVu Sans"],
                         "svg.fonttype": "path", "axes.unicode_minus": False})


def _candles(ax, ohlc: pd.DataFrame, *, half_width: float, lw: float = .75) -> None:
    for date, bar in ohlc.iterrows():
        x = mdates.date2num(date)
        op, hi, lo, close = (float(bar[c]) for c in
                             ("adj_open", "adj_high", "adj_low", "adj_close"))
        color = "#70aa98" if close >= op else "#d9847d"
        ax.vlines(x, lo, hi, color=color, linewidth=lw, zorder=2)
        body = max(abs(close - op), min(op, close) * .001)
        ax.add_patch(Rectangle((x-half_width, min(op,close)), 2*half_width,
                               body, facecolor=color, edgecolor=color,
                               linewidth=.25, zorder=3))


def _mark(ax, date: str, px: float, *, color: str, direction: int,
          offset: float, shape: str = "v", text: str | None = None) -> None:
    x = mdates.date2num(pd.Timestamp(date))
    y = float(px) + direction * offset
    ax.plot([x,x], [float(px),y], color=color, alpha=.65, lw=.8, zorder=7)
    ax.scatter([x], [y], marker=shape, color=color,
               edgecolors="white", linewidths=.7, s=55, zorder=8)
    if text:
        ax.annotate(text, (x,y), xytext=(7, 10 if direction>0 else -17),
                    textcoords="offset points", fontsize=8, color=color)


def _exit_color(reason: str) -> str:
    if reason.startswith("7%"):
        return HARD
    if reason == "MA20/60 death cross":
        return TREND
    return N20


def _full_trade_map(data: pd.DataFrame, trades: pd.DataFrame, path: Path) -> None:
    daily = data.loc[data.Date.between("2000-01-03", "2026-11-09")].set_index("Date")
    panels = [(f"{y}-01-01", f"{min(y+3,2026)}-12-31",
               f"{y}–{min(y+3,2026)}") for y in (2000,2004,2008,2012,2016,2020,2024)]
    table_height = 1.9 + .31 * (len(trades) + 1)
    fig_height = 25 + table_height
    fig, axes = plt.subplots(len(panels), 1, figsize=(20, fig_height))
    fig.patch.set_facecolor("white")
    for ax, (first, last, title) in zip(axes, panels):
        d = daily.loc[daily.index.to_series().between(first, last)]
        weeks = d.resample("W-FRI").agg({
            "adj_open": "first", "adj_high": "max", "adj_low": "min",
            "adj_close": "last"}).dropna()
        if len(weeks) == 0:
            ax.set_visible(False)
            continue
        _candles(ax, weeks, half_width=2.1)
        span = float(weeks.adj_high.max() - weeks.adj_low.min())
        off = max(.046 * span, float(weeks.adj_close.median()) * .009)
        sub = trades.loc[trades.entry_date.between(first, last)]
        sub_exit = trades.loc[trades.exit_date.between(first, last)]
        for trade in sub.itertuples(index=False):
            _mark(ax, trade.entry_date, trade.entry_price,
                  color=BUY, direction=-1, offset=off, shape="^")
        for trade in sub_exit.itertuples(index=False):
            _mark(ax, trade.exit_date, trade.exit_price,
                  color=_exit_color(trade.exit_reason), direction=1, offset=off)
        ax.set_xlim(pd.Timestamp(first)-pd.Timedelta(days=9),
                    min(pd.Timestamp(last), pd.Timestamp("2026-11-09"))+pd.Timedelta(days=9))
        ax.set_ylim(max(0,float(weeks.adj_low.min())-2*off),
                    float(weeks.adj_high.max())+2*off)
        ax.set_title(f"{title}  ·  {len(sub)} buys / {len(sub_exit)} sells",
                     loc="left", fontsize=16, weight="bold", color=INK)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=10)
    fig.suptitle("QQQ Dynamic EMA N20 | Seven weekly K-line panels and ALL trades",
                 fontsize=23, x=.075, y=1-.28/fig_height, ha="left", weight="bold")
    fig.text(.075, 1-.9/fig_height,
             "Buy/sell markers use DAILY signal + next OPEN fills; weekly candles are DISPLAY ONLY.",
             fontsize=12, color="#647487")
    keys = [
        Line2D([], [], marker="^", ls="None", color=BUY, label="Buy (daily open)"),
        Line2D([], [], marker="v", ls="None", color=N20, label="N20 protection"),
        Line2D([], [], marker="v", ls="None", color=TREND, label="EMA20/60 exit"),
        Line2D([], [], marker="v", ls="None", color=HARD, label="7% close stop"),
    ]
    fig.legend(handles=keys, ncol=4, frameon=False, fontsize=12,
               bbox_to_anchor=(.075,1-1.40/fig_height), loc="upper left")
    fig.subplots_adjust(left=.075,right=.985,top=1-2.30/fig_height,
                        bottom=(table_height+1.25)/fig_height,hspace=.55)

    # The SVG itself also contains every completed trade as searchable vector
    # text paths; the machine-readable CSV is the canonical numerical ledger.
    columns = ["#", "Signal", "BUY", "Buy px", "Entry", "Confirm 20/60",
               "Sell signal", "SELL", "Sell px", "Exit rule", "Net %", "Equity"]
    rows = []
    for tr in trades.itertuples(index=False):
        reason = ("N20 extended" if tr.exit_reason.startswith("N20") else
                  "EMA20/60 death" if tr.exit_reason=="MA20/60 death cross" else
                  "7% close stop")
        rows.append([str(tr.id), tr.entry_signal_date, tr.entry_date,
                     f"${tr.entry_price:.2f}",
                     "fast" if tr.entry_kind.startswith("FAST") else "normal",
                     str(tr.trend_confirm_date or "—"), tr.exit_signal_date,
                     tr.exit_date, f"${tr.exit_price:.2f}", reason,
                     f"{tr.ret_net:+.2%}", f"${tr.equity_after_trade:,.0f}"])
    fig.text(.075, (table_height+.65)/fig_height,
             f"Complete executed ledger | {len(trades)} closed trades | 2000–2026",
             fontsize=20, weight="bold", color=INK)
    fig.text(.075, (table_height+.22)/fig_height,
             "Adjusted execution prices, two-sided 0.05% fees in net return; 2011 detail separately audited.",
             fontsize=11, color="#647487")
    ax_table = fig.add_axes([.075, 1.05/fig_height,.91,(table_height-1.55)/fig_height])
    ax_table.axis("off")
    grid = ax_table.table(cellText=rows, colLabels=columns,
                          colWidths=[.03,.082,.082,.068,.058,.096,.084,.082,.068,.105,.065,.11],
                          bbox=[0,0,1,1], cellLoc="center")
    grid.auto_set_font_size(False)
    grid.set_fontsize(8.7)
    for (r,c),cell in grid.get_celld().items():
        cell.set_edgecolor("#dce5ed")
        cell.set_linewidth(.42)
        cell.PAD=.025
        if r==0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f3f7fa" if r%2==0 else "white")
            if c==10:
                is_win=float(trades.iloc[r-1].ret_net)>0
                cell.get_text().set_color("#267b5a" if is_win else "#ba4848")
                cell.get_text().set_weight("bold")
    fig.text(.075,.35/fig_height,
             "Monthly EMA5 uses only the LAST COMPLETED calendar month. Stops signal on close, sell next open; gaps can exceed 7%.",
             fontsize=10, color="#607487")
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,format="svg",metadata={"Title":"EMA dynamic N20 daily fills and complete historical ledger"})
    plt.close(fig)


def _detail_2011(data: pd.DataFrame, result: dict, path: Path) -> None:
    daily=data.loc[data.Date.between("2011-06-17","2011-08-19")].set_index("Date")
    dynamic_name="EMA_A2_DynamicN20_monthEMA5_close7"
    dynamic=result["runs"][dynamic_name]["trades"]
    original=result["controls"]["fast_entry_old_fixed_n20"]["trades"]
    base=result["runs"]["EMA_A2_N20_close7"]["trades"]
    picks={
        "Dynamic N20": (dynamic,"#188c72"),
        "Old N20 with fast entry": (original,"#c65050"),
        "Frozen EMA+7% baseline": (base,"#8b66ad"),
    }
    fig, ax=plt.subplots(figsize=(16,7.4))
    _candles(ax,daily,half_width=.27,lw=.85)
    local=data.loc[data.Date.between("2011-06-17","2011-08-19")].copy()
    from a2_ema_dynamic_n20 import completed_month_signals
    # EMA initialization uses full prehistory, never a sliced 2011 seed.
    alls=completed_month_signals(data)
    inds=alls.loc[alls.Date.between("2011-06-17","2011-08-19")]
    for p,col in [(10,"#d59d38"),(20,"#467c9e"),(60,"#747b8b")]:
        ax.plot(inds.Date,inds[f"ma{p}"],label=f"EMA{p}",lw=1.6,color=col)
    detail=[]
    for name,(ledger,color) in picks.items():
        hits=ledger.loc[ledger.entry_date.between("2011-06-15","2011-07-20")]
        hit=hits.iloc[0]
        detail.append((name,hit.entry_date,hit.exit_date,float(hit.ret_net)))
        ax.scatter(pd.Timestamp(hit.entry_date),hit.entry_price,marker="^",
                   facecolor=color,edgecolor="white",s=125,zorder=10)
        ax.scatter(pd.Timestamp(hit.exit_date),hit.exit_price,marker="v",
                   facecolor=color,edgecolor="white",s=125,zorder=10)
        ax.plot([pd.Timestamp(hit.entry_date),pd.Timestamp(hit.exit_date)],
                [hit.entry_price,hit.exit_price],color=color,ls="--",alpha=.62,lw=1.2)
        ax.annotate(f"{name}: {hit.ret_net:+.2%}",
                    (pd.Timestamp(hit.exit_date),hit.exit_price),
                    xytext=(10,20 if "Dynamic" in name else -28),textcoords="offset points",
                    color=color,fontsize=10,weight="bold")
    ax.axvline(pd.Timestamp("2011-07-07"),ls=":",alpha=.55,color="#555")
    ax.annotate("EMA20/60 trend confirmed\nDynamic N20 restarts",(pd.Timestamp("2011-07-07"),
                float(daily.adj_high.max())*.95),xytext=(7,-8),textcoords="offset points",
                fontsize=10,color=INK)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_xlim(pd.Timestamp("2011-06-16"),pd.Timestamp("2011-08-19"))
    ax.grid(alpha=.18)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylabel("QQQ adjusted price (USD)")
    ax.set_title("2011 loss attribution | DAILY K candles and actual next-open fills",fontsize=17,loc="left")
    ax.legend(fontsize=9,loc="lower left",frameon=False)
    fig.text(.09,.025,
             "Entry/exit triangles are executions; cross and monthly signals use only completed bars. No same-close fills.",
             fontsize=10,color="#566")
    fig.tight_layout(rect=[0,.045,1,1])
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,format="svg",metadata={"Title":"2011 dynamic N20 versus fixed N20 daily K-line audit"})
    plt.close(fig)


def render_trade_charts(data: pd.DataFrame, result: dict, assets: Path) -> None:
    _font()
    name="EMA_A2_DynamicN20_monthEMA5_close7"
    _full_trade_map(data,result["runs"][name]["trades"],
                    assets/"a2_ema_dynamic_n20_trade_map_2000_2026.svg")
    _detail_2011(data,result,assets/"a2_ema_dynamic_n20_2011_daily_candles.svg")
