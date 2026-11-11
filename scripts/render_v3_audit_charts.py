"""Render the selected EMA V3's complete trade map and account charts.

The repository deliberately does not redistribute its licensed source QQQ
OHLCV. The seven weekly K-line panels are therefore reconstructed from the
*already-published vector candle marks* of the archived fourth-option SVG.
This reuses only the old chart's visual geometry (not a new data source or
strategy calculation). Every V3 trade marker, fill price and table value comes
exclusively from the selected V3 CSV, never the older strategy's marks.

The second SVG is based solely on the committed V3 daily account-equity CSV.
Neither chart uses weekly candles to create trade signals.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
import re
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
NS = {"s": "http://www.w3.org/2000/svg"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
YEARS = (2000, 2004, 2008, 2012, 2016, 2020, 2024)
BUY = "#1c83ac"
N20 = "#e49a37"
EXTENDED = "#b97a22"
TREND = "#8961af"
DAY0 = "#d75063"
HARD = "#c65050"
INK = "#26384b"
MUTED = "#647487"
UP, DOWN = "#70aa98", "#d9847d"


@dataclass
class Panel:
    year: int
    top: float
    bottom: float
    left: float
    right: float
    x_ref: tuple[tuple[date, float], tuple[date, float]]
    y_ref: tuple[tuple[float, float], tuple[float, float]]
    xticks: list[tuple[str, float]]
    yticks: list[tuple[str, float]]
    candle_paths: dict[str, str]
    candle_count: int

    def x(self, when: str) -> float:
        (d0, x0), (d1, x1) = self.x_ref
        dt = date.fromisoformat(when).toordinal() - d0.toordinal()
        return x0 + dt * (x1 - x0) / (d1.toordinal() - d0.toordinal())

    def y(self, price: float) -> float:
        (p0, y0), (p1, y1) = self.y_ref
        return y0 + (price - p0) * (y1 - y0) / (p1 - p0)


def _tick_label(tick: ET.Element) -> str:
    labels = [node.text.strip() for node in tick.iter()
              if node.tag is ET.Comment and node.text and node.text.strip()]
    if len(labels) != 1:
        raise ValueError("Missing or ambiguous published SVG tick label")
    return labels[0]


def _ticks(axis: ET.Element, axis_id: str, direction: str) -> list[tuple[str, float]]:
    group = axis.find(f"./s:g[@id='{axis_id}']", NS)
    if group is None:
        raise ValueError(f"Missing published chart axis {axis_id}")
    ticks = []
    for tick in group:
        if not tick.get("id", "").startswith(direction + "tick_"):
            continue
        uses = [node for node in tick.iter() if str(node.tag).endswith("}use")
                and node.get(direction) is not None]
        if not uses:
            raise ValueError("Missing published tick coordinate")
        ticks.append((_tick_label(tick), float(uses[0].get(direction))))
    return ticks


def _svg_geometry(template: Path) -> list[Panel]:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(template, parser=parser).getroot()
    panels = []
    for i, year in enumerate(YEARS, start=1):
        axis = root.find(f".//s:g[@id='axes_{i}']", NS)
        if axis is None:
            raise ValueError(f"Missing archived {year} K panel")
        background = axis.find("./s:g/s:path", NS)
        if background is None:
            raise ValueError(f"Missing archived {year} K background")
        bounds = [float(v) for v in NUMBER.findall(background.get("d", ""))]
        if len(bounds) < 8:
            raise ValueError("Malformed archived axis bounds")
        left, right = min(bounds[::2]), max(bounds[::2])
        top, bottom = min(bounds[1::2]), max(bounds[1::2])
        xticks = _ticks(axis, f"matplotlib.axis_{2*i-1}", "x")
        yticks = _ticks(axis, f"matplotlib.axis_{2*i}", "y")
        # Only the actual candle geometries are reused: never copy legacy
        # trading markers, legacy trade table or legacy chart annotations.
        wick: dict[str, list[str]] = {UP: [], DOWN: []}
        body: dict[str, list[str]] = {UP: [], DOWN: []}
        for group in axis:
            id_ = group.get("id", "")
            path = group.find("./s:path", NS)
            if path is None:
                continue
            style = path.get("style", "")
            color = next((c for c in (UP, DOWN) if c in style), None)
            if not color:
                continue
            nums = [float(n) for n in NUMBER.findall(path.get("d", ""))]
            if id_.startswith("LineCollection_"):
                if len(nums) != 4 or abs(nums[0]-nums[2]) > .0001:
                    raise ValueError(f"Unexpected wick geometry in {year}")
                wick[color].append(f"M{nums[0]:.2f},{nums[1]:.2f}V{nums[3]:.2f}")
            elif id_.startswith("patch_") and path.get("clip-path"):
                if len(nums) != 8:
                    raise ValueError(f"Unexpected candle geometry in {year}")
                xs, ys = nums[::2], nums[1::2]
                x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
                body[color].append(f"M{x:.2f},{y:.2f}h{w:.2f}v{h:.2f}h{-w:.2f}Z")
        count = sum(map(len, wick.values()))
        if count < 140 or count != sum(map(len, body.values())):
            raise ValueError(f"Incomplete archived {year} weekly K backdrop")
        if len(xticks) < 2 or len(yticks) < 2:
            raise ValueError(f"Missing axis calibration ticks in {year}")
        xrefs = [(date.fromisoformat(lbl + "-01"), x) for lbl, x in xticks]
        yrefs = [(float(lbl.replace("−", "-")), y) for lbl, y in yticks]
        paths = {f"wick_{c}": "".join(wick[c]) for c in (UP, DOWN)}
        paths.update({f"body_{c}": "".join(body[c]) for c in (UP, DOWN)})
        panels.append(Panel(year, top, bottom, left, right,
                            (xrefs[0], xrefs[-1]), (yrefs[0], yrefs[-1]),
                            xticks, yticks, paths, count))
    return panels


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _validate_trades(trades: list[dict]) -> None:
    if len(trades) != 52:
        raise ValueError(f"Expected exactly 52 selected-V3 closed trades, found {len(trades)}")
    if [int(t["id"]) for t in trades] != list(range(1, 53)):
        raise ValueError("Selected V3 trade IDs must be sequential")
    if (trades[0]["entry_date"], trades[0]["exit_date"]) != ("2000-06-22", "2000-06-23"):
        raise ValueError("V3 day-zero guard trade does not match frozen ledger")
    t22 = next(t for t in trades if t["entry_date"] == "2011-10-14")
    if (t22["exit_date"], t22["exit_reason"]) != ("2011-11-21", "N20 conditional extension"):
        raise ValueError("V3 conditional N20 trade does not match frozen ledger")
    for tr in trades:
        if tr["exit_date"] < tr["entry_date"]:
            raise ValueError("Exit precedes entry")
        if tr["entry_date"] > "2026-11-09" or tr["exit_date"] > "2026-11-09":
            raise ValueError("The archived weekly K background ends on 2026-11-09")
    if any(trades[i]["entry_date"] <= trades[i-1]["exit_date"] for i in range(1,52)):
        raise ValueError("Overlapping or out-of-order all-in V3 trades")


def _exit_legend(reason: str) -> tuple[str, str]:
    return {
        "MA20/60 death cross": ("EMA20/60 death", TREND),
        "N20 protection": ("N20", N20),
        "N20 extended 1d": ("N20 +1d", N20),
        "N20 conditional extension": ("N20 to day30", EXTENDED),
        "entry-day close <= -3%": ("Day0 -3%", DAY0),
        "7% close stop": ("Net -7% stop", HARD),
    }[reason]


def _text(x: float, y: float, content: str, *, size: float = 10,
          color: str = INK, weight: int = 400, anchor: str = "start",
          extra: str = "") -> str:
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'{extra}>{escape(content)}</text>')


def _marker(panel: Panel, tr: dict, kind: str) -> str:
    entry = kind == "buy"
    when = tr["entry_date"] if entry else tr["exit_date"]
    price = float(tr["entry_price"] if entry else tr["exit_price"])
    x, y = panel.x(when), panel.y(price)
    y2 = y + (13 if entry else -13)
    color = BUY if entry else _exit_legend(tr["exit_reason"])[1]
    # Explicit marker class makes auditing all 52 entry/exit executions easy.
    points = (f"{x:.2f},{y2-5:.2f} {x-5.5:.2f},{y2+5:.2f} {x+5.5:.2f},{y2+5:.2f}"
              if entry else
              f"{x:.2f},{y2+5:.2f} {x-5.5:.2f},{y2-5:.2f} {x+5.5:.2f},{y2-5:.2f}")
    tip = (f'#{tr["id"]} {"BUY" if entry else "SELL"}: {when}, '
           f'adj open ${price:.4f}; entry {tr["entry_date"]}; '
           f'exit {tr["exit_date"]}; {tr["exit_reason"]}; '
           f'net {float(tr["ret_net"]):+.4%}')
    return (f'<g class="v3-{kind}" data-trade="{tr["id"]}"><title>{escape(tip)}</title>'
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="1.2" opacity=".82"/>'
            f'<polygon points="{points}" fill="{color}" stroke="white" '
            f'stroke-width=".85"/></g>')


def render_trade_map(template: Path, trades_path: Path, output: Path,
                     stats: dict) -> None:
    panels, trades = _svg_geometry(template), _read_csv(trades_path)
    _validate_trades(trades)
    result: list[str] = []
    result.append(_svg_header(1440, 3035, "QQQ EMA V3 complete trading map"))
    result.append('<rect width="1440" height="3035" fill="white"/>')
    result.append(_text(108, 49, "QQQ EMA V3 | Seven weekly K panels + all 52 executed trades",
                        size=25, weight=700))
    result.append(_text(108, 80,
                        "2000-01-03 to 2026-11-09  |  $100k initial  |  0.05% per side  |  close signals -> next regular open fills",
                        size=12, color=MUTED))
    result.append(_text(108, 105,
                        f'End ${float(stats["end_equity"]):,.0f}   /   CAGR {float(stats["cagr"]):.2%}   /   '
                        f'MDD {float(stats["mdd"]):.2%}   /   Payoff {float(stats["payoff_ratio"]):.2f}   /   '
                        f'{stats["wins"]} wins, {stats["losses"]} losses',
                        size=12.2, color=INK, weight=600))
    legend = [(BUY, "Buy"), (N20, "N20 exit"), (EXTENDED, "N20 to day30"),
              (TREND, "EMA20/60 exit"), (DAY0, "Day0 -3% exit")]
    lx = 108
    for c, label in legend:
        result.append(f'<circle cx="{lx+6}" cy="130" r="5.5" fill="{c}"/>')
        result.append(_text(lx+19, 134, label, size=11.5, color=MUTED))
        lx += {"Buy":120,"N20 exit":142,"N20 to day30":184,
               "EMA20/60 exit":192,"Day0 -3% exit":160}[label]

    for panel in panels:
        last_year = min(panel.year+3, 2026)
        begin, end = f"{panel.year}-01-01", f"{last_year}-12-31"
        entries = [tr for tr in trades if begin <= tr["entry_date"] <= end]
        exits = [tr for tr in trades if begin <= tr["exit_date"] <= end]
        result.append(f'<g class="v3-panel" data-period="{panel.year}-{last_year}">')
        result.append(_text(panel.left, panel.top-13,
                            f"{panel.year}-{last_year}  |  {len(entries)} buys / {len(exits)} sells",
                            size=15, weight=700))
        for label, y in panel.yticks:
            result.append(f'<line x1="{panel.left:.2f}" x2="{panel.right:.2f}" '
                          f'y1="{y:.2f}" y2="{y:.2f}" stroke="#e2e7eb" stroke-width=".65"/>')
            result.append(_text(panel.left-9, y+4, label, size=10.5, color=MUTED, anchor="end"))
        for label, x in panel.xticks:
            result.append(f'<line x1="{x:.2f}" x2="{x:.2f}" '
                          f'y1="{panel.bottom:.2f}" y2="{panel.bottom+4:.2f}" stroke="#81909d"/>')
            result.append(_text(x, panel.bottom+20, label, size=10.5, color=MUTED, anchor="middle"))
        # Both wicks and bodies are vector paths already visible in the public
        # archived chart. No adjusted QQQ OHLC rows are reconstructed or saved.
        for color in (UP, DOWN):
            result.append(f'<path d="{panel.candle_paths[f"wick_{color}"]}" fill="none" '
                          f'stroke="{color}" stroke-width=".75"/>')
            result.append(f'<path d="{panel.candle_paths[f"body_{color}"]}" '
                          f'fill="{color}" stroke="{color}" stroke-width=".23"/>')
        result.append(f'<rect x="{panel.left:.2f}" y="{panel.top:.2f}" '
                      f'width="{panel.right-panel.left:.2f}" '
                      f'height="{panel.bottom-panel.top:.2f}" fill="none" '
                      f'stroke="#abb9c4" stroke-width=".6"/>')
        for tr in entries:
            result.append(_marker(panel, tr, "buy"))
        for tr in exits:
            result.append(_marker(panel, tr, "sell"))
        result.append("</g>")

    result.append(_text(108, 1781, "Complete selected-V3 execution ledger | 52 closed trades",
                        size=20, weight=700))
    result.append(_text(108, 1805,
                        "Adjusted next-open fills; net includes round-trip fees. Source CSV retains full precision, MFE and MAE.",
                        size=11, color=MUTED))
    headers = ["#", "Buy signal", "BUY", "Buy $", "Entry", "20/60 confirm",
               "Sell signal", "SELL", "Sell $", "Exit rule", "Days", "Net %", "Equity"]
    widths = [36, 100, 89, 81, 74, 112, 104, 90, 83, 158, 63, 85, 134]
    factor = (1418-108) / sum(widths)
    widths = [v*factor for v in widths]
    edges = [108.0]
    for width in widths:
        edges.append(edges[-1]+width)
    top, rh = 1825, 21
    result.append(f'<rect x="108" y="{top}" width="1310" height="{rh}" rx="3" fill="{INK}"/>')
    for i, h in enumerate(headers):
        result.append(_text((edges[i]+edges[i+1])/2, top+14.3, h,
                            size=10.3, color="white", weight=700, anchor="middle"))
    for idx, tr in enumerate(trades):
        y0 = top + rh*(idx+1)
        special = tr["exit_reason"] in {"entry-day close <= -3%", "N20 conditional extension"}
        bg = "#fff4e4" if special else ("#f3f7fa" if idx % 2 else "white")
        result.append(f'<rect class="v3-ledger-row" data-trade="{tr["id"]}" '
                      f'x="108" y="{y0}" width="1310" height="{rh}" fill="{bg}"/>')
        reason, color = _exit_legend(tr["exit_reason"])
        entry = "FAST" if tr["entry_kind"].startswith("FAST") else "REG"
        cells = [tr["id"], tr["entry_signal_date"], tr["entry_date"],
                 f'${float(tr["entry_price"]):.2f}', entry,
                 tr["trend_confirm_date"] or "-", tr["exit_signal_date"],
                 tr["exit_date"], f'${float(tr["exit_price"]):.2f}',
                 reason, tr["hold_days"], f'{float(tr["ret_net"]):+.2%}',
                 f'${float(tr["equity_after_trade"]):,.0f}']
        result.append(f'<g><title>Trade #{tr["id"]}: {escape(tr["exit_reason"])}; '
                      f'MFE {float(tr["mfe_close"]):+.2%}; MAE {float(tr["mae_close"]):+.2%}</title>')
        for col, content in enumerate(cells):
            cell_color = ("#267b5a" if float(tr["ret_net"]) > 0 else "#ba4848") if col == 11 else (color if col == 9 else INK)
            result.append(_text((edges[col]+edges[col+1])/2, y0+14.4, str(content),
                                size=9.8 if col not in (5, 9, 12) else 9.2,
                                color=cell_color, weight=700 if col in (9,11) else 400,
                                anchor="middle"))
        result.append("</g>")
        result.append(f'<line x1="108" x2="1418" y1="{y0+rh}" y2="{y0+rh}" '
                      f'stroke="#e1e7ec" stroke-width=".6"/>')
    result.append(_text(108, 2980,
                        "V3 selected research baseline, NOT live orders. Weekly candles = display only; all trades from completed daily signals.",
                        size=11, color=MUTED))
    result.append(_text(108, 3000,
                        "The 3% entry-day guard and net 7% close stop signal at CLOSE, execute next OPEN: overnight gaps can exceed thresholds.",
                        size=11, color=MUTED))
    result.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result), encoding="utf-8")


def _svg_header(w: int, h: int, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">'
            f'<title>{escape(title)}</title>')


def render_equity_chart(daily_path: Path, output: Path, stats: dict) -> None:
    rows = _read_csv(daily_path)
    if not rows or rows[0]["Date"] != "2000-01-03" or rows[-1]["Date"] != "2026-11-09":
        raise ValueError("Selected V3 equity does not match frozen sample dates")
    if abs(float(rows[-1]["Equity"]) - float(stats["end_equity"])) > .02:
        raise ValueError("Selected V3 equity endpoint differs from summary")
    if abs(min(float(r["Drawdown"]) for r in rows)-float(stats["mdd"])) > 1e-7:
        raise ValueError("Selected V3 MDD differs from summary")
    width, height = 1440, 750
    xlo, xhi, ytop, ybottom = 110, 1390, 150, 460
    ddtop, ddbottom = 525, 671
    d0, d1 = date.fromisoformat(rows[0]["Date"]), date.fromisoformat(rows[-1]["Date"])
    def xp(d: str) -> float:
        return xlo + (date.fromisoformat(d).toordinal()-d0.toordinal()) * (xhi-xlo) / (d1.toordinal()-d0.toordinal())
    equity_max = 1_000_000
    def ey(v: float) -> float:
        return ybottom - v/equity_max * (ybottom-ytop)
    def dy(v: float) -> float:
        return ddtop - v/.25 * (ddbottom-ddtop)

    s = [_svg_header(width,height,"QQQ EMA V3 equity and daily close drawdown"),
         f'<rect width="{width}" height="{height}" fill="white"/>',
         _text(110, 55, "QQQ EMA V3 | Account equity & daily-close drawdown", size=25, weight=700),
         _text(110, 84, "Selected V3 only | 2000-01-03 to 2026-11-09 | $100,000 start | fees 0.05% per side | cash yield 0", size=12, color=MUTED),
         _text(110, 116,
               f'End ${float(stats["end_equity"]):,.0f}   CAGR {float(stats["cagr"]):.2%}   MDD {float(stats["mdd"]):.2%}   Exposure {float(stats["exposure"]):.1%}   Completed trades {stats["completed_trades"]}',
               size=13, weight=700)]
    for yv in range(0, equity_max+1, 200_000):
        y = ey(yv)
        s.append(f'<line x1="{xlo}" x2="{xhi}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e3e9ed"/>')
        s.append(_text(xlo-10, y+4, f'${yv//1000:,}k', anchor="end", color=MUTED))
    for dd in (0,-.05,-.10,-.15,-.20,-.25):
        y = dy(dd)
        s.append(f'<line x1="{xlo}" x2="{xhi}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e3e9ed"/>')
        s.append(_text(xlo-10, y+4, f'{dd:.0%}', anchor="end", color=MUTED))
    for year in (2000,2004,2008,2012,2016,2020,2024,2026):
        dt = f'{year}-01-03'
        x = xp(dt)
        s.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{ytop}" y2="{ddbottom}" stroke="#dbe4e9" stroke-dasharray="3 5"/>')
        s.append(_text(x, ddbottom+24, str(year), size=11, color=MUTED, anchor="middle"))

    # Encode each complete daily observation. The light blue bands denote the
    # selected V3 state (Invested=True), not a separate rederived trade system.
    segments = []
    start = None
    for row in rows:
        if row["Invested"] == "True" and start is None:
            start = row["Date"]
        elif row["Invested"] != "True" and start is not None:
            segments.append((start,row["Date"]))
            start = None
    if start is not None:
        segments.append((start,rows[-1]["Date"]))
    for left,right in segments:
        x, x2 = xp(left), xp(right)
        s.append(f'<rect x="{x:.2f}" y="{ytop}" width="{x2-x:.2f}" '
                 f'height="{ybottom-ytop}" fill="#1c83ac" opacity=".04"/>')
    equity_path = "".join(("M" if i == 0 else "L") + f'{xp(r["Date"]):.2f},{ey(float(r["Equity"])):.2f}' for i,r in enumerate(rows))
    dd_path = "".join(("M" if i == 0 else "L") + f'{xp(r["Date"]):.2f},{dy(float(r["Drawdown"])):.2f}' for i,r in enumerate(rows))
    s.append(f'<path d="{equity_path}" fill="none" stroke="{BUY}" stroke-width="2.1"/>')
    s.append(f'<path d="{dd_path}" fill="none" stroke="{HARD}" stroke-width="1.5"/>')
    for when,label in (("2000-06-23","Day0 -3%"),("2011-11-21","N20 +10d")):
        x = xp(when)
        s.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{ytop}" y2="{ybottom}" stroke="{EXTENDED}" stroke-width="1.2" stroke-dasharray="5 4"><title>V3 changed exit: {when} ({label})</title></line>')
    s.extend([_text(110, 143, "Daily close portfolio value (USD)", size=11, color=MUTED),
              _text(110, 512, "Daily close drawdown (peak-to-trough)", size=11, color=MUTED),
              _text(110, 726, "This is a historical in-sample simulation, not broker fills. No tax, additional opening slippage or cash interest.",
                    size=11, color=MUTED),
              "</svg>"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(s), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=ROOT/"assets/a2_ema_dynamic_n20_trade_map_2000_2026.svg")
    folder = ROOT/"research/results/a2_v3_selected_2026_11_09"
    parser.add_argument("--trades", type=Path, default=folder/"selected_v3_trades.csv")
    parser.add_argument("--equity", type=Path, default=folder/"selected_v3_daily_equity.csv")
    parser.add_argument("--summary", type=Path, default=folder/"summary.csv")
    parser.add_argument("--out-map", type=Path, default=ROOT/"assets/a2_ema_v3_trade_map_2000_2026.svg")
    parser.add_argument("--out-equity", type=Path, default=ROOT/"assets/a2_ema_v3_equity_drawdown_2000_2026.svg")
    args = parser.parse_args()
    matches = [r for r in _read_csv(args.summary) if r["version"] == "EMA_V3_ADAPTIVE_N20_DAY0_CLOSE3"]
    if len(matches) != 1:
        raise ValueError("Expected one selected V3 summary row")
    render_trade_map(args.template, args.trades, args.out_map, matches[0])
    render_equity_chart(args.equity, args.out_equity, matches[0])
    print(f"V3 trade map: {args.out_map}")
    print(f"V3 equity / drawdown: {args.out_equity}")


if __name__ == "__main__":
    main()
