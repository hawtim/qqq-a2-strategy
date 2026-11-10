"""Independent EMA A2-N20 experiment: always-on 8% trade-loss protection.

Control: frozen EMA A2-N20 (next-open N20 and EMA20/60 exits).
Variants: (1) 8% net-loss daily-close signal, sell next open, and
          (2) gap-aware intraday stop-market simulation using adjusted low.

The intraday daily-OHLC approximation assumes a fill at the stop price if
the session opened above it, or at the worse session open after a gap.
It cannot guarantee an 8% maximum loss or model intrabar spreads/slippage.
No change to the frozen SMA engine or the independent EMA implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from qqq_a2.engine import Config, load_adjusted_csv
from a2_ema_comparison import ema_signals, replay_signals


def stop_trigger_price(entry_open: float, fee: float, stop_loss: float = 0.08) -> float:
    """Price whose two-fee round-trip return is exactly ``-stop_loss``."""
    if not 0 < stop_loss < 1 or not 0 <= fee < 1:
        raise ValueError("Invalid stop or fee")
    return entry_open * (1 + fee) * (1 - stop_loss) / (1 - fee)


def replay_guarded(
    signals: pd.DataFrame,
    *,
    config: Config = Config(),
    stop_loss: float = 0.08,
    mode: str = "close_next_open",
    slippage_bps: float = 0.0,
) -> dict:
    """Retain EMA/N20 entry/exit state machine; add an always-on loss exit.

    ``close_next_open`` uses only observed completed closes and fills the next
    open, exactly like N20. ``intraday_gap_aware`` simulates a stop-market
    order active starting at the entry fill and throughout the holding period.
    When open < stop, the fill is the open, never a fictional stop-price fill.
    Close-only EMA/N20 exits pending from the prior day execute at today's open
    before any intraday hard-stop simulation.
    """
    if not 0 < stop_loss < 1:
        raise ValueError("Stop must be between zero and 100%")
    if mode not in {"close_next_open", "intraday_gap_aware"}:
        raise ValueError("Unsupported execution model")
    if not 0 <= slippage_bps < 10000:
        raise ValueError("Invalid stop slippage")
    if mode == "intraday_gap_aware" and "adj_low" not in signals.columns:
        raise ValueError("Intraday variant requires adjusted Low")

    stop_label = f"{stop_loss * 100:g}%"
    df = signals.sort_values("Date").reset_index(drop=True)
    cash, shares = float(config.initial_cash), 0.0
    position, pending = None, None
    trades, curve = [], []
    hard_stop_gap_count = 0

    def sell(position: dict, shares: float, *, px: float, i: int,
             exit_date: str, signal_date: str, reason: str,
             gap_past_stop: bool = False) -> float:
        nonlocal hard_stop_gap_count
        proceeds = shares * px * (1 - config.fee_per_side)
        if gap_past_stop:
            hard_stop_gap_count += 1
        trades.append({
            "id": len(trades) + 1,
            "entry_signal_date": position["entry_signal_date"],
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "exit_signal_date": signal_date,
            "exit_date": exit_date,
            "exit_price": float(px),
            "exit_reason": reason,
            "hold_days": int(i - position["entry_index"]),
            "ret_net": float(proceeds / position["entry_cost"] - 1),
            "mfe_close": position["max_close"] / position["entry_price"] - 1,
            "mae_close": position["min_close"] / position["entry_price"] - 1,
            "mae_low": position["min_low"] / position["entry_price"] - 1,
            "equity_after_trade": float(proceeds),
            "stop_target_price": position["stop_price"],
            "gap_below_stop": gap_past_stop,
        })
        return float(proceeds)

    for i, bar in df.iterrows():
        if bar.Date < pd.Timestamp(config.start_date):
            continue
        day = str(bar.Date.date())
        if pending is not None:
            if pending["side"] == "buy":
                if shares:
                    raise AssertionError("Cannot buy while already invested")
                entry = float(bar.adj_open)
                shares = cash / (entry * (1 + config.fee_per_side))
                position = {
                    "entry_signal_date": pending["signal_date"],
                    "entry_date": day,
                    "entry_index": int(i),
                    "entry_price": entry,
                    "entry_cost": float(cash),
                    "stop_price": stop_trigger_price(entry, config.fee_per_side, stop_loss),
                    "max_close": float(bar.adj_close),
                    "min_close": float(bar.adj_close),
                    "min_low": float(bar.adj_low) if "adj_low" in df.columns else float(bar.adj_close),
                }
                cash = 0.0
            else:
                if not shares or position is None:
                    raise AssertionError("Sell without position")
                cash = sell(position, shares, px=float(bar.adj_open), i=int(i),
                            exit_date=day, signal_date=pending["signal_date"],
                            reason=pending["reason"])
                shares, position = 0.0, None
            pending = None

        if shares:
            assert position is not None
            # For a stop hit intraday, we do not use that day's later close to
            # determine a signal, MFE or any other counterfactual action.
            if (mode == "intraday_gap_aware" and
                    float(bar.adj_low) <= position["stop_price"]):
                gap = float(bar.adj_open) < position["stop_price"]
                stop_fill = min(float(bar.adj_open), position["stop_price"])
                stop_fill *= 1 - slippage_bps / 10000
                position["min_low"] = min(position["min_low"], float(bar.adj_low))
                cash = sell(position, shares, px=stop_fill, i=int(i),
                            exit_date=day, signal_date=day,
                            reason=f"{stop_label} intraday stop (gap)" if gap
                                   else f"{stop_label} intraday stop",
                            gap_past_stop=gap)
                shares, position = 0.0, None
            else:
                position["max_close"] = max(position["max_close"], float(bar.adj_close))
                position["min_close"] = min(position["min_close"], float(bar.adj_close))
                if "adj_low" in df.columns:
                    position["min_low"] = min(position["min_low"], float(bar.adj_low))
                age = int(i - position["entry_index"])
                hard_stop_close = (mode == "close_next_open" and
                                   float(bar.adj_close) <= position["stop_price"])
                n20 = (1 <= age <= config.n20_days and bool(bar.protect_cross)
                       and float(bar.adj_close) < position["entry_price"])
                # Simultaneous close signals execute at the identical next
                # open; record ties explicitly, without assuming within-bar order.
                if hard_stop_close:
                    reason = (f"{stop_label} close stop + trend tie" if bool(bar.death)
                              else f"{stop_label} close stop + N20 tie" if n20
                              else f"{stop_label} close stop")
                    pending = {"side": "sell", "signal_date": day, "reason": reason}
                elif bool(bar.death):
                    pending = {"side": "sell", "signal_date": day,
                               "reason": "MA20/60 death cross"}
                elif n20:
                    pending = {"side": "sell", "signal_date": day,
                               "reason": "N20 protection"}
        elif bool(bar.gold):
            pending = {"side": "buy", "signal_date": day,
                       "reason": "FRESH_GOLD_CROSS"}

        curve.append({"Date": day, "Equity": cash + shares * float(bar.adj_close),
                      "Invested": bool(shares), "Pending": pending["side"] if pending else ""})

    t, e = pd.DataFrame(trades), pd.DataFrame(curve)
    if e.empty:
        raise ValueError("No trading sessions")
    e["Drawdown"] = e.Equity / e.Equity.cummax() - 1
    years = (pd.Timestamp(e.Date.iloc[-1]) - pd.Timestamp(config.start_date)).days / 365.25
    winners, losers = t[t.ret_net > 0], t[t.ret_net < 0]
    hard_stops = t.exit_reason.str.startswith(stop_label)
    stats = {
        "completed_trades": len(t),
        "wins": len(winners), "losses": len(losers),
        "win_rate": len(winners) / len(t) if len(t) else None,
        "mean_win": float(winners.ret_net.mean()) if len(winners) else None,
        "mean_loss": float(losers.ret_net.mean()) if len(losers) else None,
        "payoff_ratio": (float(winners.ret_net.mean() / abs(losers.ret_net.mean()))
                          if len(winners) and len(losers) else None),
        "profit_factor": (float(winners.ret_net.sum() / -losers.ret_net.sum())
                          if len(winners) and len(losers) else None),
        "mdd": float(e.Drawdown.min()),
        "end_equity": float(e.Equity.iloc[-1]),
        "cagr": float((e.Equity.iloc[-1] / config.initial_cash) ** (1 / years) - 1),
        "exposure": float(e.Invested.mean()),
        "average_hold_days": float(t.hold_days.mean()),
        "n20_exits": int(t.exit_reason.eq("N20 protection").sum()),
        "trend_exits": int(t.exit_reason.eq("MA20/60 death cross").sum()),
        "hard_stop_exits": int(hard_stops.sum()),
        "hard_stop_gap_exits": hard_stop_gap_count,
        "hard_stop_worse_than_8pct": int((t.loc[hard_stops, "ret_net"] < -stop_loss - 1e-9).sum()),
        "worst_trade_ret": float(t.ret_net.min()),
        "worst_hard_stop_ret": float(t.loc[hard_stops, "ret_net"].min()) if hard_stops.any() else None,
        "active_position_at_end": position is not None,
        "pending_order_at_end": pending,
    }
    return {"trades": t, "equity": e, "stats": stats,
            "open_position": position, "pending_order": pending}


def summary_row(name: str, result: dict) -> dict:
    row = {"strategy": name,
           **{k: v for k, v in result["stats"].items()
              if k != "pending_order_at_end"}}
    row.setdefault("hard_stop_exits", 0)
    row.setdefault("hard_stop_gap_exits", 0)
    row.setdefault("hard_stop_worse_than_8pct", 0)
    row.setdefault("worst_trade_ret", float(result["trades"].ret_net.min()))
    return row


def changed_entry_audit(results: dict) -> pd.DataFrame:
    """Match by actual entry date, not ordinal trade IDs after strategy changes."""
    records = []
    for name, result in results.items():
        trade = result["trades"].copy()
        if trade.entry_date.duplicated().any():
            raise AssertionError(f"Ambiguous entry dates: {name}")
        keep = ["entry_date", "exit_date", "exit_reason", "ret_net"]
        records.append(trade[keep].set_index("entry_date").add_prefix(name + "__"))
    combined = pd.concat(records, axis=1).sort_index()
    base = "EMA_original__"
    variant_names = [name for name in results if name != "EMA_original"]
    changed = pd.Series(False, index=combined.index)
    for name in variant_names:
        prefix = name + "__"
        changed |= (combined[prefix + "exit_date"].ne(combined[base + "exit_date"])
                    | combined[prefix + "ret_net"].sub(combined[base + "ret_net"]).abs().gt(1e-10))
    return combined.loc[changed].reset_index()


def period_slices(results: dict) -> pd.DataFrame:
    eras = [
        ("2000-2003 dotcom", "2000-01-03", "2003-12-31"),
        ("2004-2009 pre/post GFC", "2004-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2021", "2016-01-01", "2021-12-31"),
        ("2022-2026", "2022-01-01", "2026-11-09"),
        ("2011 case", "2011-01-01", "2011-12-31"),
        ("2015 flash crash case", "2015-01-01", "2015-12-31"),
        ("2020-2021 big trend", "2020-01-01", "2021-12-31"),
    ]
    out = []
    for label, start, end in eras:
        for name, result in results.items():
            e = result["equity"].copy()
            dates = pd.to_datetime(e.Date)
            before = e.loc[dates < start, "Equity"]
            start_equity = float(before.iloc[-1]) if len(before) else 100_000.0
            part = e.loc[(dates >= start) & (dates <= end), "Equity"]
            if part.empty:
                continue
            normalized = pd.Series([start_equity, *part.to_list()])
            out.append({"period": label, "strategy": name,
                        "period_return": float(part.iloc[-1] / start_equity - 1),
                        "period_mdd": float((normalized / normalized.cummax() - 1).min())})
    return pd.DataFrame(out)


def draw_chart(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ("#345ea7", "#e17827", "#008c80", "#b05bab")
    fig, (ax, dd) = plt.subplots(2, 1, sharex=True, figsize=(12.5, 7.4),
                                 gridspec_kw={"height_ratios": [2.6, 1.3]})
    for (name, r), color in zip(results.items(), palette):
        df = r["equity"]
        shown = df.groupby(pd.to_datetime(df.Date).dt.to_period("W-FRI"), sort=True).tail(1)
        date = pd.to_datetime(shown.Date)
        ax.plot(date, shown.Equity, label=name, lw=1.55, color=color)
        dd.plot(date, shown.Drawdown * 100, lw=1.15, color=color)
        worst = df.loc[df.Drawdown.idxmin()]
        dd.scatter(pd.Timestamp(worst.Date), worst.Drawdown * 100,
                   color=color, marker="x", s=35, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio USD (log)")
    ax.set_title("EMA A2-N20: original vs always-on 8% loss protection, 2000-2026")
    ax.legend(loc="upper left", fontsize=9)
    dd.set_ylabel("Account drawdown (%)")
    dd.set_xlabel("Year")
    for axis in (ax, dd):
        axis.grid(alpha=.20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": "2026-11-10"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", type=Path, default=ROOT / "research/results/ema_stop8_2026_11_09")
    parser.add_argument("--chart", type=Path, default=ROOT / "assets/a2_ema_stop8_equity_drawdown_2000_2026.svg")
    args = parser.parse_args()
    data = load_adjusted_csv(args.csv)
    data = data[data.Date <= pd.Timestamp(args.end)].reset_index(drop=True)
    if data.empty or data.Date.iloc[-1] != pd.Timestamp(args.end):
        raise ValueError("Requested full final bar not found")
    s = ema_signals(data)
    results = {
        "EMA_original": replay_signals(s),
        "EMA_stop8_close_next_open": replay_guarded(s, mode="close_next_open"),
        "EMA_stop8_intraday": replay_guarded(s, mode="intraday_gap_aware"),
        "EMA_stop8_intraday_slip50bp": replay_guarded(
            s, mode="intraday_gap_aware", slippage_bps=50),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, result in results.items():
        row = summary_row(name, result)
        row["data_asof"] = args.end
        rows.append(row)
        result["trades"].to_csv(args.outdir / f"{name}_trades.csv", index=False)
    pd.DataFrame(rows).to_csv(args.outdir / "summary.csv", index=False)
    changed_entry_audit(results).to_csv(args.outdir / "changed_trades.csv", index=False)
    period_slices(results).to_csv(args.outdir / "period_slices.csv", index=False)
    frames = []
    for name, result in results.items():
        e = result["equity"].copy()
        e["strategy"] = name
        frames.append(e)
    pd.concat(frames, ignore_index=True).assign(
        YearMonth=lambda x: x.Date.str[:7]
    ).groupby(["strategy", "YearMonth"], sort=True).tail(1).to_csv(
        args.outdir / "monthly_equity.csv", index=False)
    draw_chart(results, args.chart)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
