"""Fourth QQQ benchmark: EMA early recovery + completed-month EMA5 + dynamic N20.

Hypothesis: a fast EMA10/20 re-entry can precede the slow EMA20/60 trend
confirmation; starting N20 at the first fill can make its guard expire before
the new trend is established. For fast entries, keep the losing-position
EMA10/20 protection until slow confirmation and for 20 sessions thereafter.

Do not modify the frozen SMA A2 engine or the original EMA 7%-stop baseline.
Signals only use completed daily bars and the *previous completed* month;
all orders execute at the next regular session's adjusted open.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_comparison import ema_signals
from a2_ema_hard_stop8 import stop_trigger_price
from qqq_a2.engine import Config


def completed_month_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA5 on monthly closes; shift availability to NEXT month.

    A month is a calendar month with at least one observed session. The
    previous *completed* month's EMA5 is the only monthly input to a trade.
    For a partial latest month, its evolving value never enters a signal.
    """
    s = ema_signals(data).reset_index(drop=True)
    monthly = s.groupby(s.Date.dt.to_period("M"), sort=True).tail(1).copy()
    monthly["month"] = monthly.Date.dt.to_period("M")
    monthly["monthly_ema5"] = monthly.adj_close.ewm(
        span=5, adjust=False, min_periods=5).mean()
    monthly["monthly_ema5_up"] = monthly.monthly_ema5.diff() > 0
    # Use the last fully closed calendar month; no current-month final close.
    known = s.Date.dt.to_period("M") - 1
    monthly_filter = known.map(monthly.set_index("month").monthly_ema5_up)
    s["prev_completed_month_ema5_up"] = monthly_filter.eq(True).fillna(False)
    s["ema10_20_gold"] = (s.ma10 > s.ma20) & (s.ma10.shift(1) <= s.ma20.shift(1))
    return s


def replay_dynamic(
    signals: pd.DataFrame,
    *,
    config: Config = Config(),
    stop_loss: float = .07,
    fast_reentry: bool = True,
    reset_n20_at_confirmation: bool = True,
    n20_confirm_closes: int = 1,
) -> dict:
    """Replay a fully specified strategy with optional negative controls.

    "fast_reentry=False" matches the current EMA+7% baseline exactly;
    "reset_n20_at_confirmation=False" matches fast re-entry with the old
    fixed-from-buy 20-day guard. n20_confirm_closes=2/3 are *negative
    control* delays; the fourth published benchmark uses value 1.

    An N20 exit after fast entry also permits the next fresh EMA10/20 cross.
    EMA20/60 and 7% exits do so too. Normal fresh EMA20/60 gold entries are
    always allowed, including without a positive completed-month filter.
    """
    if not 0 < stop_loss < 1 or n20_confirm_closes not in (1, 2, 3):
        raise ValueError("Invalid stop/confirmation settings")
    df = signals.sort_values("Date").reset_index(drop=True)
    required = {"gold", "death", "protect_cross", "ema10_20_gold",
                "prev_completed_month_ema5_up", "ma10", "ma20", "ma60"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required signal columns: {required - set(df.columns)}")
    cash, shares, position, pending = float(config.initial_cash), 0., None, None
    eligible_fast = False
    ledger, curve = [], []
    for i, row in df.iterrows():
        if row.Date < pd.Timestamp(config.start_date):
            continue
        day = str(row.Date.date())
        if pending is not None:
            if pending["side"] == "buy":
                if shares or position is not None:
                    raise AssertionError("Double buy")
                entry = float(row.adj_open)
                shares = cash / (entry * (1 + config.fee_per_side))
                # At the OPEN, only yesterday's EMA state is knowable.
                confirmed = i > 0 and bool(df.iloc[i - 1].ma20 > df.iloc[i - 1].ma60)
                position = {
                    "entry_signal_date": pending["signal_date"],
                    "entry_date": day, "entry_index": i, "entry_price": entry,
                    "entry_cost": cash, "entry_kind": pending["kind"],
                    "trend_confirm_index": (i - 1 if confirmed else None),
                    "trend_confirm_date": (str(df.iloc[i - 1].Date.date()) if confirmed else None),
                    "ema10_below20_closes": 0,
                    "max_close": float(row.adj_close),
                    "min_close": float(row.adj_close),
                }
                cash, eligible_fast = 0., False
            else:
                if not shares or position is None:
                    raise AssertionError("Sell with no position")
                px = float(row.adj_open)
                cash = shares * px * (1 - config.fee_per_side)
                ledger.append({
                    "id": len(ledger) + 1,
                    "entry_signal_date": position["entry_signal_date"],
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "entry_kind": position["entry_kind"],
                    "trend_confirm_date": position["trend_confirm_date"],
                    "exit_signal_date": pending["signal_date"],
                    "exit_date": day, "exit_price": px,
                    "exit_reason": pending["reason"],
                    "hold_days": int(i - position["entry_index"]),
                    "ret_net": float(cash / position["entry_cost"] - 1),
                    "mfe_close": position["max_close"] / position["entry_price"] - 1,
                    "mae_close": position["min_close"] / position["entry_price"] - 1,
                    "equity_after_trade": float(cash),
                })
                eligible_fast = True
                position, shares = None, 0.
            pending = None

        if shares:
            assert position is not None
            position["max_close"] = max(position["max_close"], float(row.adj_close))
            position["min_close"] = min(position["min_close"], float(row.adj_close))
            age = i - position["entry_index"]
            is_fast = position["entry_kind"] == "FAST_EMA10_20_MONTH_EMA5"
            if is_fast and position["trend_confirm_index"] is None and bool(row.gold):
                position["trend_confirm_index"] = i
                position["trend_confirm_date"] = day
            position["ema10_below20_closes"] = (
                position["ema10_below20_closes"] + 1
                if bool(row.ma10 < row.ma20) else 0)
            n20_active = 1 <= age <= config.n20_days
            if is_fast and reset_n20_at_confirmation:
                trend_idx = position["trend_confirm_index"]
                n20_active = age >= 1 and (
                    trend_idx is None or 0 <= i - trend_idx <= config.n20_days)
            n20_signal = bool(row.protect_cross)
            if is_fast and reset_n20_at_confirmation and n20_confirm_closes > 1:
                n20_signal = (position["ema10_below20_closes"] == n20_confirm_closes)
            n20_hit = bool(n20_active and n20_signal and
                           float(row.adj_close) < position["entry_price"])
            hard_hit = float(row.adj_close) <= stop_trigger_price(
                position["entry_price"], config.fee_per_side, stop_loss)
            if hard_hit:
                label = f"{stop_loss * 100:g}% close stop"
                reason = (label + " + trend tie" if bool(row.death)
                          else label + " + N20 tie" if n20_hit else label)
                pending = {"side": "sell", "signal_date": day, "reason": reason}
            elif bool(row.death):
                pending = {"side": "sell", "signal_date": day,
                           "reason": "MA20/60 death cross"}
            elif n20_hit:
                label = (f"N20 extended {n20_confirm_closes}d" if is_fast
                         and reset_n20_at_confirmation else "N20 protection")
                pending = {"side": "sell", "signal_date": day, "reason": label}
        elif bool(row.gold) or (fast_reentry and eligible_fast and
                                bool(row.prev_completed_month_ema5_up) and
                                bool(row.ema10_20_gold)):
            normal = bool(row.gold)
            pending = {"side": "buy", "signal_date": day,
                       "kind": "REGULAR_EMA20_60" if normal else "FAST_EMA10_20_MONTH_EMA5"}
        curve.append({"Date": day, "Equity": cash + shares * float(row.adj_close),
                      "Invested": bool(shares),
                      "Pending": pending["side"] if pending is not None else ""})

    trades, equity = pd.DataFrame(ledger), pd.DataFrame(curve)
    if equity.empty:
        raise ValueError("No trading sessions")
    equity["Drawdown"] = equity.Equity / equity.Equity.cummax() - 1
    years = (pd.Timestamp(equity.Date.iloc[-1]) - pd.Timestamp(config.start_date)).days / 365.25
    winners, losers = trades[trades.ret_net > 0], trades[trades.ret_net < 0]
    stats = {
        "end_equity": float(equity.Equity.iloc[-1]),
        "cagr": float((equity.Equity.iloc[-1] / config.initial_cash) ** (1/years) - 1),
        "mdd": float(equity.Drawdown.min()),
        "completed_trades": len(trades), "wins": len(winners), "losses": len(losers),
        "win_rate": len(winners) / len(trades) if len(trades) else None,
        "payoff_ratio": float(winners.ret_net.mean() / abs(losers.ret_net.mean()))
        if len(winners) and len(losers) else None,
        "profit_factor": float(winners.ret_net.sum() / -losers.ret_net.sum())
        if len(winners) and len(losers) else None,
        "exposure": float(equity.Invested.mean()),
        "active_position_at_end": position is not None,
        "pending_order_at_end": pending,
    }
    return {"trades": trades, "equity": equity, "stats": stats,
            "open_position": position, "pending_order": pending}
