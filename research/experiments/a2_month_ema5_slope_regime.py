"""Independent month-EMA5 slope/curvature entry filters for Dynamic N20.

Research-only. Original strategy files and exit mechanics remain unchanged.
Signals use only the last completed month, except explicitly named MTD
experiments which use today's *completed daily close* to construct a provisional
monthly EMA. All orders still execute at the next regular-session open.

Run with the same lawful, privately acquired adjusted QQQ daily bars used by
the four-way research:

    python research/experiments/a2_month_ema5_slope_regime.py \
        --old /path/to/old/QQQ.csv --new /path/to/new/QQQ.csv \
        --end 2026-07-13 --outdir research/results/month_ema5_slope_2026_07_13

For the canonical November extension, append --update with the privately
downloaded 2026-07 through 2026-11 daily observations and set --end 2026-11-09.

Only derived signals, trade ledgers, aggregate statistics and charts are saved;
raw market bars are never published.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals, replay_dynamic
from a2_ema_hard_stop8 import stop_trigger_price
from qqq_a2.engine import Config, combine_rebased_sources
from qqq_a2.updates import extend_adjusted_history


VARIANTS = (
    "baseline",
    "up_only",
    "down_relief_1_abs",
    "down_relief_2_abs",
    "down_relief_2_pct",
    "up_slow_veto_1_abs",
    "up_slow_veto_2_abs",
    "both_1_abs",
    "both_2_abs",
    "both_2_pct",
    "mtd_down_relief_1_abs",
    "mtd_both_1_abs",
)


def with_month_slope_features(signals: pd.DataFrame) -> pd.DataFrame:
    """Attach strictly as-of monthly EMA5 slopes, including a labeled MTD proxy.

    The most recent partial calendar month is never mapped as a *completed*
    month for dates inside it. Initial 5-month EMA warmup matches baseline.
    """
    s = signals.copy()
    m = s.groupby(s.Date.dt.to_period("M"), sort=True).tail(1).copy()
    m["period"] = m.Date.dt.to_period("M")
    m["month_ema5"] = m.adj_close.ewm(
        span=5, adjust=False, min_periods=5).mean()
    m["slope_abs"] = m.month_ema5.diff()
    m["slope_pct"] = m.month_ema5.pct_change(fill_method=None)
    for suffix in ("abs", "pct"):
        m[f"prev_{suffix}"] = m[f"slope_{suffix}"].shift(1)
        m[f"prev2_{suffix}"] = m[f"slope_{suffix}"].shift(2)

    last_closed = s.Date.dt.to_period("M") - 1
    by_period = m.set_index("period")
    for col in ("month_ema5", "slope_abs", "slope_pct", "prev_abs", "prev_pct",
                "prev2_abs", "prev2_pct"):
        s["known_" + col] = last_closed.map(by_period[col]).astype(float)

    # At today's completed close, construct a monthly EMA for the unfinished
    # calendar month. This is a *provisional* monthly value, never a finalized
    # monthly bar and never suitable for tomorrow's stale cache without refresh.
    s["mtd_ema5"] = (s.adj_close / 3 + 2 * s.known_month_ema5 / 3)
    s["mtd_slope_abs"] = s.mtd_ema5 - s.known_month_ema5

    x = s.known_slope_abs
    p = s.known_prev_abs
    pp = s.known_prev2_abs
    x_pct = s.known_slope_pct
    p_pct = s.known_prev_pct
    pp_pct = s.known_prev2_pct
    pos_slow = (x > 0) & (x < p)
    pos_slow2 = (x > 0) & (p > 0) & (pp > 0) & (x < p) & (p < pp)
    neg_relief1 = (x < 0) & (p < 0) & (x > p)
    neg_relief2 = neg_relief1 & (pp < 0) & (p > pp)
    neg_relief2_pct = ((x_pct < 0) & (p_pct < 0) & (pp_pct < 0)
                       & (x_pct > p_pct) & (p_pct > pp_pct))
    pos_slow_pct = (x_pct > 0) & (x_pct < p_pct)

    rules = {
        "baseline": pd.Series(True, index=s.index),
        "up_only": x > 0,
        "down_relief_1_abs": (x > 0) | neg_relief1,
        "down_relief_2_abs": (x > 0) | neg_relief2,
        "down_relief_2_pct": (x_pct > 0) | neg_relief2_pct,
        "up_slow_veto_1_abs": ~pos_slow,
        "up_slow_veto_2_abs": ~pos_slow2,
        "both_1_abs": (~pos_slow) & ((x > 0) | neg_relief1),
        "both_2_abs": (~pos_slow) & ((x > 0) | neg_relief2),
        "both_2_pct": (~pos_slow_pct) & ((x_pct > 0) | neg_relief2_pct),
        "mtd_down_relief_1_abs": (
            (s.mtd_slope_abs > 0) |
            ((s.mtd_slope_abs < 0) & (x < 0) & (s.mtd_slope_abs > x))),
        "mtd_both_1_abs": (
            (s.mtd_slope_abs > x) &
            ((s.mtd_slope_abs > 0) | (x < 0))),
    }
    for name, rule in rules.items():
        s["gate_" + name] = rule.fillna(False).astype(bool)
    # Baseline fast entries still additionally require completed monthly EMA5
    # slope > 0; the new filters apply to both normal and fast entries.
    return s


def replay_gated(signals: pd.DataFrame, variant: str,
                 config: Config = Config(), stop_loss: float = .07,
                 defer_rejected_gold: bool = False,
                 hard_stop_fill: str = "next_open",
                 close_fill_adverse_slippage: float = 0.,
                 n20_day0: str = "none",
                 entry_day_close_guard: float | None = None,
                 entry_day_intraday_stop: float | None = None,
                 intraday_adverse_slippage: float = 0.,
                 regular_n20_extension: str = "none",
                 extended_n20_days: int = 30,
                 defer_rejected_fast: bool = False,
                 fast_negative_month_mode: str = "none",
                 deferred_fast_slow_state: str = "any",
                 profit_trail_pct: float | None = None,
                 profit_min_gain: float = .10,
                 profit_atr_mult: float | None = None,
                 profit_confirmation: str = "none",
                 trend_spread_atr_min: float | None = None,
                 trend_spread_exit: str = "fast_death",
                 trend_spread_reentry: str = "none",
                 trend_spread_min_gain: float = .10,
                 trend_spread_lookback: int = 1) -> dict:
    """Dedicated research replay; no changes to baseline N20/trend exits.

    The *original* gold signal is used for in-position trend confirmation;
    entry gates are consulted ONLY when flat. Suppressing the gold indicator
    globally would accidentally change Dynamic N20 exit timing.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown variant: " + variant)
    if fast_negative_month_mode not in ("none", "decelerating", "accelerating"):
        raise ValueError("Invalid negative-month FAST experiment")
    if deferred_fast_slow_state not in ("any", "bull", "bear"):
        raise ValueError("Invalid delayed FAST slow trend filter")
    if ((fast_negative_month_mode != "none" or
         deferred_fast_slow_state != "any") and not defer_rejected_fast):
        raise ValueError("Regime experiments require deferred FAST enabled")
    if trend_spread_exit not in ("fast_death", "early_narrow"):
        raise ValueError("Invalid spread exit")
    if trend_spread_reentry not in ("none", "ema10_20_gold", "ema5_10_gold"):
        raise ValueError("Invalid spread re-entry")
    if not isinstance(trend_spread_lookback, int) or not 1 <= trend_spread_lookback <= 250:
        raise ValueError("Trend spread lookback must be 1–250 complete sessions")
    if trend_spread_atr_min is not None and trend_spread_atr_min < 0:
        raise ValueError("Spread/ATR threshold must be nonnegative")
    if trend_spread_atr_min is not None and not 0 < trend_spread_min_gain < 1:
        raise ValueError("Invalid trend-spread minimum profit")
    if trend_spread_atr_min is None and trend_spread_reentry != "none":
        raise ValueError("Spread re-entry requires opt-in spread exit")
    if profit_trail_pct is not None and not 0 < profit_trail_pct < 1:
        raise ValueError("Invalid profit trail percent")
    if profit_atr_mult is not None and not profit_atr_mult > 0:
        raise ValueError("Invalid ATR multiple")
    if profit_trail_pct is not None and profit_atr_mult is not None:
        raise ValueError("Use a percentage OR ATR profit trail, not both")
    if (profit_trail_pct is not None or profit_atr_mult is not None) and not 0 < profit_min_gain < 1:
        raise ValueError("Invalid minimum favorable profit")
    if profit_confirmation not in ("none", "below_ema20", "fast_death"):
        raise ValueError("Unsupported profit-trail confirmation")
    if hard_stop_fill not in ("next_open", "same_close_oracle"):
        raise ValueError("Invalid stop fill model")
    if not 0 <= close_fill_adverse_slippage < 1:
        raise ValueError("Invalid close-fill adverse slippage")
    if n20_day0 not in ("none", "fast", "regular", "all"):
        raise ValueError("Invalid day-zero N20 entry category")
    extension_rules = ("none", "underwater_at20", "underwater_2of5",
                       "underwater_2of5_spread_down", "underwater_1of5_spread_down",
                       "underwater_2of10_spread_down", "spread_down")
    if regular_n20_extension not in extension_rules:
        raise ValueError("Invalid regular N20 extension rule")
    if extended_n20_days < config.n20_days:
        raise ValueError("Extended N20 must be at least the original N20")
    for threshold in (entry_day_close_guard, entry_day_intraday_stop):
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError("Invalid entry-day loss threshold")
    if not 0 <= intraday_adverse_slippage < 1:
        raise ValueError("Invalid intraday stop slippage")
    gate_col = "gate_" + variant
    df = signals.sort_values("Date").reset_index(drop=True)
    # Historical research-only ATR20: today's complete bar is available
    # only after close. No current/next intraday high or low informs an exit.
    if trend_spread_atr_min is not None:
        prev_close = df.adj_close.shift(1)
        true_range = pd.concat([
            df.adj_high - df.adj_low,
            (df.adj_high - prev_close).abs(),
            (df.adj_low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["spread_atr20"] = true_range.rolling(20, min_periods=20).mean()
        df["spread_in_atr"] = (df.ma20 - df.ma60) / df.spread_atr20
        # The maximum is over completed trading days, including today's
        # close. A 40-day lookback may include dates before this entry.
        df["spread_lookback_max"] = df.spread_in_atr.rolling(
            trend_spread_lookback, min_periods=1).max()
        # Strictly past-or-present closing-bar features: no future K lines.
        if trend_spread_exit == "early_narrow":
            spread = df.ma10 - df.ma20
            df["spread_early_narrow"] = (
                (spread < spread.shift(1)) &
                (spread.shift(1) < spread.shift(2)) &
                (spread.shift(2) < spread.shift(3)) &
                (df.adj_close < df.ma10) &
                (df.ma10 < df.ma10.shift(3))
            )
        if trend_spread_reentry == "ema5_10_gold":
            ema5 = df.adj_close.ewm(span=5, adjust=False).mean()
            df["spread_ema5_gold"] = (
                (ema5 > df.ma10) & (ema5.shift(1) <= df.ma10.shift(1)))
    if profit_atr_mult is not None:
        prev_close = df.adj_close.shift(1)
        true_range = pd.concat([
            df.adj_high - df.adj_low,
            (df.adj_high - prev_close).abs(),
            (df.adj_low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["profit_guard_atr20"] = true_range.rolling(20, min_periods=20).mean()
    cash, shares, position, pending = float(config.initial_cash), 0., None, None
    eligible_fast, deferred_gold, deferred_fast = False, False, False
    early_spread_reentry_eligible = False
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
                confirmed = i > 0 and bool(df.iloc[i - 1].ma20 > df.iloc[i - 1].ma60)
                position = {
                    "entry_signal_date": pending["signal_date"],
                    "entry_date": day, "entry_index": i,
                    "entry_price": entry, "entry_cost": cash,
                    "entry_kind": pending["kind"],
                    "trend_confirm_index": (i - 1 if confirmed else None),
                    "trend_confirm_date": (
                        str(df.iloc[i - 1].Date.date()) if confirmed else None),
                    "max_close": float(row.adj_close),
                    "min_close": float(row.adj_close),
                    # Only the running past closing high, known as of each day.
                    "profit_guard_peak_close": entry,
                    "n20_extension_qualified": False,
                }
                cash, eligible_fast, deferred_gold, deferred_fast = 0., False, False, False
                early_spread_reentry_eligible = False
            else:
                if not shares or position is None:
                    raise AssertionError("Sell without position")
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
                eligible_fast, deferred_gold, deferred_fast = True, False, False
                early_spread_reentry_eligible = pending["reason"].startswith(
                    "research trend spread")
                position, shares = None, 0.
            pending = None

        if shares:
            assert position is not None
            position["max_close"] = max(position["max_close"], float(row.adj_close))
            position["min_close"] = min(position["min_close"], float(row.adj_close))
            age = i - position["entry_index"]
            fast = position["entry_kind"] == "FAST_EMA10_20_MONTH_EMA5"
            if fast and position["trend_confirm_index"] is None and bool(row.gold):
                position["trend_confirm_index"] = i
                position["trend_confirm_date"] = day
            # This is an independent research switch; the published A2
            # benchmark continues to start N20 on the first FULL day held.
            # A day-zero cross is evaluated at the entry session's CLOSE and
            # can only execute at the following session's OPEN.
            entry_day_covered = age == 0 and (
                n20_day0 == "all" or
                (n20_day0 == "fast" and fast) or
                (n20_day0 == "regular" and not fast))
            if age == config.n20_days and not fast and regular_n20_extension != "none":
                recent5 = df.iloc[i - 4:i + 1]
                recent10 = df.iloc[i - 9:i + 1]
                under5 = int((recent5.adj_close < position["entry_price"]).sum())
                under10 = int((recent10.adj_close < position["entry_price"]).sum())
                spread_now = float(row.ma10 / row.ma20 - 1)
                prev = df.iloc[i - 5]
                spread_down = spread_now < float(prev.ma10 / prev.ma20 - 1)
                rules = {
                    "underwater_at20": float(row.adj_close) < position["entry_price"],
                    "underwater_2of5": under5 >= 2,
                    "underwater_2of5_spread_down": under5 >= 2 and spread_down,
                    "underwater_1of5_spread_down": under5 >= 1 and spread_down,
                    "underwater_2of10_spread_down": under10 >= 2 and spread_down,
                    "spread_down": spread_down,
                }
                position["n20_extension_qualified"] = bool(rules[regular_n20_extension])
            n20_active = (1 <= age <= config.n20_days or entry_day_covered or
                          (not fast and position["n20_extension_qualified"] and
                           config.n20_days < age <= extended_n20_days))
            if fast:
                trend_i = position["trend_confirm_index"]
                n20_active = (age >= 1 or entry_day_covered) and (
                    trend_i is None or 0 <= i - trend_i <= config.n20_days)
            n20_hit = (n20_active and bool(row.protect_cross) and
                       float(row.adj_close) < position["entry_price"])
            hard_hit = float(row.adj_close) <= stop_trigger_price(
                position["entry_price"], config.fee_per_side, stop_loss)
            # A stop-market order placed immediately after entry is approximated
            # from the entry-day LOW. Fill at the threshold is NOT guaranteed;
            # adverse slippage is an explicit sensitivity parameter. Unlike a
            # closing signal, this is executable inside the first session.
            intraday_hit = (age == 0 and entry_day_intraday_stop is not None and
                            float(row.adj_low) <= position["entry_price"] *
                            (1 - entry_day_intraday_stop))
            if intraday_hit:
                px = (position["entry_price"] * (1 - entry_day_intraday_stop) *
                      (1 - intraday_adverse_slippage))
                cash = shares * px * (1 - config.fee_per_side)
                ledger.append({
                    "id": len(ledger) + 1,
                    "entry_signal_date": position["entry_signal_date"],
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "entry_kind": position["entry_kind"],
                    "trend_confirm_date": position["trend_confirm_date"],
                    "exit_signal_date": day, "exit_date": day,
                    "exit_price": px,
                    "exit_reason": f"entry-day {entry_day_intraday_stop * 100:g}% intraday stop [modeled]",
                    "hold_days": age,
                    "ret_net": float(cash / position["entry_cost"] - 1),
                    # The daily close is AFTER the hypothetical intraday sale;
                    # attributing it as a held-period excursion would be wrong.
                    "mfe_close": float("nan"),
                    "mae_close": float("nan"),
                    "equity_after_trade": float(cash),
                })
                eligible_fast, deferred_gold, deferred_fast = True, False, False
                shares, position, pending = 0., None, None
            elif hard_hit:
                label = f"{stop_loss * 100:g}% close stop"
                reason = (label + " + trend tie" if bool(row.death)
                          else label + " + N20 tie" if n20_hit else label)
                if hard_stop_fill == "same_close_oracle":
                    # Upper bound only: the fully-known closing bar is not
                    # available until AFTER the closing auction. A real MOC
                    # order must be submitted earlier based on then-live data.
                    px = float(row.adj_close) * (1 - close_fill_adverse_slippage)
                    cash = shares * px * (1 - config.fee_per_side)
                    ledger.append({
                        "id": len(ledger) + 1,
                        "entry_signal_date": position["entry_signal_date"],
                        "entry_date": position["entry_date"],
                        "entry_price": position["entry_price"],
                        "entry_kind": position["entry_kind"],
                        "trend_confirm_date": position["trend_confirm_date"],
                        "exit_signal_date": day, "exit_date": day,
                        "exit_price": px, "exit_reason": reason + " [oracle close]",
                        "hold_days": age,
                        "ret_net": float(cash / position["entry_cost"] - 1),
                        "mfe_close": position["max_close"] / position["entry_price"] - 1,
                        "mae_close": position["min_close"] / position["entry_price"] - 1,
                        "equity_after_trade": float(cash),
                    })
                    eligible_fast, deferred_gold, deferred_fast = True, False, False
                    shares, position, pending = 0., None, None
                else:
                    pending = {"side": "sell", "signal_date": day, "reason": reason}
            elif bool(row.death):
                pending = {"side": "sell", "signal_date": day,
                           "reason": "MA20/60 death cross"}
            elif n20_hit:
                pending = {"side": "sell", "signal_date": day,
                           "reason": ("N20 extended 1d" if fast else
                                      "N20 conditional extension" if age > config.n20_days
                                      else "N20 protection")}
            elif (age == 0 and entry_day_close_guard is not None and
                  float(row.adj_close) <= position["entry_price"] *
                  (1 - entry_day_close_guard)):
                pending = {"side": "sell", "signal_date": day,
                           "reason": f"entry-day close <= -{entry_day_close_guard * 100:g}%"}
            elif trend_spread_atr_min is not None and (
                    float(row.ma20) > float(row.ma60) and
                    pd.notna(row.spread_lookback_max) and
                    float(row.spread_lookback_max) >= trend_spread_atr_min and
                    position["profit_guard_peak_close"] >=
                    position["entry_price"] * (1 + trend_spread_min_gain)):
                # Compare fast death and earlier three-day spread narrowing.
                # All signals confirmed at end of current day; fill next open.
                weak = (bool(row.protect_cross) if
                        trend_spread_exit == "fast_death" else
                        bool(row.spread_early_narrow))
                if weak:
                    pending = {
                        "side": "sell", "signal_date": day,
                        "reason": f"research trend spread {trend_spread_exit}",
                    }
            elif (profit_trail_pct is not None or profit_atr_mult is not None):
                past_peak = position["profit_guard_peak_close"]
                gain = past_peak / position["entry_price"] - 1
                close = float(row.adj_close)
                if profit_atr_mult is None:
                    drawdown_trigger = close <= past_peak * (1 - profit_trail_pct)
                else:
                    atr = float(row.profit_guard_atr20)
                    drawdown_trigger = pd.notna(atr) and past_peak - close >= profit_atr_mult * atr
                confirm = (
                    profit_confirmation == "none" or
                    (profit_confirmation == "below_ema20" and close < float(row.ma20)) or
                    (profit_confirmation == "fast_death" and bool(row.protect_cross))
                )
                if gain >= profit_min_gain and drawdown_trigger and confirm:
                    label = (f"{profit_trail_pct * 100:g}%"
                             if profit_trail_pct is not None
                             else f"{profit_atr_mult:g}ATR")
                    pending = {
                        "side": "sell", "signal_date": day,
                        "reason": f"research profit giveback {label} + {profit_confirmation}",
                    }
            # Update only after checks: the signal cannot know a future peak.
            if position is not None and shares:
                position["profit_guard_peak_close"] = max(
                    position["profit_guard_peak_close"], float(row.adj_close))
        else:
            if bool(row.death):
                deferred_gold = False
                early_spread_reentry_eligible = False
            if bool(row.gold) and not bool(row[gate_col]):
                deferred_gold = True
            regular = bool(row.gold) or (
                defer_rejected_gold and deferred_gold and
                bool(row.ma20 > row.ma60))
            # V3.1 optional: retain a FAST crossing rejected by either
            # monthly condition. Recheck only when flat; eligibility begins
            # after a SELL and is invalidated by the next EMA10/20 death.
            if defer_rejected_fast and eligible_fast:
                if bool(row.protect_cross) or not bool(row.ma10 > row.ma20):
                    deferred_fast = False
                if (bool(row.ema10_20_gold) and
                        (not bool(row.prev_completed_month_ema5_up)
                         or not bool(row[gate_col]))):
                    deferred_fast = True
            # Independent H1 research: negative completed-month EMA5 slope
            # may be improving (less negative) or accelerating (more negative).
            # The acceleration arm is a negative control, not a chosen rule.
            # All monthly inputs are from completed months only.
            negative_relief = False
            if fast_negative_month_mode != "none":
                cur = float(row.known_slope_abs)
                prev = float(row.known_prev_abs)
                if cur < 0 and prev < 0:
                    negative_relief = (
                        cur > prev if fast_negative_month_mode == "decelerating"
                        else cur < prev)
            # H2 research: filter ONLY deferred FAST candidates; fresh valid
            # crosses and regular EMA20/60 entries keep V3/V3.1 semantics.
            waiting_fast = (defer_rejected_fast and deferred_fast
                            and not bool(row.ema10_20_gold))
            slow_bull = bool(row.ma20 > row.ma60)
            slow_state_ok = (
                not waiting_fast or deferred_fast_slow_state == "any"
                or (deferred_fast_slow_state == "bull" and slow_bull)
                or (deferred_fast_slow_state == "bear" and not slow_bull))
            fast = (eligible_fast and slow_state_ok
                    and (bool(row.prev_completed_month_ema5_up)
                         or negative_relief)
                    and (bool(row.ema10_20_gold) or
                         (defer_rejected_fast and deferred_fast
                          and bool(row.ma10 > row.ma20))))
            fast_reentry = (
                early_spread_reentry_eligible and
                trend_spread_reentry != "none" and
                bool(row.ma20 > row.ma60) and
                (bool(row.ema10_20_gold) if
                 trend_spread_reentry == "ema10_20_gold"
                 else bool(row.spread_ema5_gold))
            )
            if bool(row[gate_col]) and (regular or fast) or fast_reentry:
                pending = {
                    "side": "buy", "signal_date": day,
                    "kind": ("REGULAR_EMA20_60" if
                             bool(row[gate_col]) and regular else
                             "FAST_EMA10_20_MONTH_EMA5"),
                }
        curve.append({"Date": day, "Equity": cash + shares * float(row.adj_close),
                      "Invested": bool(shares),
                      "Pending": pending["side"] if pending is not None else ""})

    trades = pd.DataFrame(ledger)
    if trades.empty:
        trades = pd.DataFrame(columns=["id", "entry_signal_date", "entry_date",
                                       "entry_price", "entry_kind", "trend_confirm_date",
                                       "exit_signal_date", "exit_date", "exit_price",
                                       "exit_reason", "hold_days", "ret_net", "mfe_close",
                                       "mae_close", "equity_after_trade"])
    equity = pd.DataFrame(curve)
    equity["Drawdown"] = equity.Equity / equity.Equity.cummax() - 1
    years = (pd.Timestamp(equity.Date.iloc[-1]) -
             pd.Timestamp(config.start_date)).days / 365.25
    w, l = trades.loc[trades.ret_net > 0], trades.loc[trades.ret_net < 0]
    stats = {
        "end_equity": float(equity.Equity.iloc[-1]),
        "cagr": float((equity.Equity.iloc[-1] / config.initial_cash) ** (1 / years) - 1),
        "mdd": float(equity.Drawdown.min()),
        "completed_trades": len(trades), "wins": len(w), "losses": len(l),
        "win_rate": len(w) / len(trades) if len(trades) else None,
        "payoff_ratio": float(w.ret_net.mean() / abs(l.ret_net.mean())) if len(w) and len(l) else None,
        "profit_factor": float(w.ret_net.sum() / -l.ret_net.sum()) if len(w) and len(l) else None,
        "exposure": float(equity.Invested.mean()),
        "worst_trade": float(trades.ret_net.min()) if len(trades) else None,
        "active_position_at_end": position is not None,
        "pending_order_at_end": pending,
    }
    return {"trades": trades, "equity": equity, "stats": stats,
            "open_position": position, "pending_order": pending}


def comparison(data: pd.DataFrame, *, end: str, config: Config = Config()) -> dict:
    data = data.loc[data.Date <= pd.Timestamp(end)].copy()
    if data.empty or str(data.Date.iloc[-1].date()) != end:
        raise ValueError("Requested endpoint not present in complete data: " + end)
    s = with_month_slope_features(completed_month_signals(data))
    baseline = replay_dynamic(s, config=config)
    runs = {v: replay_gated(s, v, config=config) for v in VARIANTS}
    # Negative control: an otherwise valid cross remains pending while its
    # 20/60 bull regime holds. This isolates the permanent missed-cross cost.
    for v in ("up_only", "down_relief_1_abs", "down_relief_2_abs",
              "up_slow_veto_1_abs", "both_1_abs", "both_2_abs",
              "both_2_pct"):
        runs[v + "__defer"] = replay_gated(
            s, v, config=config, defer_rejected_gold=True)
    import pandas.testing as pdt
    pdt.assert_frame_equal(runs["baseline"]["trades"], baseline["trades"])
    pdt.assert_frame_equal(runs["baseline"]["equity"], baseline["equity"])

    rows = []
    for variant, result in runs.items():
        st = result["stats"]
        rows.append({"variant": variant, **st,
                     "end": end,
                     "return_vs_baseline": st["end_equity"] /
                     baseline["stats"]["end_equity"] - 1})
    summary = pd.DataFrame(rows)

    sample_dates = ["2007-12-06", "2007-12-24", "2008-04-21", "2008-08-14",
                    "2009-03-30"]
    event = s.loc[s.Date.isin(pd.to_datetime(sample_dates)), [
        "Date", "gold", "known_slope_abs", "known_prev_abs",
        "known_prev2_abs", "known_slope_pct", "known_prev_pct",
        "mtd_slope_abs", *["gate_" + v for v in VARIANTS]]].copy()
    event["Date"] = event.Date.dt.strftime("%Y-%m-%d")

    trade_events = []
    selected = baseline["trades"]
    for variant, run in runs.items():
        if variant == "baseline":
            continue
        t = run["trades"]
        base_keys = set(selected.entry_signal_date)
        candidate_keys = set(t.entry_signal_date)
        for trade in selected.itertuples(index=False):
            if trade.entry_signal_date not in candidate_keys:
                trade_events.append({"variant": variant, "difference": "baseline_only",
                                     "entry_signal_date": trade.entry_signal_date,
                                     "entry_date": trade.entry_date,
                                     "exit_date": trade.exit_date,
                                     "ret_net": trade.ret_net})
        for trade in t.itertuples(index=False):
            if trade.entry_signal_date not in base_keys:
                trade_events.append({"variant": variant, "difference": "candidate_only",
                                     "entry_signal_date": trade.entry_signal_date,
                                     "entry_date": trade.entry_date,
                                     "exit_date": trade.exit_date,
                                     "ret_net": trade.ret_net})

    period_rows = []
    periods = [
        ("2000-2003", "2000-01-03", "2003-12-31"),
        ("2004-2009", "2004-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2021", "2016-01-01", "2021-12-31"),
        ("2022-end", "2022-01-01", end),
    ]
    for name, run in runs.items():
        e = run["equity"].copy()
        e["Date"] = pd.to_datetime(e.Date)
        for period, lo, hi in periods:
            inside = e.loc[e.Date.between(lo, hi)]
            if inside.empty:
                continue
            previous = e.loc[e.Date < pd.Timestamp(lo)]
            capital = float(previous.Equity.iloc[-1]) if len(previous) else config.initial_cash
            values = pd.concat([pd.Series([capital]), inside.Equity], ignore_index=True)
            period_rows.append({"variant": name, "period": period,
                                "return": float(values.iloc[-1] / capital - 1),
                                "mdd": float((values / values.cummax() - 1).min())})
    return {"signals": s, "runs": runs, "summary": summary, "event": event,
            "trade_events": pd.DataFrame(trade_events),
            "period": pd.DataFrame(period_rows)}


def chart(result: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = ["baseline", "down_relief_2_abs", "up_slow_veto_1_abs",
              "both_2_abs", "down_relief_2_abs__defer",
              "up_slow_veto_1_abs__defer", "both_2_abs__defer"]
    fig, (ax, dd) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                gridspec_kw={"height_ratios": [2.3, 1]})
    for key in chosen:
        e = result["runs"][key]["equity"].copy()
        e["Date"] = pd.to_datetime(e.Date)
        reduced = e.groupby(e.Date.dt.to_period("W-FRI")).tail(1)
        ax.plot(reduced.Date, reduced.Equity, label=key, lw=1.5)
        dd.plot(reduced.Date, reduced.Drawdown * 100, label=key, lw=1.1)
    ax.set_yscale("log")
    ax.set_ylabel("USD equity (log)")
    dd.set_ylabel("Drawdown (%)")
    dd.set_xlabel("QQQ session date")
    ax.legend(fontsize=8, ncol=2)
    for target in (ax, dd):
        target.grid(alpha=.2)
        target.spines[["top", "right"]].set_visible(False)
    fig.suptitle("QQQ Dynamic N20: month EMA5 slope & curvature entry gates")
    fig.text(.08, .01, "$100k initial; 0.05% per side; next open; cash 0%; in-sample only.",
             fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96])
    destination.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(destination, format="svg")
    plt.close(fig)


def chart_crisis(result: dict, destination: Path) -> None:
    """Chart 2007-2009 gold crosses against only-as-of monthly slope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result["signals"]
    sub = s.loc[s.Date.between("2007-01-01", "2009-12-31")].copy()
    event = result["event"].copy()
    event["Date"] = pd.to_datetime(event.Date)
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(14, 7.5),
                                      gridspec_kw={"height_ratios": [2, 1.35]})
    top.plot(sub.Date, sub.adj_close, color="#435873", lw=1.3,
             label="QQQ adjusted close")
    descriptions = {
        "2007-12-06": "07/12 #1",
        "2007-12-24": "07/12 #2",
        "2008-04-21": "08/04",
        "2008-08-14": "08/08",
        "2009-03-30": "09/03 recovery",
    }
    for _, point in event.iterrows():
        day = point.Date
        price = float(sub.loc[sub.Date == day, "adj_close"].iloc[0])
        is_rejected = not bool(point.gate_up_slow_veto_1_abs)
        color = "#c44c47" if is_rejected else "#1f8c75"
        top.scatter(day, price, marker="x" if is_rejected else "o", s=85,
                    color=color, zorder=3)
        top.annotate(descriptions[str(day.date())], (day, price),
                     xytext=(2, 11), textcoords="offset points", fontsize=9)
        bottom.scatter(day, float(point.known_slope_abs), color=color,
                       marker="x" if is_rejected else "o", s=70, zorder=3)
    bottom.plot(sub.Date, sub.known_slope_abs, drawstyle="steps-post",
                color="#7051a4", lw=1.7,
                label="Last completed month's EMA5 slope")
    bottom.axhline(0, color="black", alpha=.45, lw=.9)
    top.legend(fontsize=9, loc="upper right")
    bottom.legend(fontsize=9)
    top.set_ylabel("Adjusted price")
    bottom.set_ylabel("EMA5 slope ($ / month)")
    bottom.set_xlabel("Signal date (next-open fill)")
    for ax in (top, bottom):
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("2007-2009: rising monthly EMA5 can be losing momentum")
    fig.text(.08, .01,
             "Red X: upward-slope slowdown veto. Green dot: not vetoed. "
             "Month-slope snapshots never use the ongoing month's final close.",
             fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96])
    destination.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(destination, format="svg")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--update", type=Path, help="Local July-Nov 2026 update, with overlap")
    p.add_argument("--end", default="2026-07-13")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    if args.update:
        frozen = combine_rebased_sources(args.old, args.new, end_date="2026-07-13")
        data, extension_audit = extend_adjusted_history(
            frozen, str(args.update), frozen_cutoff="2026-07-13",
            completed_cutoff=args.end)
    else:
        data = combine_rebased_sources(args.old, args.new, end_date=args.end)
        extension_audit = None
    result = comparison(data, end=args.end)
    if args.end == "2026-11-09":
        st = result["runs"]["baseline"]["stats"]
        if not (abs(st["end_equity"] - 742702.78) < 15 and
                abs(st["mdd"] + .259975) < .0001 and
                st["completed_trades"] == 59):
            raise AssertionError("Original fourth strategy did not reproduce")
    args.outdir.mkdir(exist_ok=True, parents=True)
    result["summary"].to_csv(args.outdir / "summary.csv", index=False)
    result["period"].to_csv(args.outdir / "period_metrics.csv", index=False)
    result["event"].to_csv(args.outdir / "crisis_signal_audit.csv", index=False)
    result["trade_events"].to_csv(args.outdir / "changed_trade_events.csv", index=False)
    for name, run in result["runs"].items():
        run["trades"].to_csv(args.outdir / (name + "_trades.csv"), index=False)
    chart(result, args.outdir / "equity_drawdown.svg")
    chart_crisis(result, args.outdir / "crisis_signals.svg")
    meta = {
        "asof": args.end, "source_type": "locally acquired two-source adjusted QQQ bars",
        "signal": "month EMA5 recursive adjusted closes, previous completed month",
        "provisional_exception": "MTD EMA calculated from today's completed daily close and prior completed EMA; never final month close",
        "entry_gate_scope": "both normal and fast entries; original fast EMA10/20 monthly-positive prerequisite retained",
        "skipped_gold": "main variants require next independent fresh signal",
        "defer_control": "__defer variants may enter after gate recovers while same 20/60 bull regime persists",
        "exit_rules": "original fourth strategy: dynamic N20, 7% close stop, EMA20/60 death",
        "execution": "signal on completed daily close, fill next regular-session open",
        "fee_per_side": .0005, "initial_cash": 100000, "cash_yield": 0,
        "tax_and_slippage": "not modeled", "sample_status": "in-sample hypothesis comparison",
        "forward_extension_audit": extension_audit,
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["summary"][
        ["variant", "end_equity", "cagr", "mdd", "completed_trades",
         "win_rate", "payoff_ratio", "exposure", "return_vs_baseline"]
    ].to_string(index=False))
    print("\n2007-2009 signal gates:\n", result["event"].to_string(index=False))


if __name__ == "__main__":
    main()
