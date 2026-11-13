"""Separate full-state replay for H3 monthly-veto exceptions with weekly RV.

Weekly volatility is calculated from DAILY adjusted close log returns
but only PREVIOUSLY COMPLETED calendar weeks are exposed to signal rows.
Original H3 entry/exit, N20, day-zero and 7pct stop are fully preserved.
No changes to frozen H3, A/B profit guard or existing experimental engines.
The weekly OFF control MUST match native H3 at daily-equity/trade granularity.
All experimental entries execute at next REGULAR open.
"""
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research"/"experiments"))
from a2_month_ema5_slope_regime import VARIANTS
from a2_ema_hard_stop8 import stop_trigger_price
from qqq_a2.engine import Config

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
                 trend_spread_lookback: int = 1,
                 trend_spread_trim_fraction: float | None = None,
                 trend_spread_trim_restore: str = "ema5_10_gold",
                 fast_profit_exit_peak_min: float | None = None,
                 fast_profit_exit_slow_gap_min_pct: float | None = None,
                 fast_profit_reentry_mode: str = "none",
                 fast_profit_reentry_gap_pct: float = 2.,
                 entry_relax_mode: str = "none",
                 weekly_vol_gate: str = "none") -> dict:
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
    if fast_profit_exit_peak_min is not None and not 0 <= fast_profit_exit_peak_min < 1:
        raise ValueError("Invalid peak profit activation")
    if fast_profit_reentry_mode not in ("none", "any", "current", "seen"):
        raise ValueError("Invalid FAST profitable re-entry mode")
    if fast_profit_reentry_mode != "none" and fast_profit_exit_peak_min is None:
        raise ValueError("Profitable FAST re-entry requires enabled fast-death exit")
    if fast_profit_exit_slow_gap_min_pct is not None and fast_profit_exit_slow_gap_min_pct < 0:
        raise ValueError("Invalid wide slow spread filter")
    if fast_profit_reentry_gap_pct < 0:
        raise ValueError("Invalid narrow slow spread filter")
    if (fast_profit_exit_peak_min is not None and
            (profit_trail_pct is not None or profit_atr_mult is not None
             or trend_spread_atr_min is not None)):
        raise ValueError("Separate profit-fast-exit hypothesis from other profit exits")
    if trend_spread_trim_fraction is not None:
        if not 0 < trend_spread_trim_fraction < 1:
            raise ValueError("Trim must be partial (0,1), never full")
        if trend_spread_atr_min is None:
            raise ValueError("Trim requires enabled trend spread detector")
        if trend_spread_reentry != "none":
            raise ValueError("Partial restore and full-exit fast reentry are exclusive")
    if trend_spread_trim_restore not in ("ema5_10_gold", "ema10_20_gold"):
        raise ValueError("Invalid partial restore signal")
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
    valid_relax_modes = {
        "none", "all_no_month", "all_no_month_ema200",
        "fast_no_month", "fast_no_month_ema200", "fast_no_month_above200",
        "fast_global_veto_only_ema200", "fast_special_month_only_ema200",
        "fast_positive_slowdown_only_ema200",
        "fast_negative_recovery_only_ema200",
        "fast_no_month_ema200_slow_bull",
    }
    if entry_relax_mode not in valid_relax_modes:
        raise ValueError("Unknown entry_relax_mode: " + entry_relax_mode)
    weekly_vol_modes = {
        "none", "fast_unfiltered","slow_unfiltered",
        "fast_up","fast_up_positive","fast_up_positive_price20",
        "fast_falling_positive","slow_up","slow_up_positive",
        "both_up_positive","fast_up_positive_global_only",
    }
    if weekly_vol_gate not in weekly_vol_modes:
        raise ValueError("Invalid weekly volatility entry gate " + weekly_vol_gate)
    if weekly_vol_gate != "none" and entry_relax_mode != "none":
        raise ValueError("Isolate weekly-vol research from old entry relax modes")
    if weekly_vol_gate != "none":
        for required in ("last_completed_week_rv","prior4week_median_rv",
                         "last_completed_week_return",
                         "weekly_rv_up","weekly_rv_valid"):
            if required not in signals.columns:
                raise AssertionError("Missing as-of weekly volatility " + required)
    df = signals.sort_values("Date").reset_index(drop=True)
    df["relax_ema200"] = df.adj_close.ewm(
        span=200,adjust=False,min_periods=200).mean()
    df["relax_price_above_200"] = df.adj_close > df.relax_ema200
    df["relax_ema200_rising20"] = (
        df.relax_ema200 > df.relax_ema200.shift(20))
    df["relax_above_rising200"] = (
        df.relax_price_above_200 & df.relax_ema200_rising20)

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
        if (trend_spread_reentry == "ema5_10_gold" or
                trend_spread_trim_fraction is not None and
                trend_spread_trim_restore == "ema5_10_gold"):
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
    fast_profit_reentry_eligible = False
    fast_profit_slow_gap_seen = False
    last_fast_profit_exit = None
    fast_profit_events = []
    rebalance_events = []
    ledger, curve = [], []
    vol_extra_signal_log = []
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
                    "entry_override": pending.get("entry_override", "BASE"),
                    "trend_confirm_index": (i - 1 if confirmed else None),
                    "trend_confirm_date": (
                        str(df.iloc[i - 1].Date.date()) if confirmed else None),
                    "max_close": float(row.adj_close),
                    "min_close": float(row.adj_close),
                    # Only the running past closing high, known as of each day.
                    "profit_guard_peak_close": entry,
                    "n20_extension_qualified": False,
                    "trimmed": False, "trim_count": 0,
                    "restore_count": 0,
                }
                if pending.get("fast_profit_reentry", False):
                    fast_profit_events.append({
                        "action": "REENTRY", "entry_date": day,
                        "signal_date": pending["signal_date"],
                        "fill_date": day, "price": entry,
                        "slow_gap_pct": pending.get("slow_gap_pct"),
                        "previous_exit_date": (
                            last_fast_profit_exit["date"]
                            if last_fast_profit_exit else None),
                        "previous_exit_price": (
                            last_fast_profit_exit["price"]
                            if last_fast_profit_exit else None),
                    })
                cash, eligible_fast, deferred_gold, deferred_fast = 0., False, False, False
                early_spread_reentry_eligible = False
                fast_profit_reentry_eligible = False
                fast_profit_slow_gap_seen = False
            elif pending["side"] == "sell":
                if not shares or position is None:
                    raise AssertionError("Sell without position")
                px = float(row.adj_open)
                # In partial-position trials, uninvested cash survives the
                # final exit; the frozen full-investment case has cash == 0.
                cash += shares * px * (1 - config.fee_per_side)
                ledger.append({
                    "id": len(ledger) + 1,
                    "entry_signal_date": position["entry_signal_date"],
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "entry_kind": position["entry_kind"],
                    "entry_override": position["entry_override"],
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
                fast_profit_reentry_eligible = (
                    pending["reason"] == "research profitable FAST death")
                fast_profit_slow_gap_seen = False
                if fast_profit_reentry_eligible:
                    fast_profit_events.append({
                        "action": "EXIT", "entry_date": position["entry_date"],
                        "signal_date": pending["signal_date"],
                        "fill_date": day, "price": px,
                        "slow_gap_pct": pending.get("slow_gap_pct"),
                        "profit_at_signal_pct": pending.get("profit_at_signal_pct"),
                    })
                    last_fast_profit_exit = {"date": day, "price": px}
                position, shares = None, 0.
            elif pending["side"] == "trim":
                if position is None or position["trimmed"] or not shares:
                    raise AssertionError("Invalid partial trim")
                assert trend_spread_trim_fraction is not None
                px = float(row.adj_open)
                sold = shares * trend_spread_trim_fraction
                shares -= sold
                cash += sold * px * (1 - config.fee_per_side)
                position["trimmed"] = True
                position["trim_count"] += 1
                rebalance_events.append({
                    "entry_date": position["entry_date"], "action": "trim",
                    "signal_date": pending["signal_date"], "fill_date": day,
                    "price": px, "fraction_of_shares": trend_spread_trim_fraction,
                    "shares_delta": -sold,
                    "fee": sold * px * config.fee_per_side,
                    "equity_after_fill": cash + shares * px,
                })
            elif pending["side"] == "restore":
                if position is None or not position["trimmed"] or cash <= 0:
                    raise AssertionError("Invalid partial restore")
                px = float(row.adj_open)
                bought = cash / (px * (1 + config.fee_per_side))
                fee = bought * px * config.fee_per_side
                shares += bought
                cash = 0.
                position["trimmed"] = False
                position["restore_count"] += 1
                rebalance_events.append({
                    "entry_date": position["entry_date"], "action": "restore",
                    "signal_date": pending["signal_date"], "fill_date": day,
                    "price": px, "fraction_of_shares": float("nan"),
                    "shares_delta": bought, "fee": fee,
                    "equity_after_fill": cash + shares * px,
                })
            else:
                raise AssertionError("Unknown partial-position order")
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
                    "entry_override": position["entry_override"],
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
            elif (fast_profit_exit_peak_min is not None and
                  bool(row.protect_cross) and
                  float(row.adj_close) > position["entry_price"] *
                  (1 + config.fee_per_side) / (1 - config.fee_per_side) and
                  position["profit_guard_peak_close"] >=
                  position["entry_price"] * (1 + fast_profit_exit_peak_min) and
                  (fast_profit_exit_slow_gap_min_pct is None or
                   100 * (float(row.ma20) / float(row.ma60) - 1) >=
                   fast_profit_exit_slow_gap_min_pct)):
                # Complete daily CLOSE, not hindsight MFE. Fill next OPEN.
                pending = {
                    "side": "sell", "signal_date": day,
                    "reason": "research profitable FAST death",
                    "slow_gap_pct": 100 * (
                        float(row.ma20) / float(row.ma60) - 1),
                    "profit_at_signal_pct": 100 * (
                        float(row.adj_close) / position["entry_price"] - 1),
                }
            elif trend_spread_trim_fraction is not None and position["trimmed"]:
                # Restore at the next OPEN, and keep the ORIGINAL entry
                # price, N20 clock, hard stop and slow-death exit unchanged.
                # In particular, restoration must not reset the N20 clock.
                restore_cross = (
                    bool(row.spread_ema5_gold)
                    if trend_spread_trim_restore == "ema5_10_gold"
                    else bool(row.ema10_20_gold)
                )
                if bool(row.ma20 > row.ma60) and restore_cross:
                    pending = {
                        "side": "restore", "signal_date": day,
                        "reason": f"research restore {trend_spread_trim_restore}",
                    }
            elif trend_spread_atr_min is not None and (
                    not position["trimmed"] and
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
                        "side": ("trim" if trend_spread_trim_fraction is not None
                                 else "sell"),
                        "signal_date": day,
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
                fast_profit_reentry_eligible = False
                fast_profit_slow_gap_seen = False
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
            slow_gap_pct = 100 * (float(row.ma20) / float(row.ma60) - 1)
            if (fast_profit_reentry_eligible and slow_bull and
                    0 <= slow_gap_pct <= fast_profit_reentry_gap_pct):
                fast_profit_slow_gap_seen = True
            fast_profit_reentry = (
                fast_profit_reentry_eligible and slow_bull and
                bool(row.ema10_20_gold) and
                (fast_profit_reentry_mode == "any" or
                 (fast_profit_reentry_mode == "current" and
                  0 <= slow_gap_pct <= fast_profit_reentry_gap_pct) or
                 (fast_profit_reentry_mode == "seen" and
                  fast_profit_slow_gap_seen))
            )
            fast_reentry = (
                early_spread_reentry_eligible and
                trend_spread_reentry != "none" and
                bool(row.ma20 > row.ma60) and
                (bool(row.ema10_20_gold) if
                 trend_spread_reentry == "ema10_20_gold"
                 else bool(row.spread_ema5_gold))
            )
            original = bool(row[gate_col]) and (regular or fast)
            # RESEARCH-ONLY: allow an EARLIER entry on the CURRENT completed
            # golden-cross bar. Original entry remains fully allowed, and
            # the entire actual position state including N20/stop is rerun.
            override_fast = False
            override_regular = False
            if entry_relax_mode != "none":
                above200 = bool(row.relax_price_above_200)
                bullish200 = bool(row.relax_above_rising200)
                allowed_by_200 = (
                    entry_relax_mode in ("all_no_month", "fast_no_month")
                    or entry_relax_mode == "fast_no_month_above200" and above200
                    or bullish200)
                month_positive_slowing = bool(
                    pd.notna(row.known_slope_abs) and
                    pd.notna(row.known_prev_abs) and
                    row.known_slope_abs > 0 and
                    row.known_slope_abs < row.known_prev_abs)
                month_negative_recovery = bool(
                    pd.notna(row.known_slope_abs) and
                    pd.notna(row.known_prev_abs) and
                    row.known_slope_abs < 0 and
                    row.known_prev_abs < 0 and
                    row.known_slope_abs > row.known_prev_abs)
                fast_month_allow_original = bool(
                    row.prev_completed_month_ema5_up or negative_relief)
                slow_filter_pass = (
                    entry_relax_mode != "fast_no_month_ema200_slow_bull"
                    or slow_bull)
                if eligible_fast and bool(row.ema10_20_gold) and (
                    allowed_by_200 and slow_filter_pass):
                    if entry_relax_mode in (
                        "all_no_month","all_no_month_ema200",
                        "fast_no_month","fast_no_month_ema200",
                        "fast_no_month_above200",
                        "fast_no_month_ema200_slow_bull"):
                        override_fast = True
                    elif entry_relax_mode=="fast_global_veto_only_ema200":
                        override_fast = fast_month_allow_original
                    elif entry_relax_mode=="fast_special_month_only_ema200":
                        override_fast = bool(row[gate_col])
                    elif entry_relax_mode=="fast_positive_slowdown_only_ema200":
                        override_fast = month_positive_slowing and fast_month_allow_original
                    elif entry_relax_mode=="fast_negative_recovery_only_ema200":
                        override_fast = bool(row[gate_col]) and month_negative_recovery
                if (entry_relax_mode in
                    ("all_no_month","all_no_month_ema200") and
                    allowed_by_200 and bool(row.gold)):
                    override_regular = True
            allow_original = original or fast_reentry or fast_profit_reentry
            vol_override_fast = False
            vol_override_regular = False
            if weekly_vol_gate != "none" and not allow_original:
                fresh_fast = bool(eligible_fast and row.ema10_20_gold)
                # The original FAST eligibility is the combined effect of
                # the global monthly veto and FAST-specific monthly rule.
                fast_month_eligible = bool(
                    row.prev_completed_month_ema5_up or negative_relief)
                fast_blocked_by_month = not (
                    bool(row[gate_col]) and fast_month_eligible)
                slow_blocked_by_month = not bool(row[gate_col])
                rv_up = bool(row.weekly_rv_up)
                rv_valid = bool(row.weekly_rv_valid)
                wk_positive = bool(row.last_completed_week_return > 0)
                price_above_20 = bool(row.adj_close > row.ma20)
                gate_up = rv_valid and rv_up
                gate_up_positive = gate_up and wk_positive
                gate_up_positive_price = gate_up_positive and price_above_20
                gate_falling_positive = rv_valid and not rv_up and wk_positive
                if fresh_fast and fast_blocked_by_month:
                    vol_override_fast = (
                        weekly_vol_gate == "fast_unfiltered" or
                        weekly_vol_gate == "fast_up" and gate_up or
                        weekly_vol_gate == "fast_up_positive" and gate_up_positive or
                        weekly_vol_gate == "fast_up_positive_price20"
                        and gate_up_positive_price or
                        weekly_vol_gate == "fast_falling_positive"
                        and gate_falling_positive or
                        weekly_vol_gate == "both_up_positive"
                        and gate_up_positive or
                        weekly_vol_gate == "fast_up_positive_global_only"
                        and gate_up_positive and fast_month_eligible
                        and not bool(row[gate_col])
                    )
                if bool(row.gold) and slow_blocked_by_month:
                    vol_override_regular = (
                        weekly_vol_gate == "slow_unfiltered" or
                        weekly_vol_gate == "slow_up" and gate_up or
                        weekly_vol_gate in ("slow_up_positive",
                                           "both_up_positive")
                        and gate_up_positive
                    )
                if vol_override_fast or vol_override_regular:
                    vol_extra_signal_log.append({
                        "signal_date":day,
                        "weekly_vol_gate":weekly_vol_gate,
                        "entry_signal_kind":("FAST_EMA10_20"
                            if vol_override_fast and not vol_override_regular
                            else "REGULAR_EMA20_60"),
                        "entry_month_gate":bool(row[gate_col]),
                        "fast_month_eligible":fast_month_eligible,
                        "most_recent_completed_week_rv":float(
                            row.last_completed_week_rv),
                        "prev_four_weeks_rv_median":float(
                            row.prior4week_median_rv),
                        "vol_ratio":(float(row.last_completed_week_rv) /
                            float(row.prior4week_median_rv)
                            if row.prior4week_median_rv>0 else None),
                        "last_completed_week_return":float(
                            row.last_completed_week_return),
                        "daily_close_above_EMA20":price_above_20,
                        "month_known_slope":float(row.known_slope_abs),
                        "month_previous_slope":float(row.known_prev_abs),
                        "ema20_above_ema60":bool(row.ma20>row.ma60),
                    })
            if (allow_original or override_fast or override_regular or
                    vol_override_fast or vol_override_regular):
                from_profit_reentry = (
                    fast_profit_reentry and not original and not fast_reentry)
                override_tag = (
                    "BASE" if allow_original else
                    "RELAX_REGULAR" if override_regular else
                    "VOL_SLOW_" + weekly_vol_gate if vol_override_regular else
                    "VOL_FAST_" + weekly_vol_gate if vol_override_fast else
                    "RELAX_FAST")
                pending = {
                    "side": "buy", "signal_date": day,
                    "kind": ("REGULAR_EMA20_60" if
                             (bool(row[gate_col]) and regular) or
                             override_regular or vol_override_regular
                             else "FAST_EMA10_20_MONTH_EMA5"),
                    "fast_profit_reentry": from_profit_reentry,
                    "slow_gap_pct": slow_gap_pct,
                    "entry_override": override_tag,
                }
        curve_row = {
            "Date": day, "Equity": cash + shares * float(row.adj_close),
            "Invested": bool(shares),
            "Pending": pending["side"] if pending is not None else "",
        }
        # Keep frozen V3/H3 equity schema completely unchanged by default.
        if trend_spread_trim_fraction is not None:
            curve_row["PositionWeight"] = (
                shares * float(row.adj_close) / curve_row["Equity"])
        curve.append(curve_row)

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
            "open_position": position, "pending_order": pending,
            "rebalance_events": pd.DataFrame(rebalance_events),
            "fast_profit_events": pd.DataFrame(fast_profit_events),
            "weekly_vol_override_signals":pd.DataFrame(vol_extra_signal_log)}


