"""Independent A2-N20 re-entry study: reclaim the N20 exit price.

Only a completed close may generate a signal; next regular open executes.
The frozen A2 engine and all existing research experiments are unchanged.

Run: PYTHONPATH=src python research/experiments/a2_n20_price_reclaim_study.py
"""
from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))
import a2_n20 as frozen
from qqq_a2.engine import compute_signals, load_adjusted_csv
from a2_n20_fast_reentry_and_deadcross_buffer import (
    Variant, enriched, simulate as ma10_simulate,
)

OUT = ROOT / "research" / "results" / "price_reclaim_2026_11_10"
FEE = frozen.FEE


@dataclass(frozen=True)
class Rule:
    name: str
    level: str = ""  # "fill_open", "signal_close", "entry_open", or empty
    retry_limit: int | None = 1  # None allows repeated N20 price-reclaim entries
    wait_days: int | None = None  # None means until slow MA20/60 trend ends
    min_signal_wait: int = 0  # 0 permits recovery on the stop-fill day's close
    require_fast_bull: bool = False  # MA10 > MA20 at the re-entry signal close
    second_guard_ma: int | None = None  # After price reclaim, exit on MA20 < SMA30/50


RULES = [
    Rule("A2_frozen"),
    Rule("R_price_exit_open_once", "fill_open"),
    Rule("R_price_signal_close_once", "signal_close"),
    Rule("R_price_original_entry_once", "entry_open"),
    Rule("R_price_exit_open_once_10d", "fill_open", wait_days=10),
    Rule("R_price_exit_open_once_20d", "fill_open", wait_days=20),
    Rule("R_price_exit_open_once_wait_1_close", "fill_open", min_signal_wait=1),
    Rule("R_price_exit_open_once_wait_2_closes", "fill_open", min_signal_wait=2),
    Rule("R_price_exit_open_once_fast_bull", "fill_open", require_fast_bull=True),
    Rule("R_price_exit_open_unlimited", "fill_open", retry_limit=None),
]


def simulate(df: pd.DataFrame, rule: Rule):
    cash, shares, position, pending = 100_000.0, 0.0, None, None
    trades, equity, actions = [], [], []
    reclaim, retry_used = None, 0

    for i, bar in enumerate(df.itertuples(index=False)):
        if bar.Date < frozen.START:
            continue
        # Execute yesterday's completed-close decision at today's opening bar.
        if pending is not None:
            action, signal_i, origin = pending
            if action == "buy":
                assert position is None and shares == 0
                if origin == "FRESH_MA20_60_GOLD":
                    retry_used = 0
                else:
                    assert origin == "PRICE_RECLAIM" and reclaim is not None
                    retry_used += 1
                position = {
                    "entry_date": str(bar.Date.date()),
                    "entry_price": float(bar.adj_open),
                    "entry_i": i,
                    "capital": cash,
                    "origin": origin,
                    "reclaim_anchor": reclaim["level"] if origin == "PRICE_RECLAIM" else np.nan,
                    "reclaim_wait": (i - reclaim["exit_i"] if origin == "PRICE_RECLAIM" else np.nan),
                }
                shares = cash / (float(bar.adj_open) * (1 + FEE))
                actions.append({"date": str(bar.Date.date()), "action": "buy", "origin": origin})
                cash, reclaim = 0.0, None
            else:
                assert position is not None and shares > 0
                proceeds = shares * float(bar.adj_open) * (1 - FEE)
                trades.append({
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "entry_origin": position["origin"],
                    "reclaim_anchor": position["reclaim_anchor"],
                    "reclaim_wait": position["reclaim_wait"],
                    "exit_date": str(bar.Date.date()),
                    "exit_price": float(bar.adj_open),
                    "exit_signal_date": str(df.iloc[signal_i].Date.date()),
                    "exit_reason": origin,
                    "ret_net": proceeds / position["capital"] - 1,
                    "equity_after_trade": proceeds,
                })
                actions.append({"date": str(bar.Date.date()), "action": "sell", "origin": origin})
                if (rule.level and origin == "N20 protection" and
                    (rule.retry_limit is None or retry_used < rule.retry_limit)):
                    anchor = {
                        "fill_open": float(bar.adj_open),
                        "signal_close": float(df.iloc[signal_i].adj_close),
                        "entry_open": float(position["entry_price"]),
                    }[rule.level]
                    reclaim = {"level": anchor, "exit_i": i,
                               "exit_date": str(bar.Date.date())}
                else:
                    reclaim = None
                cash, shares, position = proceeds, 0.0, None
            pending = None

        # The MA20/60 death cross always takes priority; N20 uses the first
        # 20 FULL days after each fill and resets on every genuine re-entry.
        if position is not None:
            age = i - position["entry_i"]
            if bool(bar.death):
                pending = ("sell", i, "MA20/60 death cross")
            elif (1 <= age <= 20 and bool(bar.protect_cross) and
                  float(bar.adj_close) < position["entry_price"]):
                pending = ("sell", i, "N20 protection")
            elif (position["origin"] == "PRICE_RECLAIM" and
                  rule.second_guard_ma is not None and
                  float(bar.ma20) < float(getattr(bar, f"ma{rule.second_guard_ma}"))):
                # This is a STATE check, not another fresh cross: if the fast
                # re-entry occurs after the crossover, it still exits. Never
                # execute before the following regular session's open.
                pending = ("sell", i, f"RECLAIM_MA20_{rule.second_guard_ma}_BELOW")
        else:
            if bool(bar.gold):
                pending = ("buy", i, "FRESH_MA20_60_GOLD")
                reclaim = None
            elif reclaim is not None:
                wait = i - reclaim["exit_i"]
                if (not (float(bar.ma20) > float(bar.ma60)) or
                    (rule.wait_days is not None and wait > rule.wait_days)):
                    reclaim = None
                elif (wait >= rule.min_signal_wait and
                      float(bar.adj_close) >= reclaim["level"] and
                      (not rule.require_fast_bull or float(bar.ma10) > float(bar.ma20))):
                    pending = ("buy", i, "PRICE_RECLAIM")

        equity.append({"Date": str(bar.Date.date()),
                       "Equity": float(cash + shares * float(bar.adj_close)),
                       "Invested": bool(shares > 0)})

    t, e, a = pd.DataFrame(trades), pd.DataFrame(equity), pd.DataFrame(actions)
    e["Drawdown"] = e.Equity / e.Equity.cummax() - 1
    open_ret = (shares * float(df.iloc[-1].adj_close) * (1 - FEE) /
                position["capital"] - 1) if position is not None else None
    return t, e, a, open_ret


def metrics(name: str, t: pd.DataFrame, e: pd.DataFrame,
            a: pd.DataFrame, open_ret: float | None):
    rets = pd.concat([t.ret_net, pd.Series([open_ret])], ignore_index=True) if open_ret is not None else t.ret_net
    wins, losses = rets[rets > 0], rets[rets < 0]
    years = (pd.Timestamp(e.Date.iloc[-1]) - frozen.START).days / 365.25
    return {
        "scenario": name,
        "completed": len(t), "wins": int((t.ret_net > 0).sum()),
        "losses": int((t.ret_net < 0).sum()),
        "n20_exits": int((t.exit_reason == "N20 protection").sum()),
        "reentries": int((a.origin == "PRICE_RECLAIM").sum()),
        "win_rate_common": float((rets > 0).mean()),
        "payoff_common": float(wins.mean() / -losses.mean()),
        "pf_common": float(wins.sum() / -losses.sum()),
        "worst_trade": float(t.ret_net.min()),
        "losses_over_5pct": int((t.ret_net <= -0.05).sum()),
        "end_equity": float(e.Equity.iloc[-1]),
        "cagr": float((e.Equity.iloc[-1] / 100_000) ** (1 / years) - 1),
        "mdd": float(e.Drawdown.min()),
        "exposure": float(e.Invested.mean()),
        "2000_2002": period(e, "2000-01-03", "2002-12-31"),
        "2003_2004": period(e, "2003-01-01", "2004-12-31"),
        "2020": period(e, "2020-01-01", "2020-12-31"),
        "2022": period(e, "2022-01-01", "2022-12-31"),
    }


def period(e: pd.DataFrame, start: str, end: str):
    before = e[e.Date < start]
    base = float(before.Equity.iloc[-1]) if len(before) else 100_000.0
    return float(e[e.Date.between(start, end)].Equity.iloc[-1] / base - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, help="Local-only pre-rebased adjusted QQQ CSV")
    args = parser.parse_args()
    raw = compute_signals(load_adjusted_csv(args.merged)) if args.merged else frozen.prices()[0]
    df = enriched(raw)
    frozen_t, frozen_e, frozen_s, _ = frozen.run(raw, warm_start=False)
    out = (OUT.with_name(OUT.name + "_to_" + str(df.Date.iloc[-1].date()))
           if args.merged else OUT)
    out_metrics, out_trades, out_actions = [], [], []
    for rule in RULES:
        t, e, a, op = simulate(df, rule)
        if not rule.level:
            assert t.entry_date.tolist() == frozen_t.entry_date.tolist()
            assert t.exit_date.tolist() == frozen_t.exit_date.tolist()
            np.testing.assert_allclose(t.ret_net, frozen_t.ret_net, rtol=0, atol=1e-12)
            np.testing.assert_allclose(e.Equity, frozen_e.Equity, rtol=0, atol=1e-6)
            assert abs(e.Drawdown.min() - frozen_s["mdd"]) < 1e-10
        out_metrics.append(metrics(rule.name, t, e, a, op))
        t.insert(0, "scenario", rule.name)
        a.insert(0, "scenario", rule.name)
        out_trades.append(t)
        out_actions.append(a)
    # Existing independent MA10/20 cross study, strict natural initialization.
    t, e, a, op = ma10_simulate(df, Variant(
        "R_MA10_gold_once", fast_reentry=True,
        fast_retry_limit_per_slow_cycle=1), warm_start=False)
    out_metrics.append(metrics("R_MA10_gold_once", t, e, a.rename(columns={"type": "action"}), op))
    t.insert(0, "scenario", "R_MA10_gold_once")
    out_trades.append(t)
    m = pd.DataFrame(out_metrics)
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "summary.csv", index=False)
    pd.concat(out_trades, ignore_index=True).to_csv(out / "trades.csv", index=False)
    pd.concat(out_actions, ignore_index=True).to_csv(out / "actions.csv", index=False)
    print(m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
