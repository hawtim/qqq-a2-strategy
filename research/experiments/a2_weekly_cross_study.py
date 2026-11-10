"""One-factor comparison of weekly MA5/10 crossover exits and symmetric entries.

Inputs and account conventions reuse the verified, frozen A2-N20 baseline.
All weekly signals become observable only at a completed weekly close, and
orders execute at the next trading day's adjusted open. No intraday fills.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen


@dataclass(frozen=True)
class Strategy:
    name: str
    entry: str  # daily, weekly
    exit: str   # daily, weekly, either
    n20: bool


CASES = [
    Strategy("A2_baseline", "daily", "daily", True),
    Strategy("D20_60_in_W5_10_out_N20", "daily", "weekly", True),
    Strategy("D20_60_in_first_death_N20", "daily", "either", True),
    Strategy("W5_10_in_out", "weekly", "weekly", False),
    Strategy("W5_10_in_out_N20", "weekly", "weekly", True),
]


def with_weekly_crosses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["w_gold"] = False
    df["w_death"] = False
    df["w_bull"] = False
    week_key = df.Date.dt.to_period("W-FRI")
    week_close = df.groupby(week_key, sort=True).tail(1).copy()
    # The historical cutoff is Monday 2026-07-13. Never treat that unfinished
    # calendar week as a completed week or backfill its MA into earlier bars.
    assert df.Date.iloc[-1] == pd.Timestamp("2026-07-13")
    week_close = week_close[week_close.Date.dt.to_period("W-FRI") != week_key.iloc[-1]]
    week_close["w_ma5"] = week_close.adj_close.rolling(5, min_periods=5).mean()
    week_close["w_ma10"] = week_close.adj_close.rolling(10, min_periods=10).mean()
    week_close["w_gold"] = ((week_close.w_ma5 > week_close.w_ma10)
                             & (week_close.w_ma5.shift(1) <= week_close.w_ma10.shift(1)))
    week_close["w_death"] = ((week_close.w_ma5 < week_close.w_ma10)
                              & (week_close.w_ma5.shift(1) >= week_close.w_ma10.shift(1)))
    week_close["w_bull"] = week_close.w_ma5 > week_close.w_ma10
    # Assign weekly cross only to the actual final trading day of that week.
    for attr in ("w_gold", "w_death"):
        df.loc[week_close.index, attr] = week_close[attr].to_numpy(dtype=bool)
    # Regime is a known, last-completed-week state as of each daily close.
    state = pd.Series(np.nan, index=df.index)
    state.loc[week_close.index] = week_close.w_bull.astype(float)
    df["w_bull"] = state.ffill().fillna(0.0).astype(bool)
    assert df.w_gold[df.Date.dt.weekday < 3].sum() == 0  # Friday holiday -> Thursday possible
    return df


def simulate(df: pd.DataFrame, cfg: Strategy, warm_start: bool = True):
    cash, shares, position, pending = 100000.0, 0.0, None, None
    trades, curve = [], []
    for i, bar in enumerate(df.itertuples(index=False)):
        if bar.Date < frozen.START:
            continue
        if warm_start and bar.Date == frozen.START:
            bull = (df.iloc[i-1].ma20 > df.iloc[i-1].ma60
                    if cfg.entry == "daily" else bool(df.iloc[i-1].w_bull))
            if bull:
                pending = ("buy", i-1, "warm_start")

        if pending is not None:
            side, signal_i, reason = pending
            if side == "buy":
                assert position is None and shares == 0
                shares = cash / (float(bar.adj_open) * (1+frozen.FEE))
                position = {"entry_date":str(bar.Date.date()),
                            "entry_signal_date":str(df.iloc[signal_i].Date.date()),
                            "entry_price":float(bar.adj_open),
                            "entry_idx":i, "cash_at_entry":cash,
                            "entry_origin":reason,
                            "peak_close":float(bar.adj_close),
                            "peak_equity":shares * float(bar.adj_close)}
                cash = 0.0
            else:
                assert position is not None and shares > 0
                proceeds = shares * float(bar.adj_open) * (1-frozen.FEE)
                trades.append({**position,
                               "exit_date":str(bar.Date.date()),
                               "exit_signal_date":str(df.iloc[signal_i].Date.date()),
                               "exit_reason":reason,
                               "exit_price":float(bar.adj_open),
                               "ret_net":proceeds/position["cash_at_entry"]-1,
                               "hold_days":i-position["entry_idx"],
                               "profit_giveback_pp":
                                   100*(position["peak_equity"] / position["cash_at_entry"]
                                        - proceeds / position["cash_at_entry"]),
                               "exit_equity":proceeds})
                position, shares, cash = None, 0.0, proceeds
            pending = None

        if position is not None:
            position["peak_close"] = max(position["peak_close"],float(bar.adj_close))
            position["peak_equity"] = max(position["peak_equity"],shares*float(bar.adj_close))
            death = bool(bar.death if cfg.exit == "daily" else
                         bar.w_death if cfg.exit == "weekly" else
                         bar.death or bar.w_death)
            if death:
                exit_by_week = cfg.exit == "weekly" or (cfg.exit == "either" and bool(bar.w_death))
                pending = ("sell",i,"weekly_death" if exit_by_week else "daily_death")
            elif (cfg.n20 and 1 <= i-position["entry_idx"] <= 20 and
                  bool(bar.protect_cross) and float(bar.adj_close) < position["entry_price"]):
                pending = ("sell",i,"N20_protection")
        else:
            gold = bool(bar.gold if cfg.entry == "daily" else bar.w_gold)
            if gold:
                pending = ("buy",i,"daily_gold" if cfg.entry == "daily" else "weekly_gold")

        curve.append({"date":str(bar.Date.date()),"equity":cash+shares*float(bar.adj_close),
                      "invested":bool(shares>0)})

    t, e = pd.DataFrame(trades), pd.DataFrame(curve)
    e["dd"] = e.equity / e.equity.cummax() - 1
    wins, losses = t.loc[t.ret_net>0,"ret_net"],t.loc[t.ret_net<0,"ret_net"]
    common_returns = t.ret_net.to_list()
    if position is not None:
        common_returns.append(shares*float(df.iloc[-1].adj_close)*(1-frozen.FEE)
                              /position["cash_at_entry"]-1)
    common = pd.Series(common_returns)
    cw, cl = common[common>0],common[common<0]
    years = (pd.Timestamp(e.date.iloc[-1])-frozen.START).days/365.25
    metrics = {"case":cfg.name,"completed":len(t),"win_rate":len(wins)/len(t),
               "avg_win":wins.mean(),"avg_loss":losses.mean(),
               "payoff_completed":wins.mean()/abs(losses.mean()),
               "payoff_same_cutoff":cw.mean()/abs(cl.mean()),
               "pf_same_cutoff":cw.sum()/abs(cl.sum()),
               "mdd":e.dd.min(),"cagr":(e.equity.iloc[-1]/100000)**(1/years)-1,
               "exposure":e.invested.mean(),"end_equity":e.equity.iloc[-1],
               "worst_trade":t.ret_net.min(),
               "avg_hold_days":t.hold_days.mean(),
               "median_giveback_pp":t.loc[t.ret_net>0,"profit_giveback_pp"].median(),
               "open_at_cutoff":position is not None,
               "exit_weekly":int((t.exit_reason=="weekly_death").sum()),
               "exit_n20":int((t.exit_reason=="N20_protection").sum())}
    return t,e,metrics


def main():
    df,_ = frozen.prices()
    df = with_weekly_crosses(df)
    base_trades, _, base_stats,_ = frozen.run(df, warm_start=True)
    for cfg in CASES:
        t,e,s = simulate(df,cfg)
        if cfg.name=="A2_baseline":
            assert s["completed"]==base_stats["completed_trades"]==59
            assert np.isclose(s["cagr"],base_stats["cagr"],rtol=1e-10)
            assert np.isclose(s["mdd"],base_stats["mdd"],rtol=1e-10)
            assert (t.entry_date.to_list()==base_trades.entry_date.to_list())
            assert (t.exit_date.to_list()==base_trades.exit_date.to_list())
            assert np.allclose(t.ret_net.to_numpy(),base_trades.ret_net.to_numpy())
        print(cfg.name)
        print({k:(round(float(v),5) if isinstance(v,(float,np.floating)) else v)
               for k,v in s.items()})
        if cfg.name!="A2_baseline":
            original_best = base_trades.nlargest(10,"ret_net")
            by_entry = t.set_index("entry_date")
            print("top baseline winners matched:")
            for rec in original_best.itertuples(index=False):
                if rec.entry_date in by_entry.index:
                    hit=by_entry.loc[rec.entry_date]
                    print(rec.entry_date, round(rec.ret_net*100,2),
                          round(float(hit.ret_net)*100,2),str(hit.exit_date))
                else:
                    print(rec.entry_date,round(rec.ret_net*100,2),"not entered")
            print("worst trade:")
            print(t.nsmallest(5,"ret_net")[["entry_date","exit_date","ret_net","exit_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
