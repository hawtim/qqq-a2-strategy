"""H3 frozen shadow-ledger profit-guard EMA5/10 vs EMA5/20 backtests.

Full account, ONE additional sell + at most ONE reentry per frozen H3
trade, with unchanged *shadow* H3 entry/exit dates. Study P10/P20 peak
activation, minimum 10 fully completed sessions since entry, and
next-regular-open fills. Explicitly NOT an alteration of the frozen engine.

This "H3 shadow state" is an implementable high-level overlay concept:
the original H3 continues computing its original signals independently
while the overlay may temporarily hold cash; overlays must rejoin the
original H3 scheduled exit. It prevents the early exit from otherwise
changing FAST-eligibility, avoiding spurious apparent baseline changes.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research"/"experiments"))
from h3_volume_peak_exhaustion import load_qqq, prep as prep_raw
from h3_googl_cross_asset import load_prices as load_googl
from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_h3_trend_spread_quick_reentry import replay
from qqq_a2.engine import Config

AGE_MIN=10
FEE=.0005
ARMS=[
    ("H3_FROZEN",None,None,None),
    ("P10_X5_10_R5_10",.10,"5_10","5_10"),
    ("P10_X5_20_R5_10",.10,"5_20","5_10"),
    ("P20_X5_10_R5_10",.20,"5_10","5_10"),
    ("P20_X5_20_R5_10",.20,"5_20","5_10"),
    ("P10_X5_20_R5_20",.10,"5_20","5_20"),
    ("P20_X5_20_R5_20",.20,"5_20","5_20"),
]

def features(d, signals):
    x=d.copy()
    # All EMAs on dividend/split-adjusted closes as of completed session.
    x["ema5"]=x.adj_close.ewm(span=5,adjust=False).mean()
    x["ema10"]=x.adj_close.ewm(span=10,adjust=False,min_periods=10).mean()
    x["ema20"]=x.adj_close.ewm(span=20,adjust=False,min_periods=20).mean()
    x["ema60"]=x.adj_close.ewm(span=60,adjust=False,min_periods=60).mean()
    # Prevent an EMA initialization mismatch between the base H3
    # engine and the overlay. EMA10/20 already in signals.
    assert np.allclose(x.ema10,signals.ma10,equal_nan=True)
    assert np.allclose(x.ema20,signals.ma20,equal_nan=True)
    assert np.allclose(x.ema60,signals.ma60,equal_nan=True)
    for slow in (10,20):
        f=x.ema5; s=x[f"ema{slow}"]
        x[f"death_5_{slow}"]=(f<s)&(f.shift(1)>=s.shift(1))
        x[f"gold_5_{slow}"]=(f>s)&(f.shift(1)<=s.shift(1))
    x["trend_bull"]=x.ema20>x.ema60
    return x

def describe_giveback(x, basetrades):
    loc={str(d.date()):i for i,d in enumerate(x.Date)}
    labels={}
    for t in basetrades.itertuples(index=False):
        first=loc[t.entry_date];last=loc[t.exit_signal_date]
        held=x.iloc[first:last+1]
        gain=held.adj_close/float(t.entry_price)-1
        high=gain.cummax()
        giveback=np.where(high>=.10,(high-gain)/high,0.)
        labels[t.entry_date]={
            "h3trade_id":int(t.id),
            "base_ever_50pct_giveback":bool(np.any(giveback>=.5)),
            "base_max_peak_gain":float(high.max()),
            "base_return_net":float(t.ret_net),
        }
    return labels

def simulate(x,frozen,arm,peak_min,exit_cross,re_cross,start_date,labels):
    t=frozen["trades"]
    entries={}
    exits={}
    for row in t.itertuples(index=False):
        if row.entry_date in entries or row.exit_date in exits:
            raise AssertionError("Overlap in frozen ledger")
        entries[row.entry_date]={"entry_date":row.entry_date,
            "base_exit_date":row.exit_date,"id":int(row.id),
            "entry_price":float(row.entry_price),
            "base_net_return":float(row.ret_net)}
        exits[row.exit_date]=int(row.id)
    if frozen["open_position"] is not None:
        op=frozen["open_position"]; entry=op["entry_date"]
        if entry not in entries:
            entries[entry]={"entry_date":entry,"base_exit_date":None,
                "id":int(len(t)+1),"entry_price":float(op["entry_price"]),
                "base_net_return":None}
    cash=100000.0
    shares=0.
    active=None
    pending=None
    active_peak=0.
    guard_used=False
    guard_flat=False
    curve=[]
    ledger=[]
    actions=[]
    active_ledger_entry_cash=None
    overlay_used=0
    on_event_used=0
    false_event_used=0
    omitted_due_base_exit=0
    entry_lookup={str(dt.date()):i for i,dt in enumerate(x.Date)}
    for i,r in x.iterrows():
        day=str(r.Date.date())
        if day < start_date:
            continue
        price_open=float(r.adj_open)
        price_close=float(r.adj_close)
        # Shadow-H3 complete original exit is authoritative.
        if day in exits:
            if active is None or active["id"]!=exits[day]:
                raise AssertionError(f"Frozen exit without matching shadow trade: {day}")
            if shares:
                cash+=shares*price_open*(1-FEE)
                shares=0.
            prior_base=1.+active["base_net_return"]
            actual_cash_return=cash/active_ledger_entry_cash-1
            ledger.append({
                "arm":arm,"h3_trade_id":active["id"],"entry_date":active["entry_date"],
                "base_exit_date":day,
                "base_net_return_pct":100*active["base_net_return"],
                "overlay_net_return_pct":100*actual_cash_return,
                "relative_growth_factor":(1+actual_cash_return)/prior_base,
                "overlay_benefit_pp":100*(actual_cash_return-active["base_net_return"]),
                "guard_used":guard_used,
                "guard_still_flat_at_base_exit":guard_flat,
                **labels.get(active["entry_date"],{}),
            })
            active=None; pending=None
            guard_used=False
            guard_flat=False
            active_peak=0.
            active_ledger_entry_cash=None
        if day in entries:
            if active is not None or shares or pending is not None:
                raise AssertionError(f"Overlapping frozen entry: {day}")
            active=entries[day]
            active_ledger_entry_cash=cash
            shares=cash/(price_open*(1+FEE))
            cash=0.
            active_peak=0.
            guard_used=False
            guard_flat=False
        # Process yesterday's overlay signal at today's OPEN, unless this
        # was an original H3 exit session (then pending was nulled).
        if pending is not None:
            if active is None:
                raise AssertionError("Pending overlay without shadow H3")
            side=pending["side"]
            if side=="sell":
                if guard_used or not shares or guard_flat:raise AssertionError("Invalid guard exit")
                cash+=shares*price_open*(1-FEE); shares=0.
                guard_used=True; guard_flat=True
                overlay_used+=1
                if labels.get(active["entry_date"],{}).get("base_ever_50pct_giveback",False):
                    on_event_used+=1
                else:
                    false_event_used+=1
                actions.append({
                    "arm":arm,"ticker":"", "h3_trade_id":active["id"],
                    "entry_date":active["entry_date"],"kind":"GUARD_SELL",
                    "signal_date":pending["date"],"fill_date":day,
                    "fill_price":price_open,"baseline_exit_date":active["base_exit_date"],
                    "peak_gain_at_signal_pct":100*pending["peak_gain"],
                    "gain_at_signal_pct":100*pending["profit"],
                    "giveback_at_signal":pending["giveback"],
                    "ema20_gt60_at_signal":pending["bull"],
                    **labels.get(active["entry_date"],{}),
                })
            elif side=="buy":
                if not guard_used or not guard_flat or shares:raise AssertionError("Invalid guard reentry")
                shares=cash/(price_open*(1+FEE))
                cash=0.; guard_flat=False
                actions.append({
                    "arm":arm,"ticker":"","h3_trade_id":active["id"],
                    "entry_date":active["entry_date"],"kind":"GUARD_REENTER",
                    "signal_date":pending["date"],"fill_date":day,
                    "fill_price":price_open,"baseline_exit_date":active["base_exit_date"],
                    **labels.get(active["entry_date"],{}),
                })
            else:raise AssertionError("Unknown pending overlay")
            pending=None
        # Signal only on complete CLOSE; any original H3 exit signal
        # scheduled for tomorrow takes priority over the guard.
        if active is not None:
            active_peak=max(active_peak,price_close/active["entry_price"]-1.)
            current=price_close/active["entry_price"]-1.
            age=i-entry_lookup[active["entry_date"]]
            if re_cross and guard_flat:
                # One and only one reentry is allowed for this original H3
                # trade. Never force below previous selling price.
                if bool(r[f"gold_{re_cross}"]) and bool(r.trend_bull):
                    pending={"side":"buy","date":day}
            elif (peak_min is not None and not guard_used and shares and
                  age>=AGE_MIN and active_peak>=peak_min and
                  current>0 and bool(r.trend_bull) and
                  bool(r[f"death_{exit_cross}"])):
                # If tomorrow is already the scheduled H3 exit, no distinct
                # early exit exists.
                nextday=str(x.iloc[i+1].Date.date()) if i+1<len(x) else ""
                if nextday != active["base_exit_date"]:
                    pending={"side":"sell","date":day,"peak_gain":active_peak,
                        "profit":current,
                        "giveback":(active_peak-current)/active_peak,
                        "bull":bool(r.trend_bull)}
                else:
                    omitted_due_base_exit+=1
        curve.append({"Date":day,"Equity":cash+shares*price_close,
                      "Invested":bool(shares),
                      "BaselineShadowInvested":active is not None})
    d=pd.DataFrame(curve)
    d["Drawdown"]=d.Equity/d.Equity.cummax()-1
    years=(pd.Timestamp(d.Date.iloc[-1])-pd.Timestamp(start_date)).days/365.25
    led=pd.DataFrame(ledger)
    act=pd.DataFrame(actions)
    # This should match original H3 exactly when no overlay is active.
    stats={
        "arm":arm,"end_equity":float(d.Equity.iloc[-1]),
        "cagr":float((float(d.Equity.iloc[-1])/100000)**(1/years)-1),
        "mdd":float(d.Drawdown.min()),
        "exposure":float(d.Invested.mean()),
        "h3_base_trade_windows":len(t),
        "extra_full_sells":overlay_used,
        "extra_reentry_buys":int((act.kind=="GUARD_REENTER").sum()) if len(act) else 0,
        "unreentered_guards":int(led.guard_still_flat_at_base_exit.sum()) if len(led) else 0,
        "guards_on_bad_giveback_trades":on_event_used,
        "guards_on_other_trades":false_event_used,
        "omitted_same_h3_exit":omitted_due_base_exit,
        "position_open_at_end":bool(shares),
        "shadow_base_open_at_end":active is not None,
        "trades_improved":int((led.overlay_benefit_pp>0.000001).sum()) if len(led) else 0,
        "trades_harmed":int((led.overlay_benefit_pp<-.000001).sum()) if len(led) else 0,
        "mean_trade_pp_improvement":float(led.overlay_benefit_pp.mean()) if len(led) else None,
        "median_trade_pp_improvement":float(led.overlay_benefit_pp.median()) if len(led) else None,
        "worst_trade_pct":float(led.overlay_net_return_pct.min()) if len(led) else None,
    }
    return d,led,act,stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--end",default="2026-11-11")
    p.add_argument("--outdir",default="research/results/h3_profit_ema_fast_guard_run")
    args=p.parse_args()
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    allsummary=[]; alltrades=[]; allactions=[];manifest={"by_symbol":{}}
    for ticker in ("QQQ","GOOGL"):
        raw,prov=load_qqq(args.end) if ticker=="QQQ" else load_googl(args.end)
        # QQQ until 2026-11: 60 completed H3 windows in recent research.
        start_date="2000-01-03" if ticker=="QQQ" else "2004-08-19"
        d=prep_raw(raw)
        frozen_signals=with_month_slope_features(completed_month_signals(d))
        frozen=replay(frozen_signals,"H3_FROZEN",config=Config(start_date=start_date))
        x=features(d,frozen_signals)
        baseline_curve=frozen["equity"]
        labels=describe_giveback(x,frozen["trades"])
        manifest["by_symbol"][ticker]={
            "source":prov,"start":start_date,"end":args.end,
            "frozen_h3_stats":frozen["stats"],
            "n_completed_frozen_trades":len(frozen["trades"]),
        }
        print("H3_PROFIT_GUARD_FROZEN",ticker,json.dumps(frozen["stats"],default=str))
        for arm,peak,exc,rec in ARMS:
            curve,ledger,actions,stats=simulate(x,frozen,arm,peak,exc,rec,start_date,labels)
            # Validates our independent shadow-ledger account accounting.
            if arm=="H3_FROZEN":
                assert curve.Date.tolist()==baseline_curve.Date.tolist()
                difference=np.max(np.abs(curve.Equity.to_numpy()-baseline_curve.Equity.to_numpy()))
                if difference>.05:raise AssertionError(f"Shadow H3 bookkeeping mismatch max {difference}")
                for key in ("end_equity","cagr","mdd","exposure"):
                    if abs(stats[key]-frozen["stats"][key])>1e-7:
                        raise AssertionError(f"Base {key} mismatch {stats[key]} vs {frozen['stats'][key]}")
            stats["ticker"]=ticker
            stats["peak_activation_pct"]=peak*100 if peak is not None else None
            stats["minimum_entry_age_days"]=AGE_MIN
            stats["exit_cross"]=exc
            stats["reentry_cross"]=rec
            allsummary.append(stats)
            ledger["ticker"]=ticker
            actions["ticker"]=ticker
            if len(ledger):alltrades.append(ledger)
            if len(actions):allactions.append(actions)
            if arm!="H3_FROZEN":
                curve.to_csv(out/f"{ticker.lower()}_{arm.lower()}_equity.csv",index=False)
            print("H3_PROFIT_GUARD_RESULT",ticker,json.dumps(stats,default=str))
        (out/f"{ticker.lower()}_frozen_h3_trades.csv").write_text(frozen["trades"].to_csv(index=False))
    summary=pd.DataFrame(allsummary)
    for ticker in ("QQQ","GOOGL"):
        base=summary.loc[(summary.ticker==ticker)&(summary.arm=="H3_FROZEN")].iloc[0]
        for j in summary.index[summary.ticker==ticker]:
            summary.loc[j,"equity_vs_h3_pct"]=100*(summary.loc[j,"end_equity"]/base.end_equity-1.)
            summary.loc[j,"cagr_delta_pp"]=100*(summary.loc[j,"cagr"]-base.cagr)
            summary.loc[j,"mdd_delta_pp"]=100*(summary.loc[j,"mdd"]-base.mdd)
    summary.to_csv(out/"summary.csv",index=False)
    pd.concat(alltrades,ignore_index=True).to_csv(out/"trade_impact.csv",index=False)
    if allactions:pd.concat(allactions,ignore_index=True).to_csv(out/"overlay_actions.csv",index=False)
    manifest.update({
       "initial_cash":100000,
       "fee_per_side":FEE,
       "fill":"completed daily CLOSE decision, next regular OPEN execution",
       "rules":"Peak profit >= 10% or 20%; entry age >=10 sessions; still profitable on close; EMA20>EMA60; EMA5 crosses below EMA10 or EMA20; at most one overlay early exit & one reentry on next EMA5/10 (or 5/20) golden cross while EMA20>EMA60; if never reenter, cash until original H3 shadow exit and then next H3 new entry",
       "base_logic":"Use all original H3 entry/exit timing as an independent shadow engine while overlay may hold cash; no H3 modifications",
       "position_accounting":"All cash (no interest), next-open sell and buy fees; baseline and overlay daily accounts must match original H3 exactly in the control",
       "no_optimization_claim":True,
       "in_sample":True,
       "raw_market_bars_exported":False,
       "risk":"Extra exit can sell a long uptrend; replacing exposure at a higher price may destroy return. Gap/execution slippage, intraday prices and taxes not modeled.",
       "not_live_orders":True,
    })
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n")
    print("H3_PROFIT_GUARD_COMPLETE",summary.to_json(orient="records"))

if __name__=="__main__":main()
