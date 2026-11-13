"""H3 profit protection: crossing, rapid price breaks, distance-gated early EMA exits.

Independent cash/shares overlay on immutable frozen H3 shadow positions.
No future maxima used in any signal. Completed close -> next regular open.
Full-lifecycle comparisons (including reset reentry and up to unlimited cycles).

PRIMARY:
  A) Confirm old QQQ P10 EMA5<20 death -> EMA5>10 gold (1 cycle).
  B) Price crosses under EMA20 with 3-session fall >=1.5 ATR20;
     reenter when price crosses above EMA10. 1/2/unlimited guard cycles.
  C) Early EMA5/20 death while EMA10>EMA20>EMA60 and
     (EMA20-EMA60)/ATR20>=1.5, plus an optional fast-decline filter.

GOOGL uses peak>=20%, QQQ uses peak>=10% for primary comparison.
All arms are predeclared, in sample, and NOT selected H3 updates.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research"/"experiments"))
from h3_profit_n10_ema_guard_shadow_overlay import (
    load_qqq,load_googl,prep_raw,completed_month_signals,
    with_month_slope_features,replay,Config,features,simulate as prior_simulate,
    describe_giveback,buy_hold_comparator
)

FEE=.0005
START_AGE=10
ATR_SLOW_DISTANCE=1.5
ATR_FAST_DECLINE=1.5
ARM_DEFS={
    "H3_FROZEN":dict(exit="none",re="none",max_cycles=0),
    "OLD_P10_X5_20_R5_10":dict(exit="cross5_20",re="cross5_10",max_cycles=1),
    "PRICE20_CROSS_RPRICE10_ONCE":dict(exit="price20_cross",re="price10_cross",max_cycles=1),
    "PRICE20_FAST_RPRICE10_ONCE":dict(exit="price20_cross_fast",re="price10_cross",max_cycles=1),
    "PRICE20_FAST_RPRICE10_TWICE":dict(exit="price20_cross_fast",re="price10_cross",max_cycles=2),
    "PRICE20_FAST_RPRICE10_UNLIMITED":dict(exit="price20_cross_fast",re="price10_cross",max_cycles=9999),
    "PRICE20_FAST_STATE_RPRICE10_UNLIMITED":dict(exit="price20_fast_state",re="price10_cross",max_cycles=9999),
    "CROSS5_20_PRE10_BULL_R5_10":dict(exit="cross5_20_pre10",re="cross5_10",max_cycles=1),
    "CROSS5_20_FAR_R5_10":dict(exit="cross5_20_far",re="cross5_10",max_cycles=1),
    "CROSS5_20_PRE10_FAR_R5_10":dict(exit="cross5_20_pre10_far",re="cross5_10",max_cycles=1),
    "CROSS5_20_PRE10_FAR_FAST_R5_10":dict(exit="cross5_20_pre10_far_fast",re="cross5_10",max_cycles=1),
    "CROSS5_20_PRE10_FAST_R5_10":dict(exit="cross5_20_pre10_fast",re="cross5_10",max_cycles=1),
}

def enrich(x):
    d=x.copy()
    d["atr20"]=pd.concat([
      d.adj_high-d.adj_low,
      (d.adj_high-d.adj_close.shift(1)).abs(),
      (d.adj_low-d.adj_close.shift(1)).abs()
    ],axis=1).max(axis=1).rolling(20,min_periods=20).mean()
    d["trend_distance_atr20"]=(d.ema20-d.ema60)/d.atr20
    d["decline3_atr20"]=(d.adj_close.shift(3)-d.adj_close)/d.atr20
    d["price_below_ema20_cross"]=(d.adj_close<d.ema20)&(d.adj_close.shift(1)>=d.ema20.shift(1))
    d["price_above_ema10_cross"]=(d.adj_close>d.ema10)&(d.adj_close.shift(1)<=d.ema10.shift(1))
    d["price_fast_decline"]=d.decline3_atr20>=ATR_FAST_DECLINE
    d["pre_fast_death"]=(d.ema10>d.ema20)&d.trend_bull
    d["far_from_slow"]=d.trend_distance_atr20>=ATR_SLOW_DISTANCE
    d["cross5_20"]=d.death_5_20
    return d

def exit_condition(bar, name):
    if name=="none":return False
    if name=="cross5_20":return bool(bar.cross5_20)
    if name=="price20_cross":return bool(bar.price_below_ema20_cross)
    if name=="price20_cross_fast":return bool(bar.price_below_ema20_cross and bar.price_fast_decline)
    if name=="price20_fast_state":return bool(bar.adj_close<bar.ema20 and bar.price_fast_decline)
    if name=="cross5_20_pre10":return bool(bar.cross5_20 and bar.pre_fast_death)
    if name=="cross5_20_far":return bool(bar.cross5_20 and bar.far_from_slow)
    if name=="cross5_20_pre10_far":return bool(bar.cross5_20 and bar.pre_fast_death and bar.far_from_slow)
    if name=="cross5_20_pre10_fast":return bool(bar.cross5_20 and bar.pre_fast_death and bar.price_fast_decline)
    if name=="cross5_20_pre10_far_fast":
        return bool(bar.cross5_20 and bar.pre_fast_death and bar.far_from_slow and bar.price_fast_decline)
    raise ValueError("Unknown exit "+name)

def reentry_condition(bar,name):
    if name=="cross5_10":return bool(bar.gold_5_10)
    if name=="price10_cross":return bool(bar.price_above_ema10_cross)
    if name=="none":return False
    raise ValueError("Unknown reentry "+name)

def simulate(x,base,ticker,arm,conf,peak_gate,labels):
    start="2000-01-03" if ticker=="QQQ" else "2004-08-19"
    entries={}
    exit_days={}
    for trade in base["trades"].itertuples(index=False):
        if trade.entry_date in entries:raise AssertionError("duplicate entry")
        entries[trade.entry_date]={
            "id":int(trade.id),"date":trade.entry_date,
            "entry_open":float(trade.entry_price),"exit_date":trade.exit_date,
            "base_net_return":float(trade.ret_net),
        }
        if trade.exit_date in exit_days:raise AssertionError("duplicate exit")
        exit_days[trade.exit_date]=int(trade.id)
    if base["open_position"] is not None:
        op=base["open_position"]; day=op["entry_date"]
        if day not in entries:
            entries[day]={"id":len(base["trades"])+1,
                          "date":day,"entry_open":float(op["entry_price"]),
                          "exit_date":None,"base_net_return":None}
    index={str(dt.date()):i for i,dt in enumerate(x.Date)}
    cash=100000.; shares=0.
    active=None; pending=None
    entry_cash=None;hwm_gain=0.
    count_sells=0;count_reentries=0
    events=[];trades=[];equity=[]
    for i,b in x.iterrows():
        day=str(b.Date.date())
        if day<start:continue
        price_open=float(b.adj_open)
        close=float(b.adj_close)
        # Process the baseline's original next-open exit first. An overlay
        # trade cannot extend the original H3 holding period.
        if day in exit_days:
            if active is None or active["id"]!=exit_days[day]:
                raise AssertionError("H3 shadow mismatch exit "+day)
            if shares:
                cash+=shares*price_open*(1-FEE);shares=0.
            observed=cash/entry_cash-1
            base_return=active["base_net_return"]
            trades.append({
                "ticker":ticker,"arm":arm,"h3_id":active["id"],
                "entry_date":active["date"],"exit_date":day,
                "h3_net_return_pct":100*base_return,
                "overlay_net_return_pct":100*observed,
                "delta_return_pp":100*(observed-base_return),
                "relative_growth":(1+observed)/(1+base_return),
                "overlay_extra_exits":count_sells,
                "overlay_extra_reentries":count_reentries,
                "cash_at_h3_exit":shares==0,
                "base_ever50_giveback":labels.get(active["date"],{}).get("base_ever_50pct_giveback"),
                "base_peak_gain_pct":100*labels.get(active["date"],{}).get("base_max_peak_gain",0),
            })
            active=None;pending=None;entry_cash=None
        if day in entries:
            if active is not None or shares or pending:
                raise AssertionError("H3 shadow mismatch entry "+day)
            active=entries[day]
            entry_cash=cash
            shares=cash/(price_open*(1+FEE));cash=0.
            hwm_gain=0.;count_sells=0;count_reentries=0
        if pending:
            if active is None:raise AssertionError("pending with no H3 active trade")
            if pending["side"]=="sell":
                if not shares:raise AssertionError("double overlay sell")
                before_shares=shares
                cash+=shares*price_open*(1-FEE);shares=0.
                count_sells+=1
                if count_sells>conf["max_cycles"]:raise AssertionError("too many cycles")
                events.append({
                    "ticker":ticker,"arm":arm,"h3_id":active["id"],
                    "original_entry":active["date"],"type":"GUARD_SELL",
                    "signal_date":pending["signal_date"],"fill_date":day,
                    "adjusted_fill_price":price_open,"cycle":count_sells,
                    "base_h3_exit":active["exit_date"],
                    "peak_profit_at_signal_pct":pending["peak_profit_pct"],
                    "current_profit_at_signal_pct":pending["current_profit_pct"],
                    "giveback_at_signal":pending["giveback"],
                    "decline3_atr20":pending["decline3_atr20"],
                    "slow_distance_atr20":pending["slow_distance_atr20"],
                    "ema10_gt20":pending["ema10_gt20"],
                    "base_ever50_giveback":labels.get(active["date"],{}).get("base_ever_50pct_giveback"),
                })
            elif pending["side"]=="buy":
                if shares or not cash:raise AssertionError("double overlay buy")
                shares=cash/(price_open*(1+FEE));cash=0.
                count_reentries+=1
                events.append({
                    "ticker":ticker,"arm":arm,"h3_id":active["id"],
                    "original_entry":active["date"],"type":"GUARD_REBUY",
                    "signal_date":pending["signal_date"],"fill_date":day,
                    "adjusted_fill_price":price_open,"cycle":count_reentries,
                    "base_h3_exit":active["exit_date"],
                })
            else:raise ValueError("bad pending")
            pending=None
        if active is not None:
            hwm_gain=max(hwm_gain,close/active["entry_open"]-1)
            gain=close/active["entry_open"]-1
            age=i-index[active["date"]]
            if not shares:
                # A price crossing above EMA10 or EMA5/10 gold cross,
                # no "below last sell" requirement; trend must still be bull.
                if reentry_condition(b,conf["re"]) and bool(b.trend_bull):
                    pending={"side":"buy","signal_date":day}
            elif (conf["max_cycles"]>count_sells and age>=START_AGE and
                  hwm_gain>=peak_gate and gain>0 and bool(b.trend_bull)
                  and exit_condition(b,conf["exit"])):
                # The next H3 exit controls; don't schedule a duplicate.
                nextday=str(x.iloc[i+1].Date.date()) if i+1<len(x) else ""
                if nextday!=active["exit_date"]:
                    pending={
                        "side":"sell","signal_date":day,
                        "peak_profit_pct":100*hwm_gain,
                        "current_profit_pct":100*gain,
                        "giveback":(hwm_gain-gain)/hwm_gain,
                        "decline3_atr20":float(b.decline3_atr20),
                        "slow_distance_atr20":float(b.trend_distance_atr20),
                        "ema10_gt20":bool(b.ema10>b.ema20),
                    }
        equity.append({"Date":day,"Equity":cash+shares*close,"Invested":bool(shares)})
    curve=pd.DataFrame(equity)
    curve["Drawdown"]=curve.Equity/curve.Equity.cummax()-1
    T=(pd.Timestamp(curve.Date.iloc[-1])-pd.Timestamp(start)).days/365.25
    t=pd.DataFrame(trades)
    e=pd.DataFrame(events)
    positive=t.loc[t.overlay_net_return_pct>0] if len(t) else t
    negative=t.loc[t.overlay_net_return_pct<0] if len(t) else t
    gain_abs=positive.overlay_net_return_pct.mean() if len(positive) else None
    loss_abs=-negative.overlay_net_return_pct.mean() if len(negative) else None
    stats={
        "ticker":ticker,"arm":arm,
        "peak_gate_pct":100*peak_gate,
        "trend_atr_distance_threshold":ATR_SLOW_DISTANCE if "far" in conf["exit"] else None,
        "fast_3day_decline_atr_threshold":ATR_FAST_DECLINE if "fast" in conf["exit"] else None,
        "cycles_cap":conf["max_cycles"] if conf["max_cycles"]<9999 else "unlimited",
        "end_equity":float(curve.Equity.iloc[-1]),
        "cagr":float((curve.Equity.iloc[-1]/100000)**(1/T)-1),
        "mdd":float(curve.Drawdown.min()),
        "exposure":float(curve.Invested.mean()),
        "baseline_shadow_h3_trades":len(base["trades"]),
        "extra_exits":int((e.type=="GUARD_SELL").sum()) if len(e) else 0,
        "extra_reentries":int((e.type=="GUARD_REBUY").sum()) if len(e) else 0,
        "changed_h3_trades":int((t.overlay_extra_exits>0).sum()) if len(t) else 0,
        "repeated_guard_trades":int((t.overlay_extra_exits>=2).sum()) if len(t) else 0,
        "max_guard_exits_one_h3":int(t.overlay_extra_exits.max()) if len(t) else 0,
        "improved_h3_trades":int((t.delta_return_pp>0.000001).sum()) if len(t) else 0,
        "harmed_h3_trades":int((t.delta_return_pp<-0.000001).sum()) if len(t) else 0,
        "mean_delta_pp":float(t.delta_return_pp.mean()) if len(t) else None,
        "win_rate":float(len(positive)/len(t)) if len(t) else None,
        "payoff_ratio":float(gain_abs/loss_abs) if loss_abs and gain_abs else None,
        "profit_factor":float(positive.overlay_net_return_pct.sum()/-negative.overlay_net_return_pct.sum()) if len(negative) and len(positive) else None,
        "open_shadow_position_at_end":active is not None,
        "open_real_position_at_end":bool(shares),
    }
    return curve,t,e,stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--end",default="2026-11-11")
    p.add_argument("--outdir",default="research/results/h3_midcycle_fast_reentry")
    args=p.parse_args()
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    summaries=[];alltrades=[];allevents=[];top=[];eras=[];manifest={"parameters":{
        "age_min_full_sessions":START_AGE,
        "extra_order_fee_each_side":FEE,
        "atr20_spread_min":ATR_SLOW_DISTANCE,
        "fast_3day_decline_atr20_min":ATR_FAST_DECLINE,
        "no_h3_state_change":True,
        "original_h3_entry_exit_signal_from_frozen_engine":True,
        "close_signal_next_regular_open_fill":True,
        "one_trade_overlay_is_exactly_one_exit_one_rebuy":True,
        "price10_reentry": "completed daily close crosses from <=EMA10 to >EMA10, EMA20>EMA60; next open; no constraint against reentry above previous sell price",
        "price20_cross_exit":"completed daily close crosses from >=EMA20 to <EMA20",
        "fast":"close three sessions earlier minus current close >=1.5*ATR20 (ATR20 includes completed signal day only)",
        "pre_ema10_20_death":"EMA10>EMA20 when EMA5 crosses under EMA20",
        "far":"(EMA20-EMA60)/ATR20>=1.5, plus EMA20>EMA60",
        "trading_model":"100% invest or 100% cash, each original H3 trade independently shadowed; any early cash remains idle unless new crossing; never extend original H3 exit",
    },"by_symbol":{}}
    for ticker in ("QQQ","GOOGL"):
        gate=.10 if ticker=="QQQ" else .20
        start="2000-01-03" if ticker=="QQQ" else "2004-08-19"
        raw,prov=load_qqq(args.end) if ticker=="QQQ" else load_googl(args.end)
        d=prep_raw(raw)
        signals=with_month_slope_features(completed_month_signals(d))
        base=replay(signals,"H3_FROZEN",config=Config(start_date=start))
        x=enrich(features(d,signals))
        labels=describe_giveback(x,base["trades"])
        bh=buy_hold_comparator(x,start)
        manifest["by_symbol"][ticker]={"input":prov,"frozen_stats":base["stats"],
            "buy_hold":bh,"gate":gate,"input_rows":len(d)}
        print("MIDCYCLE_BASE",ticker,json.dumps(base["stats"],default=str))
        print("MIDCYCLE_BH",ticker,json.dumps(bh,default=str))
        for arm,conf in ARM_DEFS.items():
            curve,trades,events,stats=simulate(x,base,ticker,arm,conf,gate,labels)
            if arm=="H3_FROZEN":
                ref=base["equity"]
                assert curve.Date.tolist()==ref.Date.tolist()
                assert np.allclose(curve.Equity,ref.Equity,atol=.05,rtol=0)
                for key in ("cagr","mdd","exposure","end_equity"):
                    assert abs(stats[key]-base["stats"][key])<.05 if key=="end_equity" else abs(stats[key]-base["stats"][key])<1e-7
            if arm=="OLD_P10_X5_20_R5_10":
                # Independent old script with identically matched rules to
                # protect against false new benefit due to implementation.
                old_curve,old_t,old_act,old_stats=prior_simulate(
                    x,base,"P10_X5_20_R5_10" if ticker=="QQQ" else "P20_X5_20_R5_10",
                    gate,"5_20","5_10",start,labels)
                diff=float(np.max(abs(curve.Equity.to_numpy()-old_curve.Equity.to_numpy())))
                assert diff<.05, f"old candidate mismatch {ticker}: {diff:.8f}"
                for key in ("cagr","mdd","exposure","end_equity"):
                    assert abs(stats[key]-old_stats[key]) < (.05 if key=="end_equity" else 1e-7), (ticker,key,stats[key],old_stats[key])
            baseline_eq=float(base["stats"]["end_equity"])
            stats["equity_vs_frozen_pct"]=100*(stats["end_equity"]/baseline_eq-1)
            stats["cagr_vs_frozen_pp"]=100*(stats["cagr"]-base["stats"]["cagr"])
            stats["mdd_vs_frozen_pp"]=100*(stats["mdd"]-base["stats"]["mdd"])
            stats["cagr_vs_buyhold_pp"]=100*(stats["cagr"]-bh["cagr"])
            summaries.append(stats)
            alltrades.append(trades)
            if len(events):allevents.append(events)
            # Derived trade outcomes grouped by original H3 entry cohort.
            decade=lambda date:("2000s" if int(date[:4])<2010 else ("2010s" if int(date[:4])<2020 else "2020-2026"))
            if len(trades):
                for period,g in trades.groupby(trades.entry_date.map(decade)):
                    eras.append({"ticker":ticker,"arm":arm,"period":period,
                        "frozen_window_count":len(g),
                        "relative_wealth_factor":float(g.relative_growth.prod()),
                        "guards":int(g.overlay_extra_exits.sum()),
                        "mean_delta_return_pp":float(g.delta_return_pp.mean())})
                changed=trades.loc[trades.overlay_extra_exits>0]
                if len(changed):
                    for direction,k in [("best",False),("worst",True)]:
                        subset=changed.sort_values("delta_return_pp",ascending=k).head(5)
                        for t in subset.to_dict("records"):
                            top.append({"sample":direction,**t})
            if arm not in ("H3_FROZEN","OLD_P10_X5_20_R5_10"):
                # One monthly equity curve is sufficient for review; no raw prices.
                tmp=curve.assign(month=curve.Date.str[:7]).groupby("month",as_index=False).tail(1)
                tmp[["Date","Equity","Drawdown","Invested"]].to_csv(
                    out/f"{ticker.lower()}_{arm.lower()}_month_end_equity.csv",index=False)
            print("MIDCYCLE_ARM",json.dumps(stats,default=str))
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/"summary.csv",index=False)
    pd.concat(alltrades,ignore_index=True).to_csv(out/"trade_impact.csv",index=False)
    pd.concat(allevents,ignore_index=True).to_csv(out/"guard_actions.csv",index=False)
    pd.DataFrame(eras).to_csv(out/"era_relative_factors.csv",index=False)
    pd.DataFrame(top).to_csv(out/"top_winners_losers.csv",index=False)
    manifest["arms"]=ARM_DEFS
    manifest["source_issues"]=["Yahoo adjusted OHLC and volumes can be revised",
        "no slippage/taxes/cash yield or intraday execution",
        "in-sample param choices; variant comparisons are not OOS validation",
        "one QQQ vs one GOOGL cannot validate an asset-class-specific coefficient",
        "distances are dollar/ATR-based and not independent of market volatility",
        "A H3 shadow freeze means early exits do not reset original H3 entry eligibility; that is the defined overlay, not re-running H3 after every sale"]
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n")
    print("MIDCYCLE_RESULTS",summary.to_json(orient="records"))
    print("MIDCYCLE_SUCCEEDED",len(summary))
if __name__=="__main__":main()
