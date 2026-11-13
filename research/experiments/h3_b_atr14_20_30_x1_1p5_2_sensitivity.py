"""Audit B ATR14/20/30 * 1.0/1.5/2.0, unchanged frozen H3 shadow.

Pure sensitivity analysis: NEVER change frozen B v1.0, use copy of
feature dataframe with alternative ATR lookbacks in original overlay
engine's normalized three-day close decline field.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd,numpy as np
import h3_midcycle_fast_ema_overlay_reentry as engine
from h3_profit_guard_ab_full_report import returns_by_period
LOOKBACKS=[14,20,30]
MULT=[1.0,1.5,2.0]
START="2000-01-03";END="2026-11-11"
def main():
    a=argparse.ArgumentParser();a.add_argument("--outdir",default="research/results/h3_b_atr_sensitivity");ns=a.parse_args()
    out=Path(ns.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,source=engine.load_qqq(END)
    bars=engine.prep_raw(raw)
    sig=engine.with_month_slope_features(engine.completed_month_signals(bars))
    base=engine.replay(sig,"H3_FROZEN",config=engine.Config(start_date=START))
    assert len(base["trades"])==60
    x0=engine.enrich(engine.features(bars,sig))
    labels=engine.describe_giveback(x0,base["trades"])
    oldage=engine.START_AGE;oldthreshold=engine.ATR_FAST_DECLINE
    engine.START_AGE=0
    rows=[];events=[];yearly=[]
    try:
        tr=pd.concat([
            bars.adj_high-bars.adj_low,
            (bars.adj_high-bars.adj_close.shift(1)).abs(),
            (bars.adj_low-bars.adj_close.shift(1)).abs()],axis=1).max(axis=1)
        # frozen A same for all benchmarks; no threshold
        ca,ta,ea,sa=engine.simulate(
            x0,base,"QQQ","A_FROZEN",
            {"exit":"cross5_20","re":"cross5_10","max_cycles":1},
            .10,labels)
        assert sa["extra_exits"]==28 and sa["extra_reentries"]==18
        for length in LOOKBACKS:
            atr=tr.rolling(length,min_periods=length).mean()
            x=x0.copy()
            x["atr20"]=atr
            x["decline3_atr20"]=(x.adj_close.shift(3)-x.adj_close)/atr
            x["price_fast_decline"]=x.decline3_atr20>=oldthreshold
            for thresh in MULT:
                engine.ATR_FAST_DECLINE=thresh
                x["price_fast_decline"]=x.decline3_atr20>=thresh
                name=f"B_ATR{length}_X{thresh:.1f}"
                c,t,e,s=engine.simulate(
                    x,base,"QQQ",name,
                    {"exit":"cross5_20_pre10_fast",
                     "re":"cross5_10","max_cycles":1},
                    .10,labels)
                if length==20 and thresh==1.5:
                    assert s["extra_exits"]==22 and s["extra_reentries"]==12
                    assert abs(s["cagr"]-.110095)<.00005
                    assert abs(s["mdd"]+.242999)<.00005
                rows.append({
                  "variant":name,"atr_window":length,
                  "decline3_atr_threshold":thresh,
                  "cagr_pct":s["cagr"]*100,
                  "mdd_pct":s["mdd"]*100,
                  "end_equity":s["end_equity"],
                  "extra_sells":s["extra_exits"],
                  "extra_rebuys":s["extra_reentries"],
                  "delta_cagr_vs_frozen_B_pp":100*(s["cagr"]-.110095),
                  "delta_cagr_vs_frozen_A_pp":100*(s["cagr"]-sa["cagr"]),
                  "mdd_vs_frozen_B_pp":100*(s["mdd"]+.242999),
                })
                yy,dec=returns_by_period(c)
                for period,val in dec.items():
                    yearly.append({
                        "variant":name,"period":period,
                        "cumulative_account_return_pct":100*float(val)})
                if len(e):
                    z=e[["h3_id","type","signal_date","fill_date","cycle"]].copy()
                    z["variant"]=name
                    events.append(z)
    finally:
        engine.ATR_FAST_DECLINE=oldthreshold
        engine.START_AGE=oldage
    pd.DataFrame(rows).to_csv(out/"atr_window_threshold_sensitivity_summary.csv",index=False)
    pd.DataFrame(yearly).to_csv(out/"atr_window_threshold_era_returns.csv",index=False)
    if events:pd.concat(events,ignore_index=True).to_csv(
        out/"atr_window_threshold_actual_orders.csv",index=False)
    manifest={
        "question":"Does frozen B special sell rule derive from an optimum ATR20 * 1.5?",
        "scope":"Sensitivity only, original A and B unchanged",
        "market": "QQQ",
        "sample":[START,END],"source":source,
        "frozen_control":"ATR20 X1.5 exactly reproduces B v1.0 22/12 extras and ~11.01% CAGR",
        "grid":[LOOKBACKS,MULT],
        "TrueRange":"max(high-low, abs(high-prevClose),abs(low-prevClose))",
        "ATR_definition":"simple rolling mean of N true ranges. 3d close difference normalized by ATRN",
        "initial_cash":100000,"fee_each_side":.0005,
        "future_information":"no, completed day only and next open fills",
        "interpretation":"Do not select a best threshold from in-sample values as evidence of out-of-sample improvement",
        "A_B_frozen_unchanged":True
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,
        ensure_ascii=False,default=str)+"\n")
    print("ATR_B_SENSITIVITY",pd.DataFrame(rows).to_json(orient="records"))
    print("ATR_B_FINISHED",len(rows))
if __name__=="__main__":main()
