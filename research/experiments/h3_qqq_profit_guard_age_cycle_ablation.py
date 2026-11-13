"""H3 QQQ Profit Guard A/B sensitivity: removal of entry-age and cycle-count caps.

Frozen underlying H3 and 10% peak-profit gate. Test age >=0, >=3, >=10
fully completed sessions, and 1/2/unlimited supplemental sell+reentry
cycles per H3 base trade. All candidate definitions otherwise unchanged.
This is a repeated in-sample stress test, NOT a parameter optimization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h3_midcycle_fast_ema_overlay_reentry as engine
from h3_profit_guard_ab_full_report import compute_metrics, returns_by_period

CONF = {
    "A": {"exit":"cross5_20","re":"cross5_10"},
    "B": {"exit":"cross5_20_pre10_fast","re":"cross5_10"},
}
AGES=(10,3,0)
CAPS=(1,2,9999)
START="2000-01-03"
END="2026-11-11"
FEE=0.0005
def get_data():
    raw,provider=engine.load_qqq(END)
    d=engine.prep_raw(raw)
    sig=engine.with_month_slope_features(engine.completed_month_signals(d))
    frozen=engine.replay(sig,"H3_FROZEN",config=engine.Config(start_date=START))
    x=engine.enrich(engine.features(d,sig))
    labels=engine.describe_giveback(x,frozen["trades"])
    if len(frozen["trades"]) != 60:
        raise AssertionError("H3 60 completed trade reference missing")
    return x,frozen,provider,labels
def row_summary(curve,trades,actions,stats,age,cap,label,ref,regimes):
    m=compute_metrics(curve.Equity)
    m.update(stats)
    m["arm"]=label
    m["min_age"]=age
    m["max_cycles"]="unlimited" if cap==9999 else cap
    m["cum_wealth_vs_H3_pct"]=100*(m["end_equity"]/ref-1)
    m["cagr_vs_H3_pp"]=100*(m["cagr"]-regimes["H3_CAGR"])
    m["mdd_vs_H3_pp"]=100*(m["mdd"]-regimes["H3_MDD"])
    changed=trades.loc[trades.overlay_extra_exits>0]
    m["mean_affected_trade_delta_pp"]=float(changed.delta_return_pp.mean()) if len(changed) else None
    m["median_affected_trade_delta_pp"]=float(changed.delta_return_pp.median()) if len(changed) else None
    m["extra_sell_per_year"]=float(stats["extra_exits"]/regimes["years"])
    m["trade_windows_with_extra_sale"]=len(changed)
    m["trade_windows_with_repeated_sale"]=int((trades.overlay_extra_exits>=2).sum())
    m["trades_improved"]=int((changed.delta_return_pp>1e-6).sum())
    m["trades_harmed"]=int((changed.delta_return_pp<-1e-6).sum())
    buys=actions.loc[actions.type=="GUARD_REBUY"] if len(actions) else pd.DataFrame()
    sells=actions.loc[actions.type=="GUARD_SELL"] if len(actions) else pd.DataFrame()
    m["reenter_after_early_sell_count"]=len(buys)
    m["no_reenter_before_H3_exit_count"]=int((changed.cash_at_h3_exit).sum())
    # Relative price for every SELL=>REBUY matching original window + cycle.
    if len(buys):
        paired=sells.merge(buys,on=["h3_id","cycle"],suffixes=("_sell","_rebuy"),validate="one_to_one")
        m["rebuys_higher_price_count"]=int((paired.adjusted_fill_price_rebuy>paired.adjusted_fill_price_sell).sum())
        m["rebuys_lower_price_count"]=int((paired.adjusted_fill_price_rebuy<paired.adjusted_fill_price_sell).sum())
        m["median_rebuy_price_premium_pct"]=float((100*(paired.adjusted_fill_price_rebuy/paired.adjusted_fill_price_sell-1)).median())
    else:
        m["rebuys_higher_price_count"]=0
        m["rebuys_lower_price_count"]=0
        m["median_rebuy_price_premium_pct"]=None
    # Whole trade-level differences, not wrong per-rebuy PnL attribution.
    r=changed.loc[changed.overlay_extra_reentries>0]
    n=changed.loc[changed.overlay_extra_reentries==0]
    m["reentry_window_improved_count"]=int((r.delta_return_pp>1e-6).sum())
    m["reentry_window_harmed_count"]=int((r.delta_return_pp<-1e-6).sum())
    m["no_reentry_window_improved_count"]=int((n.delta_return_pp>1e-6).sum())
    m["no_reentry_window_harmed_count"]=int((n.delta_return_pp<-1e-6).sum())
    m["mean_delta_pp_reentry_windows"]=float(r.delta_return_pp.mean()) if len(r) else None
    m["mean_delta_pp_no_reentry_windows"]=float(n.delta_return_pp.mean()) if len(n) else None
    profit=trades.loc[trades.base_peak_gain_pct>=10].copy()
    if len(profit):
        giveback=(profit.base_peak_gain_pct-profit.overlay_net_return_pct)/profit.base_peak_gain_pct
        m["retained_profit_ge10_windows"]=len(profit)
        m["end_profit_giveback_ge50_windows"]=int((giveback>=.5).sum())
        m["median_end_profit_giveback"]=float(giveback.median())
    return m

def age_event_audit(x,base):
    loc={str(d.date()):i for i,d in enumerate(x.Date)}
    out=[]
    for t in base["trades"].itertuples(index=False):
        w=x.iloc[loc[t.entry_date]:loc[t.exit_signal_date]+1]
        peak=(w.adj_close/float(t.entry_price)-1).cummax()
        eligible=(peak>=.10)
        first=int(np.where(eligible)[0][0]) if eligible.any() else None
        out.append({
            "h3_trade_id":int(t.id),"entry_date":t.entry_date,"exit_date":t.exit_date,
            "h3_peak_10pct_reached":first is not None,
            "first_10pct_peak_age":first,
            "reached_10pct_peak_within_0_to_9_days":first is not None and first<10,
            "reached_10pct_peak_within_0_to_2_days":first is not None and first<3,
        })
    return pd.DataFrame(out)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--outdir",default="research/results/h3_qqq_age_cycle_ablation_2026_11_11")
    args=p.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    x,base,provider,labels=get_data()
    years=(pd.Timestamp(END)-pd.Timestamp(START)).days/365.25
    baseline=base["equity"]
    regimes={"H3_CAGR":base["stats"]["cagr"],
             "H3_MDD":base["stats"]["mdd"],"years":years}
    # Pure benchmark computed from original frozen H3 and BUY HOLD direct
    h3=base["stats"];h3row={"arm":"H3_FROZEN","min_age":None,"max_cycles":0,**h3}
    bh=x.loc[x.Date>=pd.Timestamp(START)]
    bh_equity=100000/(float(bh.adj_open.iloc[0])*(1+FEE))*bh.adj_close.to_numpy()
    bhstats=compute_metrics(bh_equity)
    bhstats["arm"]="Buy_and_Hold"
    summaries=[]
    actions_all=[];trades_all=[];era=[]
    by_group={}
    reference_out={}
    for label,cond in CONF.items():
      for age in AGES:
        engine.START_AGE=age
        for cap in CAPS:
            arm=f"{label}_AGE{age}_CAP{cap if cap<9999 else 'UNLIMITED'}"
            conf={**cond,"max_cycles":cap}
            curve,trades,actions,stats=engine.simulate(x,base,"QQQ",arm,conf,.10,labels)
            # Special test case reproducing already documented A and B controls.
            if age==10 and cap==1:
                chosen="OLD_P10_X5_20_R5_10" if label=="A" else "CROSS5_20_PRE10_FAST_R5_10"
                refcurve,reftrades,refactions,refstats=engine.simulate(
                    x,base,"QQQ",chosen,{**cond,"max_cycles":1},.10,labels)
                assert curve.Date.tolist()==refcurve.Date.tolist()
                assert np.allclose(curve.Equity,refcurve.Equity,rtol=0,atol=.01)
                reference_out[label]={"cagr":refstats["cagr"],"mdd":refstats["mdd"],
                    "extra_sells":refstats["extra_exits"],
                    "extra_buys":refstats["extra_reentries"]}
                if label=="A":
                    assert abs(stats["cagr"]-.1068802)<.0001
                else:
                    assert abs(stats["cagr"]-.1100949)<.0001
            row=row_summary(curve,trades,actions,stats,age,cap,label,h3["end_equity"],regimes)
            summaries.append(row)
            trades["variant"]=arm
            trades_all.append(trades)
            if len(actions):
                actions["variant"]=arm
                actions_all.append(actions)
            ann,dec=returns_by_period(curve)
            for period,val in dec.items():
                era.append({"variant":arm,"arm":label,"age_min":age,
                            "cycles":"unlimited" if cap==9999 else cap,
                            "period":period,"cumulative_return_pct":float(val)*100})
            if age in (0,10) and cap in (1,9999):
                end=curve.assign(month=curve.Date.str[:7]).groupby("month",as_index=False).tail(1)
                end.to_csv(out/f"{arm.lower()}_month_end_equity.csv",index=False)
            print("AB_AGE_CYCLE_ARM",json.dumps({
                k:row[k] for k in [
                  "arm","min_age","max_cycles","cagr","mdd",
                  "cum_wealth_vs_H3_pct","extra_exits","extra_reentries",
                  "trade_windows_with_extra_sale","trade_windows_with_repeated_sale",
                  "trades_improved","trades_harmed",
                  "reentry_window_improved_count","reentry_window_harmed_count",
                  "no_reentry_window_improved_count","no_reentry_window_harmed_count",
                  "end_profit_giveback_ge50_windows"
                ]},default=str))
    pd.DataFrame(summaries).to_csv(out/"QQQ_A_B_age_cycle_full_summary.csv",index=False)
    pd.concat(actions_all,ignore_index=True).to_csv(out/"QQQ_A_B_age_cycle_extra_fills.csv",index=False)
    pd.concat(trades_all,ignore_index=True).to_csv(out/"QQQ_A_B_age_cycle_trade_impact.csv",index=False)
    pd.DataFrame(era).to_csv(out/"QQQ_A_B_age_cycle_decades.csv",index=False)
    events=age_event_audit(x,base)
    events.to_csv(out/"QQQ_H3_first_10pct_peak_age.csv",index=False)
    assert abs(h3["end_equity"]-float(baseline.Equity.iloc[-1]))<.01
    metadata={
        "universe":"QQQ only; user requested SPY/Nikkei deferred for later validation",
        "cutoff":END,"start":START,
        "capital":100000,"provider":provider,
        "h3_base":h3,"buy_hold":bhstats,
        "arms":CONF,"entry_age_tested":[10,3,0],
        "cap_tested":[1,2,"unlimited"],"peak_profit_activation":0.10,
        "age_semantics":"age0 = allow protective sell on original H3 entry-day close; age3=from the 3rd completed session after entry; age10=original age>=10",
        "fees":FEE,"slippage":0,"cash_yield":0,"tax":0,
        "fixed_one_pass_no_parameter_tuning":True,
        "backtest_in_sample":True,
        "h3_shadow_unchanged":True,
        "no_causal_reset_of_original_H3_fast_month_n20":True,
        "reentry_never_requires_price_below_last_sell":True,
        "crossing_exit_pulse_only":"EMA5 first crosses EMA20, a strict fresh death-cross; repeated eligible crosses later may trigger again",
        "limit_removed":"unlimited means within each original H3 trade; original H3 sell date still overrides any repeated overlay",
        "reference_exact_A_B_age10_cap1":reference_out,
        "warning":"Look-ahead and cherry-picking avoided in signals; comparisons remain historically optimized in sample; benchmark timing sensitive to starting near 2000 crash",
    }
    (out/"manifest.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str)+"\n")
    print("AB_AGE_CYCLE_PEAK_EARLY",json.dumps({
        "all_H3_windows":len(events),
        "peak_10pct_ever_count":int(events.h3_peak_10pct_reached.sum()),
        "peak_10pct_before10":int(events.reached_10pct_peak_within_0_to_9_days.sum()),
        "peak_10pct_before3":int(events.reached_10pct_peak_within_0_to_2_days.sum()),
        "examples":events.loc[events.reached_10pct_peak_within_0_to_9_days].to_dict(orient="records")
    },default=str))
    print("AB_AGE_CYCLE_ALL_RESULTS",pd.DataFrame(summaries).to_json(orient="records"))
    print("AB_AGE_CYCLE_SUCCEEDED",len(summaries))
if __name__=="__main__":main()
