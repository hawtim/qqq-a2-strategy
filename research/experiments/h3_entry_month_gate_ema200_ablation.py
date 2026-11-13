"""H3 entry-moon-veto relaxation factorial, QQQ 2000-2026.

Full native H3 state machine replay (N20/7%/entry day, monthly eligibility,
deferred FAST etc), not shadow replacement and no A/B overlay. The original
H3 files and frozen A/B candidate contract remain UNCHANGED.

Precommitted stage order:
  H3 frozen;
  (1) release ALL monthly entry vetoes, including REGULAR;
  (2) allow overrides only above RISING EMA200;
  (3) focus FAST-only bypass and EMA200; 
  (4) relax global month gate only vs special FAST gate only;
  (5) selectively permit only positive month EMA5 slowdown or negative
      month EMA5 recovery when EMA200 rising;
  (6) optional EMA20>60 state restriction.
SAME unmodified H3 exits, same next-open fills, same $100k and fees.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from h3_profit_n10_ema_guard_shadow_overlay import (
  load_qqq, prep_raw, completed_month_signals,with_month_slope_features,
  replay as h3_replay, Config
)
from h3_month_gate_entry_relax_engine import replay_gated as experimental_replay
from a2_v3_selected import ENTRY_GATE,V3_PARAMETERS

START="2000-01-03"
END="2026-11-11"
OUT_OF_SAMPLE=False
ARMS=[
    ("H3_FROZEN","none","Frozen exact control"),
    ("ALL_NO_MONTH","all_no_month","Cancel global month veto for FAST & REGULAR"),
    ("ALL_NO_MONTH_EMA200","all_no_month_ema200","Cancel both monthly gates only while price>rising EMA200"),
    ("FAST_NO_MONTH","fast_no_month","Cancel both monthly gates for fresh FAST only"),
    ("FAST_NO_MONTH_EMA200","fast_no_month_ema200","FAST month bypass only if above rising EMA200"),
    ("FAST_NO_MONTH_ABOVE_EMA200","fast_no_month_above200","FAST month bypass if above EMA200 even if flat/down"),
    ("FAST_GLOBAL_ONLY_EMA200","fast_global_veto_only_ema200","FAST: waive only global positive-month deceleration, preserve FAST monthly"),
    ("FAST_SPECIAL_ONLY_EMA200","fast_special_month_only_ema200","FAST: preserve global gate, waive FAST monthly"),
    ("FAST_POSITIVE_SLOWDOWN_ONLY_EMA200","fast_positive_slowdown_only_ema200","FAST: specifically waive positive EMA5 monthly slope slowdown"),
    ("FAST_NEGATIVE_RECOVERY_ONLY_EMA200","fast_negative_recovery_only_ema200","FAST: specifically permit negative monthly EMA5 recovery under global veto"),
    ("FAST_NO_MONTH_EMA200_SLOW_BULL","fast_no_month_ema200_slow_bull","FAST month bypass above rising200 only if EMA20>EMA60"),
]
PERIODS=[
  ("2000-2009","2000-01-03","2009-12-31"),
  ("2010-2014","2010-01-01","2014-12-31"),
  ("2015-2019","2015-01-01","2019-12-31"),
  ("2020-2022","2020-01-01","2022-12-31"),
  ("2023-2026","2023-01-01",END),
]
def metrics_of_equity(equity):
    a=equity.Equity.to_numpy(dtype=float)
    start=pd.Timestamp(equity.Date.iloc[0])
    end=pd.Timestamp(equity.Date.iloc[-1])
    years=(end-start).days/365.25
    mdd=np.min(a/np.maximum.accumulate(a)-1)
    r=a[1:]/a[:-1]-1
    daily_std=np.std(r,ddof=1)
    return {
        "end_equity":float(a[-1]),"cagr":float((a[-1]/100000)**(1/years)-1),
        "mdd":float(mdd),"exposure":float(equity.Invested.mean()),
        "daily_std":float(daily_std),
        "sharpe_zero_cash":float(np.mean(r)/daily_std*np.sqrt(252)) if daily_std else None,
        "calmar":float(((a[-1]/100000)**(1/years)-1)/abs(mdd)) if mdd else None,
    }
def periods_from_equity(result):
    e=result["equity"].copy()
    e["Date"]=pd.to_datetime(e.Date)
    annual=e.assign(year=e.Date.dt.year)
    annual["r"]=annual.Equity.pct_change()
    annual.loc[annual.index[0],"r"]=annual.Equity.iloc[0]/100000-1
    a=annual.groupby("year").r.apply(lambda y:float(np.prod(1+y)-1))
    result_rows=[]
    for label,start,end in PERIODS:
        pos=np.flatnonzero((e.Date>=pd.Timestamp(start)) & (e.Date<=pd.Timestamp(end)))
        if len(pos)==0:raise AssertionError("missing period")
        last_i=int(pos[-1]);first_i=int(pos[0])
        equity_before=100000 if first_i==0 else float(e.Equity.iloc[first_i-1])
        period_curve=e.iloc[pos].copy()
        cash_pct=1-float(period_curve.Invested.mean())
        result_rows.append({
            "period":label,
            "period_return_pct":100*(float(e.Equity.iloc[last_i])/equity_before-1),
            "period_daily_mdd_pct":100*float(
                (period_curve.Equity.to_numpy() /
                np.maximum.accumulate(np.r_[equity_before,period_curve.Equity.to_numpy()])[1:]-1).min()),
            "cash_pct":100*cash_pct,
            "days":len(pos),
        })
    return a,result_rows
def outcome_trades(result):
    t=result["trades"].copy()
    if "entry_override" not in t.columns:
        t["entry_override"]="BASE"
    t["entry_override"]=t.entry_override.fillna("BASE")
    return t
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--end",default=END)
    p.add_argument("--outdir",default="research/results/h3_month_veto_entry_relax")
    args=p.parse_args()
    if args.end!=END:raise ValueError("Fixed experiment benchmark cutoff to 2026-11-11")
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,provenance=load_qqq(args.end)
    d=prep_raw(raw)
    signals=with_month_slope_features(completed_month_signals(d))
    config=Config(start_date=START)
    frozen=h3_replay(signals,"H3_FROZEN",config=config)
    assert len(frozen["trades"])==60
    control_baseline=frozen["stats"]
    runs={}
    summaries=[];trades=[];period=[];annual=[];entries=[];new_entry_diagnostics=[]
    reference_dates=frozen["equity"].Date.tolist()
    for name,mode,description in ARMS:
        run=experimental_replay(
            signals, ENTRY_GATE,config=config,**V3_PARAMETERS,
            defer_rejected_fast=True,
            fast_negative_month_mode="accelerating",
            trend_spread_min_gain=.10,
            entry_relax_mode=mode)
        assert run["equity"].Date.tolist()==reference_dates
        if name=="H3_FROZEN":
            assert np.allclose(run["equity"].Equity,frozen["equity"].Equity,rtol=0,atol=.02)
            assert run["equity"].Invested.equals(frozen["equity"].Invested)
            left=run["trades"].drop(columns=["entry_override"],errors="ignore").reset_index(drop=True)
            right=frozen["trades"].reset_index(drop=True)
            pd.testing.assert_frame_equal(left,right,check_dtype=False)
            for key in ("end_equity","cagr","mdd","completed_trades","exposure"):
                v=run["stats"][key];vref=control_baseline[key]
                assert abs(v-vref)<(.02 if key=="end_equity" else 1e-7), (key,v,vref)
        runs[name]=run
        st=run["stats"]
        eq_metrics=metrics_of_equity(run["equity"])
        assert abs(st["end_equity"]-eq_metrics["end_equity"])<.02
        assert abs(st["mdd"]-eq_metrics["mdd"])<1e-7
        t=outcome_trades(run)
        t["arm"]=name
        trades.append(t)
        new=t[t.entry_override!="BASE"]
        loser=t[t.ret_net<=0];winner=t[t.ret_net>0]
        summ={
            "arm":name,"mode":mode,"description":description,
            **st,
            "extra_waived_entries_completed":len(new),
            "extra_waived_regular_completed":int((new.entry_override=="RELAX_REGULAR").sum()),
            "extra_waived_fast_completed":int((new.entry_override=="RELAX_FAST").sum()),
            "new_waived_entry_win_count":int((new.ret_net>0).sum()),
            "new_waived_entry_loss_count":int((new.ret_net<=0).sum()),
            "total_trade_net_win_pct":float((t.ret_net>0).mean()) if len(t) else None,
            "hold_days_median":float(t.hold_days.median()) if len(t) else None,
            "worst_waived_entry_net_pct":float(100*new.ret_net.min()) if len(new) else None,
            "mean_waived_entry_net_pct":float(100*new.ret_net.mean()) if len(new) else None,
            "max_single_trade_loss_pct":100*float(t.ret_net.min()) if len(t) else None,
            "profit_factor":st["profit_factor"],
            "payoff_ratio":st["payoff_ratio"],
            "exposure":st["exposure"],
            "sharpe_zero_cash":eq_metrics["sharpe_zero_cash"],
            "calmar":eq_metrics["calmar"],
            "median_exit_reasons":t.exit_reason.value_counts().head(3).to_dict(),
            "same_sample_in_sample":True,
        }
        summ["cagr_vs_h3_pp"]=100*(st["cagr"]-control_baseline["cagr"])
        summ["mdd_vs_h3_pp"]=100*(st["mdd"]-control_baseline["mdd"])
        summ["total_wealth_vs_h3_pct"]=100*(st["end_equity"]/control_baseline["end_equity"]-1)
        summaries.append(summ)
        y,pds=periods_from_equity(run)
        for row in pds:period.append({"arm":name,**row})
        for k,v in y.items():annual.append({"arm":name,"year":int(k),"return_pct":100*float(v)})
        if len(new):
            # Each actual *executed* special entry; resulting N20, 7% and
            # trend exits all generated by the modified full state machine.
            for row in new.itertuples(index=False):
                new_entry_diagnostics.append({
                    "arm":name,"entry_signal_date":row.entry_signal_date,
                    "entry_date":row.entry_date,"entry_kind":row.entry_kind,
                    "entry_override":row.entry_override,"exit_date":row.exit_date,
                    "exit_reason":row.exit_reason,"ret_net_pct":100*row.ret_net,
                    "hold_days":row.hold_days,
                    "signal_month":row.entry_signal_date[:7]
                })
        # Daily balance + invested state available to prove no lookahead.
        run["equity"][["Date","Equity","Drawdown","Invested"]].to_csv(
            out/f"{name.lower()}_daily_equity.csv",index=False)
        print("ENTRY_RELAX_ARM",json.dumps({
            k:summ[k] for k in [
               "arm","cagr","mdd","exposure","completed_trades",
               "extra_waived_entries_completed","extra_waived_regular_completed","extra_waived_fast_completed",
               "cagr_vs_h3_pp","mdd_vs_h3_pp","total_wealth_vs_h3_pct",
               "new_waived_entry_win_count","new_waived_entry_loss_count"
            ]},default=str))
    # Buy and hold from the exact same PRICE series and fee rules.
    b=signals[signals.Date>=pd.Timestamp(START)].copy()
    bh_shares=100000/(float(b.adj_open.iloc[0])*(1+config.fee_per_side))
    bh_values=bh.adj_close.to_numpy()*bh_shares
    bh_curve=pd.DataFrame({"Date":[str(v.date()) for v in b.Date],
                           "Equity":bh_values,
                           "Invested":True})
    bh_curve["Drawdown"]=bh_curve.Equity/bh_curve.Equity.cummax()-1
    bmetrics=metrics_of_equity(bh_curve)
    brows={"arm":"BuyHold","mode":"none","description":"Buy at initial open and hold",
            **bmetrics,"completed_trades":0,"extra_waived_entries_completed":0,
            "cagr_vs_h3_pp":100*(bmetrics["cagr"]-control_baseline["cagr"]),
            "mdd_vs_h3_pp":100*(bmetrics["mdd"]-control_baseline["mdd"]),
            "total_wealth_vs_h3_pct":100*(bmetrics["end_equity"]/control_baseline["end_equity"]-1)}
    summaries.append(brows)
    _,pds=periods_from_equity({"equity":bh_curve})
    for row in pds:period.append({"arm":"BuyHold",**row})
    byear,_=periods_from_equity({"equity":bh_curve})
    for k,v in byear.items():annual.append({"arm":"BuyHold","year":int(k),"return_pct":100*float(v)})
    pd.DataFrame(summaries).to_csv(out/"full_account_summary.csv",index=False)
    pd.DataFrame(period).to_csv(out/"era_comparison.csv",index=False)
    pd.DataFrame(annual).to_csv(out/"calendar_year_returns.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(out/"all_completed_trades.csv",index=False)
    pd.DataFrame(new_entry_diagnostics).to_csv(out/"waived_actual_entry_trades.csv",index=False)
    if not pd.DataFrame(new_entry_diagnostics).empty:
        pd.DataFrame(new_entry_diagnostics).to_csv(out/"waived_actual_entry_trades.csv",index=False)
    # Simple fixed comparison of bullish-gate trading eras.
    piv=pd.DataFrame(period).pivot(index="period",columns="arm",values="period_return_pct")
    print("ENTRY_RELAX_PERIOD",piv.to_json())
    manifest={
        "purpose":"entry constraint research; NOT H3 or A/B rule changes",
        "source":provenance,
        "fixed_history":{"start":START,"end":END,"qqq_daily_rows":len(d),
                         "h3_completed_trades":len(frozen["trades"])},
        "baseline":{"cagr":control_baseline["cagr"],
                    "mdd":control_baseline["mdd"],
                    "end_equity":control_baseline["end_equity"]},
        "baseline_exact_match_daily_equity":True,
        "baseline_exact_match_original_trade_ledger":True,
        "variants":ARMS,
        "state_machine":"Independent copy of replay_gated; all H3 N20/7pct stop, entry-day stop, extended regular N20, FAST deferred signals, monthly eligibility and exits re-executed under changed entry dates",
        "control_entry_priority":"any originally authorized H3 entry remains authorized; additional bypass gold-crosses do NOT suppress baseline entry rules",
        "global_month_gate":"gate_up_slow_veto_1_abs rejects current known monthly EMA5 slope positive but below previous month slope",
        "fast_month_gate":"previous COMPLETED month EMA5 up OR special 2 negative slopes with ongoing negative acceleration",
        "ema200":"price above daily EMA200 and daily EMA200[t] above EMA200[t-20], all completed close as of signal",
        "execution":"end-of-day signal -> next regular session OPEN; full capital, 0.05pct each buy/sell, cash zero interest, tax/slippage omitted",
        "not_tested":"no profit A/B overlay, no live orders, no independent OOS/SPY/Nikkei",
        "disclaimer":"all results after substantial earlier QQQ tuning; no statistical proof, param scanning risk",
        "income_reinvestment_proxy":"supplier adjusted prices, taxes excluded",
    }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n")
    print("ENTRY_RELAX_ALL_SUMMARY",pd.DataFrame(summaries).to_json(orient="records"))
    print("ENTRY_RELAX_SUCCEEDED",len(runs))
if __name__=="__main__":main()
