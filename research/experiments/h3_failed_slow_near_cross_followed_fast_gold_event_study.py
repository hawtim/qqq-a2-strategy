"""Failed EMA20/60 near-gold from BELOW, retreat, later fresh EMA10/20 gold.

Full exploratory QQQ event study. ALL signal predicates as-of completed close.
No future data used for signal, but 20/60-day forward outcomes as EX POST
labels only. No in-sample-optimized parameters; broad/core/strict sensitivity.
Compare to all fresh EMA10/20 gold in EMA20<EMA60 (bear slow), then to
subset when original frozen H3 flat. Event outcomes: next open entry,
fast death next open sell PnL (subject to right-censor), 20/60d price
returns, 20/60d worst adjusted LOW, first-hit +8% vs -7% (non-tradable
ex-post path label). Repeated close signals overlapping not independent.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from h3_profit_n10_ema_guard_shadow_overlay import (
    load_qqq,prep_raw,completed_month_signals,
    with_month_slope_features,replay as h3_replay,Config
)
START="2000-01-03";END="2026-11-11";FEE=.0005
DEFINITIONS=[
    ("BROAD",1.0,.25,.10,.10,40),
    ("CORE",.50,.40,.20,.20,30),
    ("STRICT",.30,.60,.30,.30,30),
]
def build(bars,signals,frozen):
    s=signals.copy().sort_values("Date").reset_index(drop=True)
    s["atr20"]=pd.concat([
        s.adj_high-s.adj_low,
        (s.adj_high-s.adj_close.shift(1)).abs(),
        (s.adj_low-s.adj_close.shift(1)).abs()
    ],axis=1).max(axis=1).rolling(20,min_periods=20).mean()
    s["fast_gold"]=(s.ma10>s.ma20)&(
        s.ma10.shift(1)<=s.ma20.shift(1))
    s["fast_death"]=(s.ma10<s.ma20)&(
        s.ma10.shift(1)>=s.ma20.shift(1))
    s["slow_bear"]=(s.ma20<s.ma60)
    s["gap_atr"]=(s.ma60-s.ma20)/s.atr20
    f=frozen["equity"].copy()
    f["Date"]=pd.to_datetime(f.Date)
    s=s.merge(f[["Date","Invested"]].rename(
        columns={"Invested":"frozen_H3_invested"}),
        on="Date",how="left",validate="one_to_one")
    s.loc[s.Date<pd.Timestamp(START),"frozen_H3_invested"]=False
    s["frozen_H3_invested"]=s.frozen_H3_invested.fillna(False).astype(bool)
    assert s.Date.is_unique
    return s

def find_pattern(s,t,lookback,near,widen,drop,approach):
    """EMA20 approached EMA60 from below, failed, fell, THEN fast gold.

    As-of t: nearest approach p is MIN normalized slow-bear gap over
    [t-lookback,t-3]. Slow must remain BELOW slow60 since p.
    At p, gap <= near*ATR; five bars earlier gap was >= approach ATR
    further, showing an actual approach.
    After p: gap now increased >= widen ATR above p's gap, and EMA20
    had fallen >= drop*ATR below level at p, then t fresh FAST gold.
    """
    if t<lookback+65 or not bool(s.at[t,"fast_gold"]):
        return None
    if not bool(s.at[t,"slow_bear"]):return None
    past=s.iloc[t-lookback:t-2]
    past=past.loc[past.slow_bear & past.atr20.gt(0)]
    if past.empty:return None
    p=int(past.gap_atr.idxmin())
    if p<5 or t-p<3:return None
    if not np.isfinite(s.at[p,"gap_atr"]) or s.at[p,"gap_atr"]>near:return None
    if not bool(s.iloc[p:t+1].slow_bear.all()):return None
    denom=float(s.at[p,"atr20"]);nowatr=float(s.at[t,"atr20"])
    if not np.isfinite(denom) or not np.isfinite(nowatr):return None
    prior_gap=float(s.at[p-5,"ma60"]-s.at[p-5,"ma20"])
    p_gap=float(s.at[p,"ma60"]-s.at[p,"ma20"])
    if prior_gap-p_gap<approach*denom:return None
    nowgap=float(s.at[t,"ma60"]-s.at[t,"ma20"])
    if nowgap-p_gap<widen*nowatr:return None
    min20=float(s.iloc[p+1:t+1].ma20.min())
    if float(s.at[p,"ma20"])-min20<drop*denom:return None
    min20idx=int(s.iloc[p+1:t+1].ma20.idxmin())
    if min20idx>=t:return None
    return {
        "approach_date":str(s.at[p,"Date"].date()),
        "approach_to_signal_sessions":int(t-p),
        "approach_gap_atr":float(s.at[p,"gap_atr"]),
        "now_gap_atr":float(s.at[t,"gap_atr"]),
        "old_gap_raw":p_gap,
        "now_gap_raw":nowgap,
        "ema20_drop_since_near_atr":(float(s.at[p,"ma20"])-min20)/denom,
        "approach_confirmed_5d_prior_atr":(prior_gap-p_gap)/denom,
        "ema20_local_low_date":str(s.at[min20idx,"Date"].date()),
    }
def forward(s,t,h,buy):
    if t+h>=len(s):return {"complete":False}
    path=s.iloc[t+1:t+h+1]
    ret=float(s.at[t+h,"adj_close"]/buy-1)
    return {
        "complete":True,"close_ret_pct":ret*100,
        "peak_close_pct":100*(float(path.adj_close.max())/buy-1),
        "worst_low_pct":100*(float(path.adj_low.min())/buy-1)
    }
def first_barrier(s,t,buy,h=60,upper=.08,lower=-.07):
    if t+h>=len(s):return "CENSORED"
    for j in range(t+1,t+h+1):
        r=s.iloc[j]
        is_up=float(r.adj_high)/buy-1>=upper
        is_down=float(r.adj_low)/buy-1<=lower
        if is_up and is_down:return "BOTH_SAME_DAY_ORDER_UNKNOWN"
        if is_up:return "UPSIDE_FIRST"
        if is_down:return "DOWNSIDE_FIRST"
    return "NEITHER"
def fast_exit(s,t,buy):
    """Pure 10/20 death signal -> next open. Not an H3-native N20 trade."""
    for j in range(t+1,len(s)-1):
        if bool(s.at[j,"fast_death"]):
            sell=float(s.at[j+1,"adj_open"])
            return {
                "fast_death_exit_date":str(s.at[j+1,"Date"].date()),
                "fast_death_signal_date":str(s.at[j,"Date"].date()),
                "fast_trade_net_ret_pct":100*(
                   sell*(1-FEE)/(buy*(1+FEE))-1),
                "fast_trade_duration_sessions":int(j-t),
                "fast_trade_complete":True,
            }
    return {
        "fast_death_exit_date":"",
        "fast_death_signal_date":"",
        "fast_trade_net_ret_pct":None,
        "fast_trade_duration_sessions":None,
        "fast_trade_complete":False,
    }
def assess(s,t,definition,patt):
    buy=float(s.at[t+1,"adj_open"])
    result={
        "event_id":str(s.at[t,"Date"].date())+"_"+definition,
        "signal_date":str(s.at[t,"Date"].date()),
        "next_open_date":str(s.at[t+1,"Date"].date()),
        "definition":definition,
        "H3_was_flat_at_signal":not bool(s.at[t,"frozen_H3_invested"]),
        "month_global_gate_allows":bool(s.at[t,"gate_up_slow_veto_1_abs"]),
        "ema20_below_ema60":bool(s.at[t,"slow_bear"]),
        "fast_gold":bool(s.at[t,"fast_gold"]),
        "close_over_ema20":bool(s.at[t,"adj_close"]>s.at[t,"ma20"]),
        "next_open_buy_adj":buy,
        "h3_month_slope_known":float(s.at[t,"known_slope_abs"]),
        **(patt or {}),
    }
    for h in (20,60):
        f=forward(s,t,h,buy)
        for k,v in f.items():result[f"{k}_{h}d"]=v
    result["first_hit_60d"]=first_barrier(s,t,buy)
    result.update(fast_exit(s,t,buy))
    return result

def summarized(df,definition,case):
    part=df.loc[df.definition.eq(definition)]
    if case=="H3_FLAT":part=part.loc[part.H3_was_flat_at_signal]
    if case=="H3_FLAT_BLOCKED_GLOBAL":
        part=part.loc[part.H3_was_flat_at_signal &
                      ~part.month_global_gate_allows]
    row={
        "definition":definition,"sample":case,"n":len(part),
    }
    for h in (20,60):
        good=part.loc[part[f"complete_{h}d"]]
        vals=good[f"close_ret_pct_{h}d"].dropna()
        row.update({
            f"n_complete_{h}d":len(vals),
            f"forward_{h}d_win_pct":
                100*float(vals.gt(0).mean()) if len(vals) else None,
            f"forward_{h}d_mean_ret_pct":float(vals.mean())
                  if len(vals) else None,
            f"forward_{h}d_median_ret_pct":float(vals.median())
                  if len(vals) else None,
            f"forward_{h}d_worst_low_pct":float(
                good[f"worst_low_pct_{h}d"].min()) if len(good) else None,
            f"forward_{h}d_at_least_plus8_pct":float(
                100*vals.ge(8).mean()) if len(vals) else None,
        })
    x=part.loc[part.fast_trade_complete]
    net=x.fast_trade_net_ret_pct.dropna()
    row.update({
        "n_completed_fast_death_exits":len(net),
        "fast_death_trade_win_pct":
            100*float(net.gt(0).mean()) if len(net) else None,
        "fast_death_trade_mean_net_ret_pct":
            float(net.mean()) if len(net) else None,
        "fast_death_trade_median_net_ret_pct":
            float(net.median()) if len(net) else None,
        "fast_death_trade_worst_net_ret_pct":
            float(net.min()) if len(net) else None,
        "upper8_first":int(part.first_hit_60d.eq("UPSIDE_FIRST").sum()),
        "lower7_first":int(part.first_hit_60d.eq("DOWNSIDE_FIRST").sum()),
        "both_barriers_same_day":int(
            part.first_hit_60d.eq("BOTH_SAME_DAY_ORDER_UNKNOWN").sum()),
    })
    return row
def nonoverlap(s_events,days=60):
    z=s_events.sort_values("signal_date").copy()
    take=[]
    last=-100000
    for r in z.itertuples(index=False):
        i=int(r.signal_index)
        if i-last>days:
            take.append(r.Index if hasattr(r,"Index") else None)
            last=i
    # use as-of signal index explicitly
    entries=[]
    last=-100000
    for _,r in z.iterrows():
        if int(r.signal_index)-last>days:
            entries.append(r.to_dict());last=int(r.signal_index)
    return pd.DataFrame(entries)
def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",
         default="research/results/h3_failed_slow_near_cross_fast_gold")
    args=p.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,source=load_qqq(END)
    bars=prep_raw(raw)
    sig=with_month_slope_features(completed_month_signals(bars))
    frozen=h3_replay(sig,"H3_FROZEN",config=Config(start_date=START))
    assert len(frozen["trades"])==60
    s=build(bars,sig,frozen)
    allrows=[]
    for t in range(len(s)-1):
        if s.at[t,"Date"]<pd.Timestamp(START) or not bool(s.at[t,"fast_gold"]):
            continue
        if not bool(s.at[t,"slow_bear"]):continue
        base=assess(s,t,"ALL_FAST_SLOW_BEAR",{})
        base["signal_index"]=int(t)
        allrows.append(base)
        for name,near,widen,drop,approach,lookback in DEFINITIONS:
            patt=find_pattern(s,t,lookback,near,widen,drop,approach)
            if patt is None:continue
            row=assess(s,t,name,patt)
            row["signal_index"]=int(t)
            allrows.append(row)
    if not allrows:raise AssertionError("No eligible FAST gold signals at all")
    events=pd.DataFrame(allrows)
    events.to_csv(out/"all_fast_gold_slow_bear_and_near_miss_patterns.csv",index=False)
    defs=["ALL_FAST_SLOW_BEAR"]+[z[0] for z in DEFINITIONS]
    sums=[]
    for definition in defs:
        for case in ("ALL","H3_FLAT","H3_FLAT_BLOCKED_GLOBAL"):
            sums.append(summarized(events,definition,case))
    summary=pd.DataFrame(sums)
    summary.to_csv(out/"sample_size_win_probability_and_fast_exit_summary.csv",index=False)
    sens=[]
    # 60-session nonoverlap vs baseline. DO NOT infer accurate CIs
    # from all overlapping event-study forward windows.
    for definition in defs:
        subset=events.loc[
            events.definition.eq(definition) &
            events.H3_was_flat_at_signal]
        sel=nonoverlap(subset,60)
        if len(sel):
            z=summarized(sel,definition,"ALL")
            z["nonoverlap_group"]="H3_FLAT_GAP60"
            sens.append(z)
    pd.DataFrame(sens).to_csv(out/"nonoverlap_60session_signal_diagnostics.csv",index=False)
    # Top nominal events for manual factual inspection, not cherry-pick as edge
    examples=events.loc[(events.definition=="CORE") &
                       events.H3_was_flat_at_signal].copy()
    examples.sort_values("signal_date").to_csv(
        out/"core_events_h3_flat_timeline.csv",index=False)
    print("FAILED_SLOW_PATTERN_MAIN",
        summary.to_json(orient="records"))
    print("FAILED_SLOW_PATTERN_NONOVERLAP",
        pd.DataFrame(sens).to_json(orient="records"))
    print("FAILED_SLOW_PATTERN_CORE_CASES",
        examples[["signal_date","approach_date",
            "approach_to_signal_sessions","approach_gap_atr",
            "now_gap_atr","forward_20d_win_pct"
        ]].to_json(orient="records") if False else
        examples[["signal_date","approach_date",
            "approach_to_signal_sessions","approach_gap_atr",
            "now_gap_atr","close_ret_pct_20d",
            "worst_low_pct_20d","close_ret_pct_60d",
            "fast_trade_net_ret_pct"]].to_json(orient="records"))
    manifest={
        "question":"EMA20 approaches EMA60 from BELOW without slow gold, falls away and later fresh EMA10/20 gold: success probability?",
        "data_source":source,"period":[START,END],
        "baseline":"original frozen H3, used for flat status ONLY",
        "entry":"fresh EMA10/20 gold at complete daily close, next-session adjusted open, no monthly entry veto in event-study selection",
        "near_miss":"gap=(EMA60-EMA20)>0, near within last 30 or 40 sessions excluding last 2, min normalized gap <= near ATR20, gap was shrinking 5 sessions earlier, EMA20 subsequently drops and gap widens, never actual slow gold; later FAST gold while still slow-bear",
        "definitions":DEFINITIONS,
        "benchmark":"all fresh FAST gold with EMA20<EMA60; optionally restrict original H3 flat or global month veto",
        "success":"20/60 session close positive after next open entry, AND actual independent EMA10/20 fast death next-open exit net return as descriptive standalone cycle, AND 60d first hit +8% vs -7% outcome",
        "overlap":"forward20/60 signals can overlap, 60-session nonoverlap selection provided; no independent claims",
        "lookahead":"no in entry, forward outcomes only for EVALUATION; no live H3 full account candidate tested",
        "no_raw_market_data_redistribution":True,
        "frozen_H3_A_B_unchanged":True,
        "limits":"single QQQ history, researcher-defined morphology threshold, in-sample, multiple testing",
    }
    (out/"manifest.json").write_text(json.dumps(
        manifest,ensure_ascii=False,indent=2,default=str)+"\n")
    print("FAILED_SLOW_PATTERN_FINISHED",len(events))
if __name__=="__main__":main()
