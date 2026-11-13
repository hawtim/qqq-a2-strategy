"""QQQ frozen H3 profit guard A vs B, realized-volatility regime attribution.

No strategy switching/backfit. Exact frozen A/B age0 cap1 replay only; 20d
realized annualized close volatility computed from complete adjusted closes.
Daily B-A relative LOG-return spread attributed to PREVIOUS DAY rv20 bins
(fixed <20%, 20-30%, >=30%), to avoid day-t volatility lookahead.
Paired per-H3-window B/A log growth assigned to FIRST divergent SELL signal,
rv20 as-of its completed close. Historical association not causality.
High-vol regime does not prove B outperformance: count positive/negative
paired windows and dominant high-impact events. Also 2000/2010/2020 eras.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

import h3_midcycle_fast_ema_overlay_reentry as engine
from h3_profit_guard_ab_full_report import compute_metrics

START="2000-01-03"
END="2026-11-11"
ARMS={
    "H3":{"exit":"none","re":"none","max_cycles":0},
    "A":{"exit":"cross5_20","re":"cross5_10","max_cycles":1},
    "B":{"exit":"cross5_20_pre10_fast","re":"cross5_10","max_cycles":1},
}
VOL_LABELS=("LOW_<20%","MID_20-30%","HIGH_>=30%","UNKNOWN")
ERAS=[
  ("2000-2009","2000-01-03","2009-12-31"),
  ("2010-2019","2010-01-01","2019-12-31"),
  ("2020-2026","2020-01-01",END),
]
def vol_bin(v):
    if pd.isna(v):return "UNKNOWN"
    return "LOW_<20%" if v<.20 else "MID_20-30%" if v<.30 else "HIGH_>=30%"
def era_of(day):
    d=str(day)
    return "2000-2009" if d<"2010" else "2010-2019" if d<"2020" else "2020-2026"
def setup_daily(bars,signal):
    x=engine.enrich(engine.features(bars,signal))
    lr=np.log(x.adj_close/x.adj_close.shift(1))
    x["rv20_ann"]=lr.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)
    x["rv5_ann"]=lr.rolling(5,min_periods=5).std(ddof=1)*np.sqrt(252)
    x["vol_bin"]=x.rv20_ann.map(vol_bin)
    x["rv20_ann_prior_close"]=x.rv20_ann.shift(1)
    x["prior_close_vol_bin"]=x.rv20_ann_prior_close.map(vol_bin)
    return x
def exact_baseline(x,base,config):
    engine.START_AGE=0
    curves={}; trades={};events={};st={}
    labels=engine.describe_giveback(x,base["trades"])
    for k in ("H3","A","B"):
        c,t,e,s=engine.simulate(x,base,"QQQ","H3_AB_VOL_REGIME_"+k,
                                ARMS[k],.10,labels)
        c["Date"]=pd.to_datetime(c.Date)
        if k=="H3":
            expected=base["equity"]
            assert expected.Date.tolist()==[str(d.date()) for d in c.Date]
            assert np.allclose(c.Equity,expected.Equity,rtol=0,atol=.05)
            assert np.array_equal(c.Invested,expected.Invested)
        curves[k]=c;trades[k]=t;events[k]=e;st[k]=s
    assert int(st["A"]["extra_exits"])==28
    assert int(st["A"]["extra_reentries"])==18
    assert int(st["B"]["extra_exits"])==22
    assert int(st["B"]["extra_reentries"])==12
    assert abs(st["A"]["cagr"]-.10688)<.002
    assert abs(st["B"]["cagr"]-.11010)<.002
    return curves,trades,events,st
def full_period_table(curves):
    rows=[]
    for name,begin,end in ERAS:
        out={"period":name}
        for k,c in curves.items():
            w=c.loc[c.Date.between(pd.Timestamp(begin),pd.Timestamp(end))]
            pos0=w.index[0]
            baseval=float(c.Equity.iloc[pos0-1]) if pos0 else 100000.
            out[k+"_period_return_pct"]=100*(float(w.Equity.iloc[-1])/baseval-1)
        out["B_minus_A_simple_period_return_pp"]=(
            out["B_period_return_pct"]-out["A_period_return_pct"])
        out["B_over_A_period_relative_wealth_pct"]=100*(
            (1+out["B_period_return_pct"]/100)/
            (1+out["A_period_return_pct"]/100)-1)
        rows.append(out)
    return pd.DataFrame(rows)
def daily_regime_decompose(curves,features):
    # Use volatility as known at YESTERDAY completed close for each day
    # portfolio close-to-close move to avoid contemporaneous label bias.
    z=pd.DataFrame({"Date":curves["H3"].Date})
    for k,c in curves.items():
        z[k+"_equity"]=c.Equity.to_numpy()
        z[k+"_invested"]=c.Invested.to_numpy()
        z[k+"_dayret"]=c.Equity.pct_change().fillna(
            c.Equity.iloc[0]/100000.-1).to_numpy()
    feat=features[[
        "Date","rv20_ann","rv20_ann_prior_close",
        "prior_close_vol_bin","vol_bin","decline3_atr20",
        "trend_bull"
    ]].copy()
    z=z.merge(feat,on="Date",validate="one_to_one")
    z["B_minus_A_daily_log"]=(
        np.log1p(z.B_dayret)-np.log1p(z.A_dayret))
    z["B_over_A_wealth"]=z.B_equity/z.A_equity
    z["B_A_daily_exposure_diff"]=(
        z.B_invested.astype(int)-z.A_invested.astype(int))
    z["era"]=z.Date.map(era_of)
    group=z.groupby(["era","prior_close_vol_bin"],dropna=False).agg(
        days=("Date","size"),
        B_minus_A_log_growth=("B_minus_A_daily_log","sum"),
        n_B_only_long_days=("B_A_daily_exposure_diff",
            lambda x:int((x>0).sum())),
        n_A_only_long_days=("B_A_daily_exposure_diff",
            lambda x:int((x<0).sum())),
        mean_known_prior_rv20=("rv20_ann_prior_close","mean")
    ).reset_index()
    group["B_A_factor_pct"]=100*(np.exp(group.B_minus_A_log_growth)-1)
    overall=z.groupby("prior_close_vol_bin",dropna=False).agg(
        days=("Date","size"),
        B_minus_A_log_growth=("B_minus_A_daily_log","sum"),
        n_B_only_long_days=("B_A_daily_exposure_diff",
            lambda x:int((x>0).sum())),
        n_A_only_long_days=("B_A_daily_exposure_diff",
            lambda x:int((x<0).sum()))
    ).reset_index()
    overall["B_A_factor_pct"]=100*(np.exp(overall.B_minus_A_log_growth)-1)
    return z,group,overall
def paired_windows(trades,events,features):
    aa=trades["A"][["h3_id","entry_date","exit_date",
        "overlay_net_return_pct","overlay_extra_exits",
        "overlay_extra_reentries"]].rename(columns={
          "overlay_net_return_pct":"A_net_return_pct",
          "overlay_extra_exits":"A_extra_sells",
          "overlay_extra_reentries":"A_special_rebuys"})
    bb=trades["B"][["h3_id","entry_date","exit_date",
        "overlay_net_return_pct","overlay_extra_exits",
        "overlay_extra_reentries"]].rename(columns={
          "overlay_net_return_pct":"B_net_return_pct",
          "overlay_extra_exits":"B_extra_sells",
          "overlay_extra_reentries":"B_special_rebuys"})
    p=aa.merge(bb,on=["h3_id","entry_date","exit_date"],
        validate="one_to_one")
    p["B_minus_A_return_pp"]=p.B_net_return_pct-p.A_net_return_pct
    p["B_minus_A_log_growth"]=(
        np.log1p(p.B_net_return_pct/100)-
        np.log1p(p.A_net_return_pct/100))
    p["B_relative_win"]=p.B_minus_A_log_growth>1e-9
    p["A_relative_win"]=p.B_minus_A_log_growth< -1e-9
    p["era"]=p.entry_date.map(era_of)
    fa=events["A"].loc[events["A"].type.eq("GUARD_SELL"),
            ["h3_id","signal_date","fill_date",
             "decline3_atr20","ema10_gt20"]].rename(columns={
                 "signal_date":"A_first_sell_signal",
                 "fill_date":"A_first_sell_fill",
                 "decline3_atr20":"A_decline3_atr20_at_sell",
                 "ema10_gt20":"A_ema10_gt20_at_sell"})
    fb=events["B"].loc[events["B"].type.eq("GUARD_SELL"),
            ["h3_id","signal_date","fill_date"]].rename(columns={
                "signal_date":"B_first_sell_signal",
                "fill_date":"B_first_sell_fill"})
    assert not fa.h3_id.duplicated().any()
    assert not fb.h3_id.duplicated().any()
    p=p.merge(fa,on="h3_id",how="left",validate="one_to_one")
    p=p.merge(fb,on="h3_id",how="left",validate="one_to_one")
    p["different_extra_exit_timing"]=(
        p.A_first_sell_signal.fillna("")!=
        p.B_first_sell_signal.fillna(""))
    # First different signal is always an A extra sell:
    # B's condition is a subset of A when both are long.
    p["first_divergence_signal_date"]=p.A_first_sell_signal.where(
        p.different_extra_exit_timing)
    vol=features[["Date","rv20_ann","rv5_ann",
        "decline3_atr20","trend_bull","ma10","ma20"]].copy()
    vol["Date"]=vol.Date.dt.strftime("%Y-%m-%d")
    p=p.merge(vol.rename(columns={
        "Date":"first_divergence_signal_date",
        "rv20_ann":"rv20_at_divergence",
        "rv5_ann":"rv5_at_divergence",
        "decline3_atr20":"decline3_atr_at_divergence",
        "trend_bull":"slow_trend_bull_at_divergence",
        "ma10":"ma10_at_divergence",
        "ma20":"ma20_at_divergence"}),
        on="first_divergence_signal_date",how="left",
        validate="many_to_one")
    p["divergence_vol_bin"]=p.rv20_at_divergence.map(vol_bin)
    p["A_extra_sell_B_did_not_just_then"]=p.different_extra_exit_timing
    p["A_sell_after_B_sell"]=p.A_first_sell_signal.fillna("")>p.B_first_sell_signal.fillna("X")
    p["first_divergence_B_filter_failure"]=np.select([
        ~p.different_extra_exit_timing,
        p.A_first_sell_signal.isna(),
        p.A_ema10_gt20_at_sell.fillna(False)==False,
        p.A_decline3_atr20_at_sell<1.5,
    ],["none","A_no_sell","ema10_not_above20","decline_below_1p5_ATR"],
        default="mixed_or_other")
    keys=p.loc[p.different_extra_exit_timing].copy()
    # Interpret per-original-H3 trade log growth difference, paired
    # and normalized: not a directly causal experiment in volatility.
    summ=keys.groupby("divergence_vol_bin",dropna=False).agg(
        n_divergent_h3_windows=("h3_id","size"),
        B_better=("B_relative_win","sum"),
        A_better=("A_relative_win","sum"),
        summed_B_minus_A_log_growth=("B_minus_A_log_growth","sum"),
        mean_B_minus_A_trade_return_pp=("B_minus_A_return_pp","mean"),
        median_B_minus_A_trade_return_pp=("B_minus_A_return_pp","median"),
    ).reset_index()
    summ["B_A_relative_growth_factor_pct"]=100*(
        np.exp(summ.summed_B_minus_A_log_growth)-1)
    return p,summ
def special_intervals(events,feature):
    """A-only early exit windows and whether price recovered after sell.

    This is post-event descriptive ex-post observation, never a strategy
    signal. B may sell later; A may buy back, so don't equate the signal
    with all subsequent account advantage.
    """
    fa=events["A"].loc[events["A"].type.eq("GUARD_SELL")].copy()
    fb=events["B"].loc[events["B"].type.eq("GUARD_SELL")].copy()
    feat=feature.set_index(feature.Date.dt.strftime("%Y-%m-%d"))
    bmap={int(r.h3_id):str(r.signal_date) for r in fb.itertuples(index=False)}
    records=[]
    for r in fa.itertuples(index=False):
        a_day=str(r.signal_date)
        b_day=bmap.get(int(r.h3_id),"")
        if b_day==a_day:continue
        if a_day not in feat.index:continue
        pos=feat.index.get_loc(a_day)
        if isinstance(pos,slice):continue
        closing=float(feat.iloc[pos].adj_close)
        forward=[]
        for n in (3,5,10):
            if pos+n<len(feat):
                forward.append((n,100*(float(feat.iloc[pos+n].adj_close)/closing-1)))
        records.append({
            "h3_id":int(r.h3_id),
            "A_early_sell_signal_date":a_day,
            "A_sell_fill_date":str(r.fill_date),
            "B_sell_signal_date":b_day,
            "A_decline3_atr20":float(r.decline3_atr20),
            "ema10_gt20_at_A_signal":bool(r.ema10_gt20),
            "rv20_at_A_signal":float(feat.iloc[pos].rv20_ann),
            "vol_bin":vol_bin(feat.iloc[pos].rv20_ann),
            "forward_3d_pct_ex_post":dict(forward).get(3),
            "forward_5d_pct_ex_post":dict(forward).get(5),
            "forward_10d_pct_ex_post":dict(forward).get(10),
            "price_rebounded_5d":dict(forward).get(5,0)>0,
        })
    return pd.DataFrame(records)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outdir",default="research/results/h3_ab_volatility_regime")
    args=parser.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,provider=engine.load_qqq(END)
    bars=engine.prep_raw(raw)
    signals=engine.with_month_slope_features(
        engine.completed_month_signals(bars))
    base=engine.replay(signals,"H3_FROZEN",
                      config=engine.Config(start_date=START))
    assert len(base["trades"])==60
    x=setup_daily(bars,signals)
    curves,trades,events,st=exact_baseline(x,base,None)
    daily,era_vol,overall_vol=daily_regime_decompose(curves,x)
    pairs,pairbins=paired_windows(trades,events,x)
    unmatched=special_intervals(events,x)
    periods=full_period_table(curves)
    daily.to_csv(out/"daily_A_B_H3_account_and_20d_asof_vol.csv",index=False)
    era_vol.to_csv(out/"A_B_relative_daily_log_by_era_and_vol.csv",index=False)
    overall_vol.to_csv(out/"A_B_relative_daily_log_by_vol.csv",index=False)
    pairs.to_csv(out/"paired_H3_trade_windows_A_vs_B.csv",index=False)
    pairbins.to_csv(out/"divergence_events_A_B_vs_vol_bin.csv",index=False)
    unmatched.to_csv(out/"A_only_guard_exit_forward_price_diagnostics.csv",index=False)
    periods.to_csv(out/"actual_era_account_compounding_A_B.csv",index=False)
    action=pd.concat([events[k].assign(arm=k)
        for k in ("A","B")],ignore_index=True)
    action.to_csv(out/"all_actual_guard_sells_and_rebuys.csv",index=False)
    expected_ratio=st["B"]["end_equity"]/st["A"]["end_equity"]
    daily_log_total=float(daily.B_minus_A_daily_log.sum())
    assert abs(float(np.exp(daily_log_total))-expected_ratio)<1e-9
    assert abs(float(np.exp(pairs.B_minus_A_log_growth.sum()))-
               expected_ratio)<1e-9
    v={
        "A":{"cagr":st["A"]["cagr"],"mdd":st["A"]["mdd"],
             "end":st["A"]["end_equity"],
             "extra_sells":st["A"]["extra_exits"],
             "extra_rebuys":st["A"]["extra_reentries"]},
        "B":{"cagr":st["B"]["cagr"],"mdd":st["B"]["mdd"],
             "end":st["B"]["end_equity"],
             "extra_sells":st["B"]["extra_exits"],
             "extra_rebuys":st["B"]["extra_reentries"]},
        "H3":{"cagr":st["H3"]["cagr"],"mdd":st["H3"]["mdd"],
              "end":st["H3"]["end_equity"]},
        "B_over_A_end_factor":expected_ratio,
        "B_over_A_end_advantage_pct":100*(expected_ratio-1),
        "A_B_windows_divergent":int(pairs.different_extra_exit_timing.sum()),
        "A_B_windows_B_better":int(pairs.B_relative_win.sum()),
        "A_B_windows_A_better":int(pairs.A_relative_win.sum()),
        "A_B_windows_exact_same":int((~pairs.A_relative_win & ~pairs.B_relative_win).sum()),
        "A_only_sell_signals_not_identical_B":len(unmatched),
        "A_only_5d_rebound_count":int(unmatched.price_rebounded_5d.sum()),
        "A_only_5d_rebound_frequency":(
            float(unmatched.price_rebounded_5d.mean()) if len(unmatched)
            else None),
    }
    (out/"summary_and_integrity.json").write_text(json.dumps(
        v,ensure_ascii=False,indent=2,default=str)+"\n")
    manifest={
        "question":"Does high QQQ realized volatility conditionally favor frozen H3 profit guard B over A?",
        "dataset_source":provider,
        "history":[START,END],
        "regime_definition":"20d annualized standard deviation of 20 daily log close returns; lagged 1 day for daily PnL category, as-of signal close for divergence-event category",
        "vol_bins":"predefined LOW:<20%, MID:20%-<30%, HIGH:>=30%. No switching strategy run; no thresholds tuned",
        "relative_growth_math":"sum daily log(1+rB)-log(1+rA) across bins equals terminal log(B_end/A_end); similarly sum per-trade log growth factors",
        "account_era":"exact within-period compounded returns carried from first to last session of each era, not fresh $100k resets",
        "event_comparison":"first A extra GUARD_SELL signal different from B's; B event is necessarily a subset of A's detection predicate but executed events may differ across same H3 window",
        "ex_post_labels":"A-only sell-signal future 3/5/10 day prices for interpretation only, not entry/exit decisions",
        "controls":"frozen H3 shadow replay exactly daily, frozen A/B age0 cap1 rules; 28/18 A and 22/12 B sell/rebuy counts",
        "no_raw_ohlcv_redistribution":True,
        "no_trading_authorization":True,
        "not_causal":"volatility bins are correlated with markets and H3 holdings. 60 trades/limited divergent events, so this is exploratory sample-in association, not causal or OOS evidence",
        "frozen_AB_modified":False
    }
    (out/"manifest.json").write_text(json.dumps(manifest,
          ensure_ascii=False,indent=2,default=str)+"\n")
    print("ABVOL_STATS",json.dumps(v,default=str))
    print("ABVOL_ERAS",periods.to_json(orient="records"))
    print("ABVOL_ERA_VOL",era_vol.to_json(orient="records"))
    print("ABVOL_TOTAL_VOL",overall_vol.to_json(orient="records"))
    print("ABVOL_PAIRED_EVENT",pairbins.to_json(orient="records"))
    print("ABVOL_TOP_B",pairs.sort_values("B_minus_A_log_growth",
        ascending=False).head(12)[[
        "h3_id","entry_date","B_minus_A_return_pp",
        "B_minus_A_log_growth","A_first_sell_signal","B_first_sell_signal",
        "rv20_at_divergence","divergence_vol_bin"]].to_json(orient="records"))
    print("ABVOL_TOP_A",pairs.sort_values("B_minus_A_log_growth"
        ).head(12)[[
        "h3_id","entry_date","B_minus_A_return_pp",
        "B_minus_A_log_growth","A_first_sell_signal","B_first_sell_signal",
        "rv20_at_divergence","divergence_vol_bin"]].to_json(orient="records"))
    print("ABVOL_FINISHED",len(pairs))
if __name__=="__main__":main()
