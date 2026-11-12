"""H3 frozen trade-level volume-exhaustion attribution (QQQ and GOOGL).

Question: can falling volume near a high anticipate a large peak-to-exit
profit giveback, and can a two-stage volume+price trigger preserve profit?

Keep the H3 state machine frozen. Study candidate signals WITHIN actual
H3-held periods; candidate early sells go to cash ONLY until the original
H3 sell date (no optimistic perfect buyback; no portfolio CAGR claim).
All candidate signals are decided on completed daily bars and would fill
at the next regular open. Trades selected by realized peak giveback are
labeled only AFTER the experiment, not used in signal definitions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"research"/"experiments"))
from h3_googl_cross_asset import load_prices as load_googl, get_bytes
from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_h3_trend_spread_quick_reentry import replay
from qqq_a2.engine import Config

STRATEGIES=[
    "DRY_NEAR_HIGH_ONLY",
    "EMA5_10_DEATH_PRICE_ONLY",
    "DRY10_THEN_EMA5_10_DEATH",
    "DRY10_THEN_EMA10_BREAK",
    "DRY10_THEN_DISTRIBUTION_DOWN",
]
def load_qqq(end):
    a=int(datetime(1999,3,9,tzinfo=timezone.utc).timestamp())
    b=int(pd.Timestamp(end,tz="UTC").timestamp())+86400
    url="https://query1.finance.yahoo.com/v8/finance/chart/QQQ?"+urlencode({
        "period1":a,"period2":b,"interval":"1d","events":"div,splits",
        "includeAdjustedClose":"true"})
    payload=get_bytes(url)
    item=json.loads(payload)["chart"]["result"][0]
    quote=item["indicators"]["quote"][0]
    adj=item["indicators"]["adjclose"][0]["adjclose"]
    data=pd.DataFrame({
        "Date":[datetime.fromtimestamp(t,timezone.utc).date() for t in item["timestamp"]],
        "Open":quote["open"],"High":quote["high"],"Low":quote["low"],
        "Close":quote["close"],"Adj Close":adj,"Volume":quote["volume"]})
    data.Date=pd.to_datetime(data.Date)
    data=data.loc[data.Date<=pd.Timestamp(end)].sort_values("Date").reset_index(drop=True)
    if len(data)<6500 or data.Date.iloc[0]>pd.Timestamp("1999-03-15"):
        raise AssertionError("QQQ history incomplete")
    if str(data.Date.iloc[-1].date())!=end:raise AssertionError("QQQ final bar missing")
    return data,{"provider":"Yahoo chart API","date_end":end,"rows":len(data),
                 "sha256":hashlib.sha256(payload).hexdigest(),"basis":"dividend-and-split adjusted OHLC"}
def prep(raw):
    d=raw.copy()
    d.Date=pd.to_datetime(d.Date)
    d=d.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for c in ("Open","High","Low","Close","Adj Close","Volume"):
        if d[c].isna().any() or (d[c]<=0).any():
            raise AssertionError("Invalid OHLCV "+c)
    f=d["Adj Close"].astype(float)/d["Close"].astype(float)
    for col in ("Open","High","Low","Close"):
        d["adj_"+col.lower()]=d[col].astype(float)*f
    d["v20_prev"]=d.Volume.shift(1).rolling(20,min_periods=20).mean()
    d["v5_prev"]=d.Volume.shift(1).rolling(5,min_periods=5).mean()
    d["volume_ratio_5_20"]=d.v5_prev/d.v20_prev
    d["dry"]=d.volume_ratio_5_20 <= .80
    d["close20high"]=d.adj_close.rolling(20,min_periods=20).max()
    d["near_20high"]=d.adj_close>=d.close20high*.97
    d["dry_near_high"]=d.dry & d.near_20high
    d["dry_peak_in_last10"]=d.dry_near_high.rolling(10,min_periods=1).max().astype(bool)
    d["ema5"]=d.adj_close.ewm(span=5,adjust=False).mean()
    d["ema10"]=d.adj_close.ewm(span=10,adjust=False).mean()
    d["ema5_10_death"]=(d.ema5<d.ema10)&(d.ema5.shift(1)>=d.ema10.shift(1))
    d["ema10_break"]=(d.adj_close<d.ema10)&(d.adj_close.shift(1)>=d.ema10.shift(1))
    d["vol_spike_down"]=(d.adj_close<d.adj_close.shift(1))&(d.Volume>=d.v20_prev*1.2)
    return d
def trades_for(d,trades,symbol):
    trading_days={str(x.date()):int(i) for i,x in enumerate(d.Date)}
    signals=[]
    cases=[]
    all_daily=[]
    for t in trades.itertuples(index=False):
        enter=trading_days[t.entry_date]
        original_exit=trading_days[t.exit_date]
        signal_exit=trading_days[t.exit_signal_date]
        if original_exit<=enter:
            continue
        sub=d.iloc[enter:signal_exit+1].copy()
        sub["peak_so_far"]=sub.adj_close.cummax()
        sub["peak_profit"]=sub.peak_so_far/float(t.entry_price)-1
        sub["current_profit"]=sub.adj_close/float(t.entry_price)-1
        peak_idx=int(sub.adj_close.idxmax())
        peak=d.iloc[peak_idx]
        peak_ratio=float(peak.volume_ratio_5_20)
        giveback_pp=float(t.mfe_close-t.ret_net)*100
        target=bool(t.mfe_close>=.20 and giveback_pp>=15)
        cases.append({"symbol":symbol,"trade_id":int(t.id),"entry":t.entry_date,"exit":t.exit_date,
                      "peak_date":str(peak.Date.date()),"max_profit_pct":100*float(t.mfe_close),
                      "exit_return_pct":100*float(t.ret_net),"giveback_pp":giveback_pp,
                      "big_giveback":target,
                      "peak_volume_ratio_5_20":peak_ratio,
                      "volume_was_dry_on_peak":bool(peak.dry),
                      "volume_dry_near_high_10d_before_peak":bool(d.iloc[max(enter,peak_idx-9):peak_idx+1].dry_near_high.any())})
        # Candidate trades only become eligible after 20% observed floating profit,
        # and current profit >10% (protection, not forced loss stop).
        eligible=(sub.peak_profit>=.20)&(sub.current_profit>=.10)
        candidates={
            "DRY_NEAR_HIGH_ONLY":eligible & sub.dry_near_high,
            "EMA5_10_DEATH_PRICE_ONLY":eligible & sub.ema5_10_death,
            "DRY10_THEN_EMA5_10_DEATH":eligible & sub.dry_peak_in_last10 & sub.ema5_10_death,
            "DRY10_THEN_EMA10_BREAK":eligible & sub.dry_peak_in_last10 & sub.ema10_break,
            "DRY10_THEN_DISTRIBUTION_DOWN":eligible & sub.dry_peak_in_last10 & sub.vol_spike_down & (sub.adj_close<sub.ema10),
        }
        for i,row in sub.iterrows():
            if bool(eligible.loc[i]):
                all_daily.append({"symbol":symbol,"trade_id":int(t.id),
                    "date":str(row.Date.date()),"big_giveback":target,
                    "before_peak":i<=peak_idx,
                    "dry":bool(row.dry),"dry_near_high":bool(row.dry_near_high),
                    "vol_ratio":float(row.volume_ratio_5_20)})
        for strategy, bools in candidates.items():
            indices=bools[bools].index.tolist()
            # Last signal after original H3 exit signal adds no exit benefit;
            # exit would coincide with normal original execution.
            indices=[idx for idx in indices if idx < signal_exit]
            if not indices: continue
            trigger=int(indices[0])
            fill=trigger+1
            if fill>original_exit:continue
            early_fill=float(d.iloc[fill].adj_open)
            baseline_fill=float(d.iloc[original_exit].adj_open)
            signal=dict(symbol=symbol,trade_id=int(t.id),strategy=strategy,
                        trigger_date=str(d.iloc[trigger].Date.date()),
                        early_sell_date=str(d.iloc[fill].Date.date()),
                        baseline_exit=t.exit_date,
                        peak_date=str(peak.Date.date()),
                        days_earlier=original_exit-fill,
                        pre_peak=trigger<=peak_idx,
                        big_giveback=target,
                        actual_peak_giveback_pp=giveback_pp,
                        early_exit_net_gain_pct=100*(early_fill/float(t.entry_price)*(1-.0005)/(1+.0005)-1),
                        original_exit_net_gain_pct=100*float(t.ret_net),
                        difference_vs_baseline_exit_pct=100*(early_fill/baseline_fill-1),
                        saved_profit_pp_of_entry_price=100*(early_fill-baseline_fill)/float(t.entry_price),
                        foregone_profit_if_negative=bool(early_fill<baseline_fill),
                        volume_ratio=float(d.iloc[trigger].volume_ratio_5_20),
                        peak_profit_at_signal_pct=100*float(sub.loc[trigger,"peak_profit"]))
            signals.append(signal)
    return pd.DataFrame(cases),pd.DataFrame(signals),pd.DataFrame(all_daily)
def aggregate(cases,signals,daily,symbol):
    n_big=int(cases.big_giveback.sum())
    n_small=int((~cases.big_giveback).sum())
    result=[]
    for strategy in STRATEGIES:
        u=signals.loc[signals.strategy==strategy]
        big=u.loc[u.big_giveback] if len(u) else u
        nonbig=u.loc[~u.big_giveback] if len(u) else u
        result.append(dict(symbol=symbol,strategy=strategy,
            baseline_trades=len(cases),big_giveback_trades=n_big,other_trades=n_small,
            triggered=int(len(u)),triggered_big=int(len(big)),
            triggered_other=int(len(nonbig)),
            big_recall=float(len(big)/n_big) if n_big else None,
            big_precision=float(len(big)/len(u)) if len(u) else None,
            successful_early_sells=int((u.difference_vs_baseline_exit_pct>0).sum()) if len(u) else 0,
            early_sell_win_rate=float((u.difference_vs_baseline_exit_pct>0).mean()) if len(u) else None,
            mean_saved_vs_baseline_pct=float(u.difference_vs_baseline_exit_pct.mean()) if len(u) else None,
            median_saved_vs_baseline_pct=float(u.difference_vs_baseline_exit_pct.median()) if len(u) else None,
            mean_saved_big_pct=float(big.difference_vs_baseline_exit_pct.mean()) if len(big) else None,
            mean_saved_other_pct=float(nonbig.difference_vs_baseline_exit_pct.mean()) if len(nonbig) else None,
            early_sells_before_peak=int(u.pre_peak.sum()) if len(u) else 0,
            mean_days_early=float(u.days_earlier.mean()) if len(u) else None))
    description={
        "symbol":symbol,"all_trades":len(cases),"large_peak_givebacks":n_big,
        "big_peak_volume_dry_count":int(cases.loc[cases.big_giveback,"volume_was_dry_on_peak"].sum()),
        "big_peak_dry_near_10_count":int(cases.loc[cases.big_giveback,"volume_dry_near_high_10d_before_peak"].sum()),
        "other_peak_volume_dry_count":int(cases.loc[~cases.big_giveback,"volume_was_dry_on_peak"].sum()),
        "other_peak_dry_near_10_count":int(cases.loc[~cases.big_giveback,"volume_dry_near_high_10d_before_peak"].sum()),
    }
    return pd.DataFrame(result),description
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--end",default="2026-11-11")
    ap.add_argument("--outdir",default="research/results/h3_volume_exhaustion_local")
    args=ap.parse_args()
    output=Path(args.outdir);output.mkdir(parents=True,exist_ok=True)
    stats=[];samples=[];cases_all=[];signals_all=[];provenance={}
    for symbol in ["QQQ","GOOGL"]:
        if symbol=="QQQ":
            raw,prov=load_qqq(args.end);start="2000-01-03"
        else:
            raw,prov=load_googl(args.end);start="2004-08-19"
        d=prep(raw)
        config=Config(start_date=start)
        signals=with_month_slope_features(completed_month_signals(d))
        frozen=replay(signals,"H3_FROZEN",config=config)
        if symbol=="GOOGL":
            assert len(frozen["trades"])==58
            assert abs(frozen["stats"]["cagr"]-.1371769)<.0002
        t=frozen["trades"]
        c,s,e=trades_for(d,t,symbol)
        agg,desc=aggregate(c,s,e,symbol)
        stats.append(agg);samples.append(desc)
        cases_all.append(c);signals_all.append(s)
        provenance[symbol]={"provider":prov,"start":start,
           "frozen_h3_stats":frozen["stats"],
           "trade_count":len(t)}
        print("VOLUME_H3_SYMBOL",symbol,json.dumps(desc,default=str))
        print("VOLUME_H3_STRATEGIES",symbol,agg.to_json(orient="records"))
        print("VOLUME_H3_BIG_TRADES",symbol,c.loc[c.big_giveback].to_json(orient="records"))
    summary=pd.concat(stats,ignore_index=True)
    cases=pd.concat(cases_all,ignore_index=True)
    evs=pd.concat(signals_all,ignore_index=True)
    summary.to_csv(output/"signal_summary.csv",index=False)
    cases.to_csv(output/"trade_peak_volume.csv",index=False)
    evs.to_csv(output/"early_exit_candidates.csv",index=False)
    detail={
        "run":"H3 volume-exhaustion hypothesis, fixed-threshold exploratory diagnostic",
        "sources":provenance,"sample_summary":samples,
        "sample_selection":"All actually completed frozen H3 trades (not only profitable trades)",
        "big_giveback":"mfe_close >= 20% and (mfe_close - net realized exit return)>=15 percentage points; ex post label ONLY",
        "dry_definition":"average past 5 day volume <= 80% of average past 20 day volume; excludes current bar",
        "near_high":"current adjusted close >=97% rolling 20-day max close",
        "candidate_activation":"peak-so-far profit >=20% and current gross profit>=10%",
        "comparison":"next-open early sell vs original frozen H3 next-open exit; assumes no reentry before original exit: signal diagnostic, not executable CAGR",
        "prices":"split-and-dividend adjusted; no intraday data, no slippage, 0.05% sell side fee cancels in price comparison",
        "caution":"Vendor-adjusted Volume may not be strictly stationary around historical QQQ splits; no data-driven optimized thresholds or portfolio CAGR claimed",
        "no_raw_ohlcv_exported":True,
        "no_h3_strategy_modification":True,
    }
    (output/"manifest.json").write_text(json.dumps(detail,ensure_ascii=False,indent=2,default=str)+"\n")
    print("VOLUME_H3_FINISHED",json.dumps({"summary":summary.to_dict(orient="records"),
        "samples":samples},default=str))
if __name__=="__main__":main()
