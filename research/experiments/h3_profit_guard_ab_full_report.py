"""QQQ H3 frozen vs Buy&Hold vs Profit Guard A/B: one auditable account report.

Fixed arms selected PRIOR to this diagnostic report:
A = Peak >= 10%; EMA5 crosses below EMA20; reenter on EMA5 crosses
    above EMA10, EMA20>EMA60; only one extra sell/rebuy per H3 trade.
B = A + EMA10>EMA20 and 3-session drop >=1.5 * ATR20 at SELL signal.

All are H3 shadow overlays, not changes to frozen H3. Full daily cash
ledger, next-open fills, 0.05% per side, no taxes/slippage/cash yield.
The reported backtest is IN SAMPLE; this artifact reports not optimizes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from h3_midcycle_fast_ema_overlay_reentry import (
  load_qqq,prep_raw,completed_month_signals,with_month_slope_features,
  replay,Config,features,enrich,simulate,describe_giveback,
)
from h3_profit_n10_ema_guard_shadow_overlay import simulate as old_simulate

START="2000-01-03"
ARM_NAMES={
  "H3":"H3_FROZEN",
  "A":"OLD_P10_X5_20_R5_10",
  "B":"CROSS5_20_PRE10_FAST_R5_10",
}
CONF={
  "H3":dict(exit="none",re="none",max_cycles=0),
  "A":dict(exit="cross5_20",re="cross5_10",max_cycles=1),
  "B":dict(exit="cross5_20_pre10_fast",re="cross5_10",max_cycles=1),
}
FEE=.0005
def compute_metrics(equity):
    x=np.asarray(equity,dtype=float)
    daily=np.empty(len(x))
    daily[0]=x[0]/100000.-1
    daily[1:]=x[1:]/x[:-1]-1
    vola=np.std(daily,ddof=1)*np.sqrt(252)
    down=daily[daily<0]
    downside=np.sqrt(np.mean(np.square(np.minimum(daily,0))))*np.sqrt(252)
    end=x[-1]
    years=(pd.Timestamp("2026-11-11")-pd.Timestamp(START)).days/365.25
    cagr=(end/100000)**(1/years)-1
    running=np.maximum.accumulate(x)
    dd=x/running-1
    trough=int(np.argmin(dd))
    peak=int(np.argmax(x[:trough+1])) if trough else 0
    underwater=x<running*(1-1e-12)
    spans=[];run=0
    for y in underwater:
        if y:run+=1
        elif run:
            spans.append(run);run=0
    if run:spans.append(run)
    return {"end_equity":float(end),"cagr":float(cagr),"mdd":float(min(dd)),
        "annualized_realized_vol":float(vola),
        "sharpe_zero_cash":float(np.mean(daily)/np.std(daily,ddof=1)*np.sqrt(252)) if vola else None,
        "sortino_zero_cash":float(np.mean(daily)*252/downside) if downside else None,
        "calmar":float(cagr/abs(min(dd))) if min(dd)<0 else None,
        "longest_underwater_trading_days":max(spans,default=0),
        "drawdown_trough_array_index":trough,
        "drawdown_prior_peak_array_index":peak}
def returns_by_period(curve):
    d=curve[["Date","Equity"]].copy()
    d["Date"]=pd.to_datetime(d.Date)
    d["r"]=d.Equity.pct_change()
    d.loc[d.index[0],"r"]=d.Equity.iloc[0]/100000-1
    d["year"]=d.Date.dt.year
    d["decade"]=np.select([d.year<2010,d.year<2020],["2000-2009","2010-2019"],default="2020-2026")
    ann=d.groupby("year").r.apply(lambda s:float(np.prod(1+s)-1))
    decades=d.groupby("decade",sort=False).r.apply(lambda s:float(np.prod(1+s)-1))
    return ann,decades
def action_analysis(actions,trade_impact,dates,ticker,arm):
    if len(actions)==0:return pd.DataFrame()
    result=[]
    ix={v:i for i,v in enumerate(dates)}
    a=actions.loc[actions.type=="GUARD_SELL"].copy()
    buys=actions.loc[actions.type=="GUARD_REBUY"].copy()
    for sell in a.itertuples(index=False):
        match=buys.loc[(buys.h3_id==sell.h3_id)&(buys.cycle==sell.cycle)]
        if len(match)>1:raise AssertionError("multi matched buyback")
        b=match.iloc[0] if len(match) else None
        end=str(b.fill_date) if b is not None else str(sell.base_h3_exit)
        if end not in ix:raise AssertionError("No end fill "+end)
        row=trade_impact.loc[trade_impact.h3_id==sell.h3_id].iloc[0]
        result.append({
            "ticker":ticker,"arm":arm,"h3_trade_id":int(sell.h3_id),
            "h3_original_entry":str(sell.original_entry),
            "guard_signal_date":str(sell.signal_date),
            "guard_sell_date":str(sell.fill_date),
            "guard_sell_adjusted_price":float(sell.adjusted_fill_price),
            "rebuy_date":str(b.fill_date) if b is not None else None,
            "rebuy_adjusted_price":float(b.adjusted_fill_price) if b is not None else None,
            "base_exit_date":str(sell.base_h3_exit),
            "flat_trading_sessions":int(ix[end]-ix[str(sell.fill_date)]),
            "rebuy_vs_sell_pct":float(100*(b.adjusted_fill_price/sell.adjusted_fill_price-1)) if b is not None else None,
            "realized_trade_delta_pp":float(row.delta_return_pp),
            "base_peak_gain_pct":float(row.base_peak_gain_pct),
            "base_ever50_giveback":bool(row.base_ever50_giveback),
            "peak_gain_at_sell_pct":float(sell.peak_profit_at_signal_pct),
            "current_gain_at_sell_pct":float(sell.current_profit_at_signal_pct),
            "profit_giveback_ratio_at_sell":float(sell.giveback_at_signal),
            "fast_3day_decline_ATR20":float(sell.decline3_atr20),
            "ema20_minus_ema60_ATR20":float(sell.slow_distance_atr20),
            "ema10_above20_at_sell":bool(sell.ema10_gt20),
        })
    return pd.DataFrame(result)
def concentration(trades):
    changed=trades.loc[trades.overlay_extra_exits>0].copy()
    changed["log_relative_factor"]=np.log(changed.relative_growth)
    all_logs=float(changed.log_relative_factor.sum())
    win_logs=changed.loc[changed.log_relative_factor>0].sort_values(
        "log_relative_factor",ascending=False)
    result={
        "all_relative_wealth_factor":float(np.exp(all_logs)),
        "best_single_relative_factor":float(np.exp(float(win_logs.iloc[0].log_relative_factor))) if len(win_logs) else None,
        "wealth_gain_after_removing_best1":float(np.expm1(all_logs-float(win_logs.log_relative_factor.head(1).sum()))),
        "wealth_gain_after_removing_best3":float(np.expm1(all_logs-float(win_logs.log_relative_factor.head(3).sum()))),
        "top1_trade":str(win_logs.iloc[0].entry_date) if len(win_logs) else None,
        "top3_trade_dates":win_logs.entry_date.head(3).tolist(),
        "improved":int((changed.delta_return_pp>1e-6).sum()),
        "harmed":int((changed.delta_return_pp<-1e-6).sum()),
        "mean_gain_delta_pp_affected":float(changed.delta_return_pp.mean()),
        "median_delta_pp_affected":float(changed.delta_return_pp.median()),
    }
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--end",default="2026-11-11")
    p.add_argument("--outdir",default="research/results/h3_profit_guard_ab_report")
    args=p.parse_args()
    if args.end!="2026-11-11":raise ValueError("Fixed report cutoff 2026-11-11")
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,prov=load_qqq(args.end)
    q=prep_raw(raw)
    signals=with_month_slope_features(completed_month_signals(q))
    frozen=replay(signals,"H3_FROZEN",config=Config(start_date=START))
    if len(frozen["trades"])!=60:raise AssertionError("QQQ H3 trade count changed")
    x=enrich(features(q,signals))
    labels=describe_giveback(x,frozen["trades"])
    days=[str(z.date()) for z in x.loc[x.Date>=pd.Timestamp(START)].Date]
    ref=frozen["equity"].copy()
    b=x.loc[x.Date>=pd.Timestamp(START)].copy()
    firstprice=float(b.adj_open.iloc[0])
    bhshares=100000/(firstprice*(1+FEE))
    buyhold=pd.DataFrame({
        "Date":[str(date.date()) for date in b.Date],
        "Equity":b.adj_close.to_numpy()*bhshares,
        "Invested":True
    })
    if buyhold.Date.tolist()!=ref.Date.tolist():raise AssertionError("B&H alignment mismatch")
    allcurves={"Buy_Hold":buyhold}
    alltrades={}
    allevents={}
    statrows=[]
    for label,arm in ARM_NAMES.items():
        c,t,e,s=simulate(x,frozen,"QQQ",arm,CONF[label],.10,labels)
        assert c.Date.tolist()==ref.Date.tolist()
        assert c.Date.tolist()==buyhold.Date.tolist()
        if label=="H3":
            assert np.allclose(c.Equity,ref.Equity,rtol=0,atol=.05)
            for key in ("cagr","mdd","end_equity","exposure"):
                assert abs(float(s[key])-float(frozen["stats"][key]))<(.05 if key=="end_equity" else 1e-7)
        if label=="A":
            oldc,_,_,oldstats=old_simulate(
                x,frozen,"P10_X5_20_R5_10",.10,"5_20","5_10",START,labels)
            assert np.allclose(c.Equity,oldc.Equity,rtol=0,atol=.05)
            assert abs(s["cagr"]-oldstats["cagr"])<1e-7
        c["Drawdown"]=c.Equity/c.Equity.cummax()-1
        allcurves[label]=c
        alltrades[label]=t
        allevents[label]=e
        m=compute_metrics(c.Equity)
        m["series"]=label
        m["exposure"]=float(c.Invested.mean())
        m["completed_original_h3_trades"]=len(t)
        m["original_trade_winrate"]=float((t.overlay_net_return_pct>0).mean())
        gains=t.loc[t.overlay_net_return_pct>0].overlay_net_return_pct
        losses=t.loc[t.overlay_net_return_pct<0].overlay_net_return_pct
        m["mean_winning_trade_pct"]=float(gains.mean()) if len(gains) else None
        m["mean_losing_trade_pct"]=float(losses.mean()) if len(losses) else None
        m["payoff_ratio"]=float(gains.mean()/-losses.mean()) if len(gains) and len(losses) else None
        m["profit_factor"]=float(gains.sum()/-losses.sum()) if len(gains) and len(losses) else None
        m["extra_sells"]=int(s["extra_exits"])
        m["extra_buys"]=int(s["extra_reentries"])
        m["extra_actions"]=int(s["extra_exits"]+s["extra_reentries"])
        m["guard_affected_trades"]=int(s["changed_h3_trades"])
        peak_qualified=t[t.base_peak_gain_pct>=10].copy()
        peak_qualified["giveback_realized"]=(peak_qualified.base_peak_gain_pct-peak_qualified.overlay_net_return_pct)/peak_qualified.base_peak_gain_pct
        m["qualifying_peak10_trades"]=len(peak_qualified)
        m["realized_profit_giveback_ge50_trade_count"]=int((peak_qualified.giveback_realized>=.5).sum())
        m["realized_profit_giveback_median_ratio"]=float(peak_qualified.giveback_realized.median())
        m["realized_profit_giveback_mean_ratio"]=float(peak_qualified.giveback_realized.mean())
        m["gross_giveback_ratio_vs_h3_peak_definition"]="h3 observed high-closing-price profit vs net end-of-trade realized profit; comparability diagnostic only"
        m["mdd_prior_peak_date"]=days[m["drawdown_prior_peak_array_index"]]
        m["mdd_trough_date"]=days[m["drawdown_trough_array_index"]]
        statrows.append(m)
    m=compute_metrics(buyhold.Equity);m["series"]="Buy_Hold";m["exposure"]=1.;m["completed_original_h3_trades"]=0
    m["mdd_prior_peak_date"]=days[m["drawdown_prior_peak_array_index"]]
    m["mdd_trough_date"]=days[m["drawdown_trough_array_index"]]
    statrows.append(m)
    statdf=pd.DataFrame(statrows)
    statdf.to_csv(out/"four_strategy_stats.csv",index=False)
    table=pd.DataFrame({"Date":ref.Date})
    for k,d in allcurves.items():
        table[k+"_equity"]=d.Equity.to_numpy()
        table[k+"_drawdown"]=d.Equity.to_numpy()/np.maximum.accumulate(d.Equity.to_numpy())-1
    table.to_csv(out/"four_strategy_daily_equity.csv",index=False)
    table["Month"]=table.Date.str[:7]
    table.groupby("Month",sort=True).tail(1).drop(columns=["Month"]).to_csv(out/"four_strategy_month_end_equity.csv",index=False)
    annual={};decades={}
    for k,d in allcurves.items():
        a,z=returns_by_period(d)
        annual[k]=a
        decades[k]=z
    pd.DataFrame(annual).rename_axis("Year").to_csv(out/"four_strategy_calendar_year_returns.csv")
    pd.DataFrame(decades).rename_axis("Period").to_csv(out/"four_strategy_decade_returns.csv")
    attribution=pd.concat([v for v in alltrades.values()],ignore_index=True)
    attribution.to_csv(out/"h3_a_b_trade_comparison.csv",index=False)
    events=pd.concat([v for v in allevents.values() if len(v)],ignore_index=True)
    events.to_csv(out/"a_b_extra_actions.csv",index=False)
    detailed=pd.concat([action_analysis(allevents[k],alltrades[k],days,"QQQ",k) for k in ("A","B")],ignore_index=True)
    detailed.to_csv(out/"a_b_sell_and_reentry_diagnostics.csv",index=False)
    # Compare same original entry trade under all three replay states.
    pivot=attribution.pivot(index=["h3_id","entry_date","exit_date"],columns="arm",values="overlay_net_return_pct").reset_index()
    pivot["A_minus_H3_pp"]=pivot["OLD_P10_X5_20_R5_10"]-pivot["H3_FROZEN"]
    pivot["B_minus_H3_pp"]=pivot["CROSS5_20_PRE10_FAST_R5_10"]-pivot["H3_FROZEN"]
    pivot["B_minus_A_pp"]=pivot["CROSS5_20_PRE10_FAST_R5_10"]-pivot["OLD_P10_X5_20_R5_10"]
    pivot.to_csv(out/"h3_a_b_trade_difference.csv",index=False)
    concentrations={k:concentration(v) for k,v in alltrades.items() if k in ("A","B")}
    (out/"concentration_and_top_trades.json").write_text(
        json.dumps(concentrations,indent=2,ensure_ascii=False,default=str)+"\n")
    manifest={
      "question":"Fixed A/B QQQ H3 profit guard vs Buy&Hold, all-in account, trading and profit preservation",
      "asof":args.end,"start_date":START,"start_capital":100000,
      "provider":prov,"raw_ohlcv_redistributed":False,
      "baseline":"frozen H3 original dates and book, recreated independently without changing engine",
      "controls":{"shadow_daily_H3_exact_match":True,"A_replay_daily_matches_previous_independent_experiment":True},
      "A":"historical asof peak gross gain >=10%; entry age>=10 sessions; close > original entry; EMA20>EMA60; EMA5 crosses BELOW EMA20 -> next OPEN sell; while shadow H3 still long, EMA5 crosses ABOVE EMA10 and EMA20>EMA60 -> next OPEN rebuy; at most 1 extra sell+1 extra rebuy per H3 position",
      "B":"A with EMA10>EMA20 AND (adj_close[t-3] - adj_close[t])/ATR20[t] >= 1.5 on sell signal date",
      "common_execution":"Only as-of close signal; next regular OPEN execution, no future signal info; initial long entry and terminal original H3 sell unchanged",
      "fees":"0.05% on each buy and sell side; no cash yield, extra slippage, taxes, borrow, or short positions",
      "Buy_Hold":"first available start day adjusted open with 0.05% fee; hold adjusted dividend/reinvestment price index through cutoff close; never sell, no terminal sell fee",
      "giveback_label":"After strategy trade closes, compare net completed net trade gain to ORIGINAL H3 peak gross close gain. Ratios may exceed 100% for losses. This is not a causally predictive event and differs from intra-trade first-50%-giveback.",
      "cautions":["in sample and many preceding strategy parameter candidates, no untouched independent evaluation",
         "B&H vs H3 differences reflect initial dates and strategy exposure; ETF dividend tax omitted",
         "benchmark ends in 2026 and begins at 2000 near dotcom peak",
         "the H3 baseline continues original N20 and monthly signal shadow regardless of an overlay sale",
         "historical provider may revise price/dividend adjustments", 
         "fast filter itself may select winners ex post and not robust OOS"],
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str)+"\n")
    print("AB_FOUR_METRICS",statdf.to_json(orient="records"))
    print("AB_CONCENTRATION",json.dumps(concentrations,default=str))
    print("AB_DECADES",pd.DataFrame(decades).to_json())
    print("AB_SUCCEEDED",len(table))
if __name__=="__main__":main()
