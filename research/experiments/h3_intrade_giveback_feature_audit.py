"""H3 intra-position giveback: trade-wide quantitative audit and risk feature screen.

This is a DIAGNOSTIC, not a model or new trading strategy. Frozen H3 entry,
monthly filters, stops and exits are unchanged. We analyze all completed H3
trades on QQQ and GOOGL, 2000/2004 through 2026-11-11.

'Giveback >=50%' uses the peak CLOSE mark-to-market gain since entry,
with a 10% peak-gain floor. Interim first-cross and giveback-at-exit are
different labels. All feature observations at the 20% early-warning mark
use data available on that bar; FUTURE outcome labels never enter signals.
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
from h3_volume_peak_exhaustion import load_qqq, prep as prep_raw
from h3_googl_cross_asset import load_prices as load_googl
from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_h3_trend_spread_quick_reentry import replay
from qqq_a2.engine import Config

MIN_PEAK=.10
PROFIT_GIVEBACK_WARN=.20
PROFIT_GIVEBACK_EVENT=.50
FEATURES={
    "price_below_ema10":"Close < EMA10",
    "price_below_ema20":"Close < EMA20",
    "ema5_below_ema10":"EMA5 < EMA10",
    "rsi14_below_50":"RSI14 < 50",
    "macd_hist_negative":"MACD histogram < 0",
    "recent_high_dry_volume":"Past 10 days: volume dry while close near 20-day high",
    "volume5_under_80pct_volume20":"Previous-5d/previous-20d average volume <= 0.80",
    "rv10_over_rv60_1p3":"10d/60d realized volatility >= 1.3",
    "atr14_over_63d_med_1p3":"ATR14% >= 1.3x previous 63-day median",
    "heavy_down_day_last5":"Last 5 days include down >=1.5% on >=1.2x volume",
}
def enrich(data,signals):
    d=data.copy()
    d["h3_ema20"]=signals["ma20"].to_numpy()
    d["h3_ema60"]=signals["ma60"].to_numpy()
    d["ema20"]=d.adj_close.ewm(span=20,adjust=False).mean()
    d["ema5_below_ema10"]=d.ema5<d.ema10
    d["price_below_ema10"]=d.adj_close<d.ema10
    d["price_below_ema20"]=d.adj_close<d.ema20
    change=d.adj_close.diff()
    gain=change.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-change.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    d["rsi14"]=100-100/(1+gain/loss.replace(0,np.nan))
    d["rsi14_below_50"]=d.rsi14<50
    macd=d.adj_close.ewm(span=12,adjust=False).mean()-d.adj_close.ewm(span=26,adjust=False).mean()
    d["macd_hist"]=macd-macd.ewm(span=9,adjust=False).mean()
    d["macd_hist_negative"]=d.macd_hist<0
    d["recent_high_dry_volume"]=d.dry_near_high.rolling(10,min_periods=1).max().astype(bool)
    d["volume5_under_80pct_volume20"]=d.dry
    ret=d.adj_close.pct_change()
    d["rv10"]=ret.rolling(10,min_periods=10).std()
    d["rv60"]=ret.rolling(60,min_periods=60).std()
    d["rv_ratio"]=d.rv10/d.rv60
    d["rv10_over_rv60_1p3"]=d.rv_ratio>=1.3
    prev=d.adj_close.shift(1)
    tr=pd.concat([(d.adj_high-d.adj_low).abs(),(d.adj_high-prev).abs(),(d.adj_low-prev).abs()],axis=1).max(axis=1)
    d["atr14_pct"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/d.adj_close
    d["atr_rel"]=d.atr14_pct/d.atr14_pct.shift(1).rolling(63,min_periods=45).median()
    d["atr14_over_63d_med_1p3"]=d.atr_rel>=1.3
    down=(ret<=-.015)&(d.Volume>=d.v20_prev*1.2)
    d["heavy_down_day_last5"]=down.rolling(5,min_periods=1).max().astype(bool)
    return d

def one_symbol(symbol,end):
    raw,prov=load_qqq(end) if symbol=="QQQ" else load_googl(end)
    d=prep_raw(raw)
    signals=with_month_slope_features(completed_month_signals(d))
    frozen=replay(signals,"H3_FROZEN",config=Config(start_date="2000-01-03" if symbol=="QQQ" else "2004-08-19"))
    trades=frozen["trades"]
    assert len(trades)>=50
    if symbol=="GOOGL":
        assert len(trades)==58 and abs(frozen["stats"]["cagr"]-.1371769)<.0002
    d=enrich(d,signals)
    # Every trade is checked, no sampling on eventual performance.
    loc={str(x.date()):i for i,x in enumerate(d.Date)}
    rows=[]; warnings=[]; episodes=[]
    for t in trades.itertuples(index=False):
        first=loc[t.entry_date]
        last=loc[t.exit_signal_date]
        assert first<=last
        w=d.iloc[first:last+1].copy()
        entry=float(t.entry_price)
        w["r"]=w.adj_close/entry-1.
        w["peak_r"]=w.r.cummax()
        w["px_hwm"]=w.adj_close.cummax()
        w["price_dd"]=1.-w.adj_close/w.px_hwm
        w["giveback"]=np.where(w.peak_r>=MIN_PEAK,(w.peak_r-w.r)/w.peak_r,np.nan)
        # Sensitivity bands must require the qualifying peak AS-OF the event.
        # A later 30% peak cannot retroactively turn an earlier 11% peak
        # giveback into a qualifying 20%-peak event.
        w["giveback_peak5"]=np.where(w.peak_r>=.05,(w.peak_r-w.r)/w.peak_r,np.nan)
        w["giveback_peak20"]=np.where(w.peak_r>=.20,(w.peak_r-w.r)/w.peak_r,np.nan)
        meets=w.loc[(w.peak_r>=MIN_PEAK)&(w.giveback>=PROFIT_GIVEBACK_EVENT)]
        warn=w.loc[(w.peak_r>=MIN_PEAK)&(w.giveback>=PROFIT_GIVEBACK_WARN)]
        # Trade-level global descriptors, not a causal prediction.
        peak_idx=int(w.r.idxmax())
        peak_row=d.loc[peak_idx]
        peak_r=float(w.peak_r.max())
        interim=not meets.empty
        first_event=int(meets.index[0]) if interim else None
        first_warning=int(warn.index[0]) if not warn.empty else None
        exit_ret=float(t.ret_net)
        exit_giveback=(peak_r-exit_ret)/peak_r if peak_r>=MIN_PEAK else float("nan")
        # Since entry through actual completed-signal date; can miss gap risk
        # between signal close and next opening exit, separately in exit label.
        maxprice_dd=float(w.price_dd.max())
        event_row=d.loc[first_event] if interim else None
        # The biggest peak-r based drawdown in this trade occurs in the observed range.
        risk_max=w.giveback.dropna().max() if w.giveback.notna().any() else float("nan")
        trend_at_peak=bool(peak_row.h3_ema20>peak_row.h3_ema60)
        trend_at_event=bool(event_row.h3_ema20>event_row.h3_ema60) if interim else None
        # A 50%-giveback event can be a normal mid-cycle pullback that later
        # completely recovers. Do not equate the event with an economic loss.
        pre_event_peak_idx=int(w.loc[:first_event,"r"].idxmax()) if interim else None
        pre_event_peak_r=float(w.loc[first_event,"peak_r"]) if interim else None
        later=w.loc[first_event+1:] if interim else w.iloc[0:0]
        recovered_prior_peak=bool((later.r>=pre_event_peak_r).any()) if len(later) else False
        recovery_first=int(later.loc[later.r>=pre_event_peak_r].index[0]) if recovered_prior_peak else None
        reached_prior_profit80=bool((later.r>=.80*pre_event_peak_r).any()) if len(later) else False
        rows.append({
            "ticker":symbol,"trade_id":int(t.id),"entry_date":t.entry_date,
            "exit_signal":t.exit_signal_date,"exit_date":t.exit_date,
            "entry_kind":t.entry_kind,"exit_reason":t.exit_reason,
            "sessions_in_trade":len(w),
            "peak_date":str(peak_row.Date.date()),"max_close_profit_pct":100*peak_r,
            "exit_net_return_pct":100*exit_ret,
            "exit_profit_giveback_ratio":exit_giveback,
            "exit_giveback_ge50":bool(exit_giveback>=.5) if np.isfinite(exit_giveback) else False,
            "max_intrade_giveback_ratio":float(risk_max),
            "intrade_giveback_ge50":interim,
            "intrade_giveback_ge50_after_peak5":bool((w.giveback_peak5>=.50).any()),
            "intrade_giveback_ge50_after_peak20":bool((w.giveback_peak20>=.50).any()),
            "max_intrade_price_dd_pct":100*maxprice_dd,
            "price_dd_ge8":maxprice_dd>=.08,
            "price_dd_ge12":maxprice_dd>=.12,
            "price_dd_ge15":maxprice_dd>=.15,
            "first_20pct_giveback_date":str(d.loc[first_warning].Date.date()) if first_warning is not None else "",
            "first_50pct_giveback_date":str(event_row.Date.date()) if interim else "",
            "days_from_prior_observed_peak_to_first50":first_event-pre_event_peak_idx if interim else np.nan,
            "days_event_to_h3_exit":last-first_event if interim else np.nan,
            "new_global_peak_after_first50":bool(peak_idx>first_event) if interim else None,
            "recovered_pre_event_peak_before_h3_exit":recovered_prior_peak if interim else None,
            "regained_80pct_pre_event_profit":reached_prior_profit80 if interim else None,
            "sessions_to_recover_pre_event_peak":recovery_first-first_event if recovery_first is not None else np.nan,
            "ema20_gt_ema60_at_peak":trend_at_peak,
            "ema20_gt_ema60_at_first50":trend_at_event,
            "price_below_ema20_at_first50":bool(event_row.price_below_ema20) if interim else None,
            "rsi14_at_peak":float(peak_row.rsi14),
            "volume_ratio_at_peak":float(peak_row.volume_ratio_5_20),
            "atr_rel_at_peak":float(peak_row.atr_rel),
            "rv_ratio_at_peak":float(peak_row.rv_ratio),
        })
        if interim:
            event_d=int(first_event)
            # Use peak since entry PRIOR to the first event; not the global
            # peak which may occur in future. This prevents time-order error.
            preceding_high=int(w.loc[:event_d,"r"].idxmax())
            episodes.append({
                "ticker":symbol,"trade_id":int(t.id),
                "warning_date":str(d.loc[first_warning].Date.date()) if first_warning is not None else "",
                "local_peak_date_prior_to_event":str(d.loc[preceding_high].Date.date()),
                "event_date":str(d.loc[event_d].Date.date()),
                "recovered_pre_event_peak_before_h3_exit":recovered_prior_peak,
                "regained_80pct_pre_event_profit":reached_prior_profit80,
                "days_since_pre_event_peak":event_d-preceding_high,
                "days_until_recover_pre_event_peak":recovery_first-event_d if recovery_first is not None else np.nan,
                "price_dd_pct_at_event":100*float(w.loc[event_d,"price_dd"]),
                "peak_gain_pct_asof_event":100*float(w.loc[event_d,"peak_r"]),
                "profit_giveback_at_event":float(w.loc[event_d,"giveback"]),
                "ema20_gt_ema60_at_event":trend_at_event,
                "price_below_ema20_at_event":bool(event_row.price_below_ema20),
                "macd_hist_negative_at_event":bool(event_row.macd_hist_negative),
                "vol_ratio_at_event":float(event_row.volume_ratio_5_20),
                "rsi14_at_event":float(event_row.rsi14),
            })
        if first_warning is not None:
            wd=d.loc[first_warning]
            min_future=last-first_warning
            # The presence of 50% drawdown at this very first 20% warning means
            # no leading prediction was possible; keep explicit.
            already=bool(w.loc[first_warning,"giveback"]>=.50)
            end_5=min(first_warning+5,last)
            end_10=min(first_warning+10,last)
            end_20=min(first_warning+20,last)
            def label_by(end_idx):
                return bool(((w.index>first_warning)&(w.index<=end_idx)&(w.giveback>=.50)).any())
            base={
                "ticker":symbol,"trade_id":int(t.id),"entry_date":t.entry_date,
                "warning_date":str(wd.Date.date()),
                "warning_profit_pct":100*float(w.loc[first_warning,"r"]),
                "peak_profit_so_far_pct":100*float(w.loc[first_warning,"peak_r"]),
                "giveback_at_warning":float(w.loc[first_warning,"giveback"]),
                "already_50pct_at_warning":already,
                "remaining_sessions_until_h3_sell_signal":min_future,
                "event_before_original_h3_sell":bool(interim and first_event>first_warning),
                "next5_50pct":label_by(end_5),
                "next10_50pct":label_by(end_10),
                "next20_50pct":label_by(end_20),
                "complete_5day_observation":min_future>=5,
                "complete_10day_observation":min_future>=10,
                "complete_20day_observation":min_future>=20,
                "ma20_above_ma60":bool(wd.h3_ema20>wd.h3_ema60),
                "rsi14":float(wd.rsi14),"macd_hist":float(wd.macd_hist),
                "vol_ratio5_20":float(wd.volume_ratio_5_20),
                "atr_rel":float(wd.atr_rel),
                "rv_ratio":float(wd.rv_ratio),
                **{key:bool(wd[key]) for key in FEATURES},
            }
            warnings.append(base)
    return pd.DataFrame(rows),pd.DataFrame(warnings),pd.DataFrame(episodes),{
        "symbol":symbol,"h3_stats":frozen["stats"],"input":prov,
        "input_rows":len(d),"trades":len(trades),
        "warning_definition":"first trade day with asof gain>=10% and profit giveback>=20%",
        "event_definition":"first trade day with asof gain>=10% and profit giveback>=50%",
    }
def cohort_summary(cases,warn):
    out=[]
    for ticker,g in cases.groupby("ticker"):
        n=len(g); qual=g[g.max_close_profit_pct>=MIN_PEAK*100]
        hits=g[g.intrade_giveback_ge50]
        e=g[g.exit_giveback_ge50]
        w=warn[warn.ticker==ticker]
        out.append({"ticker":ticker,"total_completed_h3_trades":n,
            "ever_peak_profit_ge5pct":int((g.max_close_profit_pct>=5).sum()),
            "ever_peak_profit_ge10pct":len(qual),
            "ever_peak_profit_ge20pct":int((g.max_close_profit_pct>=20).sum()),
            "intrade_giveback_50_after_peak5":int(g.intrade_giveback_ge50_after_peak5.sum()),
            "intrade_profit_giveback_ge50pct":len(hits),
            "intrade_giveback_50_after_peak20":int(g.intrade_giveback_ge50_after_peak20.sum()),
            "intrade_giveback_rate_among_peak_ge10":len(hits)/len(qual) if len(qual) else None,
            "giveback_ge50_at_exit":len(e),
            "max_price_dd_ge8pct":int(g.price_dd_ge8.sum()),
            "max_price_dd_ge12pct":int(g.price_dd_ge12.sum()),
            "max_price_dd_ge15pct":int(g.price_dd_ge15.sum()),
            "event_while_ema20_gt60":int(hits.ema20_gt_ema60_at_first50.sum()),
            "event_while_price_below_ema20":int(hits.price_below_ema20_at_first50.sum()),
            "event_later_recovers_prior_peak":int(hits.recovered_pre_event_peak_before_h3_exit.sum()),
            "event_later_regains80pct_prior_profit":int(hits.regained_80pct_pre_event_profit.sum()),
            "event_then_global_peak_later":int(hits.new_global_peak_after_first50.sum()),
            "event_days_until_h3_exit_median":float(hits.days_event_to_h3_exit.median()) if len(hits) else None,
            "event_price_dd_pct_median":float(hits.max_intrade_price_dd_pct.median()) if len(hits) else None,
            "first20_warning_trades":len(w),
            "first20_warning_already50":int(w.already_50pct_at_warning.sum()) if len(w) else 0,
            "warn_to_50_before_h3_exit_count":int(w.event_before_original_h3_sell.sum()) if len(w) else 0,
            "median_peak_profit_pct_big_giveback":float(hits.max_close_profit_pct.median()) if len(hits) else None,
            "median_peak_profit_pct_other":float(g.loc[~g.intrade_giveback_ge50,"max_close_profit_pct"].median()) if len(g)-len(hits) else None,
        })
    return pd.DataFrame(out)
def indicator_screen(warn):
    result=[]
    # Main reliable fixed-horizon comparison: the first early-warning mark
    # with >=20 sessions before original H3 exit, and not already 50%.
    for ticker in ["QQQ","GOOGL"]:
        rows=warn[(warn.ticker==ticker)&(warn.complete_20day_observation)&(~warn.already_50pct_at_warning)].copy()
        for feature,desc in FEATURES.items():
            b=rows[feature]
            yes=rows[b];no=rows[~b]
            result.append({
                "ticker":ticker,"feature":feature,"desc":desc,
                "eligible_warning_trades":len(rows),
                "positive_next20_total":int(rows.next20_50pct.sum()),
                "feature_true_n":len(yes),
                "feature_true_future50":int(yes.next20_50pct.sum()),
                "feature_false_n":len(no),
                "feature_false_future50":int(no.next20_50pct.sum()),
                "p_next20_if_true":float(yes.next20_50pct.mean()) if len(yes) else None,
                "p_next20_if_false":float(no.next20_50pct.mean()) if len(no) else None,
                "risk_difference":float(yes.next20_50pct.mean()-no.next20_50pct.mean()) if len(yes) and len(no) else None,
            })
    return pd.DataFrame(result)
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--end",default="2026-11-11")
    a.add_argument("--outdir",default="research/results/h3_profit_giveback_audit_run")
    args=a.parse_args()
    pieces=[one_symbol(symbol,args.end) for symbol in ["QQQ","GOOGL"]]
    cases=pd.concat([x[0] for x in pieces],ignore_index=True)
    warnings=pd.concat([x[1] for x in pieces],ignore_index=True)
    events=pd.concat([x[2] for x in pieces],ignore_index=True)
    cohort=cohort_summary(cases,warnings)
    screen=indicator_screen(warnings)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    for name,df in [("trade_cohorts.csv",cases),("first20_warning_features.csv",warnings),
        ("first50_event_details.csv",events),("cohort_summary.csv",cohort),
        ("indicator_screen_next20.csv",screen)]:
        df.to_csv(out/name,index=False)
    prov={
        "question":"profit giveback>=50% intra-H3; severity, trend state, candidate advance-warning feature associations",
        "sources":[x[3] for x in pieces],
        "completed_trade_count":len(cases),
        "no_change_to_h3":True,
        "intratrade_event_label":"asof peak CLOSE gross return >=10% and (peak_close_return - current_close_return) / peak_close_return >=.5",
        "sensitivity_peak_floors":"5%, 10%, 20% each evaluated using as-of peak gain AT THE EVENT; no retroactive later larger peak qualification",
        "exit_label":"total peak CLOSE return vs actual H3 next-open realized net return, may include overnight exit gap and fees",
        "warning_label":"first bar of peak gain>=10% and asof giveback>=20%",
        "indicator_comparison":"trades with at least 20 remaining sessions, feature at first 20% warning; outcome first 50% crossing within 20 subsequent sessions",
        "limitations":["one first warning per H3 trade, no resampling of later pullback episodes",
          "feature differences exploratory not independent statistical proof; sample small",
          "20% warning is itself an asof drawdown, not top-before-correction forecast",
          "features may correlate and reflect price already falling",
          "QQQ raw provider-adjusted volume historical split handling may differ",
          "profit ratio unstable at small peak gains; stratify >=10%, >=20% as needed",
          "only actual H3 held days; does not test altered entry exit or CAGR",
          "no price OHLCV exports, only derived signals"],
    }
    (out/"manifest.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2,default=str)+"\n")
    print("H3_GIVEBACK_COHORT",cohort.to_json(orient="records"))
    print("H3_GIVEBACK_INDICATORS",screen.to_json(orient="records"))
    print("H3_GIVEBACK_EVENTS",events.to_json(orient="records"))
    print("H3_GIVEBACK_WARNINGS",warnings.to_json(orient="records"))
    print("H3_GIVEBACK_FINISHED",str(out))
if __name__=="__main__":main()
