"""Read-only QQQ H3 post-2010 flat spell and delayed reentry diagnostic.

Reproduce H3_FROZEN from identical daily OHLCV, and audit ALL intervals when
H3 holds cash after 2010-01-01 (not selected missed-rally examples). Trace
fresh EMA10/20 FAST crosses, EMA20/60 REGULAR crosses, monthly veto and
FAST month-slope eligibility. Build as-of features at signal, ex-post
opportunity costs and false-start controls; do not create trade orders or
new strategy returns.

Signals at day's completed close; hypothetical buys at next regular open.
Forward outcomes are EVALUATION labels only, never predicates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from h3_profit_n10_ema_guard_shadow_overlay import (
    load_qqq,prep_raw,completed_month_signals,
    with_month_slope_features,replay,Config,
)
from h3_profit_guard_ab_full_report import compute_metrics

START="2010-01-01"
END="2026-11-11"
SPLITS=[
    ("2010-2014","2010-01-01","2014-12-31"),
    ("2015-2019","2015-01-01","2019-12-31"),
    ("2020-2022","2020-01-01","2022-12-31"),
    ("2023-2026","2023-01-01","2026-11-11"),
]
def enrich(s):
    x=s.copy().reset_index(drop=True)
    x["ema200"]=x.adj_close.ewm(span=200,adjust=False,min_periods=200).mean()
    x["ema200_up20"]=x.ema200>x.ema200.shift(20)
    x["ema20_up5"]=x.ma20>x.ma20.shift(5)
    x["new20high"]=x.adj_close>x.adj_close.shift(1).rolling(20,min_periods=20).max()
    x["new60high"]=x.adj_close>x.adj_close.shift(1).rolling(60,min_periods=60).max()
    x["ret20"]=x.adj_close/x.adj_close.shift(20)-1.
    x["ret60"]=x.adj_close/x.adj_close.shift(60)-1.
    x["ath_dist"]=x.adj_close/x.adj_close.shift(1).cummax()-1.
    x["monthly_gate"]=x.gate_up_slow_veto_1_abs.astype(bool)
    cur=x.known_slope_abs;prev=x.known_prev_abs
    x["fast_month_ok"]=x.prev_completed_month_ema5_up | ((cur<0)&(prev<0)&(cur<prev))
    x["fast_new_cross"]=x.ema10_20_gold.astype(bool)
    x["slow_new_cross"]=x.gold.astype(bool)
    x["trend_bull"]=x.ma20>x.ma60
    x["above_ema20"]=x.adj_close>x.ma20
    x["above_ema200"]=x.adj_close>x.ema200
    x["fresh_price_over_ema20"]=(x.above_ema20)&(~x.above_ema20.shift(1).fillna(False))
    return x

def next_open_outcomes(x,idx,limit=60):
    """Potential signal outcome if hypothetical entry at next OPEN.
    Ex post labels; never used to change H3 timing."""
    if idx+1>=len(x):return {}
    buy=float(x.loc[idx+1,"adj_open"])
    out={"hypothetical_next_open_date":str(x.loc[idx+1,"Date"].date()),
         "hypothetical_next_open":buy}
    for h in (20,60):
        end=min(idx+h,len(x)-1)
        if end<=idx:return out
        z=x.iloc[idx+1:end+1]
        out[f"forward_{h}d_close_return_pct"]=100*(float(x.loc[end,"adj_close"])/buy-1)
        out[f"forward_{h}d_mfe_close_pct"]=100*(float(z.adj_close.max())/buy-1)
        out[f"forward_{h}d_mae_low_pct"]=100*(float(z.adj_low.min())/buy-1)
        out[f"forward_{h}d_complete"]=idx+h<len(x)
    return out

def reason_for_fast(r):
    if not bool(r.monthly_gate):return "global_month_up_slow_veto"
    if not bool(r.fast_month_ok):
        if float(r.known_slope_abs)<0 and float(r.known_prev_abs)<0:
            return "negative_month_slope_recovering_or_flat"
        if float(r.known_slope_abs)<0:
            return "negative_month_slope_new_or_mixed"
        return "fast_month_state_not_eligible"
    return "eligibility_ok_other_state_check"

def grouping(df,span,price,col="Date"):
    s,e=span
    return df[(df[col]>=pd.Timestamp(s))&(df[col]<=pd.Timestamp(e))]

def analyze(x,frozen):
    curve=frozen["equity"].copy()
    curve["Date"]=pd.to_datetime(curve.Date)
    sub=curve.loc[curve.Date>=pd.Timestamp(START)].reset_index(drop=True)
    assert len(sub)>4000
    assert str(sub.Date.iloc[-1].date())==END
    # One-to-one join on real completed session. The H3 actual state is
    # the ONLY definition of a cash interval. Do not infer from trade list.
    m=x.merge(curve[["Date","Invested"]],on="Date",how="left",validate="one_to_one")
    assert not m.loc[m.Date>=pd.Timestamp("2000-01-03"),"Invested"].isna().any()
    m["Invested"]=m.Invested.fillna(False).astype(bool)
    indices=np.flatnonzero((m.Date>=pd.Timestamp(START)).to_numpy())
    first=int(indices[0]);last=int(indices[-1])
    assert last==len(m)-1
    gaps=[];candidates=[]
    # Contiguous cash intervals in the actual H3 daily account.
    i=first
    n=0
    while i<=last:
        if bool(m.loc[i,"Invested"]):
            i+=1;continue
        j=i
        while j+1<=last and not bool(m.loc[j+1,"Invested"]):j+=1
        # Uninvested start day belongs to exit fill OPEN; next invested
        # day belongs to next entry fill OPEN. Adj open->open return is
        # a hindsight "what H3 did not own" interval, NOT a tradable alpha.
        n+=1
        end_i=(j+1 if j+1<=last else j)
        closed=bool(j<last)
        start_open=float(m.loc[i,"adj_open"])
        end_px=float(m.loc[end_i,"adj_open"] if closed else m.loc[end_i,"adj_close"])
        spell=m.iloc[i:j+1]
        first_day=str(m.loc[i,"Date"].date())
        exit_in_trades=frozen["trades"].loc[
            frozen["trades"].exit_date.eq(first_day)
        ]
        start_left_censored=(len(exit_in_trades)==0)
        base={"cash_spell_id":n,"start_date":first_day,
              "end_cash_date":str(m.loc[j,"Date"].date()),
              "next_h3_buy_date":str(m.loc[j+1,"Date"].date()) if closed else "",
              "days_cash":len(spell),"end_right_censored":not closed,
              "start_left_censored":start_left_censored,
              "last_h3_exit_reason":str(exit_in_trades.iloc[0].exit_reason) if len(exit_in_trades) else "started_before_2010",
              "cash_start_open":start_open,
              "end_reference_price":end_px,
              "gap_price_return_pct":100*(end_px/start_open-1),
              "max_price_move_from_exit_pct":100*(float(spell.adj_close.max())/start_open-1),
              "min_price_move_from_exit_pct":100*(float(spell.adj_close.min())/start_open-1),
              "price_at_next_entry_vs_spell_low_pct":100*(end_px/float(spell.adj_close.min())-1),
              "all_cash_20d_return_positive_days":int((spell.ret20>0).sum()),
              "cash_days_close_above_ema200":int(spell.above_ema200.sum()),
              "cash_days_ema20_gt60":int(spell.trend_bull.sum()),
              "cash_days_ma200_up20":int(spell.ema200_up20.sum()),
              "cash_days_new20high":int(spell.new20high.sum()),
              "cash_days_new60high":int(spell.new60high.sum()),
              "cash_days_month_gate_false":int((~spell.monthly_gate).sum()),
              "first_fast_cross":"","first_eligible_fast_cross":"",
              "first_slow_cross":"","fast_cross_n":int(spell.fast_new_cross.sum()),
              "slow_cross_n":int(spell.slow_new_cross.sum()),
              "blocked_fast_cross_n":0,
              "gate_block_cross_n":0,"monthly_fast_block_cross_n":0,
              "fast_cross_forward_20d_return_pct":np.nan,
              "first_fast_cross_to_h3_entry_trading_days":np.nan,
              "first_fast_cross_fill_to_h3_entry_price_pct":np.nan,
              "price_at_cash_start_ath_distance_pct":100*float(m.loc[i,"ath_dist"]) if pd.notna(m.loc[i,"ath_dist"]) else np.nan,
              "price_at_cash_start_ma200_bull":bool(m.loc[i,"above_ema200"] and m.loc[i,"ema200_up20"]),
        }
        first_fast_idx=None
        for k,r in spell.iterrows():
            if bool(r.fast_new_cross):
                if first_fast_idx is None:
                    first_fast_idx=k
                    base["first_fast_cross"]=str(r.Date.date())
                true_gate=bool(r.monthly_gate)
                true_month=bool(r.fast_month_ok)
                if true_gate and true_month and not base["first_eligible_fast_cross"]:
                    base["first_eligible_fast_cross"]=str(r.Date.date())
                if not(true_gate and true_month):
                    base["blocked_fast_cross_n"]+=1
                if not true_gate:base["gate_block_cross_n"]+=1
                if true_gate and not true_month:base["monthly_fast_block_cross_n"]+=1
                c={"cash_spell_id":n,"date":str(r.Date.date()),
                   "date_index":k,"signal_type":"fast_new_ema10_20_gold",
                   "global_month_gate_ok":true_gate,
                   "special_fast_month_ok":true_month,
                   "effective_fast_allowed":true_gate and true_month,
                   "blocked_reason":reason_for_fast(r),
                   "ma20_gt_ma60":bool(r.trend_bull),
                   "close_gt_ema20":bool(r.above_ema20),
                   "close_gt_ema200":bool(r.above_ema200),
                   "ema200_up20":bool(r.ema200_up20),
                   "new20high":bool(r.new20high),
                   "new60high":bool(r.new60high),
                   "ret20_pct":100*float(r.ret20),
                   "ath_distance_pct":100*float(r.ath_dist),
                   "known_slope_abs":float(r.known_slope_abs),
                   "known_prev_abs":float(r.known_prev_abs),
                   "sessions_until_actual_h3_entry":end_i-k if closed else np.nan,
                   **next_open_outcomes(m,k)}
                if k+1<=last and closed:
                    c["eventual_h3_buy_open_over_candidate_open_pct"]=100*(end_px/float(m.loc[k+1,"adj_open"])-1)
                else:c["eventual_h3_buy_open_over_candidate_open_pct"]=np.nan
                candidates.append(c)
            if bool(r.slow_new_cross) and not base["first_slow_cross"]:
                base["first_slow_cross"]=str(r.Date.date())
        if first_fast_idx is not None:
            base["first_fast_cross_to_h3_entry_trading_days"]=end_i-first_fast_idx if closed else np.nan
            if first_fast_idx+1<=last:
                cpx=float(m.loc[first_fast_idx+1,"adj_open"])
                base["first_fast_cross_fill_to_h3_entry_price_pct"]=100*(end_px/cpx-1)
                if first_fast_idx+20<len(m):
                    base["fast_cross_forward_20d_return_pct"]=100*(float(m.loc[first_fast_idx+20,"adj_close"])/cpx-1)
        gaps.append(base)
        i=j+1
    gaps=pd.DataFrame(gaps)
    candidates=pd.DataFrame(candidates)
    assert len(gaps) and (gaps.days_cash>0).all()
    assert int(gaps.days_cash.sum())==int((~sub.Invested).sum()), (gaps.days_cash.sum(),(~sub.Invested).sum())
    # Snapshot period comparison: all H3 signal states inherited, no restart.
    bar=m.loc[m.Date>=pd.Timestamp("2009-12-31")].copy()
    h=curve.loc[curve.Date>=pd.Timestamp("2009-12-31")].copy()
    assert len(bar)==len(h)
    bhshares=100000/(float(m.loc[m.Date>=pd.Timestamp("2000-01-03"),"adj_open"].iloc[0])*1.0005)
    bar["bh_equity"]=bar.adj_close.to_numpy()*bhshares
    bar["h3_equity"]=h.Equity.to_numpy()
    period=[]
    for name,start,end in SPLITS:
        current=bar.loc[(bar.Date>=pd.Timestamp(start))&(bar.Date<=pd.Timestamp(end))]
        firstidx=int(current.index[0]);lastidx=int(current.index[-1])
        before=bar.loc[bar.index<firstidx].iloc[-1]
        lastrow=bar.loc[lastidx]
        period.append({"period":name,"first_session":str(current.Date.iloc[0].date()),
                       "last_session":str(lastrow.Date.date()),"trading_sessions":len(current),
                       "cash_days":int((~current.Invested).sum()),
                       "cash_fraction":float((~current.Invested).mean()),
                       "new20high_while_cash_days":int((~current.Invested & current.new20high).sum()),
                       "new60high_while_cash_days":int((~current.Invested & current.new60high).sum()),
                       "ma20_gt_ma60_while_cash_days":int((~current.Invested & current.trend_bull).sum()),
                       "cash_when_ema200_up_and_close_above_days":int((~current.Invested & current.above_ema200 & current.ema200_up20).sum()),
                       "H3_account_period_return_pct":100*(float(lastrow.h3_equity)/float(before.h3_equity)-1),
                       "BH_account_period_return_pct":100*(float(lastrow.bh_equity)/float(before.bh_equity)-1),
                       "QQQ_price_close_period_return_pct":100*(float(lastrow.adj_close)/float(before.adj_close)-1)})
    period=pd.DataFrame(period)
    # Causal descriptive market state at first FAST gold after H3 exit.
    # Relative ranks conditioned on 20/60d completed forward windows.
    signals=pd.DataFrame()
    if len(candidates):
        signals=candidates.copy()
        def cls(row):
            if not row.forward_60d_complete:return "censored"
            ret=row.forward_60d_close_return_pct
            mae=row.forward_60d_mae_low_pct
            if ret>=10 and mae>=-8:return "strong_rally_limited_drawdown_ex_post"
            if ret>=10:return "strong_rally_with_drawdown_ex_post"
            if ret<=0:return "not_up_in_60d_ex_post"
            return "small_positive_60d_ex_post"
        signals["outcome_60d_ex_post"]=signals.apply(cls,axis=1)
    return gaps,candidates,period,signals,bar

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--outdir",default="research/results/h3_qqq_post2010_flat_entry_audit")
    args=p.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,provider=load_qqq(END)
    d=prep_raw(raw)
    s=with_month_slope_features(completed_month_signals(d))
    frozen=replay(s,"H3_FROZEN",config=Config(start_date="2000-01-03"))
    assert len(frozen["trades"])==60
    x=enrich(s)
    gaps,events,period,outcomes,bar=analyze(x,frozen)
    data={
        "total_gaps_post2010":len(gaps),
        "total_cash_days":int(gaps.days_cash.sum()),
        "sum_cash_gaps_positive":int((gaps.gap_price_return_pct>0).sum()),
        "sum_cash_gaps_gt5pct":int((gaps.gap_price_return_pct>5).sum()),
        "sum_cash_gaps_gt10pct":int((gaps.gap_price_return_pct>10).sum()),
        "sum_cash_gaps_lt_minus5pct":int((gaps.gap_price_return_pct< -5).sum()),
        "cash_gap_price_return_positive_total_pct":float(gaps.loc[gaps.gap_price_return_pct>0,"gap_price_return_pct"].sum()),
        "cash_gap_price_return_negative_total_pct":float(gaps.loc[gaps.gap_price_return_pct<0,"gap_price_return_pct"].sum()),
        "fast_cross_all":len(events),
        "fast_cross_blocked":int((~events.effective_fast_allowed).sum()) if len(events) else 0,
        "fast_cross_month_gate_veto":int((~events.global_month_gate_ok).sum()) if len(events) else 0,
        "fast_cross_special_month_block":int((events.global_month_gate_ok & ~events.special_fast_month_ok).sum()) if len(events) else 0,
        "fast_cross_eligible":int(events.effective_fast_allowed.sum()) if len(events) else 0,
        "full_60d_evaluable_crosses":int(events.forward_60d_complete.sum()) if len(events) else 0,
        "first_fast_in_gap_count":int(gaps.first_fast_cross.ne("").sum()),
        "flat_gaps_starting_at_h3_actual_exit":int((~gaps.start_left_censored).sum()),
        "gaps_end_at_h3_real_buy":int((~gaps.end_right_censored).sum()),
        "years":END[0:4],
        "h3_frozen_stats":frozen["stats"],
        "provider":provider,
    }
    # Period regime features and relative wealth attribution. The product of
    # flat OPEN-to-OPEN price factors is interpretable only when ALL flat
    # episodes lie fully inside the fixed period. Otherwise explicitly mark
    # it as a non-reconcilable partial-period diagnostic.
    regime=[]
    attribution=[]
    for row in period.itertuples(index=False):
        start=int(str(row.first_session)[:4]);end=int(str(row.last_session)[:4])
        subset=gaps[(gaps.start_date>=row.first_session)&(gaps.start_date<=row.last_session)]
        signal_sub=events[(events.date>=row.first_session)&(events.date<=row.last_session)] if len(events) else events
        feature={
            "period":row.period,"cash_days":int(row.cash_days),
            "cash_days_as_pct_of_all_sessions":100*float(row.cash_fraction),
            "cash_days_above_rising_ema200":int(row.cash_when_ema200_up_and_close_above_days),
            "share_cash_above_rising_ema200_pct":100*row.cash_when_ema200_up_and_close_above_days/row.cash_days if row.cash_days else None,
            "cash_days_ema20_gt60":int(row.ma20_gt_ma60_while_cash_days),
            "share_cash_ema20_gt60_pct":100*row.ma20_gt_ma60_while_cash_days/row.cash_days if row.cash_days else None,
            "fast_gold_crosses_cash":len(signal_sub),
            "fast_gold_blocked_cash":int((~signal_sub.effective_fast_allowed).sum()) if len(signal_sub) else 0,
            "fast_gold_global_month_veto_cash":int((~signal_sub.global_month_gate_ok).sum()) if len(signal_sub) else 0,
            "fast_gold_special_month_veto_cash":int((signal_sub.global_month_gate_ok & ~signal_sub.special_fast_month_ok).sum()) if len(signal_sub) else 0,
        }
        regime.append(feature)
        clean=bool((not subset.end_right_censored.any()) and
                   (subset.next_h3_buy_date.dropna().le(row.last_session).all()) and
                   (not subset.start_left_censored.any()) and
                   (sum(subset.days_cash)==row.cash_days))
        product=float(np.prod(1.+subset.gap_price_return_pct/100.))
        actual_ratio=float((1.+row.BH_account_period_return_pct/100.)/
                           (1.+row.H3_account_period_return_pct/100.))
        attribution.append({
            "period":row.period,"clean_open_to_open_cash_gap_partition":clean,
            "flat_interval_count":len(subset),"flat_open_to_open_qqq_growth_factor":product,
            "BH_over_H3_account_growth_factor":actual_ratio,
            "ratio_BH_over_H3_to_flat_price_product":actual_ratio/product,
            "flat_spells_positive_price_count":int((subset.gap_price_return_pct>0).sum()),
            "flat_spells_negative_price_count":int((subset.gap_price_return_pct<0).sum()),
            "flat_spells_gt5pct":int((subset.gap_price_return_pct>5).sum()),
            "interpretation":"Not a tradable return: pure missing exposure proxy" if clean else "Partition crosses reporting boundary; do NOT interpret product as complete period attribution",
        })
    regime=pd.DataFrame(regime)
    attribution=pd.DataFrame(attribution)
    signal_stats=[]
    for kind,group in [
        ("all_fast_gold",events),
        ("blocked_fast_gold",events[~events.effective_fast_allowed]),
        ("allowed_fast_gold",events[events.effective_fast_allowed]),
        ("global_month_veto",events[~events.global_month_gate_ok]),
        ("fast_month_veto_only",events[events.global_month_gate_ok & ~events.special_fast_month_ok]),
    ]:
        valid=group[group.forward_60d_complete]
        signal_stats.append({
            "kind":kind,"count":len(group),"complete60d_count":len(valid),
            "future60_positive_count":int((valid.forward_60d_close_return_pct>0).sum()),
            "future60_ge10pct_count":int((valid.forward_60d_close_return_pct>=10).sum()),
            "future60_low_below_minus8pct_count":int((valid.forward_60d_mae_low_pct<=-8).sum()),
            "future60_return_median_pct":float(valid.forward_60d_close_return_pct.median()) if len(valid) else None,
            "forward_60d_mae_low_median_pct":float(valid.forward_60d_mae_low_pct.median()) if len(valid) else None,
            "bought_on_next_h3_entry_count":int((group.sessions_until_actual_h3_entry==1).sum()),
        })
    signal_stats=pd.DataFrame(signal_stats)
    for file,frame in [
        ("cash_era_regime_feature_summary.csv",regime),
        ("flat_open_to_open_price_vs_H3_BH_gap.csv",attribution),
        ("fast_cross_block_reason_and_forward_risk_summary.csv",signal_stats),
        ("cash_intervals_all_2010_2026.csv",gaps),
        ("cash_intervals_sorted_by_price_missed.csv",gaps.sort_values("gap_price_return_pct",ascending=False)),
        ("fast_crosses_during_cash.csv",events),
        ("forward_signal_outcome_labels.csv",outcomes),
        ("period_2010_2014_2015_2019_2020_2022_2023_2026.csv",period),
    ]:
        frame.to_csv(out/file,index=False)
    manifest={
      "scope":"All H3_FROZEN cash intervals overlapping 2010-01-01 to 2026-11-11, QQQ",
      "baseline_and_price":"H3 frozen replay on same provider-adjusted OHLC; no H3 code changes",
      "market_data_provenance":provider,
      "price_gap_return":"Cash interval first day adjusted OPEN to next H3 buy OPEN (or latest CLOSE for censored gap), adjusted proxy. Not net alpha.",
      "cash_day_count":"Every original H3 daily equity row with Invested=False from 2010-01-01, including sell-fill session",
      "fast_cross_definition":"fresh EMA10/20 gold on completed close while H3 original status flat",
      "global_month_gate":"up_slow_veto_1_abs, calculated from last completed month as-of that day",
      "fast_month_gate":"completed prior month EMA5 positive slope OR 2 negative slopes with second more negative; must also be after H3 sell to have FAST eligibility",
      "caveat":"A freshly eligible FAST cross may still be blocked by shadow deferred state and pending orders. No reconstructed engine order was sent. All true FAST signal gates among uninvested days are counted but not declared orders.",
      "forward_20_60":"hypothetical next-open buy, after-the-fact 20/60 completed session close and low risk; evaluation labels only, not usable to trigger order ex ante",
      "left_censor":"First interval begins when analysis starts 2010 even if original exit was earlier; first partial spell flag.",
      "right_censor":"last interval without later H3 original buy ends at latest daily CLOSE; flag.",
      "cash_compounding":"Gap returns never summed as a portfolio return; conditional total percent is just descriptive. Actual account period return uses initial baseline equity and calendar period endpoints.",
      "lookahead_prevented":"Event signals use as-of completed EMA and month data. 20/60 forward price outcomes and before-entry maximum/minimum are exclusively ex-post labels.",
      "not_attempted":"No new entry, removing monthly filters, optimized long-only account CAGR/MDD; no modification of A/B frozen profit guards.",
      "history":"2010+ diagnosis already selected by user as investigation; cross-era contrast observational, not random experiment.",
    }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n")
    print("FLAT_POST2010_MAIN",json.dumps({k:v for k,v in data.items() if k not in ("h3_frozen_stats","provider")},default=str))
    print("FLAT_POST2010_PERIOD",period.to_json(orient="records"))
    print("FLAT_POST2010_REGIME",regime.to_json(orient="records"))
    print("FLAT_POST2010_RELATIVE_GAP_DECOMP",attribution.to_json(orient="records"))
    print("FLAT_POST2010_SIGNAL_STATS",signal_stats.to_json(orient="records"))
    print("FLAT_POST2010_TOP_GAPS",gaps.sort_values("gap_price_return_pct",ascending=False).head(15).to_json(orient="records"))
    print("FLAT_POST2010_FAST_OUTCOMES",outcomes.groupby(["blocked_reason","outcome_60d_ex_post"]).size().to_json() if len(outcomes) else "{}")
    print("FLAT_POST2010_TOP_FAST",events.sort_values("eventual_h3_buy_open_over_candidate_open_pct",ascending=False).head(15).to_json(orient="records") if len(events) else "[]")
    (out/"metrics.json").write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+"\n")
    print("FLAT_POST2010_FINISHED",str(out))
if __name__=="__main__":
    main()
