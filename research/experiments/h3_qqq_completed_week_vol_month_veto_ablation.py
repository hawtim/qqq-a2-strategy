"""H3 QQQ: completed-week realized volatility to qualify rejected monthly-gate entries.

A small predeclared comparison of ONLY already-blocked fresh EMA10/20 and
EMA20/60 gold crosses, with rigorous non-leaking previous-completed-week
realized volatility and direction. Full independent H3 state machine replay:
all exits, FAST/N20, fees, all altered subsequent trades are recomputed.

Control MUST match the actual frozen H3 DAILY equity and complete ledger.
No overlay A/B, no raw licensed OHLC redistribution, no live trading.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from h3_profit_n10_ema_guard_shadow_overlay import (
    load_qqq, prep_raw, completed_month_signals,
    with_month_slope_features, replay as h3_replay, Config
)
from h3_weekly_realized_vol_month_veto_engine import (
    replay_gated as experimental_replay
)
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS

START="2000-01-03"
END="2026-11-11"
ARMS=[
 ("H3_FROZEN","none","Exact frozen H3 no weekly override"),
 ("FAST_NO_WEEK_FILTER","fast_unfiltered","FAST 10/20 new gold: unrestricted monthly bypass"),
 ("SLOW_NO_WEEK_FILTER","slow_unfiltered","SLOW 20/60 new gold: unrestricted monthly bypass"),
 ("FAST_WEEK_RV_UP","fast_up","FAST if prior completed week RV > 4 earlier weeks median"),
 ("FAST_WEEK_RV_UP_POS","fast_up_positive","FAST vol up + prior completed week return positive"),
 ("FAST_WEEK_RV_UP_POS_PRICE20","fast_up_positive_price20","FAST vol up + weekly return positive + daily close > EMA20"),
 ("FAST_WEEK_RV_FALL_POS","fast_falling_positive","Negative control FAST vol not rising but weekly return positive"),
 ("FAST_WEEK_RV_UP_POS_GLOBAL","fast_up_positive_global_only","FAST veto global only, preserve original FAST monthly eligibility"),
 ("SLOW_WEEK_RV_UP","slow_up","SLOW 20/60 new gold vol up"),
 ("SLOW_WEEK_RV_UP_POS","slow_up_positive","SLOW 20/60 new gold vol up + positive week return"),
 ("BOTH_WEEK_RV_UP_POS","both_up_positive","FAST or SLOW fresh blocked gold with expanding weekly vol + up week"),
]
ERAS=[
 ("2000-2009","2000-01-03","2009-12-31"),
 ("2010-2014","2010-01-01","2014-12-31"),
 ("2015-2019","2015-01-01","2019-12-31"),
 ("2020-2022","2020-01-01","2022-12-31"),
 ("2023-2026","2023-01-01",END),
]

def weekly_features(signal_df):
    """Only a previous COMPLETE ISO-like Mon-Sun week reaches current week.

    Volatility is sqrt(SUM of squared adjusted-close daily log returns)
    including each Monday previous-Friday gap. Prior comparison uses median
    of FOUR full weeks BEFORE that completed week (no current-week leakage).
    Calendar weeks with holidays retain fewer sessions; no future session
    is used to infer what happened in the present week.
    """
    x=signal_df.sort_values("Date").reset_index(drop=True).copy()
    x["week_id"]=x.Date.dt.to_period("W-SUN")
    logret=np.log(x.adj_close/x.adj_close.shift(1))
    x["sq_logret"]=logret.pow(2)
    wk=x.groupby("week_id",sort=True).agg(
        adj_close=("adj_close","last"),
        last_bar=("Date","last"),
        sq_returns=("sq_logret","sum"),
        n_trading_days=("Date","size"),
    )
    wk["weekly_rv"]=np.sqrt(wk.sq_returns)
    wk["weekly_return"]=wk.adj_close.pct_change()
    wk["median_prior4"]=wk.weekly_rv.shift(1).rolling(
        4,min_periods=4).median()
    wk["last_completed_week_rv"]=wk.weekly_rv.shift(1)
    wk["prior4week_median_rv"]=wk.median_prior4.shift(1)
    wk["last_completed_week_return"]=wk.weekly_return.shift(1)
    wk["weekly_rv_valid"]=(
        wk.last_completed_week_rv.notna() &
        wk.prior4week_median_rv.notna() &
        (wk.prior4week_median_rv>0) &
        wk.last_completed_week_return.notna())
    wk["weekly_rv_up"]=(
        wk.weekly_rv_valid &
        (wk.last_completed_week_rv>wk.prior4week_median_rv))
    for col in (
        "last_completed_week_rv","prior4week_median_rv",
        "last_completed_week_return","weekly_rv_up","weekly_rv_valid"):
        x[col]=x.week_id.map(wk[col])
    x["weekly_rv_up"]=x.weekly_rv_up.fillna(False).astype(bool)
    x["weekly_rv_valid"]=x.weekly_rv_valid.fillna(False).astype(bool)
    # Extra leak check: the weekly return / volatility released on day t
    # must be based exclusively on weeks ending BEFORE the day t week.
    previous_week_lastbar=wk.last_bar.shift(1)
    x["last_known_week_end_bar"]=x.week_id.map(previous_week_lastbar)
    assert bool((x.loc[x.weekly_rv_valid,"last_known_week_end_bar"] <
                 x.loc[x.weekly_rv_valid,"Date"]).all())
    wk["rv_rising_at_week_close"]=(
        wk.weekly_rv>wk.median_prior4).where(wk.median_prior4.notna())
    wk["prior_week_weekly_return"]=wk.weekly_return
    return x,wk

def period_stats(result):
    e=result["equity"].copy()
    e.Date=pd.to_datetime(e.Date)
    res=[]
    for label,start,end in ERAS:
        pos=np.flatnonzero((e.Date>=pd.Timestamp(start)) &
                            (e.Date<=pd.Timestamp(end)))
        fst=int(pos[0]);lst=int(pos[-1])
        wealth_before=100000.0 if fst==0 else float(e.Equity.iloc[fst-1])
        er=e.iloc[pos]
        drawdown=er.Equity.to_numpy()/np.maximum.accumulate(
            np.r_[wealth_before,er.Equity.to_numpy()])[1:]-1
        res.append({"period":label,
                    "period_return_pct":float(
                        100*(e.Equity.iloc[lst]/wealth_before-1)),
                    "period_mdd_pct":float(100*drawdown.min()),
                    "cash_days":int((~er.Invested).sum())})
    return res

def baseline_veto_diagnostics(x,frozen):
    """All fresh blocked gold crosses while actual FROZEN H3 holds cash.

    Forward performance is a hindsight-only EVALUATION LABEL. The strategy
    cannot access it; it is never passed to replay entry calculations.
    """
    f=frozen["equity"].copy()
    f.Date=pd.to_datetime(f.Date)
    y=x.merge(f[["Date","Invested"]],on="Date",how="left",
              validate="one_to_one")
    records=[]
    start_i=int(np.flatnonzero(
        (y.Date>=pd.Timestamp(START)).to_numpy())[0])
    for i in range(start_i,len(y)-1):
        r=y.iloc[i]
        if bool(r.Invested):continue
        fast_month_ok=bool(r.prev_completed_month_ema5_up) or (
            float(r.known_slope_abs)<0 and
            float(r.known_prev_abs)<0 and
            float(r.known_slope_abs)<float(r.known_prev_abs))
        blocked_global=not bool(r.gate_up_slow_veto_1_abs)
        for kind,hit,blocked in [
            ("FAST10_20",bool(r.ema10_20_gold),
             blocked_global or not fast_month_ok),
            ("SLOW20_60",bool(r.gold),blocked_global)]:
            if not hit or not blocked:continue
            end_i=min(len(y)-1,i+20)
            buy=float(y.loc[i+1,"adj_open"])
            forward=float(y.loc[end_i,"adj_close"])/buy-1
            future_low=float(y.loc[i+1:end_i,"adj_low"].min())/buy-1
            records.append({
                "signal_date":str(r.Date.date()),"kind":kind,
                "blocked_global":blocked_global,
                "blocked_fast_specific":not fast_month_ok if
                    kind=="FAST10_20" else False,
                "previous_completed_week_end":(
                    str(r.last_known_week_end_bar.date())
                    if pd.notna(r.last_known_week_end_bar) else ""),
                "rv_valid":bool(r.weekly_rv_valid),
                "weekly_rv_rising":bool(r.weekly_rv_up),
                "previous_full_week_return_positive":bool(
                    r.last_completed_week_return>0),
                "rv_ratio":float(r.last_completed_week_rv/
                    r.prior4week_median_rv)
                    if r.weekly_rv_valid else None,
                "previous_full_week_return_pct":float(
                    100*r.last_completed_week_return),
                "signal_close_price_above_ema20":bool(r.adj_close>r.ma20),
                "after_20days_evaluation_close_return_pct":100*forward,
                "after_20days_evaluation_low_drawdown_pct":100*future_low,
                "eval_forward20_complete":bool(i+20<len(y)),
            })
    return pd.DataFrame(records)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--end",default=END)
    p.add_argument("--outdir",default="research/results/h3_qqq_week_vol_month_veto")
    a=p.parse_args()
    if a.end!=END:
        raise ValueError("Frozen dataset cutoff is 2026-11-11")
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,source=load_qqq(END)
    d=prep_raw(raw)
    signals=with_month_slope_features(completed_month_signals(d))
    x,weekly=weekly_features(signals)
    config=Config(start_date=START)
    original=h3_replay(signals,"H3_FROZEN",config=config)
    assert len(original["trades"])==60
    assert len(original["equity"])>=6700
    diagnostics=baseline_veto_diagnostics(x,original)
    diagnostics.to_csv(out/"all_h3_flat_month_blocked_crosses_and_20day_outcomes.csv",index=False)
    # ONLY weekly as-of aggregates, not raw market data.
    summary=[];era=[];trades=[];events=[];daily={}
    for name,gate,description in ARMS:
        run=experimental_replay(
            x,ENTRY_GATE,config=config,**V3_PARAMETERS,
            defer_rejected_fast=True,
            fast_negative_month_mode="accelerating",
            trend_spread_min_gain=.10,
            weekly_vol_gate=gate)
        if run["equity"].Date.tolist()!=original["equity"].Date.tolist():
            raise AssertionError("Trade dates/calendar misaligned")
        if name=="H3_FROZEN":
            assert np.allclose(run["equity"].Equity,
                               original["equity"].Equity,atol=1e-8,rtol=0)
            assert np.array_equal(
                run["equity"].Invested,original["equity"].Invested)
            cols=[c for c in original["trades"].columns
                  if c in run["trades"].columns]
            pd.testing.assert_frame_equal(
                run["trades"][cols].reset_index(drop=True),
                original["trades"][cols].reset_index(drop=True),
                check_dtype=False)
        t=run["trades"].copy()
        if "entry_override" not in t.columns:t["entry_override"]="BASE"
        t["entry_override"]=t.entry_override.fillna("BASE")
        new=t.loc[t.entry_override.str.startswith("VOL_")].copy()
        if len(new):
            assert new.entry_kind.isin([
                "FAST_EMA10_20_MONTH_EMA5",
                "REGULAR_EMA20_60"]).all()
        if len(run["weekly_vol_override_signals"]):
            assert len(run["weekly_vol_override_signals"])>=len(new)
        n20exits=new.exit_reason.str.contains("N20|close stop|entry-day",
                                            case=False,na=False)
        s=run["stats"]
        summary.append({
            "arm":name,"gate":gate,"description":description,
            "cagr_pct":100*float(s["cagr"]),
            "mdd_pct":100*float(s["mdd"]),
            "end_equity":float(s["end_equity"]),
            "exposure_pct":100*float(s["exposure"]),
            "all_completed_trades":int(s["completed_trades"]),
            "additional_exempt_completed":len(new),
            "extra_wins":int((new.ret_net>0).sum()),
            "extra_losses":int((new.ret_net<=0).sum()),
            "extra_N20_or_loss_stop":int(n20exits.sum()),
            "extra_median_ret_pct":float(new.ret_net.median()*100)
                   if len(new) else None,
            "extra_worst_ret_pct":float(new.ret_net.min()*100)
                   if len(new) else None,
            "cagr_diff_h3_pp":100*(float(s["cagr"])-float(
                original["stats"]["cagr"])),
            "mdd_diff_h3_pp":100*(float(s["mdd"])-float(
                original["stats"]["mdd"])),
        })
        for period in period_stats(run):
            era.append({"arm":name,**period})
        t["arm"]=name
        trades.append(t)
        new["arm"]=name
        if len(new):new.to_csv(
            out/f"{name.lower()}_extra_completed_entry_trades.csv",index=False)
        v=run["weekly_vol_override_signals"]
        if len(v):
            v=v.copy();v["arm"]=name;events.append(v)
        run["equity"].to_csv(out/f"{name.lower()}_daily_equity.csv",
                             index=False)
        daily[name]=run["equity"]
        print("WEEK_RV_ARM",json.dumps(summary[-1],default=str))
    if len(events):
        pd.concat(events,ignore_index=True).to_csv(
            out/"all_actual_week_vol_override_signals.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(
        out/"all_completed_trades.csv",index=False)
    ss=pd.DataFrame(summary)
    ss.to_csv(out/"full_account_summary.csv",index=False)
    ee=pd.DataFrame(era)
    ee.to_csv(out/"period_return_comparison.csv",index=False)
    # Baseline blocked signal diagnostic is descriptive, NOT a
    # pre-selection of eventual winning or losing trades.
    if len(diagnostics):
        group=diagnostics.assign(
            candidate_up_pos=(
                diagnostics.weekly_rv_rising &
                diagnostics.previous_full_week_return_positive &
                diagnostics.rv_valid),
            lookahead_positive20=(
                diagnostics.after_20days_evaluation_close_return_pct>0)
        ).groupby(["kind","candidate_up_pos"],dropna=False).agg(
            n=("signal_date","size"),
            positive_20d=("lookahead_positive20","sum"),
            future20_mean_pct=(
                "after_20days_evaluation_close_return_pct","mean"),
            future20_min_low_pct=(
                "after_20days_evaluation_low_drawdown_pct","min"),
        ).reset_index()
        group.to_csv(out/"blocked_signal_week_vol_diagnostic.csv",
                     index=False)
        print("WEEK_RV_DIAGNOSTIC",group.to_json(orient="records"))
        chosen=diagnostics.loc[
            diagnostics.weekly_rv_rising &
            diagnostics.previous_full_week_return_positive &
            diagnostics.rv_valid]
        print("WEEK_RV_CASES_2014_2024",
            chosen.loc[
                chosen.signal_date.str.startswith(("2014","2024"))
            ].to_json(orient="records"))
    print("WEEK_RV_PERIOD",ee.pivot(
        index="period",columns="arm",
        values="period_return_pct").to_json())
    manifest={
        "purpose":"QQQ H3 monthly veto override gated by rising weekly realized volatility and direction. All entrants unknown future, no hindsight selection.",
        "period":[START,END],
        "source":source,
        "n_raw_rows":len(d),
        "n_days":len(original["equity"]),
        "frozen_h3_trade_count":len(original["trades"]),
        "baseline_exact_daily_and_ledger_match":True,
        "weekly_definition":"weekly RV=sqrt(sum(square(1-day adjusted close log returns))) including previous Friday-to-Monday jump, for previous COMPLETED calendar Mon-Sun week; no intraweek volatility ever exposed",
        "weekly_previous_four_median":"median of 4 completed weeks preceding the last completed week; weekly up if last completed RV > this median; no numeric thresholds optimized",
        "weekly_positive_week":"last completed week's close-to-previous completed week's close return>0",
        "week_rv_units":"nonannualized realized log-volatility; ratio level relative to prior 4 weeks controls scale",
        "precedence":"old H3 allowed entries first, extra overrides only for fresh crosses that MONTH eligibility rejected",
        "fast_eligible":"original H3 eligible_fast remains a mandatory state; FAST-specific month gate may be waived except in global-only arm",
        "execution":"completed daily close->next regular open; .0005 per-side fee, no tax, no additional slippage, zero cash yield",
        "risk":"same native H3 exits and N20 protective state recomputed after actual new entry, no A/B overlay",
        "negative_control":"falling weekly RV + positive weekly return on FAST",
        "sample_in":"These are follow-up hypotheses after examining the same QQQ 2000-2026 history. Do not promote in-sample uplift to evidence of statistical edge.",
        "A_B_frozen_unchanged":True,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,
                    ensure_ascii=False,indent=2,default=str)+"\n")
    print("WEEK_RV_FINISHED",len(ARMS))
if __name__=="__main__":
    main()
