"""Versioned, READ-ONLY H3 Profit Guard A/B v1.0 research entrypoint.

This script locks age=0, cap=1 for two SHADOW H3 profit protection
candidates. Runs QQQ H3, A, B and Buy&Hold in one provider snapshot,
asserts precise age=10 historic replay identity and frozen H3 identity,
emits derived account/trade/order evidence. Does NOT place orders or
modify baseline H3.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"research"/"experiments"))

import h3_midcycle_fast_ema_overlay_reentry as engine
from h3_profit_n10_ema_guard_shadow_overlay import buy_hold_comparator

CONFIG_PATH=ROOT/"research"/"config"/"h3_profit_guard_ab_v1.json"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--outdir",default="research/results/h3_profit_guard_ab_v1_frozen")
    args=p.parse_args()
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    data=CONFIG_PATH.read_bytes()
    config=json.loads(data)
    assert config["status"]=="research_candidate_frozen"
    assert config["ticker"]=="QQQ"
    assert config["baseline"]=="H3_FROZEN"
    assert config["shared"]["initial_hold_age_min_sessions"]==0
    assert config["shared"]["max_extra_exits_per_base_position"]==1
    assert config["shared"]["max_extra_reentries_per_base_position"]==1
    assert config["start"]=="2000-01-03"
    assert config["end"]=="2026-11-11"
    v={e["name"]:e for e in config["variants"]}
    assert set(v)=={"A","B"}
    assert v["A"]["exit_rule"]=="cross5_20"
    assert v["B"]["exit_rule"]=="cross5_20_pre10_fast"
    assert all(e["max_cycles"]==1 and e["min_entry_age_sessions"]==0
               and e["peak_min"]==0.10 for e in v.values())
    assert all(e["reentry_rule"]=="cross5_10" for e in v.values())

    raw,provider=engine.load_qqq(config["end"])
    data_bars=engine.prep_raw(raw)
    signals=engine.with_month_slope_features(engine.completed_month_signals(data_bars))
    frozen=engine.replay(signals,"H3_FROZEN",config=engine.Config(start_date=config["start"]))
    assert len(frozen["trades"])==60
    x=engine.enrich(engine.features(data_bars,signals))
    labels=engine.describe_giveback(x,frozen["trades"])
    comparison={}
    executions=[]
    trades=[]
    day_values=[]
    reference_date_list=frozen["equity"].Date.tolist()
    # Explicitly re-run with both age=0 and age=10 for regression equivalence.
    # A different future dataset may legitimately diverge at early signals,
    # in which case freeze cannot claim the old historical results unchanged.
    for label in ("A","B"):
        rule=v[label]
        conf={"exit":rule["exit_rule"],"re":rule["reentry_rule"],"max_cycles":1}
        engine.START_AGE=0
        curve,t,a,stats=engine.simulate(x,frozen,"QQQ",rule["id"],conf,.10,labels)
        assert curve.Date.tolist()==reference_date_list
        assert int(stats["extra_exits"])==int(config["benchmark_sample"][f"{label}_expected_extra_exits"])
        assert int(stats["extra_reentries"])==int(config["benchmark_sample"][f"{label}_expected_extra_reentries"])
        assert abs(float(stats["cagr"])-float(config["benchmark_sample"][f"{label}_expected_cagr_approx"]))<.0005
        assert abs(float(stats["mdd"])-float(config["benchmark_sample"][f"{label}_expected_mdd_approx"]))<.0005
        engine.START_AGE=10
        old,told,aold,sold=engine.simulate(x,frozen,"QQQ",rule["id"]+"_AGE10_CHECK",conf,.10,labels)
        assert old.Date.tolist()==curve.Date.tolist()
        difference=float(np.max(abs(curve.Equity.to_numpy()-old.Equity.to_numpy())))
        if difference>1e-5:raise AssertionError(f"{label} age0 != age10: max daily cash difference {difference:.6f}")
        assert len(a)==len(aold)
        assert len(t)==len(told)
        assert (a[["h3_id","type","signal_date","fill_date","cycle"]].reset_index(drop=True).equals(
            aold[["h3_id","type","signal_date","fill_date","cycle"]].reset_index(drop=True)))
        comparison[label]={**stats,
                           "version_id":rule["id"],
                           "age0_vs_age10_max_daily_equity_difference_usd":difference}
        day_values.append(curve[["Date","Equity","Drawdown"]].rename(
            columns={"Equity":label+"_equity","Drawdown":label+"_drawdown"}).set_index("Date"))
        t=t.copy();t["version_id"]=rule["id"];trades.append(t)
        if len(a):
            a=a.copy();a["version_id"]=rule["id"];executions.append(a)
    engine.START_AGE=10

    base_curve=frozen["equity"]
    assert base_curve.Date.tolist()==reference_date_list
    engine.START_AGE=0
    hc,ht,ha,hs=engine.simulate(
        x,frozen,"QQQ","H3_FROZEN",{"exit":"none","re":"none","max_cycles":0},.10,labels)
    assert np.allclose(hc.Equity,base_curve.Equity,atol=.05,rtol=0)
    assert abs(hs["cagr"]-frozen["stats"]["cagr"])<1e-8
    engine.START_AGE=10
    bh=buy_hold_comparator(x,config["start"])
    ref=x.loc[x.Date>=pd.Timestamp(config["start"])].copy()
    share_count=config["entry_cash_usd"]/(float(ref.adj_open.iloc[0])*(1.+config["fee_per_side"]))
    bh_curve=pd.DataFrame({"Date":[str(dt.date()) for dt in ref.Date],
                           "Equity":ref.adj_close.to_numpy()*share_count})
    assert bh_curve.Date.tolist()==reference_date_list
    base=hc[["Date","Equity","Drawdown"]].rename(
        columns={"Equity":"H3_equity","Drawdown":"H3_drawdown"}).set_index("Date")
    bh_curve["Drawdown"]=bh_curve.Equity/bh_curve.Equity.cummax()-1
    bh_curve=bh_curve.rename(
        columns={"Equity":"BuyHold_equity","Drawdown":"BuyHold_drawdown"}).set_index("Date")
    tab=base.join([bh_curve]+day_values,validate="one_to_one")
    tab.to_csv(out/"four_way_daily_equity.csv")
    summaries=[
      {"strategy":"BuyHold","version_id":"QQQ_BUYHOLD","cagr":bh["cagr"],
       "mdd":bh["mdd"],"end_equity":bh["end_equity"],"extra_exits":0,"extra_reentries":0},
      {"strategy":"H3","version_id":"H3_FROZEN","cagr":hs["cagr"],
       "mdd":hs["mdd"],"end_equity":hs["end_equity"],"extra_exits":0,"extra_reentries":0},
    ]
    for label in ("A","B"):
        v= comparison[label]
        summaries.append({"strategy":label,"version_id":v["version_id"],
          "cagr":v["cagr"],"mdd":v["mdd"],"end_equity":v["end_equity"],
          "extra_exits":v["extra_exits"],"extra_reentries":v["extra_reentries"],
          "age0_vs_age10_max_diff":v["age0_vs_age10_max_daily_equity_difference_usd"]})
    pd.DataFrame(summaries).to_csv(out/"frozen_ab_four_way_summary.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(out/"frozen_ab_trade_impact.csv",index=False)
    pd.concat(executions,ignore_index=True).to_csv(out/"frozen_ab_extra_actions.csv",index=False)
    manifest={
      "version":"H3_PROFIT_GUARD_AB_V1",
      "state":"research candidate frozen, no live orders",
      "config_path":str(CONFIG_PATH.relative_to(ROOT)),
      "config_sha256":hashlib.sha256(data).hexdigest(),
      "asof":config["end"],
      "provider":provider,
      "h3_original_completed_trades":len(frozen["trades"]),
      "frozen_h3_daily_identity":True,
      "age0_equals_age10_for_current_qqq_sample":True,
      "data_raw_ohlcv_not_redistributed":True,
      "results":comparison,
      "h3":hs,
      "buy_hold":bh,
      "sample_in_only":True,
      "note":"Shadow baseline original H3 state is uninterrupted by overlays; special reentry is not a fresh normal H3 entry.",
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+"\n")
    print("FROZEN_AB_V1_FOUR_WAY",pd.DataFrame(summaries).to_json(orient="records"))
    print("FROZEN_AB_V1_SUCCESS",len(tab))

if __name__=="__main__":
    main()
