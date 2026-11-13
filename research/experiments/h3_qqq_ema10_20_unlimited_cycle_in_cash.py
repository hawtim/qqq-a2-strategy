"""QQQ H3 shadow-flat EMA10/20 cyclic recovery: no monthly gate.

The actual intent: during formerly long H3 CASH spells run repeated fresh
EMA10/20 GOLD -> NEXT OPEN buy; fresh EMA10/20 DEATH -> NEXT OPEN sell,
with UNLIMITED cycles (unlike previous once-only N20 FAST reentry).
Two shadow variants distinguish (a) use crosses ONLY while original H3
shadow is flat vs (b) let EMA10/20 cross trade even when H3 shadow long,
always respecting original H3 entry/exit events as hard boundaries.
A pure EMA10/20 repeated crossover benchmark and a shadow-cash optional
7% close-price loss-stop provide independent negative controls.

Frozen H3 original signals, N20 exits, monthly gate, and frozen A/B
profit guard code are never modified. Original frozen H3 is replayed
before simulating the overlay; cash/shares/fees next-open fills are actual
full-account. Baseline hybrid control MUST match frozen H3 daily equity,
exposure and end-wealth within rounding. No lookahead, no cash interest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from h3_profit_n10_ema_guard_shadow_overlay import (
    load_qqq, prep_raw, completed_month_signals,
    with_month_slope_features, replay, Config
)
from h3_profit_guard_ab_full_report import compute_metrics

START = "2000-01-03"
END = "2026-11-11"
FEE = .0005
CAPITAL = 100000.
# H3_SHADOW_ONLY: baseline. SHADOW_FLAT: EMA cycles only when H3 virtual
# strategy is flat, handoff to original H3 at its next scheduled entry.
# SHADOW_ALL: crossed EMA may exit H3 holdings too, but never overrides
# original H3 forced exits or scheduled starts; this becomes much noisier.
# EMA_ONLY: entirely standalone crossover system independent from H3.
ARMS = [
  ("H3_FROZEN", "shadow_none", None),
  ("H3_FLAT_10_20_CYCLE", "shadow_flat", None),
  ("H3_FLAT_10_20_CYCLE_STOP7", "shadow_flat", .07),
  ("H3_ALL_10_20_CYCLE", "shadow_all", None),
  ("EMA10_20_ALWAYS", "ema_only", None),
]
ERAS = [
  ("2000-2009","2000-01-03","2009-12-31"),
  ("2010-2014","2010-01-01","2014-12-31"),
  ("2015-2019","2015-01-01","2019-12-31"),
  ("2020-2022","2020-01-01","2022-12-31"),
  ("2023-2026","2023-01-01",END),
]

def signals_from_frozen(s):
    x=s.sort_values("Date").reset_index(drop=True).copy()
    x["fresh_fast_gold"]=(x.ma10>x.ma20)&(x.ma10.shift(1)<=x.ma20.shift(1))
    x["fresh_fast_death"]=(x.ma10<x.ma20)&(x.ma10.shift(1)>=x.ma20.shift(1))
    x["month_gate"]=x.gate_up_slow_veto_1_abs.astype(bool)
    x["slow_bull"]=x.ma20>x.ma60
    assert x.Date.is_unique
    return x

def frozen_open_events(baseline):
    e={};v={}
    for r in baseline["trades"].itertuples(index=False):
        if r.entry_date in e or r.exit_date in v:
            raise AssertionError("overlap of H3 trade fills")
        e[r.entry_date]={"trade_id":int(r.id),"kind":r.entry_kind}
        v[r.exit_date]={"trade_id":int(r.id),"reason":r.exit_reason}
    if baseline["open_position"] is not None:
        r=baseline["open_position"]
        e[r["entry_date"]]={"trade_id":len(baseline["trades"])+1,
                            "kind":r["entry_kind"]}
    return e,v

def simulate(x,base,name,mode,stop_pct):
    bentries,bexits=frozen_open_events(base)
    cash=CAPITAL
    shares=0.
    physical_entry_price=None
    physical_entry_date=None
    physical_origin=None
    pending=None
    virtual_h3_long=False
    virtual_h3_id=None
    actions=[]; trades=[]; curve=[]
    fees=0.; n_buy=0;n_sell=0
    def make_action(kind,price,date,signal_date,reason,shadow_long):
        return {
          "arm":name,"action":kind,
          "signal_date":signal_date,"fill_date":date,
          "fill_price":float(price),"reason":reason,
          "shadow_h3_long":bool(shadow_long),
          "virtual_h3_id":virtual_h3_id,
        }

    for i,r in x.iterrows():
        if r.Date < pd.Timestamp(START):continue
        day=str(r.Date.date())
        open_px=float(r.adj_open)
        close_px=float(r.adj_close)
        is_shadow=mode!="ema_only"
        # Executions at today's OPEN; original frozen H3 open events have
        # priority. Any yesterday's extra pending order is cancelled if
        # the original H3 itself trades at the same open.
        if is_shadow and day in bexits:
            if not virtual_h3_long or virtual_h3_id!=bexits[day]["trade_id"]:
                raise AssertionError(f"shadow H3 EXIT mismatch on {day}")
            if shares:
                before=cash+shares*open_px
                proceeds=shares*open_px*(1-FEE)
                fees+=shares*open_px*FEE
                cash+=proceeds
                n_sell+=1
                actions.append(make_action("SELL",open_px,day,
                  "H3_ORIGINAL", "H3_ORIGINAL_EXIT",True))
                trades.append({"arm":name,"entry_date":physical_entry_date,
                  "exit_date":day,"entry_price":physical_entry_price,
                  "exit_price":open_px,"entry_origin":physical_origin,
                  "exit_reason":"H3_ORIGINAL_EXIT",
                  "gross_price_return_pct":100*(open_px/physical_entry_price-1)})
                shares=0.
                physical_entry_date=None
                physical_entry_price=None
                physical_origin=None
            virtual_h3_long=False
            virtual_h3_id=None
            pending=None
        if is_shadow and day in bentries:
            if virtual_h3_long:
                raise AssertionError(f"shadow double H3 entry {day}")
            virtual_h3_long=True
            virtual_h3_id=bentries[day]["trade_id"]
            # If existing ETF shares were bought in shadow cash before
            # original H3 start, keep them. No fictional sale/rebuy fees.
            # H3's future original exit remains the authoritative stop.
            if not shares:
                fee=cash*FEE/(1+FEE)
                shares=cash/(open_px*(1+FEE))
                cash=0.
                fees+=fee
                n_buy+=1
                physical_entry_date=day
                physical_entry_price=open_px
                physical_origin="H3_ORIGINAL"
                actions.append(make_action("BUY",open_px,day,
                  "H3_ORIGINAL","H3_ORIGINAL_ENTRY",True))
            else:
                actions.append(make_action("HANDOFF",open_px,day,
                  "H3_ORIGINAL","CARRY_FAST_SHARES_INTO_H3",True))
            pending=None
        if pending is not None:
            side=pending["side"]
            # Only eligible shadow/all crossover may proceed.
            if side=="buy" and shares==0:
                fee=cash*FEE/(1+FEE)
                shares=cash/(open_px*(1+FEE))
                cash=0.
                fees+=fee
                n_buy+=1
                physical_entry_date=day
                physical_entry_price=open_px
                physical_origin=pending["reason"]
                actions.append(make_action("BUY",open_px,day,
                  pending["signal_date"],pending["reason"],virtual_h3_long))
            elif side=="sell" and shares>0:
                fees+=shares*open_px*FEE
                proceeds=shares*open_px*(1-FEE)
                cash+=proceeds
                n_sell+=1
                actions.append(make_action("SELL",open_px,day,
                  pending["signal_date"],pending["reason"],virtual_h3_long))
                trades.append({"arm":name,"entry_date":physical_entry_date,
                  "exit_date":day,"entry_price":physical_entry_price,
                  "exit_price":open_px,"entry_origin":physical_origin,
                  "exit_reason":pending["reason"],
                  "gross_price_return_pct":100*(open_px/physical_entry_price-1)})
                shares=0.
                physical_entry_date=None
                physical_entry_price=None
                physical_origin=None
            else:
                raise AssertionError("pending mismatched side")
            pending=None

        # Record close signal; no fill on this same close.
        if mode=="ema_only":
            in_scope=True
        elif mode=="shadow_all":
            in_scope=True
        elif mode=="shadow_flat":
            in_scope=not virtual_h3_long
        elif mode=="shadow_none":
            in_scope=False
        else:raise AssertionError("unknown mode")
        if in_scope:
            if shares and bool(r.fresh_fast_death):
                pending={"side":"sell","signal_date":day,
                         "reason":"FAST_EMA10_20_DEATH"}
            elif (shares and stop_pct is not None
                  and physical_entry_price is not None
                  and close_px/physical_entry_price-1 <= -stop_pct
                  and not virtual_h3_long):
                # Only overlay cash-spell risk control; H3 original hard
                # stops remain shadow H3 when original is invested.
                pending={"side":"sell","signal_date":day,
                         "reason":f"FAST_CLOSE_STOP_{stop_pct:.2f}"}
            elif not shares and bool(r.fresh_fast_gold):
                pending={"side":"buy","signal_date":day,
                         "reason":"FAST_EMA10_20_GOLD"}
        curve.append({"Date":day,"Equity":cash+shares*close_px,
                      "Invested":bool(shares),
                      "ShadowH3Invested":bool(virtual_h3_long),
                      "Pending":pending["side"] if pending is not None else "",
                      "MonthGateAllows":bool(r.month_gate),
                      "SlowBull":bool(r.slow_bull)})

    d=pd.DataFrame(curve)
    d["Drawdown"]=d.Equity/d.Equity.cummax()-1
    years=(pd.Timestamp(d.Date.iloc[-1])-pd.Timestamp(START)).days/365.25
    return d,pd.DataFrame(trades),pd.DataFrame(actions),{
        "arm":name,
        "cagr":(d.Equity.iloc[-1]/CAPITAL)**(1/years)-1,
        "mdd":d.Drawdown.min(),
        "end_equity":d.Equity.iloc[-1],
        "exposure":d.Invested.mean(),
        "cash_days":int((~d.Invested).sum()),
        "extra_cash_exposure_days":int((d.Invested & ~d.ShadowH3Invested).sum()),
        "original_h3_missed_days":int((~d.Invested & d.ShadowH3Invested).sum()),
        "buy_orders":n_buy,"sell_orders":n_sell,
        "round_trips":len(trades),
        "fees_paid_usd":fees,"pending_end":pending,
        "shadow_h3_trade_windows":len(base["trades"]),
        "no_month_gate_used_for_fast":True if mode!="shadow_none" else False,
        "no_limit_on_fast_cycles":True if mode!="shadow_none" else False,
        "standalone_no_h3_stop":mode=="ema_only",
    }

def by_eras(d):
    x=d.copy()
    x["Date"]=pd.to_datetime(x.Date)
    out=[]
    for label,start,end in ERAS:
        idx=np.flatnonzero((x.Date>=pd.Timestamp(start)) &
                            (x.Date<=pd.Timestamp(end)))
        first=int(idx[0]);last=int(idx[-1])
        before=CAPITAL if first==0 else float(x.Equity.iloc[first-1])
        sub=x.iloc[idx]
        out.append({"period":label,"total_return_pct":
                   100*(float(sub.Equity.iloc[-1])/before-1),
                   "cash_pct":100*(1-sub.Invested.mean()),
                   "extra_cash_exposure_days":int(
                       (sub.Invested & ~sub.ShadowH3Invested).sum()),
                   "trading_days":len(sub)})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--outdir",default="research/results/h3_qqq_cash_fast_1020_cycle")
    p.add_argument("--end",default=END)
    a=p.parse_args()
    if a.end!=END:raise ValueError("fixed historical cutoff 2026-11-11")
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    raw,provenance=load_qqq(END)
    bars=prep_raw(raw)
    signals=with_month_slope_features(completed_month_signals(bars))
    x=signals_from_frozen(signals)
    base=replay(signals,"H3_FROZEN",config=Config(start_date=START))
    assert len(base["trades"])==60
    summaries=[];periods=[];acts=[];trades=[]
    curves={}
    for name,mode,stop in ARMS:
        d,t,a,s=simulate(x,base,name,mode,stop)
        ref=base["equity"]
        assert d.Date.tolist()==ref.Date.tolist()
        if name=="H3_FROZEN":
            delta=float(np.max(abs(d.Equity.to_numpy()-ref.Equity.to_numpy())))
            assert delta<.05, delta
            assert np.array_equal(d.Invested,ref.Invested)
            assert abs(float(s["cagr"])-float(base["stats"]["cagr"]))<.00001
            assert abs(float(s["mdd"])-float(base["stats"]["mdd"]))<.00001
            assert abs(float(s["exposure"])-float(base["stats"]["exposure"]))<.00001
            assert s["buy_orders"]==len(base["trades"])
            assert s["sell_orders"]==len(base["trades"])
        curves[name]=d
        s["cagr_vs_h3_pp"]=100*(s["cagr"]-base["stats"]["cagr"])
        s["mdd_vs_h3_pp"]=100*(s["mdd"]-base["stats"]["mdd"])
        s["end_wealth_vs_h3_pct"]=100*(s["end_equity"]/base["stats"]["end_equity"]-1)
        summaries.append(s)
        for er in by_eras(d):periods.append({"arm":name,**er})
        if len(t):
            trades.append(t)
        if len(a):
            acts.append(a)
        d.to_csv(out/f"{name.lower()}_daily_equity.csv",index=False)
        print("FAST_CYCLE_ARM",json.dumps(s,default=float))
    table=pd.DataFrame(summaries)
    pd.DataFrame(summaries).to_csv(out/"full_account_summary.csv",index=False)
    pd.DataFrame(periods).to_csv(out/"era_returns_and_exposure.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(out/"all_closed_real_trades.csv",index=False)
    pd.concat(acts,ignore_index=True).to_csv(out/"all_executed_orders.csv",index=False)
    print("FAST_CYCLE_PERIOD",pd.DataFrame(periods).pivot(
        index="period",columns="arm",values="total_return_pct").to_json())
    # audit orders in cash transition: genuine EMA10/20 rotation, not just
    # one buy/sell. Count repeated cycles and admissions inside each flat
    # segment of the original H3 shadow.
    b=curves["H3_FROZEN"].copy()
    b["cash_spell_id"]=(b.ShadowH3Invested.ne(
                        b.ShadowH3Invested.shift(1))).cumsum()
    cash_spells=b.loc[~b.ShadowH3Invested,["Date","cash_spell_id"]]
    action_df=pd.concat(acts,ignore_index=True)
    shadow_cash_fills=action_df[
        action_df.arm.eq("H3_FLAT_10_20_CYCLE") &
        action_df.reason.isin(["FAST_EMA10_20_GOLD","FAST_EMA10_20_DEATH"])]
    cash_tags=shadow_cash_fills.merge(
        cash_spells,left_on="fill_date",right_on="Date",how="left",
        validate="many_to_one")
    if len(cash_tags):
        counts=cash_tags.groupby(["cash_spell_id","action"]).size().unstack(fill_value=0)
        counts["round_trips_completed"]=counts.get("SELL",0)
        counts.sort_values("round_trips_completed",ascending=False).to_csv(
            out/"repeat_fast_cycles_by_h3_cash_spell.csv")
        print("FAST_CYCLE_REPEAT_SPELLS",counts.sort_values(
            "round_trips_completed",ascending=False).head(12).to_json())
    daily=curves["H3_FLAT_10_20_CYCLE"].merge(
        curves["H3_FROZEN"][["Date","Equity"]],
        on="Date",suffixes=("","_h3"),validate="one_to_one")
    daily["RelativeWealth"]=daily.Equity/daily.Equity_h3
    daily.to_csv(out/"hybrid_vs_frozen_h3_daily_relative_wealth.csv",index=False)
    manifest={
        "sample":"QQQ", "start":START,"end":END,"provider":provenance,
        "baseline":"H3_FROZEN original native replay, shadow opens and exits fixed by original H3 full state machine",
        "baseline_identity_asserted":True,
        "signals":"fresh EMA10 crosses EMA20 or vice versa, as-of complete close; NEXT OPEN execution; fee .05% each side",
        "month_veto":"NO month entry veto for FAST overlay, original H3 unchanged",
        "H3_FLAT_10_20_CYCLE":"FAST crossover overlay operates only when original H3 virtual account cash; original H3 entry overrides overlay pending and if shares held hands them to baseline without selling; original H3 exit is mandatory",
        "H3_FLAT_10_20_CYCLE_STOP7":"same with complete closing-price 7% loss from actual FAST entry, next-open sell only if still original shadow cash",
        "H3_ALL_10_20_CYCLE":"new gold/ death orders may operate when original H3 virtual account long; original H3 buy & original H3 exit have open execution priority",
        "EMA10_20_ALWAYS":"pure repeated cross long/cash; no H3 original entry/exits, no N20 or hard stops",
        "H3_exit_shadow_authority":"Original H3 exits remain authoritative even if overlay already flat. Original H3 N20 is driven by virtual original entry prices, not overlay earlier fill price. Thus hybrid is an overlay with immutable original H3 shadow; it is not a native new H3 strategy with reinitialized N20 on every FAST buy.",
        "cash_fee":"initial USD100k, no cash interest, no taxes, no extra slippage, 0.05% per executed buy or sell",
        "lookahead":"no future signal, use complete daily bar then next-open; only original H3 state derived with complete same historical bars",
        "warning":"the H3 shadow comparator is a design choice; all-cycle strategy has different regime and does not preserve H3 risk benefit. All results heavily in-sample",
        "A_B_frozen_unchanged":True,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,
                        indent=2,default=str)+"\n")
    print("FAST_CYCLE_COMPLETED",len(table))

if __name__=="__main__":
    main()
