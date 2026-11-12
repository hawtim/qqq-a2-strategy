"""Forensic why-no-FAST audit for selected GOOGL H3 flat intervals.

Uses the same Yahoo data loader and FROZEN H3 replay. Does NOT change or
reimplement trade decisions: signals are audited against the actual H3 ledger.
Only derived candidate events and reason codes are saved; not raw OHLCV.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))
from h3_googl_cross_asset import load_prices, START
from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_h3_trend_spread_quick_reentry import replay
from qqq_a2.engine import Config

WINDOWS = [
    ("IPO initial wait", "2004-08-19", "2005-05-03"),
    ("2007 recovery", "2007-08-22", "2007-10-02"),
    ("2009 rebound", "2009-01-20", "2009-02-06"),
    ("2024 rebound", "2024-03-06", "2024-04-02"),
    ("2025 rebound", "2025-05-12", "2025-05-29"),
    ("2026 March-April", "2026-03-09", "2026-04-15"),
    ("2026 July-October", "2026-07-23", "2026-10-14"),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-11-11")
    ap.add_argument("--outdir", default="research/results/h3_googl_fast_audit_local")
    args=ap.parse_args()
    raw, prov=load_prices(args.end)
    factor=raw["Adj Close"]/raw.Close
    data=raw.copy()
    for key in ("Open","High","Low","Close"):
        data["adj_"+key.lower()]=data[key]*factor
    signals=with_month_slope_features(completed_month_signals(data))
    run=replay(signals,"H3_FROZEN",config=Config(start_date=START))
    ledger=run["trades"]; equity=run["equity"]
    assert len(ledger)==58
    assert abs(run["stats"]["end_equity"]-1741724.7114545524)<1, "Frozen H3 drift"
    assert signals.Date.dt.strftime("%Y-%m-%d").tolist()==equity.Date.tolist()
    buy_dates=set(ledger.entry_date)
    exit_dates=set(ledger.exit_date)
    entry_signal_by_date={r.entry_signal_date:r for r in ledger.itertuples()}
    desired={}
    for label,a,b in WINDOWS:
        matching=ledger.loc[ledger.entry_date==b]
        assert len(matching)==1,(label,b,len(matching))
        if a!=START:
            assert len(ledger.loc[ledger.exit_date==a])==1,(label,a)
        desired[label]={"name":label,"start":a,"next_entry":b,"next_kind":matching.iloc[0].entry_kind,
                        "next_signal":matching.iloc[0].entry_signal_date}
    seen_exit=False;deferred_fast=False;deferred_gold=False;events=[];validation=[]
    prev_invested=False
    for i, row in signals.iterrows():
        day=str(row.Date.date())
        if day<START:continue
        invested=bool(equity.iloc[i].Invested)
        if invested:
            deferred_fast=False;deferred_gold=False
            prev_invested=True
            continue
        # A SELL is filled at the open; immediately allows fresh FAST eligibility.
        if day in exit_dates:
            seen_exit=True;deferred_fast=False;deferred_gold=False
        if bool(row.death):
            deferred_gold=False
        gate=bool(row.gate_up_slow_veto_1_abs)
        monthly_up=bool(row.prev_completed_month_ema5_up)
        cur=float(row.known_slope_abs);prev=float(row.known_prev_abs)
        neg_accel=bool(cur<0 and prev<0 and cur<prev)
        recent_gold=bool(row.ema10_20_gold)
        if bool(row.gold) and not gate:
            deferred_gold=True
        if seen_exit:
            if bool(row.protect_cross) or not bool(row.ma10>row.ma20):
                deferred_fast=False
            if recent_gold and (not monthly_up or not gate):
                deferred_fast=True
        fast_structural=bool(recent_gold or (deferred_fast and bool(row.ma10>row.ma20)))
        can_fast=bool(seen_exit and fast_structural and (monthly_up or neg_accel) and gate)
        regular_structural=bool(row.gold or (deferred_gold and row.ma20>row.ma60))
        can_regular=bool(regular_structural and gate)
        if day in entry_signal_by_date:
            t=entry_signal_by_date[day]
            validation.append({"date":day,"expected":t.entry_kind,
                               "fast":can_fast,"regular":can_regular,
                               "match": bool(can_regular if t.entry_kind=="REGULAR_EMA20_60" else can_fast and not can_regular)})
        if recent_gold or (deferred_fast and can_fast) or bool(row.gold):
            reasons=[]
            if not seen_exit:reasons.append("NO_FAST_ELIGIBILITY_BEFORE_FIRST_EXIT")
            if recent_gold and not monthly_up and not neg_accel:
                reasons.append("FAST_MONTHLY_SLOPE_NOT_UP_OR_ACCEL_NEGATIVE")
            if recent_gold and not gate:
                reasons.append("MONTHLY_SLOWING_UP_VETO")
            if recent_gold and can_fast and can_regular:
                reasons.append("REGULAR_PRIORITY")
            if recent_gold and can_fast and not can_regular:
                reasons.append("FAST_ACCEPTED")
            if not recent_gold and can_fast and deferred_fast:
                reasons.append("DEFERRED_FAST_RELEASED")
            if bool(row.gold) and gate:reasons.append("REGULAR_ACCEPTED_OR_WAITING_PRIORITY")
            events.append({
                "date":day,"fresh_fast_cross":recent_gold,
                "fresh_regular_cross":bool(row.gold),
                "monthly_up":monthly_up,
                "slope_current":cur,"slope_previous":prev,
                "negative_accelerating":neg_accel,
                "monthly_gate":gate,"fast_eligibility":bool(seen_exit),
                "fast_deferred_active":bool(deferred_fast),
                "fast_pass":can_fast,"regular_pass":can_regular,
                "label":"|".join(reasons) if reasons else "STRUCTURAL_EVENT_ONLY",
                "is_actual_entry_signal":day in entry_signal_by_date,
                "actual_entry_kind":entry_signal_by_date[day].entry_kind if day in entry_signal_by_date else "",
            })
        prev_invested=False
    assert all(x["match"] for x in validation), [x for x in validation if not x["match"]]
    assert len(validation)==len(ledger),len(validation)

    all_windows=[]
    for label,a,b in WINDOWS:
        ev=[x for x in events if a<=x["date"]<b]
        fresh=[x for x in ev if x["fresh_fast_cross"]]
        blocked=[x for x in fresh if not x["fast_pass"]]
        accepted=[x for x in fresh if x["fast_pass"]]
        all_windows.append({
            **desired[label],"fresh_fast_crosses":len(fresh),
            "fast_cross_dates":[x["date"] for x in fresh],
            "blocked_fast_crosses":len(blocked),
            "accepted_fresh_fast":len(accepted),
            "no_fast_eligibility":sum("NO_FAST_ELIGIBILITY_BEFORE_FIRST_EXIT" in x["label"] for x in fresh),
            "monthly_slope_block":sum("FAST_MONTHLY_SLOPE_NOT_UP_OR_ACCEL_NEGATIVE" in x["label"] for x in fresh),
            "monthly_slowing_veto":sum("MONTHLY_SLOWING_UP_VETO" in x["label"] for x in fresh),
            "deferred_release":sum("DEFERRED_FAST_RELEASED" in x["label"] for x in ev),
            "events":ev
        })
    out=Path(args.outdir);out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(events).to_csv(out/"flat_signal_events.csv",index=False)
    pd.DataFrame([{k:v for k,v in q.items() if k!="events"} for q in all_windows]).to_csv(out/"window_summary.csv",index=False)
    (out/"window_details.json").write_text(json.dumps({
        "ticker":"GOOGL","strategy":"H3_FROZEN",
        "data_end":prov["date_end"],"data_sha256":prov["sha256"],
        "runner":"h3_googl_fast_opportunity_audit.py",
        "validation":{"signal_matches":len(validation),"matched":sum(x["match"] for x in validation)},
        "windows":all_windows,
        "note":"Only completed-close as-of data; next-open fills; no raw OHLCV exported"
    },ensure_ascii=False,indent=2,default=str)+"\n")
    print("H3_FAST_AUDIT_FINISHED",json.dumps({
        "validation":len(validation),"source_sha256":prov["sha256"],
        "windows":all_windows
    },ensure_ascii=False,default=str))

if __name__=="__main__":
    main()
