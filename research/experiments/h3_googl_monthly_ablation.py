"""GOOGL H3 frozen monthly-filter ablation, no strategy parameter tuning.

Exclusively vary two completed-month entry gates on a copy of daily SIGNALS:
  G: gate_up_slow_veto_1_abs (positive EMA5 slope but slowing)
  F: prev_completed_month_ema5_up (FAST positive-month requirement).
To remove the F filter, force it True: the frozen H3 "negative accelerating"
allowance becomes redundant, while all FAST cross/defer/N20/exit rules remain.

No monthly history is passed to execution after gating. All data has the
same end date, adjusted basis, cash/fees, and IPO initialization as frozen H3.
This is a counterfactual in-sample ablation, NOT a selected strategy.
No raw OHLCV is committed, only derived outputs.
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

from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_h3_trend_spread_quick_reentry import replay
from h3_googl_cross_asset import START, load_prices
from qqq_a2.engine import Config

ARMS = [
    ("H3_FROZEN", False, False),
    ("NO_SLOWING_VETO", True, False),
    ("NO_FAST_MONTH_SLOPE", False, True),
    ("NO_MONTHLY_FILTERS", True, True),
]
FLAT_WINDOWS = [
    ("IPO first entry", "2004-08-19", "2005-05-03"),
    ("2007 recovery", "2007-08-22", "2007-10-02"),
    ("2009 rebound", "2009-01-20", "2009-02-06"),
    ("2024 rebound", "2024-03-06", "2024-04-02"),
    ("2025 rebound", "2025-05-12", "2025-05-29"),
    ("2026 spring", "2026-03-09", "2026-04-15"),
    ("2026 summer-fall", "2026-07-23", "2026-10-14"),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--end", default="2026-11-11")
    p.add_argument("--outdir", default="research/results/h3_googl_monthly_ablation_local")
    args = p.parse_args()

    raw, provenance = load_prices(args.end)
    data = raw.copy()
    ratio = data["Adj Close"].astype(float) / data["Close"].astype(float)
    for field in ("Open", "High", "Low", "Close"):
        data["adj_" + field.lower()] = data[field].astype(float) * ratio
    signals = with_month_slope_features(completed_month_signals(data))
    config = Config(start_date=START)

    runs, summary_rows, attribution_rows, annual_rows = {}, [], [], []
    for name, no_veto, no_fast_slope in ARMS:
        x = signals.copy(deep=True)
        if no_veto:
            x.loc[:, "gate_up_slow_veto_1_abs"] = True
        if no_fast_slope:
            x.loc[:, "prev_completed_month_ema5_up"] = True
        result = replay(x, "H3_FROZEN", config=config)
        runs[name] = result
        stats = result["stats"]
        trades = result["trades"]
        summary_rows.append({
            "arm": name, "disable_global_up_slow_veto": no_veto,
            "disable_fast_month_slope_gate": no_fast_slope, **stats,
            "fast_entries": int((trades.entry_kind == "FAST_EMA10_20_MONTH_EMA5").sum()),
            "regular_entries": int((trades.entry_kind == "REGULAR_EMA20_60").sum()),
            "first_entry_date": str(trades.entry_date.iloc[0]),
            "last_exit_date": str(trades.exit_date.iloc[-1]),
            "data_end": provenance["date_end"], "data_rows": provenance["rows"],
        })
        curves = result["equity"].copy()
        curves["year"] = curves.Date.str[:4].astype(int)
        end_by_year = curves.groupby("year").Equity.last()
        prev_year_end = config.initial_cash
        for yr, last in end_by_year.items():
            annual_rows.append({"arm": name, "year": int(yr),
                                "calendar_equity_return": float(last/prev_year_end-1),
                                "year_end_equity": float(last)})
            prev_year_end = last

    base=runs["H3_FROZEN"]
    assert len(base["trades"]) == 58, "Frozen H3 completed trade count changed"
    assert abs(base["stats"]["cagr"] - .1371769) < .0002
    assert abs(base["stats"]["end_equity"] / 1741724.7114545524 - 1) < 1e-4
    assert abs(base["stats"]["mdd"] + .3549377112350106) < 1e-4
    for name, result in runs.items():
        if result["equity"].Date.tolist() != base["equity"].Date.tolist():
            raise AssertionError("Different trading calendar: " + name)

    for label, beginning, next_buy in FLAT_WINDOWS:
        for name, result in runs.items():
            eq=result["equity"]
            window=eq.loc[(eq.Date >= beginning) & (eq.Date < next_buy)]
            if len(window)<1:raise AssertionError("Missing window")
            rows=result["trades"]
            new_entries=rows.loc[(rows.entry_date >= beginning)&(rows.entry_date < next_buy)]
            attribution_rows.append({
                "window": label, "window_start": beginning,
                "baseline_next_buy": next_buy,"arm":name,
                "sessions":len(window),
                "exposed_sessions":int(window.Invested.sum()),
                "exposed_fraction":float(window.Invested.mean()),
                "first_new_entry_before_baseline":str(new_entries.entry_date.iloc[0]) if len(new_entries) else "",
                "first_new_entry_kind":str(new_entries.entry_kind.iloc[0]) if len(new_entries) else "",
                "new_entries_before_baseline":len(new_entries),
            })
    summary=pd.DataFrame(summary_rows)
    a=summary.set_index("arm")
    end_base=float(a.loc["H3_FROZEN","end_equity"])
    summary["ending_equity_vs_frozen_pct"]=100*(summary.end_equity/end_base-1)
    summary["cagr_diff_pp"]=100*(summary.cagr-float(a.loc["H3_FROZEN","cagr"]))
    summary["mdd_diff_pp"]=100*(summary.mdd-float(a.loc["H3_FROZEN","mdd"]))
    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"summary.csv",index=False)
    pd.DataFrame(attribution_rows).to_csv(out/"flat_window_exposure.csv",index=False)
    pd.DataFrame(annual_rows).to_csv(out/"annual_equity_returns.csv",index=False)
    for name,result in runs.items():
        result["trades"].to_csv(out/f"{name.lower()}_trades.csv",index=False)
        result["equity"].to_csv(out/f"{name.lower()}_equity.csv",index=False)
    meta={
        "ticker":"GOOGL","strategy":"QQQ EMA V3.1-H3 FROZEN execution",
        "data_source":provenance,
        "start_date":START, "end_date":provenance["date_end"],
        "ablated_signal_columns":{
            "NO_SLOWING_VETO":"set gate_up_slow_veto_1_abs=True",
            "NO_FAST_MONTH_SLOPE":"set prev_completed_month_ema5_up=True",
            "NO_MONTHLY_FILTERS":"both above",
        },
        "unchanged":"EMA crosses, deferred FAST, N20, close-stop, first-day guard, fees, close-to-next-open execution, first-FAST-after-exit eligibility",
        "in_sample_and_single_company":True,
        "price_history_exported":False,
        "note":"No first-ever FAST entry added; regular first-entry eligibility frozen. Baseline may not be bit-for-bit identical if vendor backadjusts prior bars."
    }
    (out/"manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str)+"\n")
    print("H3_GOOGL_MONTHLY_ABLATION_RESULT",summary.to_json(orient="records"))
    print("H3_GOOGL_MONTHLY_ABLATION_WINDOWS",pd.DataFrame(attribution_rows).to_json(orient="records"))
    print("H3_GOOGL_MONTHLY_ABLATION_YEAR_RETURNS",pd.DataFrame(annual_rows).to_json(orient="records"))

if __name__=="__main__":
    main()
