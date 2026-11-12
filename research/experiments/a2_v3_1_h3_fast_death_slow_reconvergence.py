"""Research: profit-only EMA10/20 death exits and converged EMA20/60 FAST re-entry.

Inputs MUST be a separately obtained, legally usable, full-history, consistent
adjusted QQQ daily OHLCV file; no raw price data is added to the repository.
Date cutoff Nov 11 2026 (last independently audited snapshot at authorship).
All event predicates use completed bars; all fills are NEXT regular open.
This is an in-sample hypothesis test, not an optimized trade strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import replay_gated, with_month_slope_features
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS
from a2_v3_1_cross_asset import _load, _summary
from a2_v3_1_regime_research import period_metrics
from qqq_a2.engine import Config

# Chosen as broad comparators prior to evaluating the outcomes.
# 0.5/1/2% distance means positive EMA20/60 separation in percentage points,
# not ATR multiples and not a hindsight-dependent distance minimum.
ARMS = {
    "H3_FROZEN": {},
    "PROFIT_FAST_DEATH_EXIT_ONLY": dict(
        fast_profit_exit_peak_min=0.),
    "PROFIT_FAST_DEATH_RE_FAST_ANY": dict(
        fast_profit_exit_peak_min=0., fast_profit_reentry_mode="any"),
    "PROFIT_FAST_DEATH_RE_FAST_NEAR_0_5": dict(
        fast_profit_exit_peak_min=0., fast_profit_reentry_mode="current",
        fast_profit_reentry_gap_pct=.5),
    "PROFIT_FAST_DEATH_RE_FAST_NEAR_1": dict(
        fast_profit_exit_peak_min=0., fast_profit_reentry_mode="current",
        fast_profit_reentry_gap_pct=1.),
    "PROFIT_FAST_DEATH_RE_FAST_NEAR_2": dict(
        fast_profit_exit_peak_min=0., fast_profit_reentry_mode="current",
        fast_profit_reentry_gap_pct=2.),
    "PROFIT_FAST_DEATH_RE_TOUCH_1": dict(
        fast_profit_exit_peak_min=0., fast_profit_reentry_mode="seen",
        fast_profit_reentry_gap_pct=1.),
    "PROFIT_FAST_DEATH_RE_TOUCH_2": dict(
        fast_profit_exit_peak_min=0., fast_profit_reentry_mode="seen",
        fast_profit_reentry_gap_pct=2.),
    "PEAK10_PROFIT_FAST_DEATH_RE_FAST_ANY": dict(
        fast_profit_exit_peak_min=.10, fast_profit_reentry_mode="any"),
    "PEAK10_PROFIT_FAST_DEATH_RE_FAST_NEAR_2": dict(
        fast_profit_exit_peak_min=.10, fast_profit_reentry_mode="current",
        fast_profit_reentry_gap_pct=2.),
    "PEAK10_PROFIT_FAST_DEATH_RE_TOUCH_2": dict(
        fast_profit_exit_peak_min=.10, fast_profit_reentry_mode="seen",
        fast_profit_reentry_gap_pct=2.),
    "PROFIT_GAP2_FAST_DEATH_RE_FAST_NEAR_1": dict(
        fast_profit_exit_peak_min=0., fast_profit_exit_slow_gap_min_pct=2.,
        fast_profit_reentry_mode="current", fast_profit_reentry_gap_pct=1.),
    "PROFIT_GAP2_FAST_DEATH_RE_FAST_NEAR_2": dict(
        fast_profit_exit_peak_min=0., fast_profit_exit_slow_gap_min_pct=2.,
        fast_profit_reentry_mode="current", fast_profit_reentry_gap_pct=2.),
    "PROFIT_GAP2_FAST_DEATH_RE_TOUCH_2": dict(
        fast_profit_exit_peak_min=0., fast_profit_exit_slow_gap_min_pct=2.,
        fast_profit_reentry_mode="seen", fast_profit_reentry_gap_pct=2.),
    "PEAK10_GAP2_FAST_DEATH_RE_TOUCH_2": dict(
        fast_profit_exit_peak_min=.10, fast_profit_exit_slow_gap_min_pct=2.,
        fast_profit_reentry_mode="seen", fast_profit_reentry_gap_pct=2.),
}

KNOWN = {
    ("QQQ", "2026-11-11"): (1257933.89348445, -.24299910819915604, 60),
    ("QQQ", "2026-07-13"): (1323429.0451020638, -.2429991081991566, 58),
}


def replay(signals: pd.DataFrame, name: str,
           config: Config = Config()) -> dict:
    if name not in ARMS:
        raise ValueError("Unregistered experiment")
    return replay_gated(
        signals, ENTRY_GATE, config=config,
        **V3_PARAMETERS, defer_rejected_fast=True,
        fast_negative_month_mode="accelerating",
        **ARMS[name],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, choices=["QQQ"])
    p.add_argument("--csv", required=True,
                   help="Locally obtained fully adjusted daily OHLCV, all history")
    p.add_argument("--old")
    p.add_argument("--new")
    p.add_argument("--update")
    p.add_argument("--end", default="2026-11-11")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    data, source = _load(args)
    signals = with_month_slope_features(completed_month_signals(data))
    runs = {name: replay(signals, name) for name in ARMS}
    base = runs["H3_FROZEN"]
    reference = KNOWN.get((args.symbol, args.end))
    if reference:
        st = base["stats"]
        assert abs(st["end_equity"] - reference[0]) < .02, "H3 baseline equity drift"
        assert abs(st["mdd"] - reference[1]) < 1e-8, "H3 baseline MDD drift"
        assert st["completed_trades"] == reference[2], "H3 count drift"

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary, eras, events = [], [], []
    for name, run in runs.items():
        r = run["fast_profit_events"]
        ex = (r.action.eq("EXIT").sum() if not r.empty else 0)
        re = (r.action.eq("REENTRY").sum() if not r.empty else 0)
        summary.append({
            **_summary(name, args.symbol, args.end, run),
            "profit_fast_exits": int(ex),
            "narrow_fast_reentries": int(re),
            "actual_buy_sell_orders": (
                2 * len(run["trades"])
                + int(run["stats"]["active_position_at_end"])),
        })
        eras.extend({"symbol": args.symbol, "arm": name, **era}
                    for era in period_metrics(run["equity"]))
        if not r.empty:
            data = r.copy()
            data.insert(0, "arm", name)
            events.append(data)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(eras).to_csv(out / "era_comparison.csv", index=False)
    if events:
        pd.concat(events, ignore_index=True).to_csv(
            out / "executed_profit_fast_events.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "symbol": args.symbol, "end": args.end, "source": source,
        "hypothesis": "profit-positive fast death exit, then slow gap convergence and fresh fast gold",
        "snapshot_rights": "no raw OHLCV redistributed",
        "arms": ARMS, "all_signals_close_completed": True,
        "all_fills_next_regular_open": True,
        "fees_per_side": .0005, "additional_slippage": 0.,
        "tax": "not modeled", "cash_interest": 0.,
        "monthly_gate": "prior completed-month EMA5 only",
        "fast_reentry_without_monthly_veto": True,
        "regular_H3_entry_priority": True,
        "slow_cross_invalidates_extra_reentry": True,
        "research_only": True,
        "warning": "QQQ has repeatedly informed hypothesis search; not out of sample",
    }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print(pd.DataFrame(summary)[
        ["version", "end_equity", "cagr", "mdd",
         "completed_trades", "profit_fast_exits", "narrow_fast_reentries"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
