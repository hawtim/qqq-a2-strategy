"""Research-only replay: adaptive regular-entry N20 and first-day 3% loss.

This deliberately keeps frozen SMA V1 and the existing V2 default untouched.
All as-of signals occur at an already-completed close; next-open execution is
used unless the scenario is explicitly labeled ``intraday_stop_model``.

Example (private, legally acquired local price sources):

    python research/experiments/a2_dynamic_n20_and_day0_loss_study.py \
        --old /local/old/QQQ.csv --new /local/new/QQQ.csv \
        --update /local/qqq_update.csv --end 2026-11-09 \
        --outdir research/results/a2_dynamic_n20_day0_2026_11_09

Do not publish raw OHLCV prices; this exports derived trade events and metrics.
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
from a2_month_ema5_slope_regime import with_month_slope_features, replay_gated
from qqq_a2.engine import combine_rebased_sources
from qqq_a2.updates import extend_adjusted_history


VARIANT_ARGS = {
    "V2_frozen": {},
    "N20_2of5_down_to25": {
        "regular_n20_extension": "underwater_2of5_spread_down", "extended_n20_days": 25},
    "N20_2of5_down_to30": {
        "regular_n20_extension": "underwater_2of5_spread_down", "extended_n20_days": 30},
    "N20_2of5_down_to35": {
        "regular_n20_extension": "underwater_2of5_spread_down", "extended_n20_days": 35},
    "N20_2of10_down_to30": {
        "regular_n20_extension": "underwater_2of10_spread_down"},
    "N20_spread_down_only": {"regular_n20_extension": "spread_down"},
    "day0_close_1pct_negative_control": {"entry_day_close_guard": .01},
    "day0_close_2pct": {"entry_day_close_guard": .02},
    "day0_close_3pct": {"entry_day_close_guard": .03},
    "day0_close_3p5pct_negative_control": {"entry_day_close_guard": .035},
    "day0_intraday_stop_3pct_ideal": {"entry_day_intraday_stop": .03},
    "day0_intraday_stop_3pct_slip25bp": {
        "entry_day_intraday_stop": .03, "intraday_adverse_slippage": .0025},
    "day0_intraday_stop_3pct_slip50bp": {
        "entry_day_intraday_stop": .03, "intraday_adverse_slippage": .005},
    "N20_dynamic_and_day0_close_3pct": {
        "regular_n20_extension": "underwater_2of5_spread_down",
        "entry_day_close_guard": .03},
    "stop4_control": {"stop_loss": .04},
    "stop4_plus_N20_dynamic": {
        "stop_loss": .04, "regular_n20_extension": "underwater_2of5_spread_down"},
    "stop4_plus_day0_close_3pct": {
        "stop_loss": .04, "entry_day_close_guard": .03},
}


def compare_trades(baseline: pd.DataFrame, candidate: pd.DataFrame,
                   variant: str) -> list[dict]:
    a, b = baseline.set_index("entry_date"), candidate.set_index("entry_date")
    changed = []
    for key in sorted(set(a.index) | set(b.index)):
        old = a.loc[key] if key in a.index else None
        new = b.loc[key] if key in b.index else None
        if old is not None and new is not None:
            if (old.exit_date == new.exit_date and
                    abs(old.ret_net - new.ret_net) < 1e-10):
                continue
        changed.append({
            "variant": variant, "entry_date": key,
            "old_exit_date": old.exit_date if old is not None else "",
            "new_exit_date": new.exit_date if new is not None else "",
            "old_return_pct": round(float(old.ret_net) * 100, 5) if old is not None else None,
            "new_return_pct": round(float(new.ret_net) * 100, 5) if new is not None else None,
            "old_reason": old.exit_reason if old is not None else "",
            "new_reason": new.exit_reason if new is not None else "",
        })
    return changed


def audit_entry_day(signals: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    indexed = signals.set_index(signals.Date.dt.strftime("%Y-%m-%d"))
    rows = []
    for t in trades.itertuples(index=False):
        i = indexed.index.get_loc(t.entry_date)
        current, next_day = signals.iloc[i], signals.iloc[i + 1]
        rows.append({
            "entry_date": t.entry_date, "entry_kind": t.entry_kind,
            "day0_close_loss_pct": round((current.adj_close / t.entry_price - 1) * 100, 5),
            "day0_low_loss_pct": round((current.adj_low / t.entry_price - 1) * 100, 5),
            "next_open_loss_pct": round((next_day.adj_open / t.entry_price - 1) * 100, 5),
            "base_trade_return_pct": round(t.ret_net * 100, 5),
            "close_3pct_trigger": current.adj_close / t.entry_price <= .97,
            "intraday_3pct_reachable": current.adj_low / t.entry_price <= .97,
        })
    return pd.DataFrame(rows)


def audit_age20(signals: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    indexed = signals.set_index(signals.Date.dt.strftime("%Y-%m-%d"))
    out = []
    for t in trades.itertuples(index=False):
        if t.hold_days <= 20 or t.entry_kind != "REGULAR_EMA20_60":
            continue
        i = indexed.index.get_loc(t.entry_date)
        window = signals.iloc[i + 16:i + 21]
        day20, day15 = signals.iloc[i + 20], signals.iloc[i + 15]
        five_under = int((window.adj_close < t.entry_price).sum())
        spread_down = day20.ma10 / day20.ma20 < day15.ma10 / day15.ma20
        out.append({
            "entry_date": t.entry_date,
            "day20_close_pnl_pct": round((day20.adj_close / t.entry_price - 1) * 100, 5),
            "last_5_underwater_closes": five_under,
            "ema10_20_spread_falling": bool(spread_down),
            "extension_qualifies": bool(five_under >= 2 and spread_down),
            "base_trade_return_pct": round(t.ret_net * 100, 5),
            "base_trade_exit_date": t.exit_date,
        })
    return pd.DataFrame(out)


def period_metrics(runs: dict[str, dict], end: str) -> pd.DataFrame:
    windows = [
        ("2000-2009", "2000-01-03", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2021", "2016-01-01", "2021-12-31"),
        ("2022-end", "2022-01-01", end),
    ]
    out = []
    for name, run in runs.items():
        eq = run["equity"].copy()
        eq["Date"] = pd.to_datetime(eq.Date)
        for label, start, finish in windows:
            sample = eq.loc[eq.Date.between(start, finish)]
            previous = eq.loc[eq.Date < pd.Timestamp(start)]
            if sample.empty:
                continue
            initial = float(previous.Equity.iloc[-1]) if len(previous) else 100_000.
            values = pd.concat([pd.Series([initial]), sample.Equity], ignore_index=True)
            out.append({"variant": name, "period": label,
                        "period_return_pct": (values.iloc[-1] / initial - 1) * 100,
                        "period_mdd_pct": (values / values.cummax() - 1).min() * 100})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--update", required=True)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    old = combine_rebased_sources(args.old, args.new, end_date="2026-07-13")
    data, _ = extend_adjusted_history(old, args.update)
    data = data.loc[data.Date <= pd.Timestamp(args.end)].copy()
    if data.empty or str(data.Date.iloc[-1].date()) != args.end:
        raise ValueError("Missing specified completed last day")
    signals = with_month_slope_features(completed_month_signals(data))

    runs = {name: replay_gated(signals, "up_slow_veto_1_abs",
                               defer_rejected_gold=True, n20_day0="all", **options)
            for name, options in VARIANT_ARGS.items()}
    base = runs["V2_frozen"]["trades"]
    summary = pd.DataFrame([{"variant": name, **run["stats"]}
                            for name, run in runs.items()])
    if args.end == "2026-11-09":
        assert len(base) == 52
        assert abs(runs["V2_frozen"]["stats"]["end_equity"] - 835956.9516093095) < .1

    changes = [item for name, run in runs.items() if name != "V2_frozen"
               for item in compare_trades(base, run["trades"], name)]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(changes).to_csv(out / "changed_trades.csv", index=False)
    audit_entry_day(signals, base).to_csv(out / "entry_day_audit.csv", index=False)
    audit_age20(signals, base).to_csv(out / "age20_audit.csv", index=False)
    period_metrics(runs, args.end).to_csv(out / "subperiod_metrics.csv", index=False)
    (out / "methodology.json").write_text(json.dumps({
        "start_date": "2000-01-03", "end_date": args.end,
        "status": "in-sample exploratory; not validated for live use",
        "v2": "up_slow_veto_1_abs, defer gold, N20 day0 all",
        "base_stop_loss": .07, "fee_per_side": .0005,
        "fill": "next regular open except labeled intraday stop approximations",
        "intraday_stop": "entry-day low <= 97% of entry open; assumed threshold fill minus slippage; no minute validation",
        "adaptive_n20": "at age20, last 5 daily closes with >=2 below entry and EMA10/EMA20 spread declines over 5 days, extend only until age25/30/35; fresh cross and losing close still required",
        "variants": VARIANT_ARGS,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary[["variant", "end_equity", "cagr", "mdd", "completed_trades",
                   "payoff_ratio", "worst_trade"]].to_string(index=False))
    print("Changed trade records:", len(changes))


if __name__ == "__main__":
    main()
