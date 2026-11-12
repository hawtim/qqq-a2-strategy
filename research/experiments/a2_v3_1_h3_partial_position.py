"""H3 study: partial de-risking vs frozen buy/hold-to-slow-death.

Signals are completed daily bars; ALL buys, trims, restores and exits fill at
NEXT regular trading session's OPEN with 0.05% per-side commissions.
Trimming does not reset original entry date, N20 clock, or 7% close stop.
No raw OHLCV exported, and neither QQQ nor SPY is pristine OOS.
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

# Pre-declared comparisons, not adaptive threshold optimization.
ARMS = {
    "H3_FROZEN": {},
    "CURRENT_1_5_TRIM25_RE5": dict(
        trend_spread_atr_min=1.5, trend_spread_trim_fraction=.25),
    "CURRENT_1_5_TRIM50_RE5": dict(
        trend_spread_atr_min=1.5, trend_spread_trim_fraction=.50),
    "CURRENT_1_5_TRIM75_RE5": dict(
        trend_spread_atr_min=1.5, trend_spread_trim_fraction=.75),
    "CURRENT_1_5_TRIM50_RE10": dict(
        trend_spread_atr_min=1.5, trend_spread_trim_fraction=.50,
        trend_spread_trim_restore="ema10_20_gold"),
    "PEAK40_2_5_TRIM25_RE5": dict(
        trend_spread_atr_min=2.5, trend_spread_lookback=40,
        trend_spread_trim_fraction=.25),
    "PEAK40_2_5_TRIM50_RE5": dict(
        trend_spread_atr_min=2.5, trend_spread_lookback=40,
        trend_spread_trim_fraction=.50),
    "PEAK40_2_5_TRIM75_RE5": dict(
        trend_spread_atr_min=2.5, trend_spread_lookback=40,
        trend_spread_trim_fraction=.75),
    "PEAK40_2_5_TRIM50_RE10": dict(
        trend_spread_atr_min=2.5, trend_spread_lookback=40,
        trend_spread_trim_fraction=.50,
        trend_spread_trim_restore="ema10_20_gold"),
    "PEAK40_2_5_TRIM50_EARLY_RE5": dict(
        trend_spread_atr_min=2.5, trend_spread_lookback=40,
        trend_spread_trim_fraction=.50, trend_spread_exit="early_narrow"),
}
KNOWN_FROZEN = {
    ("QQQ", "2026-07-13"): (1323429.0451020633, -.2429991081991566, 58),
    ("SPY", "2026-07-23"): (815227.2926550363, -.2597177247450284, 54),
}


def replay(signals: pd.DataFrame, name: str,
           config: Config = Config()) -> dict:
    if name not in ARMS:
        raise ValueError(f"Unknown research arm {name}")
    return replay_gated(
        signals, ENTRY_GATE, config=config,
        **V3_PARAMETERS,
        defer_rejected_fast=True,
        fast_negative_month_mode="accelerating",
        trend_spread_min_gain=.10,
        **ARMS[name],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, choices=("QQQ", "SPY"))
    p.add_argument("--csv", help="Local adjusted daily OHLCV")
    p.add_argument("--old", help="Old QQQ daily source")
    p.add_argument("--new", help="New QQQ daily source")
    p.add_argument("--update", help="Optional additional adjusted bars")
    p.add_argument("--end", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    data, provenance = _load(args)
    signals = with_month_slope_features(completed_month_signals(data))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    result = {arm: replay(signals, arm) for arm in ARMS}
    frozen = result["H3_FROZEN"]
    known = KNOWN_FROZEN.get((args.symbol, args.end))
    if known:
        stats = frozen["stats"]
        assert abs(stats["end_equity"] - known[0]) < .02, "H3 terminal drift"
        assert abs(stats["mdd"] - known[1]) < 1e-8, "H3 max-DD drift"
        assert stats["completed_trades"] == known[2], "H3 trade-count drift"

    summary, periods, actions = [], [], []
    for name, r in result.items():
        s = r["stats"]
        orders = (2 * s["completed_trades"]
                  + int(s["active_position_at_end"])
                  + len(r["rebalance_events"]))
        weighted_exposure = (
            float(r["equity"].PositionWeight.mean())
            if "PositionWeight" in r["equity"] else s["exposure"])
        summary.append({
            **_summary(name, args.symbol, args.end, r),
            "order_count": orders,
            "trim_orders": int((r["rebalance_events"].action == "trim").sum())
            if not r["rebalance_events"].empty else 0,
            "restore_orders": int((r["rebalance_events"].action == "restore").sum())
            if not r["rebalance_events"].empty else 0,
            "weighted_exposure": weighted_exposure,
        })
        periods.extend({"symbol": args.symbol, "arm": name, **x}
                       for x in period_metrics(r["equity"]))
        if not r["rebalance_events"].empty:
            rows = r["rebalance_events"].copy()
            rows.insert(0, "symbol", args.symbol)
            rows.insert(1, "arm", name)
            actions.append(rows)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(periods).to_csv(out / "era_comparison.csv", index=False)
    if actions:
        pd.concat(actions, ignore_index=True).to_csv(
            out / "partial_orders.csv", index=False)

    (out / "manifest.json").write_text(json.dumps({
        "symbol": args.symbol, "cutoff": args.end, "sources": provenance,
        "arms": ARMS, "initial_capital": 100000,
        "fee_per_side": .0005, "slippage": 0.,
        "uninvested_cash_interest": 0.,
        "execution": "day t completed close signal -> day t+1 open fill",
        "partial_exit": "fraction of remaining shares, not total portfolio equity",
        "restore": "spend all accumulated cash to restore equity exposure",
        "entry_stop": "unchanged original entry price; trim/refill never resets N20",
        "distance": "(EMA20 - EMA60) / ATR20",
        "distance_lookback": "current day or trailing 40 completed days incl today",
        "peak_gain": "10% from original entry based only on already seen closes",
        "research_only": True, "no_raw_OHLCV_exported": True,
        "OOS": False,
    }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print(pd.DataFrame(summary)[
        ["version", "end_equity", "cagr", "mdd", "completed_trades",
         "trim_orders", "restore_orders", "weighted_exposure"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
