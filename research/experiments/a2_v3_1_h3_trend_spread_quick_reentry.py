"""QQQ/SPY H3 experiment: extended EMA20/60 spread + fast exit/re-entry.

Run on user-supplied OHLCV, not on chart images. All strategies share
the same H3 historical entry/exit baseline; only these experimental
extra exit/re-entry arms are enabled explicitly. CLOSE -> next OPEN.
No future peak, no same-close oracle, no intraday fill assumptions.
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


ARMS = {
    "H3_FROZEN": {},
    "EXIT_FAST_DEATH_NO_RE": {
        "trend_spread_atr_min": 1.5,
        "trend_spread_exit": "fast_death",
    },
    "FAST_DEATH_RE_EMA10_20": {
        "trend_spread_atr_min": 1.5,
        "trend_spread_exit": "fast_death",
        "trend_spread_reentry": "ema10_20_gold",
    },
    "FAST_DEATH_RE_EMA5_10": {
        "trend_spread_atr_min": 1.5,
        "trend_spread_exit": "fast_death",
        "trend_spread_reentry": "ema5_10_gold",
    },
    "EARLY_NARROW_RE_EMA10_20": {
        "trend_spread_atr_min": 1.5,
        "trend_spread_exit": "early_narrow",
        "trend_spread_reentry": "ema10_20_gold",
    },
    "EARLY_NARROW_RE_EMA5_10": {
        "trend_spread_atr_min": 1.5,
        "trend_spread_exit": "early_narrow",
        "trend_spread_reentry": "ema5_10_gold",
    },
    "FAST_DEATH_RE_EMA5_10_D2_5": {
        "trend_spread_atr_min": 2.5,
        "trend_spread_exit": "fast_death",
        "trend_spread_reentry": "ema5_10_gold",
    },
    "FAST_DEATH_RE_EMA5_10_NO_DIST": {
        "trend_spread_atr_min": 0.0,
        "trend_spread_exit": "fast_death",
        "trend_spread_reentry": "ema5_10_gold",
    },
}
KNOWN_BASE = {
    ("QQQ", "2026-07-13"): (1323429.0451020633, -.2429991081991566, 58),
    ("SPY", "2026-07-23"): (815227.2926550363, -.2597177247450284, 54),
}


def replay(signals: pd.DataFrame, arm: str,
           config: Config = Config()) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown trend-spread research arm")
    return replay_gated(
        signals, ENTRY_GATE, config=config, **V3_PARAMETERS,
        defer_rejected_fast=True,
        fast_negative_month_mode="accelerating",
        trend_spread_min_gain=.10,
        **ARMS[arm],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv")
    p.add_argument("--old")
    p.add_argument("--new")
    p.add_argument("--update")
    p.add_argument("--end", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    data, provenance = _load(args)
    signals = with_month_slope_features(completed_month_signals(data))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, eras, modified = [], [], []
    base = replay(signals, "H3_FROZEN")
    known = KNOWN_BASE.get((args.symbol, args.end))
    if known:
        st = base["stats"]
        assert abs(st["end_equity"] - known[0]) < .02
        assert abs(st["mdd"] - known[1]) < 1e-8
        assert st["completed_trades"] == known[2]
    for name in ARMS:
        result = base if name == "H3_FROZEN" else replay(signals, name)
        trades = result["trades"]
        extra_exits = (
            int(trades.exit_reason.str.startswith("research trend spread").sum())
            if not trades.empty else 0
        )
        rows.append({
            **_summary(name, args.symbol, args.end, result),
            "additional_fast_exits": extra_exits,
        })
        eras.extend(
            {"symbol": args.symbol, "arm": name, **r}
            for r in period_metrics(result["equity"]))
        if name != "H3_FROZEN":
            t = trades.loc[trades.exit_reason.str.startswith(
                "research trend spread")].copy()
            if not t.empty:
                t.insert(0, "arm", name)
                modified.append(t)
    pd.DataFrame(rows).to_csv(outdir / "summary.csv", index=False)
    pd.DataFrame(eras).to_csv(outdir / "era_comparison.csv", index=False)
    if modified:
        pd.concat(modified, ignore_index=True).to_csv(
            outdir / "research_early_exit_trades.csv", index=False)
    (outdir / "manifest.json").write_text(json.dumps({
        "symbol": args.symbol, "data_end": args.end,
        "source": provenance, "versions": ARMS,
        "reference": "H3_FROZEN (no experimental flags)",
        "max_atr_spread": "(EMA20 - EMA60) / ATR20",
        "profit_activation": "prior highest held closing mark >= entry open * 1.10",
        "early_narrow": "3 consecutive daily decreases in EMA10-EMA20 spread + price below EMA10 + EMA10 below its value 3 days ago",
        "fast_death": "daily EMA10 crosses below EMA20 on complete closing bar",
        "reentry": "after a specially attributed exit, allow a new fast-cross signal while EMA20>EMA60, until next slow death",
        "fee_each_side": .0005, "next_open_fill": True,
        "raw_ohlcv_exported": False,
        "interpretation": "In-sample ablation only; no OOS or chosen parameters",
    }, ensure_ascii=False, indent=2, default=str) + "\n",
       encoding="utf-8")
    print(pd.DataFrame(rows)[["version", "cagr", "mdd", "completed_trades",
                              "additional_fast_exits"]].to_string(index=False))


if __name__ == "__main__":
    main()
