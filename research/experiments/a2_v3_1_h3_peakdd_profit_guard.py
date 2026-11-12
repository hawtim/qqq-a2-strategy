"""Exploratory H3 profit-giveback exits (no rule chosen, V3 unchanged).

Input: user-supplied, lawfully obtained adjusted OHLCV. As-of daily close
signals, next regular-session open fills. Running PEAK uses preceding CLOSES;
no future maximum MFE or ex-post Peak DD enters the rule.
Out: derived summary, altered trade ledger, regime outcomes and metadata.
This sample was extensively studied and is NOT untouched out of sample.
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


# These are *comparators*, not an optimizer grid and not promoted settings.
ARMS = {
    "H3_FROZEN": {},
    "TRAIL_8_AFTER_10": {"profit_trail_pct": .08},
    "TRAIL_10_AFTER_10": {"profit_trail_pct": .10},
    "TRAIL_12_AFTER_10": {"profit_trail_pct": .12},
    "TRAIL_8_BELOW_EMA20": {
        "profit_trail_pct": .08, "profit_confirmation": "below_ema20"},
    "TRAIL_8_FAST_DEATH": {
        "profit_trail_pct": .08, "profit_confirmation": "fast_death"},
    "TRAIL_3ATR_BELOW_EMA20": {
        "profit_atr_mult": 3, "profit_confirmation": "below_ema20"},
}

# Known snapshots protect reproducibility, not future expected results.
KNOWN_FROZEN = {
    ("QQQ", "2026-07-13"): (1323429.0451020633, -.2429991081991566, 58),
    ("SPY", "2026-07-23"): (815227.2926550363, -.2597177247450284, 54),
}


def replay(signals: pd.DataFrame, arm: str,
           config: Config = Config()) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown pre-registered arm: {arm}")
    return replay_gated(
        signals, ENTRY_GATE, config=config,
        **V3_PARAMETERS, defer_rejected_fast=True,
        fast_negative_month_mode="accelerating",
        profit_min_gain=.10, **ARMS[arm],
    )


def original_trade_path_delta(baseline: dict, alternative: dict,
                              name: str) -> pd.DataFrame:
    """This is a matched entry-date audit, NOT an additive equity attribution."""
    base = baseline["trades"].set_index("entry_date")
    alt = alternative["trades"].set_index("entry_date")
    x = base[["exit_date", "ret_net"]].add_prefix("h3_").join(
        alt[["exit_date", "ret_net", "exit_reason"]].add_prefix("arm_"),
        how="outer",
    ).reset_index()
    x["arm"] = name
    x["matched_entry"] = x.h3_exit_date.notna() & x.arm_exit_date.notna()
    x["changed"] = (
        x.h3_exit_date.fillna("") != x.arm_exit_date.fillna("")
    ) | (
        x.h3_ret_net.sub(x.arm_ret_net).abs().fillna(float("inf")) > 1e-9
    )
    return x.loc[x.changed].copy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", help="One locally adjusted complete OHLCV CSV")
    p.add_argument("--old", help="Old QQQ source")
    p.add_argument("--new", help="New QQQ source")
    p.add_argument("--update", help="Optional appended adjusted observations")
    p.add_argument("--end", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    data, source = _load(args)
    signals = with_month_slope_features(completed_month_signals(data))
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    runs = {key: replay(signals, key) for key in ARMS}
    frozen = runs["H3_FROZEN"]
    reference = KNOWN_FROZEN.get((args.symbol, args.end))
    if reference is not None:
        expected_equity, expected_mdd, expected_trades = reference
        stats = frozen["stats"]
        assert abs(stats["end_equity"] - expected_equity) < .02
        assert abs(stats["mdd"] - expected_mdd) < 1e-8
        assert stats["completed_trades"] == expected_trades

    summary = []
    era = []
    changes = []
    for key, value in runs.items():
        summary.append({
            **_summary(key, args.symbol, args.end, value),
            "extra_profit_protection_exits": int(
                value["trades"].exit_reason.str.startswith(
                    "research profit giveback").sum()),
        })
        era.extend({"arm": key, "symbol": args.symbol, **r}
                   for r in period_metrics(value["equity"]))
        if key != "H3_FROZEN":
            changes.append(original_trade_path_delta(frozen, value, key))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(era).to_csv(output / "era_comparison.csv", index=False)
    if changes:
        pd.concat(changes, ignore_index=True).to_csv(
            output / "changed_trades.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "symbol": args.symbol,
        "end": args.end, "initial_cash": 100_000,
        "fee_per_side": .0005, "cash_interest": 0, "taxes": "not modeled",
        "additional_slippage": 0,
        "execution": "close signal -> next regular open fill",
        "arms": ARMS, "frozen_base_version": "H3_NEGATIVE_ACCELERATING_MONTH_EMA5_FAST",
        "profit_activation": "prior intratrade closing peak >= entry open * 1.10",
        "trail": "previous attained closing peak (as of today), not ex-post MFE",
        "priority": "7% net close stop > EMA20/60 death > N20 > day0 3% > research giveback",
        "unselected": True,
        "warning": "Both QQQ and SPY influenced prior research; neither is pristine OOS",
        "source": source, "raw_ohlcv_not_exported": True,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary)[["version", "end_equity", "cagr", "mdd",
                                 "completed_trades", "extra_profit_protection_exits"]
                                ].to_string(index=False))


if __name__ == "__main__":
    main()
