"""V3.1 cross-asset hypothesis A: deferred FAST crossing, not a V3 replacement.

Use legal local adjusted OHLCV. All conditions use completed bars, fills occur
at the next regular open. Keep selected V3's N20, stop, and slow exit unchanged.
Raw market bars are NEVER written to the output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import replay_gated, with_month_slope_features
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS, replay_selected_v3
from qqq_a2.engine import Config, combine_rebased_sources, load_adjusted_csv
from qqq_a2.updates import extend_adjusted_history


VERSION = "EMA_V3_1_DEFERRED_FAST_CROSS_ASSET"
HYPOTHESIS = (
    "After an actual exit, a fresh EMA10/20 gold cross rejected by the "
    "previous-completed-month EMA5-up condition or monthly slowdown veto "
    "remains eligible while EMA10>EMA20. Buy next regular open when BOTH "
    "monthly conditions recover. Invalidate at a new EMA10/20 death cross. "
    "Never reuse a cross that happened while still invested."
)


def replay_v3_1(signals: pd.DataFrame,
                config: Config = Config()) -> dict:
    """Freeze A. The only change from V3 is deferred rejected FAST entry."""
    return replay_gated(
        signals, ENTRY_GATE, config=config,
        **V3_PARAMETERS, defer_rejected_fast=True,
    )


def _source_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if args.csv and (args.old or args.new or args.update):
        raise ValueError("Use either --csv OR --old/--new with optional --update")
    if args.csv:
        data = load_adjusted_csv(args.csv)
        sources = {"mode": "already_combined_adjusted",
                   "sha256": {"csv": _source_sha256(args.csv)}}
    elif args.old and args.new:
        data = combine_rebased_sources(args.old, args.new,
                                      end_date="2026-07-13")
        sources = {
            "mode": "rebased_qqq_sources",
            "sha256": {"old": _source_sha256(args.old),
                       "new": _source_sha256(args.new)},
        }
        if args.update:
            data, audit = extend_adjusted_history(data, args.update)
            sources["sha256"]["update"] = _source_sha256(args.update)
            sources["extension_audit"] = audit
    else:
        raise ValueError("Provide --csv or BOTH --old and --new")
    if data.Date.duplicated().any():
        raise ValueError("Duplicate dates")
    data = data.loc[data.Date <= pd.Timestamp(args.end)].reset_index(drop=True)
    if data.empty or str(data.Date.iloc[-1].date()) != args.end:
        raise ValueError("Requested --end is missing; no fabricated bars")
    if data.Date.iloc[0] >= pd.Timestamp("2000-01-03"):
        raise ValueError("Insufficient pre-2000 EMA warmup")
    return data, sources


def _summary(name: str, symbol: str, end: str, result: dict) -> dict:
    trades = result["trades"]
    fast = trades[trades.entry_kind == "FAST_EMA10_20_MONTH_EMA5"]
    open_trade = result["open_position"]
    open_fast = bool(open_trade is not None and
                     open_trade["entry_kind"] == "FAST_EMA10_20_MONTH_EMA5")
    return {
        "version": name, "symbol": symbol, "data_asof": end,
        **{k: v for k, v in result["stats"].items()
           if k != "pending_order_at_end"},
        "completed_fast_trades": len(fast),
        "fast_entries_including_open": len(fast) + int(open_fast),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True,
                        help="E.g. QQQ or SPY; not used as a filter")
    parser.add_argument("--csv", help="One locally adjusted Yahoo-style file")
    parser.add_argument("--old", help="Locally held older QQQ history")
    parser.add_argument("--new", help="Locally held newer QQQ history")
    parser.add_argument("--update", help="Optional properly rebased recent OHLCV")
    parser.add_argument("--end", required=True, help="Last COMPLETE YYYY-MM-DD bar")
    parser.add_argument("--outdir", default="research/results/v3_1_cross_asset_local")
    args = parser.parse_args()
    data, source_audit = _load(args)
    signals = with_month_slope_features(completed_month_signals(data))
    config = Config()
    base = replay_selected_v3(signals, config=config)
    candidate = replay_v3_1(signals, config=config)

    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        _summary("EMA_V3_SELECTED_REFERENCE", args.symbol, args.end, base),
        _summary(VERSION, args.symbol, args.end, candidate),
    ]).to_csv(output / "summary.csv", index=False)
    base["trades"].to_csv(output / "v3_base_trades.csv", index=False)
    candidate["trades"].to_csv(output / "v3_1_candidate_trades.csv", index=False)
    base["equity"].to_csv(output / "v3_base_daily_equity.csv", index=False)
    candidate["equity"].to_csv(output / "v3_1_candidate_daily_equity.csv",
                               index=False)

    base_legs = base["trades"].set_index("entry_date")
    candidate_legs = candidate["trades"].set_index("entry_date")
    audit = base_legs[["exit_date", "exit_reason", "ret_net"]].add_prefix(
        "v3_").join(
        candidate_legs[["exit_date", "exit_reason", "ret_net"]].add_prefix(
            "v3_1_"), how="outer"
    ).reset_index()
    changed = (
        audit.v3_exit_date.fillna("") != audit.v3_1_exit_date.fillna("")
    ) | (
        audit.v3_ret_net.sub(audit.v3_1_ret_net).abs().fillna(float("inf")) > 1e-10
    )
    audit.loc[changed].to_csv(output / "changed_trades.csv", index=False)
    candidate["trades"].loc[
        ~candidate["trades"].entry_date.isin(base["trades"].entry_date)
    ].to_csv(output / "candidate_only_completed_trades.csv", index=False)

    metadata = {
        "version": VERSION, "status": "research candidate / NOT selected V3",
        "hypothesis": HYPOTHESIS, "symbol": args.symbol,
        "start": config.start_date, "data_asof": args.end,
        "cash_initial": config.initial_cash, "fee_per_side": config.fee_per_side,
        "cash_yield": 0, "slippage": 0, "taxes": "excluded",
        "execution": "completed close signal / next regular open fill",
        "base_parameters": {"entry_gate": ENTRY_GATE, **V3_PARAMETERS},
        "candidate_only_parameter": {"defer_rejected_fast": True},
        "base_chosen_before_cross_asset_test": True,
        "in_sample_warning": "SPY informed the candidate; QQQ is not a pristine holdout",
        "no_original_market_bars_published": True,
        "source": source_audit,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(pd.read_csv(output / "summary.csv")[
        ["version", "end_equity", "cagr", "mdd", "completed_trades",
         "payoff_ratio", "fast_entries_including_open"]
    ].to_string(index=False))
    print(f"Changed completed-entry rows: {int(changed.sum())}")


if __name__ == "__main__":
    main()
