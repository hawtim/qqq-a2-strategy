"""Selected QQQ EMA V3 research baseline; no live order submission.

V3 adds two independently audited exit guards to the unmodified V2 entry
state machine: conditional extension of REGULAR N20 to day 30, and a -3%
entry-day CLOSE loss signal (next-open fill). The 7% net close stop remains.

Use only lawfully obtained local adjusted QQQ OHLCV. Output contains derived
trade/account research results, never the source daily price bars.
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
from a2_dynamic_n20_and_day0_loss_study import compare_trades
from qqq_a2.engine import Config, combine_rebased_sources
from qqq_a2.updates import extend_adjusted_history


VERSION = "EMA_V3_ADAPTIVE_N20_DAY0_CLOSE3"
ENTRY_GATE = "up_slow_veto_1_abs"
V3_PARAMETERS = {
    "defer_rejected_gold": True,
    "n20_day0": "all",
    "stop_loss": .07,
    "regular_n20_extension": "underwater_2of5_spread_down",
    "extended_n20_days": 30,
    "entry_day_close_guard": .03,
}


def replay_selected_v3(signals: pd.DataFrame,
                       config: Config = Config()) -> dict:
    """The one authoritative selected *research* version; all knobs fixed."""
    return replay_gated(signals, ENTRY_GATE, config=config, **V3_PARAMETERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--update", required=True)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", default="research/results/a2_v3_selected_2026_11_09")
    args = parser.parse_args()

    base = combine_rebased_sources(args.old, args.new, end_date="2026-07-13")
    data, audit = extend_adjusted_history(base, args.update)
    data = data.loc[data.Date <= pd.Timestamp(args.end)].copy()
    if data.empty or str(data.Date.iloc[-1].date()) != args.end:
        raise ValueError("Requested end date is not the last complete price bar")
    signals = with_month_slope_features(completed_month_signals(data))

    v2 = replay_gated(signals, ENTRY_GATE,
                      defer_rejected_gold=True, n20_day0="all")
    v3 = replay_selected_v3(signals)
    # Deliberate negative control only: a grid-selected historical 4% stop
    # must not leak into the V3 reference rules.
    stop4 = replay_gated(signals, ENTRY_GATE, defer_rejected_gold=True,
                         n20_day0="all", stop_loss=.04)

    if args.end == "2026-11-09":
        assert len(v2["trades"]) == len(v3["trades"]) == 52
        assert abs(v2["stats"]["end_equity"] - 835956.9516093095) < .10
        assert abs(v3["stats"]["end_equity"] - 878932.617139986) < .10

    changed = compare_trades(v2["trades"], v3["trades"], VERSION)
    if args.end == "2026-11-09":
        assert [row["entry_date"] for row in changed] == [
            "2000-06-22", "2011-10-14"]

    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"version": "EMA_V2_FROZEN_REFERENCE", **v2["stats"]},
        {"version": VERSION, **v3["stats"]},
        {"version": "STOP4_IN_SAMPLE_NEGATIVE_CONTROL", **stop4["stats"]},
    ])
    summary.to_csv(output / "summary.csv", index=False)
    v3["trades"].to_csv(output / "selected_v3_trades.csv", index=False)
    v3["equity"].to_csv(output / "selected_v3_daily_equity.csv", index=False)
    pd.DataFrame(changed).to_csv(
        output / "changed_trades_vs_v2.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "selected_strategy_version": VERSION,
        "status": "user-selected research/paper baseline; no live-trading authority",
        "selected_at": "2026-11-11", "start_date": "2000-01-03",
        "data_asof": args.end, "ticker": "QQQ", "initial_cash": 100_000,
        "fee_per_side": .0005, "cash_yield": 0,
        "execution": "completed-close signal; next regular-session open fill",
        "parameters": {"entry_gate": ENTRY_GATE, **V3_PARAMETERS},
        "known_drawbacks": [
            "2015 V2 EMA crossing changes with tiny historical adjusted-close perturbation",
            "2011 conditional N20 and globally tighter 4% close stop trigger on the same day",
            "only one V2 trade changed by each new V3 guard; historical sample selection",
            "no intraday/minute execution validation, no tax/interest or extra fill slippage",
        ],
        "stop4": "in-sample negative control only; NOT part of selected V3",
        "data_extension": audit,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(summary[["version", "end_equity", "cagr", "mdd",
                   "completed_trades", "payoff_ratio", "worst_trade"]].to_string(index=False))


if __name__ == "__main__":
    main()
