"""V3.1 H1/H2: predeclared monthly-slope and slow-regime FAST ablations.

This is an investigative comparison, NOT a selection of the best in-sample
variant. The V3 and V3.1-A references are both kept exactly unchanged.
Q/Q and SPY daily OHLCV are user-provided locally; never publish raw bars.
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
from a2_v3_selected import ENTRY_GATE, V3_PARAMETERS, replay_selected_v3
from a2_v3_1_cross_asset import replay_v3_1, _load, _summary
from qqq_a2.engine import Config


VARIANTS = {
    "V3_FROZEN": {},
    "V3_1_A_FROZEN": {"defer_rejected_fast": True},
    "H1_DOWN_DECELERATING": {
        "defer_rejected_fast": True,
        "fast_negative_month_mode": "decelerating",
    },
    "H1_DOWN_ACCELERATING_CONTROL": {
        "defer_rejected_fast": True,
        "fast_negative_month_mode": "accelerating",
    },
    "H2_DEFERRED_FAST_SLOW_BULL": {
        "defer_rejected_fast": True,
        "deferred_fast_slow_state": "bull",
    },
    "H2_DEFERRED_FAST_SLOW_BEAR": {
        "defer_rejected_fast": True,
        "deferred_fast_slow_state": "bear",
    },
}


def replay_variant(signals: pd.DataFrame, name: str,
                   config: Config = Config()) -> dict:
    """No modification to exits, costs, baseline monthly gate or regular entry."""
    if name not in VARIANTS:
        raise ValueError(f"Unknown research arm: {name}")
    if name == "V3_FROZEN":
        return replay_selected_v3(signals, config)
    if name == "V3_1_A_FROZEN":
        return replay_v3_1(signals, config)
    return replay_gated(signals, ENTRY_GATE, config=config,
                        **V3_PARAMETERS, **VARIANTS[name])


def period_metrics(equity: pd.DataFrame, config: Config = Config()) -> list[dict]:
    """Calendar regimes, ex-post reporting only; not used for trading decisions."""
    periods = [
        ("2000-2003", "2000-01-03", "2003-12-31"),
        ("2004-2009", "2004-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2021", "2016-01-01", "2021-12-31"),
        ("2022-onward", "2022-01-01", "9999-12-31"),
    ]
    e = equity.copy()
    e["Date"] = pd.to_datetime(e.Date)
    records = []
    for period, start, stop in periods:
        part = e[(e.Date >= pd.Timestamp(start)) & (e.Date <= pd.Timestamp(stop))]
        if part.empty:
            continue
        before = e[e.Date < pd.Timestamp(start)]
        start_eq = (float(before.Equity.iloc[-1]) if not before.empty
                    else config.initial_cash)
        cum = float(part.Equity.iloc[-1]) / start_eq - 1
        running_peak = part.Equity.cummax().clip(lower=start_eq)
        local_dd = float((part.Equity / running_peak - 1).min())
        records.append({
            "period": period, "cumulative_return": cum,
            "local_drawdown_with_entry_mark": local_dd,
            "end_date": str(part.Date.iloc[-1].date()),
        })
    return records


def trade_context(signals: pd.DataFrame, name: str, result: dict) -> pd.DataFrame:
    """Derived trade-level evidence; NO OHLCV prices exported."""
    t = result["trades"].copy()
    context = signals.set_index(signals.Date.dt.strftime("%Y-%m-%d"))
    records = []
    for tr in t.itertuples(index=False):
        row = context.loc[tr.entry_signal_date]
        slope = float(row.known_slope_abs)
        previous = float(row.known_prev_abs)
        records.append({
            "variant": name,
            "entry_signal_date": tr.entry_signal_date,
            "entry_date": tr.entry_date,
            "exit_date": tr.exit_date,
            "entry_kind": tr.entry_kind,
            "entry_source": ("FAST_NEW_CROSS" if bool(row.ema10_20_gold) else
                             "FAST_DEFERRED" if "FAST" in tr.entry_kind else
                             "REGULAR"),
            "slow_bull_at_signal": bool(row.ma20 > row.ma60),
            "monthly_slope": slope, "previous_month_slope": previous,
            "negative_decelerating": bool(slope < 0 and previous < 0
                                           and slope > previous),
            "negative_accelerating": bool(slope < 0 and previous < 0
                                           and slope < previous),
            "monthly_gate": bool(row["gate_" + ENTRY_GATE]),
            "monthly_up": bool(row.prev_completed_month_ema5_up),
            "exit_reason": tr.exit_reason,
            "net_trade_return": float(tr.ret_net),
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--csv")
    parser.add_argument("--old")
    parser.add_argument("--new")
    parser.add_argument("--update")
    parser.add_argument("--end", required=True)
    parser.add_argument("--outdir", default="research/results/v3_1_regime_local")
    args = parser.parse_args()
    data, provenance = _load(args)
    signals = with_month_slope_features(completed_month_signals(data))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary, periods, contexts = [], [], []
    for name in VARIANTS:
        result = replay_variant(signals, name)
        summary.append(_summary(name, args.symbol, args.end, result))
        periods.extend({"variant": name, "symbol": args.symbol, **r}
                       for r in period_metrics(result["equity"]))
        contexts.append(trade_context(signals, name, result))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(periods).to_csv(out / "period_metrics.csv", index=False)
    pd.concat(contexts, ignore_index=True).to_csv(
        out / "trade_context.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "symbol": args.symbol, "end": args.end, "variants": VARIANTS,
        "baseline_frozen": True, "no_intraday_fills": True,
        "no_live_execution": True, "source": provenance,
        "hypothesis_h1": "Faster entry on negative but improving completed-month EMA5 slope",
        "negative_control": "Negative and becoming more negative: do not promote based on in-sample ranking",
        "hypothesis_h2": "For deferred FAST only, slow EMA20/60 positive/negative at entry signal",
        "warning": "Two strongly correlated securities and historical parameter iteration are NOT independent out-of-sample validation",
        "raw_market_data_in_output": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary)[["version", "end_equity", "cagr", "mdd",
                                 "completed_trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
