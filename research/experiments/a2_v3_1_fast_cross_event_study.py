"""As-of event audit: after-close EMA10/20 gold cross, next-open forward returns.

For diagnosis only. Future labels are NOT fed into strategy predicates.
No raw OHLCV is written to disk. Correlated and overlapping events are NOT
independent proof of predictive edge.
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
from a2_month_ema5_slope_regime import with_month_slope_features
from a2_v3_1_cross_asset import _load


def classify_monthly_slope(slope: float, prior: float) -> str:
    """Classification knows only the previous TWO fully completed months."""
    if slope < 0 and prior < 0:
        if slope > prior:
            return "negative_decelerating"
        if slope < prior:
            return "negative_accelerating"
        return "negative_equal"
    if slope > 0:
        return "positive_decelerating" if slope < prior else "positive_other"
    return "other"


def event_outcomes(signals: pd.DataFrame, *,
                   start="2000-01-03",
                   horizons=(5, 20, 60)) -> pd.DataFrame:
    """Buy at the next adjusted open; mark only at observed future closes."""
    s = signals.sort_values("Date").reset_index(drop=True)
    output = []
    for i in range(len(s)):
        row = s.iloc[i]
        if row.Date < pd.Timestamp(start) or not bool(row.ema10_20_gold):
            continue
        for h in horizons:
            if i + h >= len(s):
                continue
            entry = float(s.iloc[i + 1].adj_open)
            marks = s.iloc[i + 1:i + h + 1].adj_close.astype(float)
            output.append({
                "signal_date": str(row.Date.date()),
                "entry_date": str(s.iloc[i + 1].Date.date()),
                "horizon_sessions": h,
                "completed_month_bucket": classify_monthly_slope(
                    float(row.known_slope_abs), float(row.known_prev_abs)),
                "slow_bull_at_signal": bool(row.ma20 > row.ma60),
                "month_slope": float(row.known_slope_abs),
                "previous_month_slope": float(row.known_prev_abs),
                "forward_return_open_to_close": float(marks.iloc[-1] / entry - 1),
                "min_close_mark_return": float(marks.min() / entry - 1),
                "era": ("2000s" if row.Date.year < 2010 else
                        "2010s" if row.Date.year < 2020 else "2020s"),
            })
    return pd.DataFrame(output)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (h, bucket), group in events.groupby(
            ["horizon_sessions", "completed_month_bucket"]):
        ret = group.forward_return_open_to_close
        records.append({
            "horizon_sessions": h, "bucket": bucket,
            "count": len(group), "mean_forward_return": float(ret.mean()),
            "median_forward_return": float(ret.median()),
            "positive_share": float((ret > 0).mean()),
            "mean_worst_close_mark": float(group.min_close_mark_return.mean()),
            "year_2000s": int((group.era == "2000s").sum()),
            "year_2010s": int((group.era == "2010s").sum()),
            "year_2020s": int((group.era == "2020s").sum()),
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
    parser.add_argument("--outdir", default="research/results/v3_1_event_local")
    args = parser.parse_args()
    raw, provenance = _load(args)
    signals = with_month_slope_features(completed_month_signals(raw))
    event = event_outcomes(signals)
    stats = summarize_events(event)
    stats.insert(0, "symbol", args.symbol)
    folder = Path(args.outdir)
    folder.mkdir(parents=True, exist_ok=True)
    stats.to_csv(folder / "cross_event_summary.csv", index=False)
    event.to_csv(folder / "cross_event_outcomes.csv", index=False)
    (folder / "metadata.json").write_text(json.dumps({
        "symbol": args.symbol, "asof": args.end,
        "source": provenance, "horizons": [5, 20, 60],
        "entry": "signal after completed close, next completed regular open",
        "evaluation": "hold to forward completed closing mark; NO exits",
        "limitation": "Overlapping correlated events; descriptive conditional returns only",
        "no_raw_ohlcv_output": True,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
