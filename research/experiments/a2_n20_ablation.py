"""Exact N20 ablation: every original entry against the no-N20 counterfactual.

Uses frozen A2 engine, same historical data, fills, fees and starting equity.
The 59 entry dates remain identical, so every trade may be compared directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen


def audit() -> tuple[pd.DataFrame, dict]:
    df, _ = frozen.prices()
    with_stop, _, metrics_with, _ = frozen.run(df, warm_start=True)
    without_stop, _, metrics_without, _ = frozen.run(df, warm_start=True, early_stop=False)

    assert len(with_stop) == len(without_stop) == 59
    assert with_stop.entry_date.tolist() == without_stop.entry_date.tolist()

    cols = ["entry_date", "exit_date", "exit_reason", "ret_net"]
    joined = with_stop[cols].merge(without_stop[cols], on="entry_date", validate="one_to_one",
                                   suffixes=("_N20", "_no_N20"))
    unchanged = joined.exit_reason_N20 != "N20 protection"
    assert np.allclose(joined.loc[unchanged,"ret_net_N20"],
                       joined.loc[unchanged,"ret_net_no_N20"])
    assert (joined.loc[~unchanged,"exit_reason_no_N20"]=="MA20/60 death cross").all()
    joined["delta"] = joined.ret_net_N20 - joined.ret_net_no_N20

    events = joined.loc[~unchanged].copy()
    threshold = 1e-10
    positive = events[events.delta > threshold]
    negative = events[events.delta < -threshold]
    assert len(events) == 26 and len(positive) == 17 and len(negative) == 9
    assert len(with_stop[with_stop.ret_net < 0]) == 35
    assert len(without_stop[without_stop.ret_net < 0]) == 31

    summary = {
        "same_entry_dates": True,
        "N20_exits": len(events),
        "N20_helped": len(positive), "N20_hurt": len(negative),
        "helped_pct": len(positive) / len(events),
        "saved_pp_sum": positive.delta.sum()*100,
        "foregone_pp_sum": negative.delta.sum()*100,
        "net_pp_sum": events.delta.sum()*100,
        "saved_dollars_100k_each": positive.delta.sum()*100000,
        "foregone_dollars_100k_each": negative.delta.sum()*100000,
        "net_dollars_100k_each": events.delta.sum()*100000,
        "end_equity_N20": metrics_with["end_equity"],
        "end_equity_no_N20": metrics_without["end_equity"],
        "end_equity_delta": metrics_with["end_equity"]-metrics_without["end_equity"],
        "mdd_N20":metrics_with["mdd"], "mdd_no_N20":metrics_without["mdd"],
        "payoff_N20":metrics_with["payoff_ratio"],
        "payoff_no_N20":metrics_without["payoff_ratio"],
        "average_loss_N20":metrics_with["avg_loss"],
        "average_loss_no_N20":metrics_without["avg_loss"],
    }
    return events, summary


if __name__ == "__main__":
    tab, info = audit()
    print("A2-N20 vs removing N20 (all 59 entries directly matched):")
    for k,v in info.items():
        print(f"{k}: {v}")
    print("\nAll 26 N20 triggers (positive delta means N20 helped):")
    print(tab[["entry_date","exit_date_N20","ret_net_N20","exit_date_no_N20",
               "ret_net_no_N20","delta"]].to_string(index=False))
