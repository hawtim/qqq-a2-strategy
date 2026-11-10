"""Audit the 20 A2 emergency exits at QQQ -3%, volume >= 1.2 * prior20.

Do not change the A2 execution rules or optimize a new threshold here.
Features are strictly available by the event close. Future outcomes are only
used as descriptive labels, never for selecting an in-sample trade.

Run: python research/experiments/a2_3pct_volume12_postshock_attribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import a2_n20 as frozen
from a2_leverage_and_ma_tangle_study import enrich
from a2_ma200_tqqq_and_5pct_crash_stop import run_variant

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
THRESHOLD = 0.03
VOLUME_MULTIPLE = 1.2
FORWARD_WINDOWS = (2, 3, 5, 10, 20)


def get_records() -> tuple[pd.DataFrame, dict]:
    raw, source_meta = frozen.prices()
    df = enrich(raw)
    df["volume_ratio_prior20"] = df.Volume / df.Volume.shift(1).rolling(20).mean()
    df["vol_ratio20"] = df["volume_ratio_prior20"]  # engine's existing field
    df["daily_return"] = df.adj_close.pct_change()
    assert df.Date.is_monotonic_increasing and df.Date.is_unique
    assert df.loc[(df.Date >= frozen.START) & (df.Date <= frozen.END),
                  ["adj_open", "adj_high", "adj_low", "adj_close", "Volume",
                   "volume_ratio_prior20"]].notna().all().all()
    assert (df.loc[df.Date >= frozen.START, "Volume"] > 0).all()
    assert (df.loc[df.Date >= frozen.START, "adj_low"] <=
            df.loc[df.Date >= frozen.START, "adj_high"]).all()

    # No TQQQ data is accessed by the QQQ-only version of the executor.
    base, base_curve, _, base_stats = run_variant(
        df, None, "qqq", None, start=frozen.START, end=frozen.END)
    modified, modified_curve, _, modified_stats = run_variant(
        df, None, "qqq", THRESHOLD, start=frozen.START, end=frozen.END,
        volume_threshold=VOLUME_MULTIPLE)
    assert len(base) == 58 and len(modified) == 59
    shocks = modified[modified.exit_reason == "QQQ 1d crash stop"].copy()
    assert len(shocks) == 20
    assert shocks.entry_date.is_unique
    orig = base.set_index("entry_date")
    dates = pd.Index(df.Date)
    records = []

    for trade in shocks.itertuples(index=False):
        signal_date = pd.Timestamp(trade.exit_signal_date)
        idx = dates.get_loc(signal_date)
        entry_idx = dates.get_loc(pd.Timestamp(trade.entry_date))
        bar = df.iloc[idx]
        assert 0 <= entry_idx <= idx and idx + max(FORWARD_WINDOWS) < len(df)
        assert bar.daily_return <= -THRESHOLD + 1e-12
        assert bar.volume_ratio_prior20 >= VOLUME_MULTIPLE - 1e-12
        assert pd.Timestamp(trade.exit_date) == df.iloc[idx + 1].Date
        prior_close = float(df.iloc[idx - 1].adj_close)
        entry_close_peak = float(df.iloc[entry_idx:idx + 1].adj_close.max())
        row = {
            "entry_date": trade.entry_date,
            "signal_date": trade.exit_signal_date,
            "sell_open_date": trade.exit_date,
            "signal_close": float(bar.adj_close),
            "signal_day_return": float(bar.daily_return),
            "volume_ratio20": float(bar.volume_ratio_prior20),
            "signal_low": float(bar.adj_low),
            "hold_days_before_signal": idx - entry_idx,
            "unrealized_gain_at_signal": float(bar.adj_close / trade.entry_price - 1),
            "drawdown_from_holding_peak_at_signal": float(bar.adj_close / entry_close_peak - 1),
            "ma60_10d_slope": float(bar.ma60 / df.iloc[idx - 10].ma60 - 1),
            "ma60_20d_slope": float(bar.ma60 / df.iloc[idx - 20].ma60 - 1),
            "ma200_20d_slope": float(bar.ma200 / df.iloc[idx - 20].ma200 - 1),
            "close_vs_ma10": float(bar.adj_close / bar.ma10 - 1),
            "close_vs_ma20": float(bar.adj_close / bar.ma20 - 1),
            "close_vs_ma60": float(bar.adj_close / bar.ma60 - 1),
            "close_vs_ma200": float(bar.adj_close / bar.ma200 - 1),
            "ma10_ma20_spread": float(bar.ma10 / bar.ma20 - 1),
            "ma20_ma60_spread": float(bar.ma20 / bar.ma60 - 1),
            "atr14_pct_of_close": float(bar.atr14 / bar.adj_close),
            "prior10d_return_excl_signal": float(
                df.iloc[idx - 1].adj_close / df.iloc[idx - 11].adj_close - 1),
            "crosses_prior20": int(bar.cross_prev20),
            "sell_next_open_vs_signal_close": float(
                df.iloc[idx + 1].adj_open / bar.adj_close - 1),
            "new_trade_net_return": float(trade.ret_net),
            "original_trade_net_return": np.nan,
            "original_exit_date": None,
            "net_trade_improvement_pp": np.nan,
            "saved_trade": None,
        }
        if trade.entry_date in orig.index:
            baseline = orig.loc[trade.entry_date]
            delta = float(trade.ret_net - baseline.ret_net)
            row.update({
                "original_trade_net_return": float(baseline.ret_net),
                "original_exit_date": baseline.exit_date,
                "net_trade_improvement_pp": 100 * delta,
                "saved_trade": "better" if delta > 1e-8 else "worse" if delta < -1e-8 else "unchanged",
            })
        # Future price paths; no forward fields are used above.
        for n in FORWARD_WINDOWS:
            future = df.iloc[idx + 1:idx + n + 1]
            row[f"fwd_close_{n}d"] = float(future.iloc[-1].adj_close / bar.adj_close - 1)
            row[f"fwd_low_{n}d"] = float(future.adj_low.min() / bar.adj_close - 1)
            row[f"fwd_high_{n}d"] = float(future.adj_high.max() / bar.adj_close - 1)
            row[f"new_low_vs_signal_low_{n}d"] = bool(future.adj_low.min() < bar.adj_low)
            row[f"fully_recovered_precollapse_close_{n}d"] = bool(
                (future.adj_close >= prior_close).any())
        first5 = df.iloc[idx + 1:idx + 6]
        next15 = df.iloc[idx + 6:idx + 21]
        row["first5_rebound_then_new_low_by20"] = bool(
            (first5.adj_close > bar.adj_close).any() and
            (next15.adj_low < bar.adj_low).any())
        records.append(row)

    events = pd.DataFrame(records).sort_values("signal_date").reset_index(drop=True)
    scored = events[events.saved_trade.notna()].copy()
    assert len(scored) == 19
    assert int((scored.saved_trade == "better").sum()) == 12
    assert int((scored.saved_trade == "worse").sum()) == 7
    assert (events.fwd_low_20d <= events.fwd_high_20d).all()

    # Only pre-event, qualitative / round-number groupings. An analyst may
    # inspect their associations with the ex-post outcome; no rule is selected.
    gates = {
        "volume>=1.5x": events.volume_ratio20 >= 1.5,
        "volume>=2.0x": events.volume_ratio20 >= 2.0,
        "ma60 slope10<=0": events.ma60_10d_slope <= 0,
        "ma200 slope20<=0": events.ma200_20d_slope <= 0,
        "close<ma60": events.close_vs_ma60 < 0,
        "close<ma200": events.close_vs_ma200 < 0,
        "loss at signal": events.unrealized_gain_at_signal < 0,
        "gain at signal>=8%": events.unrealized_gain_at_signal >= 0.08,
        "holding age<=20": events.hold_days_before_signal <= 20,
        "QQQ day drop>=4%": events.signal_day_return <= -0.04,
    }
    slices = []
    for name, mask in gates.items():
        a = events.loc[mask]
        b = events.loc[~mask]
        for flag, subgroup in ((True, a), (False, b)):
            valid = subgroup[subgroup.saved_trade.notna()]
            slices.append({
                "condition": name,
                "condition_met": flag,
                "events": len(subgroup),
                "scorable_events": len(valid),
                "earlier_exit_better": int((valid.saved_trade == "better").sum()),
                "earlier_exit_worse": int((valid.saved_trade == "worse").sum()),
                "better_rate": float((valid.saved_trade == "better").mean()) if len(valid) else np.nan,
                "median_fwd5": float(subgroup.fwd_close_5d.median()) if len(subgroup) else np.nan,
                "median_fwd20": float(subgroup.fwd_close_20d.median()) if len(subgroup) else np.nan,
                "next10_new_low_count": int(subgroup.new_low_vs_signal_low_10d.sum()),
            })
    summary = pd.DataFrame(slices)

    # Unselected historical exposure is the needed context: a rebound or a
    # lower low can also be common on ordinary days. This is descriptive, not
    # an independent randomized control group (daily windows overlap).
    invested_dates = pd.to_datetime(base_curve.loc[base_curve.Invested, "Date"])
    in_original_position = df.Date.isin(invested_dates)
    date_index = np.arange(len(df))
    eligible_horizon = ((date_index + 20 < len(df)) &
                        (df.Date >= frozen.START).to_numpy() &
                        (df.Date <= frozen.END).to_numpy())
    cohorts = {
        "all_market_days": eligible_horizon,
        "original_a2_position_days": eligible_horizon & in_original_position.to_numpy(),
        "original_a2_position_drop>=3%": (
            eligible_horizon & in_original_position.to_numpy() &
            (df.daily_return <= -0.03).to_numpy()),
        "first_qualifying_stop_events": (
            eligible_horizon & df.Date.isin(pd.to_datetime(events.signal_date)).to_numpy()),
    }
    benchmarks = []
    for label, mask in cohorts.items():
        indices = date_index[mask]
        row = {"cohort": label, "days": len(indices)}
        for n in (5, 10, 20):
            day_returns = np.array([
                df.iloc[i + n].adj_close / df.iloc[i].adj_close - 1 for i in indices])
            new_lows = np.array([
                df.iloc[i + 1:i + n + 1].adj_low.min() < df.iloc[i].adj_low
                for i in indices])
            row[f"fwd{n}_positive_rate"] = float((day_returns > 0).mean())
            row[f"fwd{n}_median_return"] = float(np.median(day_returns))
            row[f"fwd{n}_new_low_rate"] = float(new_lows.mean())
        benchmarks.append(row)
    benchmarks = pd.DataFrame(benchmarks)
    assert int(benchmarks.loc[benchmarks.cohort=="first_qualifying_stop_events", "days"].iloc[0]) == 20
    assert np.isclose(float(benchmarks.loc[benchmarks.cohort=="first_qualifying_stop_events", "fwd5_positive_rate"].iloc[0]),
                      float((events.fwd_close_5d > 0).mean()))

    outputs = {
        "a2_3pct_volume12_postshock_event_attribution.csv": events,
        "a2_3pct_volume12_postshock_feature_slices.csv": summary,
        "a2_3pct_volume12_postshock_exposure_benchmarks.csv": benchmarks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(OUT / name, index=False)

    print("REPRODUCTION")
    print({"source_rows": source_meta["sample_rows"],
           "signal_count": len(events), "matched": len(scored),
           "original_end": round(base_stats["end_equity"], 2),
           "modified_end": round(modified_stats["end_equity"], 2),
           "better": int((scored.saved_trade == "better").sum()),
           "worse": int((scored.saved_trade == "worse").sum())})
    print("\nALL EVENTS")
    important = ["signal_date", "signal_day_return", "volume_ratio20",
                 "ma60_10d_slope", "close_vs_ma60", "unrealized_gain_at_signal",
                 "fwd_close_5d", "fwd_close_10d", "fwd_close_20d",
                 "fwd_low_10d", "saved_trade", "net_trade_improvement_pp"]
    print(events[important].to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    print("\nBINARY FUTURE / PAIRED OUTCOME SUMMARY")
    for n in (5, 10, 20):
        print(n, "fwd close positive", int((events[f"fwd_close_{n}d"] > 0).sum()),
              "new low vs trigger day", int(events[f"new_low_vs_signal_low_{n}d"].sum()),
              "fully recovered prior-day close",int(events[f"fully_recovered_precollapse_close_{n}d"].sum()),
              "median future close return",round(float(events[f"fwd_close_{n}d"].median()),4))
    print("first 5d bounce then lower low day6..20",int(events.first5_rebound_then_new_low_by20.sum()))
    print("\nFEATURE SLICES")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    print("\nBENCHMARKS")
    print(benchmarks.to_string(index=False,float_format=lambda x:f"{x:.3f}"))


if __name__ == "__main__":
    get_records()
