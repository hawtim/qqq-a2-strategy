"""Study *marginal* holding returns after a volume-confirmed QQQ crash rebound.

This is deliberately independent of A2 signals and tests the hypothesis that
once a rebound retraces X% of a one-day crash, its further upside has poor
payoff. All conditions are observable ex ante and execution is next-day open
for close-confirmed events, or an already resting limit for intraday touches.

Rows represent event/threshold/window/mode combinations, not independent
trades; a 23-session, online nonoverlap tag is provided for robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
THRESHOLDS = (.25, .50, .60, .70, .75, .80, .90, 1.00)
WINDOWS = (2, 5, 10)
HORIZONS = (5, 10, 20)


def event_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    data, _ = frozen.prices()
    data["volume_ratio"] = data.Volume / data.Volume.shift(1).rolling(20).mean()
    rows = []
    shock_rows = []
    last_nonoverlap = -100000
    # 10 observation sessions plus one next-day execution and 20 more sessions.
    for i in range(201, len(data) - max(WINDOWS) - max(HORIZONS) - 2):
        shock = data.iloc[i]
        prev = data.iloc[i - 1]
        if shock.Date < frozen.START or shock.Date > frozen.END:
            continue
        ret = float(shock.adj_close / prev.adj_close - 1)
        volume_ratio = float(shock.volume_ratio)
        if not (ret <= -0.03 and volume_ratio >= 1.2):
            continue

        base = {
            "shock_date": str(shock.Date.date()),
            "shock_return": ret,
            "volume_ratio": volume_ratio,
            "pre_shock_close": float(prev.adj_close),
            "shock_close": float(shock.adj_close),
            "shock_low": float(shock.adj_low),
            "ma20_below_ma60": bool(shock.ma20 < shock.ma60),
            "above_ma200": bool(shock.adj_close > shock.ma200),
            "nonoverlap23": bool(i - last_nonoverlap > 22),
        }
        shock_rows.append(base)
        if base["nonoverlap23"]:
            last_nonoverlap = i

        crash_amount = float(prev.adj_close - shock.adj_close)
        assert crash_amount > 0
        for threshold in THRESHOLDS:
            target = float(shock.adj_close + threshold * crash_amount)
            for mode in ("close_next_open", "intraday_limit"):
                first_hit = None
                for j in range(1, max(WINDOWS) + 1):
                    day = data.iloc[i + j]
                    reached = day.adj_close >= target if mode == "close_next_open" else day.adj_high >= target
                    if reached:
                        first_hit = j
                        break
                if first_hit is None:
                    continue

                hit_day = data.iloc[i + first_hit]
                if mode == "close_next_open":
                    sale_i = i + first_hit + 1
                    sale_price = float(data.iloc[sale_i].adj_open)
                    sale_date = str(data.iloc[sale_i].Date.date())
                    # Same-day opening fill precedes that day's later low.
                    first_risk_i = sale_i
                else:
                    sale_i = i + first_hit
                    sale_price = max(float(hit_day.adj_open), target)
                    sale_date = str(hit_day.Date.date())
                    # We do not know whether that day's low preceded the limit fill.
                    first_risk_i = sale_i + 1

                for window in WINDOWS:
                    if first_hit > window:
                        continue
                    for horizon in HORIZONS:
                        # End price is at the close horizon sessions later. For a
                        # next-open exit, this includes the exit session's close.
                        end_i = sale_i + horizon - (mode == "close_next_open")
                        if end_i >= len(data):
                            continue
                        end_day = data.iloc[end_i]
                        future = data.iloc[first_risk_i:end_i + 1]
                        forward_ret = float(end_day.adj_close / sale_price - 1)
                        min_from_exit = float(future.adj_low.min() / sale_price - 1)
                        max_from_exit = float(future.adj_high.max() / sale_price - 1)
                        rows.append({
                            **base,
                            "mode": mode,
                            "threshold": threshold,
                            "hit_window": window,
                            "hit_delay_sessions": first_hit,
                            "hit_date": str(hit_day.Date.date()),
                            "exit_date": sale_date,
                            "target": target,
                            "sale_price": sale_price,
                            "horizon": horizon,
                            "end_date": str(end_day.Date.date()),
                            "future_return": forward_ret,
                            "future_low_from_exit": min_from_exit,
                            "future_high_from_exit": max_from_exit,
                            "retest_shock_low": bool(future.adj_low.min() < shock.adj_low),
                        })
    return pd.DataFrame(rows), pd.DataFrame(shock_rows)


def metrics(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    filters = {
        "all": lambda d: d.shock_date.notna(),
        "nonoverlap23": lambda d: d.nonoverlap23,
        "bull_cross": lambda d: ~d.ma20_below_ma60,
        "bear_cross": lambda d: d.ma20_below_ma60,
        "bull_cross_nonoverlap23": lambda d: ~d.ma20_below_ma60 & d.nonoverlap23,
        "bear_cross_nonoverlap23": lambda d: d.ma20_below_ma60 & d.nonoverlap23,
        "pre_2010": lambda d: d.shock_date < "2010-01-01",
        "post_2010": lambda d: d.shock_date >= "2010-01-01",
        "post_2010_nonoverlap23": lambda d: (d.shock_date >= "2010-01-01") & d.nonoverlap23,
        "post_2015": lambda d: d.shock_date >= "2015-01-01",
        "post_2015_nonoverlap23": lambda d: (d.shock_date >= "2015-01-01") & d.nonoverlap23,
    }
    for cohort, criterion in filters.items():
        cohort_mask = criterion(df)
        cohort_events = events.loc[criterion(events)]
        for (mode, threshold, window, horizon), g in df.loc[cohort_mask].groupby(
                ["mode", "threshold", "hit_window", "horizon"]):
            returns = g.future_return.to_numpy(dtype=float)
            gains = returns[returns > 0]
            losses = returns[returns < 0]
            rows.append({
                "cohort": cohort,
                "mode": mode,
                "threshold": threshold,
                "hit_window": window,
                "horizon": horizon,
                "shocks_in_cohort": len(cohort_events),
                "hits": len(g),
                "hit_rate": len(g) / len(cohort_events) if len(cohort_events) else np.nan,
                "mean_delay": g.hit_delay_sessions.mean(),
                "continue_win_rate": float(np.mean(returns > 0)),
                "continue_positive_count": int(len(gains)),
                "continue_negative_count": int(len(losses)),
                "mean_further_return": float(returns.mean()),
                "median_further_return": float(np.median(returns)),
                "p10_further_return": float(np.quantile(returns, .1)),
                "p90_further_return": float(np.quantile(returns, .9)),
                "avg_gain": float(np.mean(gains)) if len(gains) else np.nan,
                "avg_loss": float(np.mean(losses)) if len(losses) else np.nan,
                "payoff": float(np.mean(gains) / -np.mean(losses)) if len(gains) and len(losses) else np.nan,
                "profit_factor": float(np.sum(gains) / -np.sum(losses)) if len(losses) else np.nan,
                "retest_low_rate": float(g.retest_shock_low.mean()),
                "mean_worst_excursion": float(g.future_low_from_exit.mean()),
                "p10_worst_excursion": float(g.future_low_from_exit.quantile(.10)),
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, shocks = event_grid()
    stats = metrics(rows, shocks)
    rows.to_csv(OUT / "qqq_shock_recovery_takeprofit_rows.csv", index=False)
    shocks.to_csv(OUT / "qqq_shock_recovery_takeprofit_shocks.csv", index=False)
    stats.to_csv(OUT / "qqq_shock_recovery_takeprofit_stats.csv", index=False)
    print("Shocks", len(shocks), "nonoverlap23", int(shocks.nonoverlap23.sum()),
          "grid rows", len(rows))
    for window in (2, 5, 10):
        print(f"\nCLOSE confirmed: hits within {window} sessions; continuing 20 sessions")
        show = stats.loc[(stats.cohort == "nonoverlap23") &
                         (stats["mode"] == "close_next_open") &
                         (stats.hit_window == window) & (stats.horizon == 20)]
        print(show[["threshold", "shocks_in_cohort", "hits", "continue_positive_count",
                    "continue_win_rate", "mean_further_return", "median_further_return",
                    "payoff", "retest_low_rate", "p10_further_return"]]
              .to_string(index=False, float_format=lambda n: f"{n:.3f}"))
    print("\n80% CLOSE confirmed post-2010 vs pre-2010 (5d window)")
    sel = stats.loc[(stats["mode"] == "close_next_open") &
                    (stats.threshold == .80) & (stats.hit_window == 5) &
                    stats.cohort.isin(["pre_2010", "post_2010", "post_2010_nonoverlap23",
                                     "post_2015", "post_2015_nonoverlap23", "all"])]
    print(sel[["cohort", "horizon", "hits", "continue_positive_count", "continue_win_rate",
               "mean_further_return", "median_further_return", "payoff", "profit_factor",
               "p10_further_return"]].to_string(index=False, float_format=lambda n:f"{n:.3f}"))


if __name__ == "__main__":
    main()
