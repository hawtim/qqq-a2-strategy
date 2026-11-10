"""N20 reclaim-window (1-4 weeks) x second-exit MA20/30 or MA20/50.

Independent, post-hoc research. No mutation of the frozen A2 engine.
Only uses completed daily closes, next-session opening fills, and 5bp/side.

Run:
 PYTHONPATH=src python research/experiments/a2_n20_second_stop_windows.py \
   --merged sources/qqq_merged_updated_local.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from a2_n20_price_reclaim_study import Rule, frozen, metrics, simulate
from a2_n20_fast_reentry_and_deadcross_buffer import enriched
from qqq_a2.engine import compute_signals, load_adjusted_csv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "research" / "results" / "reclaim_window_second_stop_2026_11_10"


def one_sided_fisher_greater(a_wins: int, a_size: int, b_wins: int, b_size: int) -> float:
    """Exact null Pr[successes in A >= observed], conditional on totals."""
    wins = a_wins + b_wins
    total = a_size + b_size
    denom = math.comb(total, a_size)
    return sum(math.comb(wins, k) * math.comb(total - wins, a_size - k)
               for k in range(a_wins, min(wins, a_size) + 1)
               if 0 <= a_size - k <= total - wins) / denom


def recovery_statistics(no_limit: pd.DataFrame) -> dict:
    r = no_limit[no_limit.entry_origin == "PRICE_RECLAIM"].copy()
    r["signal_wait_days"] = r.reclaim_wait.astype(int) - 1
    fast = r.loc[r.signal_wait_days <= 5, "ret_net"].to_numpy()
    slow = r.loc[r.signal_wait_days > 5, "ret_net"].to_numpy()
    rng = np.random.default_rng(43)
    bootstrap = np.array([
        rng.choice(fast, len(fast), replace=True).mean() -
        rng.choice(slow, len(slow), replace=True).mean()
        for _ in range(10_000)
    ])
    return {
        "note": "Exploratory, overlapping trades; small n and in-sample selection bias.",
        "reentry_events": len(r),
        "first_week": {"n": len(fast), "winners": int((fast > 0).sum()),
                       "average_ret": float(fast.mean())},
        "after_first_week": {"n": len(slow), "winners": int((slow > 0).sum()),
                             "average_ret": float(slow.mean())},
        "one_sided_fisher_p_fast_has_higher_win_rate": one_sided_fisher_greater(
            int((fast > 0).sum()), len(fast), int((slow > 0).sum()), len(slow)),
        "mean_fast_minus_slow_ret": float(fast.mean() - slow.mean()),
        "bootstrap_95pct_mean_diff_ci": [float(x) for x in np.percentile(
            bootstrap, [2.5, 97.5])],
    }


def run(data_path: Path, outdir: Path) -> tuple[pd.DataFrame, dict]:
    raw = compute_signals(load_adjusted_csv(data_path))
    frozen_trades, frozen_equity, _, _ = frozen.run(raw, warm_start=False)
    df = enriched(raw)
    for period in (30, 50):
        df[f"ma{period}"] = df.adj_close.rolling(period, min_periods=period).mean()

    results: dict[str, pd.DataFrame] = {}
    metrics_rows, changed_rows, trade_frames = [], [], []
    scenarios = [Rule("A2_original")]
    scenarios += [Rule(f"reclaim_{days}d_{guard or 'none'}", "fill_open",
                       wait_days=days, second_guard_ma=guard)
                  for days in (5, 10, 15, 20) for guard in (None, 30, 50)]
    scenarios.append(Rule("reclaim_no_limit", "fill_open"))
    for rule in scenarios:
        t, e, actions, open_ret = simulate(df, rule)
        if rule.name == "A2_original":
            assert t.entry_date.tolist() == frozen_trades.entry_date.tolist()
            assert t.exit_date.tolist() == frozen_trades.exit_date.tolist()
            np.testing.assert_allclose(t.ret_net, frozen_trades.ret_net, rtol=0, atol=1e-12)
            np.testing.assert_allclose(e.Equity, frozen_equity.Equity, rtol=0, atol=1e-6)
        results[rule.name] = t
        row = metrics(rule.name, t, e, actions, open_ret)
        worst = e.loc[e.Drawdown.idxmin()]
        row["worst_dd_date"] = worst.Date
        row["guard_exits"] = int(t.exit_reason.str.startswith("RECLAIM_MA20_").sum())
        metrics_rows.append(row)
        tagged = t.copy()
        tagged.insert(0, "scenario", rule.name)
        trade_frames.append(tagged)

    for days in (5, 10, 15, 20):
        initial = results[f"reclaim_{days}d_none"]
        old = initial.loc[initial.entry_origin == "PRICE_RECLAIM"].set_index("entry_date")
        for guard in (30, 50):
            guarded = results[f"reclaim_{days}d_{guard}"]
            new = guarded.loc[guarded.entry_origin == "PRICE_RECLAIM"].set_index("entry_date")
            assert new.index.equals(old.index), "Guards must never change re-entry signals"
            for date, trade in new.iterrows():
                old_trade = old.loc[date]
                if trade.exit_date == old_trade.exit_date:
                    continue
                changed_rows.append({
                    "window_days": days, "second_guard": guard,
                    "entry_date": date, "original_exit": old_trade.exit_date,
                    "guard_exit": trade.exit_date, "guard_exit_reason": trade.exit_reason,
                    "original_net_return": old_trade.ret_net,
                    "guard_net_return": trade.ret_net,
                    "delta_return_percentage_points": 100 * (trade.ret_net - old_trade.ret_net),
                    "original_was_winner": bool(old_trade.ret_net > 0),
                })

    full = results["reclaim_no_limit"]
    stats = recovery_statistics(full)
    events = full.loc[full.entry_origin == "PRICE_RECLAIM"].copy()
    events["signal_wait_days"] = events.reclaim_wait.astype(int) - 1
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(outdir / "summary.csv", index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(outdir / "trades.csv", index=False)
    pd.DataFrame(changed_rows).to_csv(outdir / "changed_reentries.csv", index=False)
    events.to_csv(outdir / "recovery_events.csv", index=False)
    (outdir / "recovery_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pd.DataFrame(metrics_rows), stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True, type=Path,
                        help="User-supplied, adjusted OHLCV CSV, no download")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary, stats = run(args.merged, args.outdir)
    print(summary[["scenario", "completed", "reentries", "cagr", "mdd",
                   "payoff_common", "pf_common", "end_equity", "guard_exits"]]
          .to_string(index=False, float_format=lambda f: f"{f:.4f}"))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Derived, no raw OHLCV output: {args.outdir}")


if __name__ == "__main__":
    main()
