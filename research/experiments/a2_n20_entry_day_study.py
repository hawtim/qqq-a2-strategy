"""N20 day-zero protection ablation on QQQ fourth and monthly-defer systems.

The only changed predicate is whether a fresh EMA10/20 bearish cross with a
closing loss on the BUY day may trigger N20. Signals form after the completed
session, orders fill at the NEXT session's adjusted open. No same-close exits.

    python research/experiments/a2_n20_entry_day_study.py \
      --old /local/old/QQQ.csv --new /local/new/QQQ.csv \
      --update /local/update.csv --end 2026-11-09 \
      --outdir research/results/a2_n20_day0_2026_11_09
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from a2_ema_dynamic_n20 import completed_month_signals, replay_dynamic
from a2_ema_hard_stop8 import stop_trigger_price
from a2_month_ema5_slope_regime import replay_gated, with_month_slope_features
from qqq_a2.engine import Config, combine_rebased_sources
from qqq_a2.updates import extend_adjusted_history


CONTEXTS = {
    "fourth_original": {"variant": "baseline", "defer": False},
    "month_slowdown_defer": {"variant": "up_slow_veto_1_abs", "defer": True},
}
MODES = ("none", "fast", "regular", "all")
PERIODS = (
    ("2000-2003", "2000-01-03", "2003-12-31"),
    ("2004-2009", "2004-01-01", "2009-12-31"),
    ("2010-2015", "2010-01-01", "2015-12-31"),
    ("2016-2021", "2016-01-01", "2021-12-31"),
    ("2022-2026", "2022-01-01", "2026-11-09"),
)


def _periods(run: dict, *, config: Config) -> list[dict]:
    equity = run["equity"].copy()
    equity["Date"] = pd.to_datetime(equity.Date)
    result = []
    for name, lo, hi in PERIODS:
        inside = equity.loc[equity.Date.between(lo, hi)]
        if inside.empty:
            continue
        earlier = equity.loc[equity.Date < pd.Timestamp(lo)]
        start_value = float(earlier.Equity.iloc[-1]) if len(earlier) else config.initial_cash
        levels = pd.concat([pd.Series([start_value]), inside.Equity], ignore_index=True)
        result.append({"period": name, "return": levels.iloc[-1] / start_value - 1,
                       "mdd": (levels / levels.cummax() - 1).min()})
    return result


def _trade_differences(base: pd.DataFrame, candidate: pd.DataFrame,
                       *, context: str, mode: str) -> list[dict]:
    by_baseline = {str(t.entry_date): t for t in base.itertuples(index=False)}
    by_candidate = {str(t.entry_date): t for t in candidate.itertuples(index=False)}
    differences = []
    for day in sorted(set(by_baseline) | set(by_candidate)):
        old = by_baseline.get(day)
        new = by_candidate.get(day)
        if (old is not None and new is not None and
            old.exit_date == new.exit_date and
            abs(float(old.ret_net) - float(new.ret_net)) < 1e-12):
            continue
        differences.append({
            "context": context, "mode": mode, "entry_date": day,
            "entry_kind_baseline": getattr(old, "entry_kind", None),
            "entry_kind_new": getattr(new, "entry_kind", None),
            "old_exit_signal": getattr(old, "exit_signal_date", None),
            "new_exit_signal": getattr(new, "exit_signal_date", None),
            "old_exit_date": getattr(old, "exit_date", None),
            "new_exit_date": getattr(new, "exit_date", None),
            "old_reason": getattr(old, "exit_reason", None),
            "new_reason": getattr(new, "exit_reason", None),
            "old_return": getattr(old, "ret_net", None),
            "new_return": getattr(new, "ret_net", None),
            "change_type": "changed_same_entry" if old is not None and new is not None
                           else "baseline_only" if old is not None else "candidate_only",
        })
    return differences


def _day0_exits(signals: pd.DataFrame, base: dict, candidate: dict,
                *, context: str, mode: str, config: Config) -> list[dict]:
    prices = signals.set_index(signals.Date.dt.strftime("%Y-%m-%d"))
    base_by_entry = {str(t.entry_date): t for t in base["trades"].itertuples(index=False)}
    result = []
    for trade in candidate["trades"].itertuples(index=False):
        if not (trade.entry_date == trade.exit_signal_date and
                trade.exit_reason.startswith("N20")):
            continue
        # This is a *day-zero* N20 event, not a later stop forced by the
        # 7%-loss / trend precedence (those never have an 'N20' primary label).
        bar = prices.loc[trade.entry_date]
        assert bool(bar.protect_cross)
        assert float(bar.adj_close) < trade.entry_price
        assert float(bar.adj_close) > stop_trigger_price(
            trade.entry_price, config.fee_per_side, .07)
        prev = base_by_entry.get(trade.entry_date)
        result.append({
            "context": context, "mode": mode,
            "entry_signal_date": trade.entry_signal_date,
            "entry_date": trade.entry_date,
            "entry_kind": trade.entry_kind,
            "day0_close_vs_entry": float(bar.adj_close) / trade.entry_price - 1,
            "next_open_exit": trade.exit_date,
            "new_exit_return": trade.ret_net,
            "old_exit_date": getattr(prev, "exit_date", None),
            "old_exit_reason": getattr(prev, "exit_reason", None),
            "old_exit_return": getattr(prev, "ret_net", None),
            "old_trade_would_have_won": (
                bool(prev.ret_net > 0) if prev is not None else None),
            "old_mfe_close": getattr(prev, "mfe_close", None),
            "old_mae_close": getattr(prev, "mae_close", None),
            "saved_vs_old_pp": (
                (trade.ret_net - prev.ret_net) * 100 if prev is not None else None),
        })
    return result


def _entry_day_opportunities(signals: pd.DataFrame, baseline: dict,
                             *, context: str, config: Config) -> list[dict]:
    """Enumerate *every* historical original entry, including the last open one.

    This separates a day-zero fresh bearish cross from the actionable N20
    subset (close below fill price), avoiding the misleading denominator of
    counting only stops that happened to be fired in modified backtests.
    """
    by_date = signals.set_index(signals.Date.dt.strftime("%Y-%m-%d"))
    entries = baseline["trades"].to_dict("records")
    if baseline["open_position"] is not None:
        entries.append(baseline["open_position"])
    output = []
    for trade in entries:
        row = by_date.loc[trade["entry_date"]]
        p = float(trade["entry_price"])
        close_ret = float(row.adj_close) / p - 1
        cross = bool(row.protect_cross)
        actionable = cross and float(row.adj_close) < p
        hard = float(row.adj_close) <= stop_trigger_price(
            p, config.fee_per_side, .07)
        output.append({
            "context": context,
            "entry_signal_date": trade["entry_signal_date"],
            "entry_date": trade["entry_date"],
            "entry_kind": trade["entry_kind"],
            "entry_day_return_close": close_ret,
            "day0_fresh_ema10_20_death": cross,
            "day0_fresh_death_and_loss": actionable,
            "day0_n20_actionable_primary": actionable and not hard and not bool(row.death),
            "day0_hard_stop_priority": hard,
            "day0_ema20_60_death_priority": bool(row.death),
            "original_exit_date": trade.get("exit_date"),
            "original_trade_net": trade.get("ret_net"),
        })
    return output


def compare(data: pd.DataFrame, *, end: str,
            config: Config = Config()) -> dict:
    data = data.loc[data.Date <= pd.Timestamp(end)].copy()
    if data.empty or str(data.Date.iloc[-1].date()) != end:
        raise ValueError("Final completed session missing: " + end)
    signals = with_month_slope_features(completed_month_signals(data))
    control = replay_dynamic(signals, config=config)
    runs: dict = {}
    metrics, periods, changed, day0, entry_day = [], [], [], [], []
    for context, strategy in CONTEXTS.items():
        runs[context] = {}
        for mode in MODES:
            run = replay_gated(signals, strategy["variant"], config=config,
                               defer_rejected_gold=strategy["defer"],
                               n20_day0=mode)
            runs[context][mode] = run
            if context == "fourth_original" and mode == "none":
                pdt.assert_frame_equal(run["trades"], control["trades"])
                pdt.assert_frame_equal(run["equity"], control["equity"])
            baseline = runs[context]["none"]
            st = run["stats"]
            worst_idx = run["equity"].Drawdown.idxmin()
            metrics.append({
                "context": context, "n20_day0": mode,
                "end": end, **st,
                "mdd_trough_date": run["equity"].loc[worst_idx, "Date"],
                "return_vs_context_baseline": (
                    st["end_equity"] / baseline["stats"]["end_equity"] - 1),
                "return_vs_original_fourth": (
                    st["end_equity"] / control["stats"]["end_equity"] - 1),
            })
            for period in _periods(run, config=config):
                periods.append({"context": context, "n20_day0": mode, **period})
            if mode != "none":
                changed.extend(_trade_differences(
                    baseline["trades"], run["trades"], context=context, mode=mode))
                day0.extend(_day0_exits(
                    signals, baseline, run, context=context, mode=mode, config=config))
        entry_day.extend(_entry_day_opportunities(
            signals, runs[context]["none"], context=context, config=config))

    return {"signals": signals, "runs": runs,
            "metrics": pd.DataFrame(metrics), "periods": pd.DataFrame(periods),
            "changes": pd.DataFrame(changed), "day0": pd.DataFrame(day0),
            "entry_day": pd.DataFrame(entry_day)}


def small_close_perturbation_check(data: pd.DataFrame) -> pd.DataFrame:
    """A one-bar stress test, never a correction to the acquired source.

    The 2015 day-zero cross is extremely narrow. Test tiny hypothetical
    adjusted-close differences; do not infer actual vendor discrepancies.
    """
    focus = pd.Timestamp("2015-08-18")
    if not bool((data.Date == focus).any()):
        return pd.DataFrame()
    records = []
    for perturb in (-.02, -.01, 0., .01, .02):
        scenario = data.copy()
        scenario.loc[scenario.Date == focus, "adj_close"] += perturb
        signals = with_month_slope_features(completed_month_signals(scenario))
        day0 = signals.loc[signals.Date == focus].iloc[0]
        day1 = signals.loc[signals.Date == pd.Timestamp("2015-08-19")].iloc[0]
        for mode in ("none", "fast"):
            run = replay_gated(signals, "up_slow_veto_1_abs",
                               defer_rejected_gold=True, n20_day0=mode)
            match = run["trades"].loc[run["trades"].entry_date == "2015-08-18"]
            trade = match.iloc[0] if not match.empty else None
            records.append({
                "hypothetical_2015_08_18_close_delta_usd": perturb,
                "n20_day0": mode,
                "day0_ema10_minus_20": float(day0.ma10 - day0.ma20),
                "day0_fresh_death": bool(day0.protect_cross),
                "day1_fresh_death": bool(day1.protect_cross),
                "2015_entry_exit": trade.exit_date if trade is not None else None,
                "2015_entry_exit_reason": trade.exit_reason if trade is not None else None,
                "2015_entry_trade_return": trade.ret_net if trade is not None else None,
                "end_equity": run["stats"]["end_equity"],
                "mdd": run["stats"]["mdd"],
            })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--update", type=Path)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.update:
        data, audit = extend_adjusted_history(
            combine_rebased_sources(args.old, args.new, end_date="2026-07-13"),
            str(args.update), frozen_cutoff="2026-07-13", completed_cutoff=args.end)
    else:
        data = combine_rebased_sources(args.old, args.new, end_date=args.end)
        audit = None
    result = compare(data, end=args.end)
    sensitivity = small_close_perturbation_check(data)
    fourth = result["runs"]["fourth_original"]["none"]["stats"]
    if args.end == "2026-11-09":
        assert abs(fourth["end_equity"] - 742702.78) < 15
        assert abs(fourth["mdd"] + .259975) < .0001
        assert fourth["completed_trades"] == 59

    args.outdir.mkdir(exist_ok=True, parents=True)
    for file_name, frame in (
        ("summary.csv", result["metrics"]),
        ("period_metrics.csv", result["periods"]),
        ("changed_trades.csv", result["changes"]),
        ("day0_n20_triggers.csv", result["day0"]),
        ("entry_day_opportunities.csv", result["entry_day"]),
        ("tiny_close_perturbation.csv", sensitivity),
    ):
        frame.to_csv(args.outdir / file_name, index=False)
    for context, group in result["runs"].items():
        for mode, run in group.items():
            run["trades"].to_csv(args.outdir / f"{context}_{mode}_trades.csv",
                                 index=False)
    metadata = {
        "asof": args.end,
        "initial_cash": 100000,
        "execution": "complete-day close signal / next regular session adjusted open",
        "day0_scope": "first entry session only; fresh EMA10/20 death cross and close < adjusted buy open",
        "n20_day1_to_day20": "unchanged; not shortened when day0 covered",
        "priority": "7% close stop > EMA20/60 death > N20",
        "entry_variant": CONTEXTS,
        "switches": MODES,
        "return_basis": "QQQ split/dividend adjusted OHLC, USD, cash earns 0",
        "fee_per_side": .0005,
        "research_status": "in-sample ablation, not live or out-of-sample validation",
        "no_raw_price_data_committed": True,
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("old", args.old), ("new", args.new),
                               ("update", args.update)) if path is not None
        },
        "extension_audit": audit,
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print(result["metrics"][[
        "context", "n20_day0", "end_equity", "cagr", "mdd",
        "completed_trades", "wins", "losses", "payoff_ratio", "profit_factor",
        "worst_trade", "exposure", "return_vs_context_baseline"
    ]].to_string(index=False))
    print("\nNew N20 day-zero exits and original outcomes:\n",
          result["day0"].to_string(index=False))
    print("\nAll changed trade rows:", len(result["changes"]))


if __name__ == "__main__":
    main()
