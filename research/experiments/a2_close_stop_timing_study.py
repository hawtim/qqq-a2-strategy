"""QQQ 7% close-stop timing: next-open reality vs same-close oracle bound.

Use locally acquired data only; do not commit raw price history. The
same-close execution deliberately assumes knowledge of the FINAL close and
is NOT an implementable trading signal with a closing-auction market order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from a2_ema_dynamic_n20 import completed_month_signals
from a2_month_ema5_slope_regime import with_month_slope_features, replay_gated
from qqq_a2.engine import combine_rebased_sources
from qqq_a2.updates import extend_adjusted_history


def compare_fills(data: pd.DataFrame) -> dict:
    s = with_month_slope_features(completed_month_signals(data))
    runs = {}
    for strategy, entry_variant, wait in (
        ("fourth", "baseline", False),
        ("up_slow_defer", "up_slow_veto_1_abs", True),
    ):
        runs[strategy + "__next_open"] = replay_gated(
            s, entry_variant, defer_rejected_gold=wait)
        runs[strategy + "__same_close"] = replay_gated(
            s, entry_variant, defer_rejected_gold=wait,
            hard_stop_fill="same_close_oracle")
        for rate in (.0025, .005, .01, .015):
            runs[f"{strategy}__same_close_slippage_{rate:g}"] = replay_gated(
                s, entry_variant, defer_rejected_gold=wait,
                hard_stop_fill="same_close_oracle",
                close_fill_adverse_slippage=rate)

    summary, stops = [], []
    for key, run in runs.items():
        st = run["stats"]
        summary.append({"scenario": key, **st,
                        "stops": int(run["trades"].exit_reason.str.contains(
                            "7% close stop", regex=False).sum())})

    for strategy in ("fourth", "up_slow_defer"):
        original = runs[strategy + "__next_open"]
        oracle = runs[strategy + "__same_close"]
        t0, t1 = original["trades"], oracle["trades"]
        if not t0.entry_signal_date.equals(t1.entry_signal_date):
            raise AssertionError("Different entry-event paths; stop attribution not paired")
        if not t0.exit_signal_date.equals(t1.exit_signal_date):
            raise AssertionError("Different exit-event paths; stop attribution not paired")
        for a, b in zip(t0.itertuples(index=False), t1.itertuples(index=False)):
            if "7% close stop" not in a.exit_reason:
                continue
            if "7% close stop" not in b.exit_reason:
                raise AssertionError("Oracle hard stop changed signal category")
            changed = (1 + b.ret_net) / (1 + a.ret_net)
            stops.append({
                "strategy": strategy,
                "entry_signal_date": a.entry_signal_date,
                "entry_date": a.entry_date,
                "stop_signal_date": a.exit_signal_date,
                "actual_exit_date": a.exit_date,
                "oracle_exit_date": b.exit_date,
                "actual_next_open_price": a.exit_price,
                "oracle_same_close_price": b.exit_price,
                "overnight_open_gap_pct": a.exit_price / b.exit_price - 1,
                "actual_trade_return": a.ret_net,
                "oracle_trade_return": b.ret_net,
                "trade_change_pp": (b.ret_net - a.ret_net) * 100,
                "end_equity_if_only_this_trade_oracle":
                    original["stats"]["end_equity"] * changed,
                "single_trade_end_delta":
                    original["stats"]["end_equity"] * (changed - 1),
            })
    return {"runs": runs, "summary": pd.DataFrame(summary),
            "stops": pd.DataFrame(stops)}


def render_chart(result: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("fourth__next_open", "fourth__same_close",
             "up_slow_defer__next_open", "up_slow_defer__same_close")
    colors = ("#4c8ba0", "#9ac3cf", "#c87938", "#39a883")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    for key, color in zip(names, colors):
        e = result["runs"][key]["equity"].copy()
        e["Date"] = pd.to_datetime(e.Date)
        reduced = e.groupby(e.Date.dt.to_period("W-FRI")).tail(1)
        label = key.replace("__", " / ").replace("_", " ")
        top.plot(reduced.Date, reduced.Equity, lw=1.5, color=color, label=label)
        bottom.plot(reduced.Date, reduced.Drawdown * 100,
                    lw=1.2, color=color)
    top.set_yscale("log")
    top.set_ylabel("Account equity ($, log)")
    bottom.set_ylabel("Drawdown (%)")
    bottom.set_xlabel("QQQ trading session")
    top.legend(fontsize=9, ncol=2)
    for ax in (top, bottom):
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("7% close stop: next-open fill vs hindsight same-close fill")
    fig.text(.07, .01,
             "$100k initial; 0.05% per side; oracle sees finalized close, so NOT executable as-is.",
             fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96])
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, format="svg")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--update", required=True, type=Path)
    p.add_argument("--end", default="2026-11-09")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    frozen = combine_rebased_sources(args.old, args.new, end_date="2026-07-13")
    data, audit = extend_adjusted_history(
        frozen, str(args.update), frozen_cutoff="2026-07-13",
        completed_cutoff=args.end)
    result = compare_fills(data)
    args.outdir.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(args.outdir / "summary.csv", index=False)
    result["stops"].to_csv(args.outdir / "stop_trade_audit.csv", index=False)
    for key in ("fourth__next_open", "fourth__same_close",
                "up_slow_defer__next_open", "up_slow_defer__same_close"):
        result["runs"][key]["trades"].to_csv(
            args.outdir / (key + "_trades.csv"), index=False)
    render_chart(result, args.outdir / "equity_drawdown.svg")
    meta = {
        "end": args.end, "source_update_audit": audit,
        "same_close_oracle": "know finalized close, then transact at it; impossible live without intraday forecast or order committed before close",
        "slippage": "applied ONLY to same-close 7% stop fills, all other fills unchanged",
        "stop_trigger": "realized closed daily bar net loss >= 7%, not intraday touch",
        "other_rules": "identical Dynamic N20 strategy, regular monthly EMA5 fast reentry, and up-slope-veto-defer research candidate",
        "research_status": "in-sample counterfactual, not executable proof",
    }
    (args.outdir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(result["summary"][
        ["scenario", "end_equity", "cagr", "mdd", "completed_trades", "stops"]
    ].to_string(index=False))
    print("\nSTOP EVENTS:\n", result["stops"].to_string(index=False))


if __name__ == "__main__":
    main()
