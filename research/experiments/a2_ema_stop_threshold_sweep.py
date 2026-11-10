"""Predeclared EMA A2-N20 5/6/7/7.5/8/9% loss-stop sensitivity sweep.

Uses the exact frozen EMA signals and execution models defined by the prior
hard-stop experiment. Does not alter baseline entry, exit, or reentry rules.
The threshold 'winner' is descriptive sample-in and cannot be deployed as
out-of-sample evidence. Historical raw OHLCV is never exported.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from qqq_a2.engine import Config, load_adjusted_csv
from a2_ema_comparison import ema_signals, replay_signals
from a2_ema_hard_stop8 import replay_guarded, period_slices, summary_row

THRESHOLDS = (0.05, 0.06, 0.07, 0.075, 0.08, 0.09)
MODES = ("close_next_open", "intraday_gap_aware")
START_SENSITIVITY = ("2004-01-02", "2010-01-04", "2016-01-04")


def pct_label(pct: float) -> str:
    return f"{pct * 100:g}"


def grid(signals: pd.DataFrame, config: Config = Config(),
         stress: bool = True) -> dict[str, dict]:
    """Freeze the exact candidate list before inspecting any result."""
    results = {"EMA_original": replay_signals(signals, config)}
    for pct in THRESHOLDS:
        for mode in MODES:
            label = f"EMA_{mode}_stop{pct_label(pct)}"
            results[label] = replay_guarded(
                signals, config=config, stop_loss=pct, mode=mode)
        if stress:
            label = f"EMA_intraday_slip50bp_stop{pct_label(pct)}"
            results[label] = replay_guarded(
                signals, config=config, stop_loss=pct,
                mode="intraday_gap_aware", slippage_bps=50)
    return results


def make_summary(results: dict, cutoff: str) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        row = summary_row(name, result)
        row["data_asof"] = cutoff
        row["stop_threshold"] = (float(name.split("_stop")[-1]) / 100
                                 if "_stop" in name else None)
        row["stop_execution"] = ("no_stop" if name == "EMA_original" else
                                 "intraday_gap_aware_50bp" if "slip50bp" in name else
                                 "intraday_gap_aware" if "intraday" in name else
                                 "close_next_open")
        # The reusable 8% study names this field for that single test;
        # generalize its semantics for arbitrary stop levels here.
        row["stop_exits_worse_than_threshold"] = row.pop("hard_stop_worse_than_8pct", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def affected_trades(results: dict) -> pd.DataFrame:
    """Changed outcomes by entry date; do not compare ordinal trade IDs."""
    base = results["EMA_original"]["trades"].set_index("entry_date")
    base = base[["exit_date", "exit_reason", "ret_net"]]
    out = []
    for name, result in results.items():
        if name == "EMA_original":
            continue
        t = result["trades"].set_index("entry_date")
        t = t[["exit_date", "exit_reason", "ret_net"]]
        joined = base.join(t, how="outer", lsuffix="_original", rsuffix="_candidate")
        changed = (joined.exit_date_original.ne(joined.exit_date_candidate) |
                   joined.ret_net_original.sub(joined.ret_net_candidate).abs().gt(1e-10))
        for day, row in joined.loc[changed].iterrows():
            out.append({"scenario": name, "entry_date": day,
                        "original_exit": row.exit_date_original,
                        "variant_exit": row.exit_date_candidate,
                        "original_reason": row.exit_reason_original,
                        "variant_reason": row.exit_reason_candidate,
                        "original_net": row.ret_net_original,
                        "variant_net": row.ret_net_candidate,
                        "net_difference": row.ret_net_candidate - row.ret_net_original})
    return pd.DataFrame(out)


def starting_date_sensitivity(signals: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Independent capital reset at later first sessions, retaining warm-up bars.

    This checks dependence on the dotcom entry path, not genuine held-out data:
    the same 2000-2026 history and prechosen threshold grid remain visible.
    """
    out = []
    for start in START_SENSITIVITY:
        results = grid(signals, config=Config(start_date=start), stress=False)
        rows = make_summary(results, cutoff)
        for row in rows.to_dict("records"):
            out.append({"start_date": start, **row})
    return pd.DataFrame(out)


def draw_chart(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = summary.loc[summary.strategy == "EMA_original"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.1))
    styles = {"close_next_open": ("#d16b27", "Close / next open"),
              "intraday_gap_aware": ("#178579", "Intraday / gap-aware"),
              "intraday_gap_aware_50bp": ("#b05bab", "Intraday + 50bp slippage")}
    for mode, (color, label) in styles.items():
        s = summary.loc[summary.stop_execution == mode].sort_values("stop_threshold")
        x = s.stop_threshold * 100
        ax1.plot(x, s.cagr * 100, marker="o", color=color, label=label, lw=2)
        ax2.plot(x, s.mdd * 100, marker="o", color=color, label=label, lw=2)
    ax1.axhline(float(ref.cagr) * 100, color="#416599", ls="--", lw=1.6,
                label="Original EMA (no stop)")
    ax2.axhline(float(ref.mdd) * 100, color="#416599", ls="--", lw=1.6,
                label="Original EMA (no stop)")
    ax1.set_ylabel("CAGR (%)  higher is better")
    ax2.set_ylabel("Max drawdown (%)  closer to 0 is better")
    for axis in (ax1, ax2):
        axis.set_xlabel("Net loss stop threshold (% of entry capital)")
        axis.set_xticks([5, 6, 7, 7.5, 8, 9])
        axis.set_xticklabels(["5", "6", "7", "7.5", "8", "9"])
        axis.grid(alpha=.24)
    ax1.legend(loc="best", fontsize=8)
    fig.suptitle("QQQ EMA A2-N20  |  Fixed stop threshold sensitivity  |  2000-2026", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": "2026-11-10"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "research/results/ema_stop_sweep_2026_11_09")
    parser.add_argument("--chart", type=Path,
                        default=ROOT / "assets/a2_ema_stop_threshold_sweep_2000_2026.svg")
    args = parser.parse_args()
    data = load_adjusted_csv(args.csv)
    data = data.loc[data.Date <= pd.Timestamp(args.end)].reset_index(drop=True)
    if data.empty or data.Date.iloc[-1] != pd.Timestamp(args.end):
        raise ValueError("Requested final daily bar is missing")
    signals = ema_signals(data)
    results = grid(signals)
    summary = make_summary(results, args.end)
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.outdir / "summary.csv", index=False)
    affected_trades(results).to_csv(args.outdir / "changed_trades.csv", index=False)
    period_slices(results).to_csv(args.outdir / "period_slices.csv", index=False)
    starting_date_sensitivity(signals, args.end).to_csv(
        args.outdir / "start_date_sensitivity.csv", index=False)
    for name, result in results.items():
        result["trades"].to_csv(args.outdir / f"{name}_trades.csv", index=False)
    draw_chart(summary, args.chart)
    print(summary[["strategy", "end_equity", "cagr", "mdd", "worst_trade_ret",
                   "completed_trades", "hard_stop_exits", "hard_stop_gap_exits"]].to_string(index=False))


if __name__ == "__main__":
    main()
