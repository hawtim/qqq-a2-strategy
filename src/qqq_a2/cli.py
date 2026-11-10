"""Read-only reproducible backtesting / signal CLI. No broker orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import Config, combine_rebased_sources, load_adjusted_csv, run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(prog="qqq-a2", description="QQQ A2-N20 frozen rule evaluator (no trading)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--csv", help="User-provided Date/Open/High/Low/Close/Adj Close daily CSV")
    inputs.add_argument("--old", help="Older source CSV, used with --new for rebased source pair")
    parser.add_argument("--new", help="Newer source CSV for rebased pair")
    parser.add_argument("--start", default="2000-01-03", help="Simulation date, with earlier history for SMA warmup")
    parser.add_argument("--end", help="Optional inclusive cutoff date for historical pair")
    parser.add_argument("--out", type=Path, default=Path(".a2-output"), help="Output directory (gitignored)")
    parser.add_argument("--initial-cash", type=float, default=100_000)
    args = parser.parse_args()
    if args.old and not args.new:
        parser.error("--old requires --new")
    if args.new and not args.old:
        parser.error("--new requires --old")
    bars = (load_adjusted_csv(args.csv) if args.csv else
            combine_rebased_sources(args.old, args.new, end_date=args.end))
    result = run_backtest(bars, Config(start_date=args.start, initial_cash=args.initial_cash))
    args.out.mkdir(parents=True, exist_ok=True)
    result["trades"].to_csv(args.out / "trades.csv", index=False)
    result["equity"].to_csv(args.out / "equity.csv", index=False)
    report = {k: v for k, v in result.items() if k not in ("trades", "equity")}
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
