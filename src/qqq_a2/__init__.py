"""QQQ A2-N20: a frozen, reproducible daily trend-following strategy."""

from .engine import compute_signals, load_adjusted_csv, run_backtest

__all__ = ["compute_signals", "load_adjusted_csv", "run_backtest"]
__version__ = "1.0.0"
