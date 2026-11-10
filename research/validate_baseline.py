"""Independent consistency checks for the frozen A2-N20 baseline."""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "research")
import a2_n20 as a2


d, m = a2.prices()
t, equity, stats, active = a2.run(d, warm_start=True)
strict, strict_eq, strict_stats, strict_active = a2.run(d, warm_start=False)
assert len(d) == 6877
assert len(t) == 59 and len(strict) == 58
assert sum(t.entry_origin.str.startswith("WARM_START")) == 1
assert all(t.iloc[1:].entry_origin == "FRESH_GOLD_CROSS")
assert stats["wins"] == 24 and stats["losses"] == 35
assert abs(stats["payoff_ratio"] - 3.95) < .01
assert abs(stats["profit_factor"] - 2.71) < .01
assert abs(stats["mdd"] + .203) < .001
assert abs(stats["cagr"] - .0566) < .0001
assert abs(stats["end_equity"] - 430834) < 50
assert stats["longest_cash_days"] == 262
assert bool(active is not None and strict_active is not None)
assert (d[["adj_open", "adj_high", "adj_low", "adj_close"]] > 0).all().all()
assert (d.adj_high >= d[["adj_open", "adj_close"]].max(axis=1)).all()
assert (d.adj_low <= d[["adj_open", "adj_close"]].min(axis=1)).all()

bydate = d.set_index("Date")
dates = list(d.Date)
pos = {v: i for i, v in enumerate(dates)}

for row in t.itertuples():
    entry_day = pd.Timestamp(row.entry_date)
    exit_day = pd.Timestamp(row.exit_date)
    entry_signal = pd.Timestamp(row.signal_date)
    exit_signal = pd.Timestamp(row.exit_signal_date)
    assert pos[entry_day] == pos[entry_signal] + 1
    assert pos[exit_day] == pos[exit_signal] + 1
    assert abs(bydate.at[entry_day, "adj_open"] - row.entry_price) < 1e-8
    assert abs(bydate.at[exit_day, "adj_open"] - row.exit_price) < 1e-8
    independently_recomputed_return = row.exit_price * (1 - a2.FEE) / (row.entry_price * (1 + a2.FEE)) - 1
    assert abs(independently_recomputed_return - row.ret_net) < 1e-10
    if row.entry_origin == "FRESH_GOLD_CROSS":
        assert bool(bydate.at[entry_signal, "gold"])
    else:
        assert row.id == 1 and entry_day == a2.START
        assert bydate.at[entry_signal, "ma20"] > bydate.at[entry_signal, "ma60"]
    if row.exit_reason == "MA20/60 death cross":
        assert bool(bydate.at[exit_signal, "death"])
    else:
        assert row.exit_reason == "N20 protection"
        elapsed = pos[exit_signal] - pos[entry_day]
        assert 1 <= elapsed <= 20
        assert bool(bydate.at[exit_signal, "protect_cross"])
        assert bydate.at[exit_signal, "adj_close"] < row.entry_price

# A flat-to-flat account balance must equal the sequential product of trade
# returns. The last outstanding position is excluded from this cash balance.
cash_reconstruction = 100000 * float(np.prod(1 + t.ret_net.to_numpy()))
day_before_active = dates[pos[pd.Timestamp(active["entry_date"])] - 1]
observed_cash_before_active = float(equity.loc[equity.Date == str(day_before_active.date()), "Equity"].iloc[0])
assert abs(cash_reconstruction - observed_cash_before_active) < 1e-7

# Check one transaction that distinguishes 20 days AFTER the buy from an
# inclusive 20-day window beginning ON the buy day.
twentytwo_aug = t[t.entry_date == "2022-08-01"].iloc[0]
assert twentytwo_aug.exit_date == "2022-08-30"
assert twentytwo_aug.exit_reason == "N20 protection"

print("PASS: 59 completed (1 synthetic initialization + 58 fresh-cross exits), 1 open")
print("PASS: signal and fill dates, signal predicates, N20 window, fees, cash compounding")
print("PASS: price bounds, equity metrics, overlap and archived baseline within rounding")
print(f"end equity {stats['end_equity']:.6f}; cash before last open {cash_reconstruction:.6f}")
