"""Test timing and fill invariants of the independent A2 modules."""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0,"research/experiments")
import a2_stops_and_entry_filters as experiments
import a2_n20 as frozen


raw,_ = frozen.prices()
df = experiments.prepare(raw)
dates = list(df.Date)
index = {d:i for i,d in enumerate(dates)}
baseline,curve,stats,_ = frozen.run(raw,warm_start=True)
main,curve2,s,_ = experiments.simulate(df,"BASE")
assert len(main) == 59
assert np.allclose(main.ret_net,baseline.ret_net,atol=1e-12)
assert np.allclose(curve.Equity,curve2.Equity,atol=1e-8)
assert abs(s["payoff"] - stats["payoff_ratio"]) < 1e-10

for name in experiments.CASES:
    if name == "BASE_A2_N20":
        continue
    t,e,met,blocked = experiments.simulate(df,name,**experiments.CASES[name])
    assert (e.Equity > 0).all()
    assert not t.entry_date.duplicated().any()
    assert (t.hold_days > 0).all()
    for trade in t.itertuples():
        entry = pd.Timestamp(trade.entry_date)
        if trade.entry_origin == "WARM_START":
            assert entry == frozen.START
        else:
            assert bool(df.iloc[index[entry]-1].gold)
            assert pd.Timestamp(trade.entry_signal_date) == dates[index[entry]-1]
        exit_date = pd.Timestamp(trade.exit_date)
        assert abs(trade.ret_net - (trade.exit_price*(1-frozen.FEE) /
                                      (trade.entry_price*(1+frozen.FEE)) -1)) < 1e-10
        if trade.exit_reason.startswith("PROFIT_MA_STOP"):
            assert trade.stop_armed_date is not None
            assert pd.Timestamp(trade.stop_armed_date) < exit_date
            assert trade.hold_days > 0
            assert np.isfinite(trade.stop_hit_level) if trade.exit_reason != "PROFIT_MA_CLOSE" else True
            if trade.exit_reason == "PROFIT_MA_STOP_TOUCH":
                bar=df.iloc[index[exit_date]]
                assert bar.adj_low <= trade.stop_hit_level + 1e-8
                assert bar.adj_open > trade.stop_hit_level - 1e-8
                assert abs(trade.exit_price-trade.stop_hit_level) < 1e-8
            if trade.exit_reason == "PROFIT_MA_STOP_GAP":
                bar=df.iloc[index[exit_date]]
                assert bar.adj_open <= trade.stop_hit_level + 1e-8
                assert abs(trade.exit_price-bar.adj_open) < 1e-8
        else:
            assert pd.Timestamp(trade.exit_signal_date)==dates[index[exit_date]-1]
    if name.startswith("F_"):
        gate = experiments.CASES[name]["gate"]
        for row in blocked.itertuples():
            signal = df.iloc[index[pd.Timestamp(row.signal_date)]]
            assert bool(signal.gold) and gate.blocked(signal)
            assert pd.Timestamp(row.intended_entry_date) == dates[index[pd.Timestamp(row.signal_date)]+1]
    if name.startswith("S_"):
        assert len(blocked)==0

print(f"PASS: baseline equivalence and {len(experiments.CASES)-1} independent scenarios")
print("PASS: every entry signal, blocked gold cross, trade return, stop touch/gap fill, no same-day activation")
