"""Invariants for A2 leverage attribution and causal N20 tangle sensitivity.

Run: python research/experiments/validate_a2_leverage_tangle.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from a2_leverage_and_ma_tangle_study import OUT, enrich, frozen, simulate


def validate():
    prices,_=frozen.prices()
    x=enrich(prices)
    baseline,equity,_,_=frozen.run(prices)
    sim_t,sim_e=simulate(x)
    assert len(baseline)==len(sim_t)==58
    assert (baseline.entry_date.values==sim_t.entry_date.values).all()
    assert (baseline.exit_date.values==sim_t.exit_date.values).all()
    assert np.allclose(baseline.ret_net,sim_t.ret_net,atol=1e-12)
    assert np.allclose(equity.Equity,sim_e.Equity,atol=1e-8)

    lev=pd.read_csv(OUT/'a2_leverage_all_trades.csv')
    winners=lev[lev.qqq_net>0]
    assert len(lev)==58 and len(winners)==24
    assert len(winners[winners.tqqq_data_quality=='ACTUAL_ADJ_OPEN'])==13
    assert len(winners[winners.tqqq_data_quality=='BEFORE_TQQQ_LAUNCH'])==10
    assert len(winners[winners.tqqq_data_quality=='OPEN_GAP_ESTIMATED'])==1
    assert (winners.loc[winners.entry_date=='2024-09-30','tqqq_actual_prices_net'].iloc[0]<0)

    ev=pd.read_csv(OUT/'a2_n20_tangle_at_stop.csv')
    assert len(ev)==25
    assert int(ev.recover5.sum())==2
    assert int((ev.cross_prev20>=2).sum())==3
    assert int((ev.gap_atr<=.25).sum())==24
    assert ev.loc[ev.entry_date=='2003-03-24','cross_prev20'].iloc[0]==2
    # All entry-time features include only prior dates; verify one manually.
    sig=x.index[x.Date==pd.Timestamp('2003-04-08')][0]
    preceding=x.iloc[sig-20:sig]
    assert int((preceding.protect_cross|preceding.ma10_gold).sum())==2

    metrics=pd.read_csv(OUT/'a2_n20_tangle_delay_metrics.csv')
    assert (metrics.trades==58).all()
    a=metrics.set_index('variant')
    assert np.isclose(a.loc['A2_N20_original','mdd'],sim_e.Drawdown.min())
    assert a.loc['N20_wait8_all','mdd'] < a.loc['A2_N20_original','mdd']

    account=pd.read_csv(OUT/'a2_same_signals_account_metrics.csv')
    exact=account[account.valuation_scope=='exact_open_period'].set_index('asset')
    assert exact.loc['QQQ','mdd'] > exact.loc['TQQQ','mdd']
    assert exact.loc['QQQ','end'] == exact.loc['TQQQ','end'] == '2025-10-28'
    print('PASS: baseline parity, 24 winners, TQQQ quality, 25 N20 events, tangle delay, same-date accounts')


if __name__=='__main__':
    validate()
