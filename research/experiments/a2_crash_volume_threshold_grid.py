"""Causal grid: QQQ 1-day close drop 3-6% with QQQ 20d prior volume multiple.

Never chooses thresholds ex post as a final strategy. A2 entry, N20, slow
death and next-*new*-gold reentry remain frozen. Each tested condition is
computed at the signal-day close; all exits fill at next trading-day open.

Run: python research/experiments/a2_crash_volume_threshold_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,str(ROOT/'research'/'experiments'))
import a2_n20 as frozen
from a2_leverage_and_ma_tangle_study import enrich,tqqq_price_map
from a2_ma200_tqqq_and_5pct_crash_stop import run_variant,EXACT_END

OUT=ROOT/'research'/'results'/'experiments_2026_11_09'
THRESHOLDS=(.03,.035,.04,.045,.05,.055,.06)
VOLUME_MULTIPLIERS=(None,1.2,1.5,2.,3.)


def event_dates(df,t,e,levels):
    """Count daily shock observations during original (not modified) A2 exposure.

    This is descriptive, not a chosen signal. Volume average excludes current
    day and returns use only past-to-today close. Event dates can cluster.
    """
    baseline=pd.DataFrame(e).set_index('Date')
    out=[]
    for threshold in THRESHOLDS:
        for vol in VOLUME_MULTIPLIERS:
            eligible=(df.qqq_day_return<=-threshold)
            if vol is not None:eligible &= (df.vol_ratio20>=vol)
            obs=df.loc[eligible,['Date','qqq_day_return','vol_ratio20']]
            inpos=obs[obs.Date.dt.strftime('%Y-%m-%d').isin(baseline[baseline.Invested].index)]
            out.append({'cohort':levels,'drop_pct':threshold,'volume_ratio_min':vol if vol is not None else 0.,
                        'all_market_days':len(obs),'in_baseline_a2_position_days':len(inpos),
                        'example_event_dates':'|'.join(inpos.Date.dt.strftime('%Y-%m-%d').head(8))})
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,_=frozen.prices();df=enrich(raw)
    df['qqq_day_return']=df.adj_close.pct_change()
    df['vol_ratio20']=df.Volume/df.Volume.shift(1).rolling(20).mean()
    assert (df.loc[df.Date=='2020-09-03','vol_ratio20'].iloc[0] > 3.0)
    price=tqqq_price_map(df)

    cohorts=[('full_qqq','qqq',frozen.START,frozen.END),
             ('exact_qqq','qqq',pd.Timestamp('2010-02-11'),EXACT_END),
             ('exact_tqqq','tqqq',pd.Timestamp('2010-02-11'),EXACT_END),
             ('exact_ma200_switch','above200_tqqq',pd.Timestamp('2010-02-11'),EXACT_END)]
    allstats=[];trade_detail=[];events=[]
    for cohort,mode,start,end in cohorts:
        base_t,base_e,_,base_s=run_variant(df,price,mode,None,end=end,start=start)
        base_s.update({'cohort':cohort,'config':'baseline','threshold':0.,'volume_min':0.})
        allstats.append(base_s)
        events.extend(event_dates(df[(df.Date>=start)&(df.Date<=end)],base_t,base_e,cohort))
        for threshold in THRESHOLDS:
            for volume in VOLUME_MULTIPLIERS:
                t,e,_,s=run_variant(df,price,mode,threshold,end=end,start=start,
                                    volume_threshold=volume)
                s.update({'cohort':cohort,'config':f'P{threshold:.3f}_V{volume or 0:.2f}',
                          'threshold':threshold,'volume_min':volume or 0.})
                allstats.append(s)
                if (threshold in (.03,.04,.05) and volume in (None,1.5,2.)):
                    subset=t.copy()
                    subset.insert(0,'cohort',cohort)
                    subset.insert(1,'config',s['config'])
                    trade_detail.append(subset)
    a=pd.DataFrame(allstats)
    b=pd.DataFrame(events)
    t=pd.concat(trade_detail,ignore_index=True)
    a.to_csv(OUT/'a2_crash_volume_grid_metrics.csv',index=False)
    b.to_csv(OUT/'a2_crash_volume_original_exposure_events.csv',index=False)
    t.to_csv(OUT/'a2_crash_volume_focus_trades.csv',index=False)

    # Validation: volume filter MUST be a subset of no-volume trigger for any
    # one cohort/threshold at a fixed price path *before strategy changes*.
    for (cohort,pct),g in b.groupby(['cohort','drop_pct']):
        base=g.loc[g.volume_ratio_min==0,'in_baseline_a2_position_days'].iloc[0]
        assert (g.in_baseline_a2_position_days<=base).all(),(cohort,pct)
    # Original A2 on 2000–2026 and the previous independently audited experiment.
    fbase=a[(a.cohort=='full_qqq')&(a.config=='baseline')].iloc[0]
    _,_,frozen_stats,_=frozen.run(raw)
    np.testing.assert_allclose([fbase.payoff,fbase.mdd,fbase.cagr],
                               [frozen_stats['payoff_ratio'],frozen_stats['mdd'],frozen_stats['cagr']],
                               atol=1e-10)
    old=pd.read_csv(OUT/'a2_ma200_crash_scenario_metrics.csv')
    for cohort,scenario in [('exact_qqq','qqq_crash5'),('exact_tqqq','tqqq_crash5'),
                             ('exact_ma200_switch','above200_tqqq_crash5'),('full_qqq','qqq_full_crash5')]:
        new=a[(a.cohort==cohort)&(a.threshold==.05)&(a.volume_min==0)].iloc[0]
        prev=old[old.scenario==scenario].iloc[0]
        np.testing.assert_allclose([new.payoff,new.mdd,new.cagr,new.end_equity],
                                   [prev.payoff,prev.mdd,prev.cagr,prev.end_equity],atol=1e-10)
    print('FULL QQQ GRID')
    print(a[(a.cohort=='full_qqq')&(a.threshold.isin((0,.03,.035,.04,.045,.05,.055,.06)))]
          [['config','crash_exits','payoff','mdd','cagr','end_equity']].to_string(index=False,
           float_format=lambda x:f'{x:.3f}'))
    print('\n2010-2025 MA200 SWITCH GRID')
    print(a[a.cohort=='exact_ma200_switch'][['config','crash_exits','payoff','mdd','cagr','end_equity']]
          .to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nVOLUME EXAMPLES')
    print(df[df.Date.isin(pd.to_datetime(['2001-05-29','2020-06-11','2020-09-03']))]
          [['Date','qqq_day_return','vol_ratio20']].to_string(index=False,
           float_format=lambda x:f'{x:.3f}'))


if __name__=='__main__':main()
