"""Frozen 25 A2/B shock watches: two-day rebound features vs forward outcomes.

Definition: shock event at t, only data <= t+2 may be used to stratify.
Forward windows begin at t+3 (strictly after two complete observation days).
No thresholds are optimized and no trading signals are changed.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'research'/'experiments')]
import a2_n20 as frozen
from a2_leverage_and_ma_tangle_study import enrich

OUT=ROOT/'research'/'results'/'experiments_2026_11_09'
HORIZONS=(5,10,20)


def get_records():
    raw,source_info=frozen.prices()
    df=enrich(raw)
    df['vol_ratio20']=df.Volume/df.Volume.shift(1).rolling(20).mean()
    prior_events=pd.read_csv(OUT/'a2_shock_three_mechanisms_events.csv')
    bevents=prior_events[(prior_events.scope=='full_qqq') & (prior_events.mechanism=='B')]
    starts=bevents[(bevents.action=='signal') & (bevents.reason=='WATCH_START')]
    decisions=bevents[(bevents.action=='decision') & (bevents.reason.str.startswith('WATCH'))]
    assert len(starts)==25 and starts.date.is_unique
    assert len(decisions)==23 and decisions.signal_date.is_unique
    bydate={str(b.Date.date()):i for i,b in enumerate(df.itertuples(index=False))}
    rows=[]
    for event in starts.itertuples(index=False):
        i=bydate[event.date]
        assert i+2+max(HORIZONS)<len(df)
        pre=df.iloc[i-1]; shock=df.iloc[i]; b1=df.iloc[i+1];b2=df.iloc[i+2]
        assert (shock.adj_close/pre.adj_close-1)<=-.03+1e-10
        assert shock.vol_ratio20>=1.2-1e-10
        drop=float(pre.adj_close-shock.adj_close)
        assert drop>0
        row={'signal_date':event.date,
             'decision_date':str(b2.Date.date()),
             'recorded_B_outcome':'interrupted_by_A2',
             'shock_return':float(shock.adj_close/pre.adj_close-1),
             'shock_volume_ratio':float(shock.vol_ratio20),
             'shock_close':float(shock.adj_close),
             'pre_shock_close':float(pre.adj_close),
             'shock_low':float(shock.adj_low),
             'observation_close':float(b2.adj_close),
             # Fraction of the entire shock close-to-close price gap recovered.
             'recovery_frac_2d':float((b2.adj_close-shock.adj_close)/drop),
             'max_recovery_frac_2d':float((max(b1.adj_close,b2.adj_close)-shock.adj_close)/drop),
             'observation_return':float(b2.adj_close/shock.adj_close-1),
             'rebound_positive_sessions':int(b1.adj_close>shock.adj_close)+int(b2.adj_close>b1.adj_close),
             'rebound_avg_volume_vs_shock':float((b1.Volume+b2.Volume)/(2*shock.Volume)),
             'rebound_avg_volume_vs_pre20':float((b1.Volume+b2.Volume)/(2*(shock.Volume/shock.vol_ratio20))),
             'rebound_day1_volume_vs_shock':float(b1.Volume/shock.Volume),
             'rebound_day2_volume_vs_shock':float(b2.Volume/shock.Volume),
             'above_ma10_day2':bool(b2.adj_close>b2.ma10),
             'above_ma20_day2':bool(b2.adj_close>b2.ma20),
             'above_ma60_day2':bool(b2.adj_close>b2.ma60),
             'ma20_above_ma60_day2':bool(b2.ma20>b2.ma60),
             'ma20_slope10_day2':float(b2.ma20/df.iloc[i-8].ma20-1),
             'ma60_slope10_day2':float(b2.ma60/df.iloc[i-8].ma60-1),
             'two_day_early_break_shock_low':bool(df.iloc[i+1:i+3].adj_low.min()<shock.adj_low)}
        match=decisions[decisions.signal_date==event.date]
        if len(match):
            label=match.iloc[0].reason
            row['recorded_B_outcome']=('recovered' if label=='WATCH_RECOVERED' else 'exit')
            assert row['recorded_B_outcome']==('recovered' if b2.adj_close>=shock.adj_close else 'exit')
        for h in HORIZONS:
            future=df.iloc[i+3:i+3+h]
            assert len(future)==h
            row[f'future_break_shock_low_{h}d']=bool(future.adj_low.min()<shock.adj_low)
            row[f'future_return_{h}d']=float(future.iloc[-1].adj_close/b2.adj_close-1)
            row[f'future_max_gain_{h}d']=float(future.adj_high.max()/b2.adj_close-1)
            row[f'future_min_decline_{h}d']=float(future.adj_low.min()/b2.adj_close-1)
            row[f'future_positive_{h}d']=bool(future.iloc[-1].adj_close>b2.adj_close)
            row[f'future_gain5_any_{h}d']=bool(future.adj_high.max()>=b2.adj_close*1.05)
            row[f'future_gain10_any_{h}d']=bool(future.adj_high.max()>=b2.adj_close*1.10)
            # Alternative: future break of *observation* low, not shock low.
            row[f'future_break_observation_low_{h}d']=bool(future.adj_low.min()<b2.adj_low)
            row[f'future_ma20_60_death_any_{h}d']=bool(future.death.any())
            row[f'future_close_under_ma60_end_{h}d']=bool(future.iloc[-1].adj_close<future.iloc[-1].ma60)
        rows.append(row)
    data=pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True)
    assert data.recorded_B_outcome.value_counts().to_dict()=={'recovered':14,'exit':9,'interrupted_by_A2':2}
    return data,source_info


def comparisons(data):
    results=[]
    groups={
        'B_2d_closing_recovered':data.recovery_frac_2d>=0,
        'B_2d_closing_not_recovered':data.recovery_frac_2d<0,
        'recovered_25pct':data.recovery_frac_2d>=.25,
        'recovered_50pct':data.recovery_frac_2d>=.5,
        'recovered_75pct':data.recovery_frac_2d>=.75,
        'recovered_100pct':data.recovery_frac_2d>=1.,
        'recovered_but_below_MA20':(data.recovery_frac_2d>=0)&(~data.above_ma20_day2),
        'recovered_and_above_MA20':(data.recovery_frac_2d>=0)&(data.above_ma20_day2),
        'rebound_volume_below_shock':data.rebound_avg_volume_vs_shock<1,
        'rebound_volume_atleast_shock':data.rebound_avg_volume_vs_shock>=1,
        'above_ma20_day2':data.above_ma20_day2,
        'below_ma20_day2':~data.above_ma20_day2,
        'above_ma60_day2':data.above_ma60_day2,
        'below_ma60_day2':~data.above_ma60_day2,
    }
    for name,mask in groups.items():
        t=data.loc[mask]
        d={'group':name,'n':len(t),
           'median_recovery_frac_2d':float(t.recovery_frac_2d.median()) if len(t) else np.nan,
           'median_avg_volume_vs_shock':float(t.rebound_avg_volume_vs_shock.median()) if len(t) else np.nan}
        for h in HORIZONS:
            d[f'low_break_{h}d_count']=int(t[f'future_break_shock_low_{h}d'].sum())
            d[f'low_break_{h}d_rate']=float(t[f'future_break_shock_low_{h}d'].mean()) if len(t) else np.nan
            d[f'avg_forward_return_{h}d']=float(t[f'future_return_{h}d'].mean()) if len(t) else np.nan
            d[f'median_forward_return_{h}d']=float(t[f'future_return_{h}d'].median()) if len(t) else np.nan
            d[f'gain5_any_{h}d_count']=int(t[f'future_gain5_any_{h}d'].sum())
            d[f'gain10_any_{h}d_count']=int(t[f'future_gain10_any_{h}d'].sum())
            d[f'ma20_60_death_any_{h}d_count']=int(t[f'future_ma20_60_death_any_{h}d'].sum())
        results.append(d)
    return pd.DataFrame(results)


def statistical_checks(data):
    # The primary (pre-selected) target is a new intraday low in 20 trading
    # days strictly AFTER two observation days; 5/10d are secondary outcomes.
    result=[]
    X={'recovery_frac_2d':data.recovery_frac_2d,
       'max_recovery_frac_2d':data.max_recovery_frac_2d,
       'rebound_avg_volume_vs_shock':data.rebound_avg_volume_vs_shock,
       'rebound_avg_volume_vs_pre20':data.rebound_avg_volume_vs_pre20,
       'ma60_slope10_day2':data.ma60_slope10_day2,
       'shock_return':data.shock_return}
    for name,x in X.items():
        for h in HORIZONS:
            flag=data[f'future_break_shock_low_{h}d'].astype(int)
            rho,p=spearmanr(x,flag)
            result.append({'feature':name,'horizon':h,'rho':rho,'two_sided_pvalue_exploratory':p,
                           'n':len(flag)})
    return pd.DataFrame(result)


def exact_gate_check(data):
    rows=[]
    for name,flag in {
        '2d_close_above_shock_close':data.recovery_frac_2d>=0,
        '2d_recovery_atleast_quarter':data.recovery_frac_2d>=.25,
        '2d_recovery_atleast_half':data.recovery_frac_2d>=.5,
        '2d_recovery_atleast_full':data.recovery_frac_2d>=1,
        '2d_close_above_ma20':data.above_ma20_day2,
        '2d_rebound_volume_below_shock':data.rebound_avg_volume_vs_shock<1,
    }.items():
        for h in HORIZONS:
            y=data[f'future_break_shock_low_{h}d']
            a=int((flag & y).sum());b=int((flag & ~y).sum());c=int((~flag & y).sum());d=int((~flag & ~y).sum())
            _,p=fisher_exact([[a,b],[c,d]],alternative='two-sided')
            rows.append({'gate':name,'horizon':h,'gate_yes_n':a+b,'gate_yes_breaks':a,
                         'gate_no_n':c+d,'gate_no_breaks':c,'fisher_two_sided_p_exploratory':p})
    return pd.DataFrame(rows)


def stability_slices(data):
    """Pre-specified era checks and non-overlapping 20-day event windows."""
    r=[]
    df,_=frozen.prices()
    idx={str(d.date()):i for i,d in enumerate(df.Date)}
    last_included_idx=-10**9
    episodes=[]
    for ev in data.itertuples(index=False):
        j=idx[ev.signal_date]
        if j-last_included_idx>22:
            episodes.append(ev.signal_date)
            last_included_idx=j
    data=data.copy()
    data['episode_representative']=data.signal_date.isin(episodes)
    masks={'all_events':pd.Series(True,index=data.index),
           'start_pre_2010':pd.to_datetime(data.signal_date)<pd.Timestamp('2010-01-01'),
           'start_since_2010':pd.to_datetime(data.signal_date)>=pd.Timestamp('2010-01-01'),
           'one_event_per_23sessions':data.episode_representative}
    for label,mask in masks.items():
        d=data.loc[mask]
        weak=d.recovery_frac_2d<.25
        for h in HORIZONS:
            low=d[f'future_break_shock_low_{h}d']
            # Fisher cannot determine dependence/causality and is unadjusted
            # for multiple thresholds/outcomes; provide as descriptive only.
            table=[[int((weak&low).sum()),int((weak&~low).sum())],
                   [int((~weak&low).sum()),int((~weak&~low).sum())]]
            pv=fisher_exact(table)[1]
            r.append({'cohort':label,'horizon':h,'n':len(d),
                      'weak_under25_n':int(weak.sum()),'weak_under25_breaks':table[0][0],
                      'strong_ge25_n':int((~weak).sum()),'strong_ge25_breaks':table[1][0],
                      'fisher_p_unadjusted_exploratory':pv})
    return pd.DataFrame(r),episodes


def main():
    data,info=get_records()
    compare=comparisons(data)
    correlation=statistical_checks(data)
    gates=exact_gate_check(data)
    stable,episodes=stability_slices(data)
    data.to_csv(OUT/'a2_B_25_rebound_event_rows.csv',index=False)
    compare.to_csv(OUT/'a2_B_25_rebound_group_comparisons.csv',index=False)
    correlation.to_csv(OUT/'a2_B_25_rebound_spearman.csv',index=False)
    gates.to_csv(OUT/'a2_B_25_rebound_fisher.csv',index=False)
    stable.to_csv(OUT/'a2_B_25_rebound_stability_slices.csv',index=False)
    print('Source:',info)
    print('N',len(data),'B decisions:',data.recorded_B_outcome.value_counts().to_dict())
    print('\nB label stats')
    print(data.groupby('recorded_B_outcome').agg(
        n=('signal_date','count'),recovery_med=('recovery_frac_2d','median'),
        volume_med=('rebound_avg_volume_vs_shock','median'),
        above_ma20=('above_ma20_day2','sum'),
        break5=('future_break_shock_low_5d','sum'),
        break10=('future_break_shock_low_10d','sum'),
        break20=('future_break_shock_low_20d','sum'),
        gain5_by20=('future_gain5_any_20d','sum'),
        gain10_by20=('future_gain10_any_20d','sum'),
        death_by20=('future_ma20_60_death_any_20d','sum'),
        fwd20_med=('future_return_20d','median')).to_string(float_format=lambda x:f'{x:.3f}'))
    print('\nSelected groups')
    names=['B_2d_closing_recovered','B_2d_closing_not_recovered','recovered_25pct',
           'recovered_50pct','recovered_75pct','recovered_100pct',
           'recovered_but_below_MA20','recovered_and_above_MA20',
           'rebound_volume_below_shock','rebound_volume_atleast_shock']
    sel=compare[compare.group.isin(names)]
    print(sel[['group','n','low_break_5d_count','low_break_10d_count','low_break_20d_count',
               'median_forward_return_20d','gain5_any_20d_count','gain10_any_20d_count',
               'ma20_60_death_any_20d_count']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nRank correlations')
    print(correlation[correlation.horizon==20].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nFisher')
    print(gates[gates.horizon==20].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nEra / nonoverlap stability')
    print(stable.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('Independent episode representatives:',len(episodes),episodes)
    print('\nALL 25')
    print(data[['signal_date','recorded_B_outcome','shock_return','recovery_frac_2d',
                'rebound_avg_volume_vs_shock','above_ma20_day2','above_ma60_day2',
                'future_break_shock_low_5d','future_break_shock_low_10d',
                'future_break_shock_low_20d','future_return_20d']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))


if __name__=='__main__':main()
