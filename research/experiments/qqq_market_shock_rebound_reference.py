"""Unconditional QQQ crash-volume two-day rebound reference, separate from A2.

Select all eligible days and a nonoverlapping, online 23-session cooled-down
cohort. Stratify by slow-trend state already known at shock day.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
import a2_n20 as frozen

OUT=ROOT/'research'/'results'/'experiments_2026_11_09'


def collect():
    data,_=frozen.prices()
    data['vol_ratio20']=data.Volume/data.Volume.shift(1).rolling(20).mean()
    records=[];last_nonoverlap=-10**9
    for i in range(60,len(data)-22):
        s=data.iloc[i];pre=data.iloc[i-1]
        if s.Date<frozen.START or s.Date>frozen.END:continue
        if not (s.adj_close/pre.adj_close-1<=-.03 and s.vol_ratio20>=1.2):continue
        future=data.iloc[i+3:i+23]
        t2=data.iloc[i+2]
        fraction=float((t2.adj_close-s.adj_close)/(pre.adj_close-s.adj_close))
        record={'shock_date':str(s.Date.date()),
                'shock_return':float(s.adj_close/pre.adj_close-1),
                'volume_ratio20':float(s.vol_ratio20),
                'recovery_fraction':fraction,
                'above25pct':bool(fraction>=.25),
                'above50pct':bool(fraction>=.50),
                'below75pct':bool(fraction<.75),
                'slow_bear_at_shock':bool(s.ma20<s.ma60),
                'slow_bear_at_obs_end':bool(t2.ma20<t2.ma60),
                'nonoverlap_primary':bool(i-last_nonoverlap>22),
                'forward_break_5d':bool(future.iloc[:5].adj_low.min()<s.adj_low),
                'forward_break_10d':bool(future.iloc[:10].adj_low.min()<s.adj_low),
                'forward_break_20d':bool(future.adj_low.min()<s.adj_low),
                'forward_low_vs_shock_low_20d':float(future.adj_low.min()/s.adj_low-1),
                'forward_break_shock_low_by_3pct_20d':bool(future.adj_low.min()<s.adj_low*.97),
                'forward_break_shock_low_by_5pct_20d':bool(future.adj_low.min()<s.adj_low*.95),
                'forward_ret_20d':float(future.iloc[-1].adj_close/t2.adj_close-1),
                'forward_end_negative_20d':bool(future.iloc[-1].adj_close<t2.adj_close),
                'shock_low':float(s.adj_low),
                'pre_shock_close':float(pre.adj_close),
                'post_observation_close':float(t2.adj_close)}
        records.append(record)
        if record['nonoverlap_primary']:last_nonoverlap=i
    return pd.DataFrame(records)


def describe(df):
    result=[]
    filters={
        'all_events':df.index==df.index,
        'nonoverlap':df.nonoverlap_primary,
        'slow_bear_all':df.slow_bear_at_shock,
        'slow_bear_nonoverlap':df.slow_bear_at_shock & df.nonoverlap_primary,
        'slow_bull_nonoverlap':~df.slow_bear_at_shock & df.nonoverlap_primary,
    }
    for name,mask in filters.items():
        sub=df.loc[mask]
        for gate_name,gate in {'all':sub.index==sub.index,
                               'recovery_ge25':sub.above25pct,
                               'recovery_ge50':sub.above50pct,
                               'recovery_ge50_lt75':sub.above50pct & sub.below75pct,
                               'recovery_lt25':~sub.above25pct}.items():
            x=sub.loc[gate]
            row={'cohort':name,'gate':gate_name,'n':len(x)}
            for h in [5,10,20]:row[f'new_low_{h}d']=int(x[f'forward_break_{h}d'].sum())
            row['negative_close_20d']=int(x.forward_end_negative_20d.sum())
            row['new_low_minus3pct_20d']=int(x.forward_break_shock_low_by_3pct_20d.sum())
            row['new_low_minus5pct_20d']=int(x.forward_break_shock_low_by_5pct_20d.sum())
            row['median_return_20d']=float(x.forward_ret_20d.median()) if len(x) else np.nan
            result.append(row)
    return pd.DataFrame(result)


def main():
    all_events=collect();stats=describe(all_events)
    all_events.to_csv(OUT/'qqq_market_shock_rebound_events.csv',index=False)
    stats.to_csv(OUT/'qqq_market_shock_rebound_stats.csv',index=False)
    assert all_events.shock_date.is_unique
    print('TOTAL event',len(all_events),'nonoverlap',all_events.nonoverlap_primary.sum())
    print(stats[['cohort','gate','n','new_low_5d','new_low_20d',
                 'new_low_minus3pct_20d','new_low_minus5pct_20d','negative_close_20d',
                 'median_return_20d']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))


if __name__=='__main__':main()
