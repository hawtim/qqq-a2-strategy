"""A2 baseline vs two-day shock observation vs one quick re-entry.

Signals use QQQ adjusted closes and QQQ volume, and fill next trading-day
adjusted opening price. No future observations enter an earlier decision.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research'),str(ROOT/'research'/'experiments')]
import a2_n20 as frozen
from a2_leverage_and_ma_tangle_study import enrich,tqqq_price_map,actual_tqqq_open_or_gap_estimate
from a2_ma200_tqqq_and_5pct_crash_stop import run_variant

OUT=ROOT/'research'/'results'/'experiments_2026_11_09'
FEE=frozen.FEE


def simulate(df,tprice,mechanism='A',asset='QQQ',start=frozen.START,end=frozen.END):
    assert mechanism in ('A','B','C') and asset in ('QQQ','TQQQ')
    cash=100000.;shares=0.;position=None;pending=None
    watch=None;retry=None;retry_consumed=False
    trades=[];events=[];equity=[]

    def price(i,side):
        b=df.iloc[i]
        if asset=='QQQ':return float(b.adj_open if side=='open' else b.adj_close)
        if side=='close':return float(tprice.loc[b.Date,'t_close'])
        value,origin=actual_tqqq_open_or_gap_estimate(tprice,df,i)
        assert origin!='ESTIMATED_3X_OVERNIGHT_GAP',(b.Date,origin)
        return float(value)

    for i,b in enumerate(df.itertuples(index=False)):
        if b.Date<start:continue
        if b.Date>end:break
        day=str(b.Date.date())
        if pending:
            action,signal_i,reason=pending
            op=price(i,'open')
            if action=='buy':
                assert position is None
                debit=cash;shares=cash/(op*(1+FEE));cash=0.
                position={'entry_date':day,'entry_idx':i,'entry_price':op,
                          'entry_signal_date':str(df.iloc[signal_i].Date.date()),
                          'qqq_reference_entry_open':float(b.adj_open),'debit':debit,
                          'entry_reason':reason,'asset':asset}
                if reason=='FAST_RETRY':retry_consumed=True;retry=None
                elif reason=='FRESH_MA20_60_GOLD':retry_consumed=False;retry=None
            else:
                assert position is not None
                cash=shares*op*(1-FEE)
                trades.append({**position,'exit_date':day,'exit_signal_date':str(df.iloc[signal_i].Date.date()),
                               'exit_price':op,'exit_reason':reason,'hold_days':i-position['entry_idx'],
                               'ret_net':cash/position['debit']-1,'equity_after_trade':cash})
                position=None;shares=0.;watch=None
                if reason!='SHOCK_EXIT':retry=None
            events.append({'mechanism':mechanism,'date':day,'action':action,'reason':reason,
                           'signal_date':str(df.iloc[signal_i].Date.date()),'open':op})
            pending=None

        shock=(i>=1 and float(b.adj_close/df.iloc[i-1].adj_close-1)<=-.03
               and np.isfinite(float(b.vol_ratio20)) and float(b.vol_ratio20)>=1.2)
        if position is not None:
            age=i-position['entry_idx']
            n20=(1<=age<=20 and bool(b.protect_cross) and
                 float(b.adj_close)<position['qqq_reference_entry_open'])
            if bool(b.death):
                pending=('sell',i,'MA20/60 death cross');watch=None
            elif n20:
                pending=('sell',i,'N20 protection');watch=None
            elif mechanism=='C' and shock:
                pending=('sell',i,'SHOCK_EXIT')
                retry=({'reference':float(df.iloc[i-1].adj_close),'shock_signal':day}
                       if not retry_consumed else None)
                events.append({'mechanism':mechanism,'date':day,'action':'signal','reason':'SHOCK_EXIT',
                               'signal_date':day,'open':np.nan})
            elif mechanism=='B':
                if watch is not None and i==watch['expiry']:
                    if float(b.adj_close)<watch['shock_close']:
                        pending=('sell',i,'WATCH_2D_EXIT')
                    events.append({'mechanism':mechanism,'date':day,'action':'decision',
                                   'reason':'WATCH_2D_EXIT' if pending else 'WATCH_RECOVERED',
                                   'signal_date':watch['shock_date'],'open':np.nan})
                    watch=None
                elif watch is None and shock:
                    watch={'expiry':i+2,'shock_close':float(b.adj_close),'shock_date':day}
                    events.append({'mechanism':mechanism,'date':day,'action':'signal',
                                   'reason':'WATCH_START','signal_date':day,'open':np.nan})
        else:
            if bool(b.gold):
                pending=('buy',i,'FRESH_MA20_60_GOLD')
            elif mechanism=='C' and retry is not None:
                if bool(b.death):retry=None
                elif (float(b.ma20)>float(b.ma60) and
                      float(b.adj_close)>=retry['reference']):
                    pending=('buy',i,'FAST_RETRY')
                    events.append({'mechanism':mechanism,'date':day,'action':'signal',
                                   'reason':'FAST_RETRY','signal_date':retry['shock_signal'],'open':np.nan})
        value=cash+(shares*price(i,'close') if position else 0.)
        equity.append({'Date':day,'Equity':value,'Invested':position is not None,
                       'Asset':asset if position is not None else 'CASH'})
    t=pd.DataFrame(trades);e=pd.DataFrame(equity)
    e['DD']=e.Equity/e.Equity.cummax()-1
    wins=t.loc[t.ret_net>0,'ret_net'];losses=t.loc[t.ret_net<0,'ret_net']
    years=(pd.Timestamp(e.Date.iloc[-1])-pd.Timestamp(e.Date.iloc[0])).days/365.25
    cash_streak=max_cash=loss_streak=max_losses=0
    for invested in e.Invested:
        cash_streak=0 if invested else cash_streak+1;max_cash=max(max_cash,cash_streak)
    for r in t.ret_net:
        loss_streak=loss_streak+1 if r<0 else 0;max_losses=max(max_losses,loss_streak)
    stat={'mechanism':mechanism,'asset':asset,'start':str(start.date()),'end':str(end.date()),
          'trades':len(t),'wins':len(wins),'losses':len(losses),'winrate':len(wins)/len(t),
          'avg_win':wins.mean(),'avg_loss':losses.mean(),'payoff':wins.mean()/(-losses.mean()),
          'pf':wins.sum()/(-losses.sum()),'mdd':e.DD.min(),
          'mdd_date':e.loc[e.DD.idxmin(),'Date'],
          'cagr':(e.Equity.iloc[-1]/100000)**(1/years)-1,
          'end_equity':e.Equity.iloc[-1],'exposure':e.Invested.mean(),
          'longest_cash_sessions':max_cash,'max_single_loss':t.ret_net.min(),
          'max_consecutive_losses':max_losses,
          'shock_exits':int((t.exit_reason=='SHOCK_EXIT').sum()),
          'watch_exits':int((t.exit_reason=='WATCH_2D_EXIT').sum()),
          'fast_retry_entries':int((t.entry_reason=='FAST_RETRY').sum())+
                               int(position is not None and position['entry_reason']=='FAST_RETRY'),
          'watch_starts':sum(x['reason']=='WATCH_START' for x in events),
          'watch_recoveries':sum(x['reason']=='WATCH_RECOVERED' for x in events),
          'open_asset':asset if position else 'CASH'}
    return t,e,pd.DataFrame(events),stat


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,_=frozen.prices();df=enrich(raw)
    df['vol_ratio20']=df.Volume/df.Volume.shift(1).rolling(20).mean()
    tprice=tqqq_price_map(df)
    all_stats=[];all_trades=[];all_events=[]
    scopes=[('full_qqq','QQQ',frozen.START,frozen.END),
            ('matched_qqq','QQQ',pd.Timestamp('2010-02-11'),pd.Timestamp('2025-10-28')),
            ('matched_tqqq','TQQQ',pd.Timestamp('2010-02-11'),pd.Timestamp('2025-10-28'))]
    for scope,asset,start,end in scopes:
        for m in ('A','B','C'):
            t,e,v,s=simulate(df,tprice,m,asset,start,end)
            s['scope']=scope;all_stats.append(s)
            t.insert(0,'scope',scope);t.insert(1,'mechanism',m);all_trades.append(t)
            v.insert(0,'scope',scope);all_events.append(v)
            e.to_csv(OUT/f'a2_shock_{scope}_{m}_equity.csv',index=False)
            if m=='A':
                standard,eq,_,ss=run_variant(df,tprice,asset_mode=asset.lower(),
                                              start=start,end=end)
                assert len(t)==len(standard),(scope,len(t),len(standard))
                np.testing.assert_allclose(t.ret_net,standard.ret_net,atol=1e-12)
                np.testing.assert_allclose(e.Equity,eq.Equity,atol=1e-8)
                np.testing.assert_allclose([s['payoff'],s['mdd'],s['end_equity']],
                                           [ss['payoff'],ss['mdd'],ss['end_equity']],atol=1e-9)
            print(scope,m,'trades',s['trades'],'payoff',round(s['payoff'],3),
                  'mdd',round(s['mdd'],4),'cagr',round(s['cagr'],4),
                  'end',round(s['end_equity'],2),'shock',s['shock_exits'],
                  'watch',s['watch_exits'],'retry',s['fast_retry_entries'],flush=True)
    metrics=pd.DataFrame(all_stats)
    trades=pd.concat(all_trades,ignore_index=True)
    events=pd.concat(all_events,ignore_index=True)
    metrics.to_csv(OUT/'a2_shock_three_mechanisms_metrics.csv',index=False)
    trades.to_csv(OUT/'a2_shock_three_mechanisms_trades.csv',index=False)
    events.to_csv(OUT/'a2_shock_three_mechanisms_events.csv',index=False)
    print('\nFOCUS 2001/2020')
    sub=trades[(trades.scope=='full_qqq') & (trades.entry_date.isin(['2001-05-10','2020-05-01']))]
    print(sub[['mechanism','entry_date','exit_date','exit_reason','ret_net']].to_string(index=False))


if __name__=='__main__':main()
