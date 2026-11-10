"""Two independent A2 experiments: entry-day MA200 asset choice and 1d -5% crash exits.

Only true MA20/60 gold crosses start positions, only new true gold crosses
restart after any exit. N20 is unchanged. A crash stop is QQQ adjusted-close
day return <= -5% (not the held asset's return), next trading-day OPEN sell.
No reversal to short. QQQ vs TQQQ asset is fixed for the whole trade.

Run: python research/experiments/a2_ma200_tqqq_and_5pct_crash_stop.py
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
from a2_leverage_and_ma_tangle_study import enrich,tqqq_price_map,actual_tqqq_open_or_gap_estimate

OUT=ROOT/'research'/'results'/'experiments_2026_11_09'
EXACT_END=pd.Timestamp('2025-10-28')
FEE=frozen.FEE


def run_variant(df, tprice, asset_mode='qqq', crash_threshold=None,
                end=EXACT_END, start=pd.Timestamp('2010-02-11'),volume_threshold=None):
    assert asset_mode in ('qqq','tqqq','above200_tqqq')
    cash=100000.;shares=0.;position=None;pending=None
    equity=[];trades=[];events=[]
    for i,bar in enumerate(df.itertuples(index=False)):
        if bar.Date<start:continue
        if bar.Date>end:break
        day=str(bar.Date.date())
        if pending is not None:
            action,signal_i,reason,choice=pending
            if action=='buy':
                assert position is None and cash>0
                if choice=='QQQ':opening=float(bar.adj_open);origin='actual_qqq'
                else:opening,origin=actual_tqqq_open_or_gap_estimate(tprice,df,i)
                assert np.isfinite(opening) and origin!='ESTIMATED_3X_OVERNIGHT_GAP', (day,origin)
                debit=cash
                shares=cash/(opening*(1+FEE));cash=0.
                position={'asset':choice,'entry_date':day,'entry_idx':i,
                          'entry_signal_date':str(df.iloc[signal_i].Date.date()),
                          'entry_ma200_above':bool(df.iloc[signal_i].adj_close>df.iloc[signal_i].ma200),
                          'entry_price':opening,
                          'qqq_reference_entry_open':float(bar.adj_open),'debit':debit}
                events.append({'date':day,'action':'buy','asset':choice,'reason':reason,'open_price':opening})
            else:
                assert position is not None
                asset=position['asset']
                if asset=='QQQ':opening=float(bar.adj_open)
                else:opening,origin=actual_tqqq_open_or_gap_estimate(tprice,df,i)
                assert np.isfinite(opening) and (asset!='TQQQ' or origin!='ESTIMATED_3X_OVERNIGHT_GAP')
                cash=shares*opening*(1-FEE)
                trades.append({**position,'exit_date':day,
                               'exit_signal_date':str(df.iloc[signal_i].Date.date()),
                               'exit_price':opening,'exit_reason':reason,
                               'hold_days':i-position['entry_idx'],
                               'ret_net':cash/position['debit']-1,
                               'equity_after_trade':cash})
                events.append({'date':day,'action':'sell','asset':asset,'reason':reason,'open_price':opening})
                position=None;shares=0.
            pending=None

        if position is not None:
            age=i-position['entry_idx']
            # Preserve A2's original N20 signal in QQQ terms irrespective
            # of which instrument was bought. This isolates asset selection.
            is_n20=(1<=age<=20 and bool(bar.protect_cross) and
                    float(bar.adj_close)<position['qqq_reference_entry_open'])
            if bool(bar.death):
                pending=('sell',i,'MA20/60 death cross',position['asset'])
            elif is_n20:
                pending=('sell',i,'N20 protection',position['asset'])
            elif crash_threshold is not None and i>=1:
                qqq_r=float(bar.adj_close/df.iloc[i-1].adj_close-1)
                vol_ok=(volume_threshold is None or
                        (np.isfinite(float(bar.vol_ratio20)) and
                         float(bar.vol_ratio20)>=volume_threshold))
                if qqq_r <= -crash_threshold and vol_ok:
                    pending=('sell',i,'QQQ 1d crash stop',position['asset'])
        elif bool(bar.gold):
            choice=('QQQ' if asset_mode=='qqq' else 'TQQQ' if asset_mode=='tqqq'
                    else 'TQQQ' if float(bar.adj_close)>float(bar.ma200) else 'QQQ')
            pending=('buy',i,'FRESH_MA20_60_GOLD',choice)
        closeprice=float(bar.adj_close) if position is not None and position['asset']=='QQQ' else (
            float(tprice.loc[bar.Date,'t_close']) if position is not None else 0.)
        equity.append({'Date':day,'Equity':cash+shares*closeprice,
                       'Invested':position is not None,
                       'Asset':position['asset'] if position is not None else 'CASH'})
    t=pd.DataFrame(trades);e=pd.DataFrame(equity)
    e['DD']=e.Equity/e.Equity.cummax()-1
    e['Peak']=e.Equity.cummax()
    nwin=int((t.ret_net>0).sum());nloss=int((t.ret_net<0).sum())
    wins=t[t.ret_net>0].ret_net;losses=t[t.ret_net<0].ret_net
    years=(pd.Timestamp(e.Date.iloc[-1])-pd.Timestamp(e.Date.iloc[0])).days/365.25
    longest_cash=streak=0;longest_loss=loss_streak=0
    for state in e.Invested:
        streak=streak+1 if not state else 0;longest_cash=max(longest_cash,streak)
    for ret in t.ret_net:
        loss_streak=loss_streak+1 if ret<0 else 0;longest_loss=max(longest_loss,loss_streak)
    s={'asset_mode':asset_mode,'crash_threshold':crash_threshold if crash_threshold is not None else 0.,
       'volume_threshold':volume_threshold if volume_threshold is not None else 0.,
       'start':str(start.date()),
       'end':str(end.date()),'trades':len(t),'wins':nwin,'losses':nloss,
       'winrate':nwin/len(t),'avg_win':float(wins.mean()),'avg_loss':float(losses.mean()),
       'payoff':float(wins.mean()/(-losses.mean())),'pf':float(wins.sum()/(-losses.sum())),
       'mdd':float(e.DD.min()),'mdd_date':str(e.loc[e.DD.idxmin(),'Date']),
       'cagr':float((e.Equity.iloc[-1]/1e5)**(1/years)-1),
       'end_equity':float(e.Equity.iloc[-1]),'exposure':float(e.Invested.mean()),
       'longest_cash_sessions':longest_cash,'max_single_loss':float(t.ret_net.min()),
       'max_consecutive_losses':longest_loss,
       'tqqq_entries':int((t.asset=='TQQQ').sum())+int(position is not None and position['asset']=='TQQQ'),
       'qqq_entries':int((t.asset=='QQQ').sum())+int(position is not None and position['asset']=='QQQ'),
       'crash_exits':int((t.exit_reason=='QQQ 1d crash stop').sum()),
       'n20_exits':int((t.exit_reason=='N20 protection').sum()),
       'open_asset':position['asset'] if position is not None else 'CASH'}
    return t,e,pd.DataFrame(events),s


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,_=frozen.prices();df=enrich(raw);tprice=tqqq_price_map(df)
    df['vol_ratio20']=df.Volume/df.Volume.shift(1).rolling(20).mean()
    allstats=[];alltrades=[];allevents=[]
    modes=('qqq','tqqq','above200_tqqq')
    thresholds=(None,.05)
    cases=[(m,v) for m in modes for v in thresholds]
    cases.extend([('above200_tqqq',p) for p in (.04,.06)])
    cases.extend([('qqq',p) for p in (.04,.06)])
    for mode,stop in cases:
        name=mode+('_base' if stop is None else f'_crash{int(stop*100)}')
        trades,equity,events,stats=run_variant(df,tprice,mode,stop)
        stats['scenario']=name
        allstats.append(stats)
        trades.insert(0,'scenario',name);alltrades.append(trades)
        events.insert(0,'scenario',name);allevents.append(events)
        if name in ('qqq_base','tqqq_base','above200_tqqq_base','above200_tqqq_crash5'):
            equity.to_csv(OUT/f'a2_ma200_crash_{name}_equity.csv',index=False)
    # Full 2000–2026 QQQ-only stress check, independent of nonexistent pre-2010 TQQQ.
    for threshold in (None,.04,.05,.06):
        name='qqq_full_'+('base' if threshold is None else f'crash{int(threshold*100)}')
        tr0,e0,ev0,s0=run_variant(df,tprice,'qqq',threshold,
                                  end=frozen.END,start=frozen.START)
        s0['scenario']=name;allstats.append(s0)
        tr0.insert(0,'scenario',name);alltrades.append(tr0)
        ev0.insert(0,'scenario',name);allevents.append(ev0)
        if threshold is None:
            ft,fe,_,_=frozen.run(raw)
            assert len(tr0)==len(ft)
            np.testing.assert_allclose(tr0.ret_net,ft.ret_net,atol=1e-12)
            np.testing.assert_allclose(e0.Equity,fe.Equity,atol=1e-8)
    mt=pd.DataFrame(allstats)
    tr=pd.concat(alltrades,ignore_index=True)
    ev=pd.concat(allevents,ignore_index=True)
    mt.to_csv(OUT/'a2_ma200_crash_scenario_metrics.csv',index=False)
    tr.to_csv(OUT/'a2_ma200_crash_all_trades.csv',index=False)
    ev.to_csv(OUT/'a2_ma200_crash_all_actions.csv',index=False)
    # Compare originals with earlier independently audited same-signals account.
    prior=pd.read_csv(OUT/'a2_same_signals_account_metrics.csv')
    for mode,asset in [('qqq','QQQ'),('tqqq','TQQQ')]:
        a=mt[mt.scenario==f'{mode}_base'].iloc[0]
        expected=prior[(prior.asset==asset)&(prior.valuation_scope=='exact_open_period')].iloc[0]
        np.testing.assert_allclose([a.mdd,a.cagr,a.end_equity],
                                   [expected.mdd,expected.cagr,expected.end_equity],rtol=0,atol=1e-7)
    # Entry-only MA200 selection changes the traded asset, never A2 timestamps.
    original_qqq=tr[tr.scenario=='qqq_base']
    selected=tr[tr.scenario=='above200_tqqq_base']
    assert original_qqq.entry_date.tolist()==selected.entry_date.tolist()
    assert original_qqq.exit_date.tolist()==selected.exit_date.tolist()
    assert (selected.loc[selected.asset=='TQQQ','entry_ma200_above']).all()
    assert (~selected.loc[selected.asset=='QQQ','entry_ma200_above']).all()
    # Explicitly catch the historical single-day crash signal and next-open fill.
    sub=tr[(tr.scenario=='above200_tqqq_crash5')&(tr.exit_reason=='QQQ 1d crash stop')]
    assert len(sub)==1
    assert (sub.iloc[0].exit_signal_date,sub.iloc[0].exit_date)==('2020-09-03','2020-09-04')
    full5=tr[(tr.scenario=='qqq_full_crash5')&(tr.exit_reason=='QQQ 1d crash stop')]
    assert set(full5.exit_signal_date)=={'2001-05-29','2020-09-03'}
    print(mt[['scenario','trades','tqqq_entries','crash_exits','payoff','pf','mdd','cagr','end_equity','max_consecutive_losses','open_asset']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\n2020-05-01 focused trade')
    print(tr[tr.entry_date=='2020-05-01'][['scenario','asset','entry_ma200_above','exit_date','exit_reason','ret_net']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))


if __name__=='__main__':main()
