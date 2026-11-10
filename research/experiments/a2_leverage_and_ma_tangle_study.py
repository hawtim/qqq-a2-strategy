"""Matched A2 winning trades on actual TQQQ, and causal N20 MA10/20 chop experiments.

Run: python research/experiments/a2_leverage_and_ma_tangle_study.py

Data boundaries: actual TQQQ adjusted opens through 2025-10-28 from two public
GitHub datasets; actual TQQQ adjusted closes to 2026-07-13 (Holochart/Yahoo).
For trades crossing the missing 2025-10 onward adjusted-open interval, only
the boundary opens are approximated using 3x QQQ overnight gap.
Pre-2010 synthetic is not a TQQQ backtest. All variants remain exploratory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
FEE = frozen.FEE


def enrich(base):
    df = base.copy()
    prev = df.adj_close.shift(1)
    tr = pd.concat([(df.adj_high-df.adj_low), (df.adj_high-prev).abs(),
                    (df.adj_low-prev).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["ma10_gold"] = (df.ma10 > df.ma20) & (df.ma10.shift(1) <= df.ma20.shift(1))
    for n in (10, 20, 30):
        # Prior n days, NOT including current signal day.
        df[f"cross_prev{n}"] = df.protect_cross.shift(1).rolling(n).sum() + df.ma10_gold.shift(1).rolling(n).sum()
    df["gap_atr"] = (df.ma10-df.ma20).abs()/df.atr14
    df["gap_pct"] = (df.ma10/df.ma20-1).abs()
    return df


def tqqq_price_map(df):
    holo = json.loads((ROOT/"sources/tqqq_holo_adjclose_2010_2026.json").read_text())
    actual_close = pd.DataFrame({"Date": holo["date"], "t_close": holo["funds"]["TQQQ"]["adjClose"]})
    actual_close["Date"] = pd.to_datetime(actual_close.Date)
    actual_close = actual_close[actual_close.Date <= frozen.END].set_index("Date")
    actual_close["t_open"] = np.nan
    actual_close["open_origin"] = "unavailable"

    # A historical source with six-decimal adjusted OHLC ends in 2021.
    full_open = pd.read_csv(ROOT/"sources/tqqq_sunethan_2010_2021.csv", parse_dates=["Date"]).set_index("Date")
    common = actual_close.index.intersection(full_open.index)
    f = actual_close.loc[common,"t_close"] / full_open.loc[common,"Adj Close"]
    assert float(f.std()/f.mean()) < 1e-5, "Historical TQQQ close basis mismatch"
    actual_close.loc[common,"t_open"] = full_open.loc[common,"Open"] * full_open.loc[common,"Adj Close"] / full_open.loc[common,"Close"] * f
    actual_close.loc[common,"open_origin"] = "actual_sunethan_2010_2021"

    # Later vendor series is two-decimal split-adjusted. Align each OPEN to its
    # identical-date Yahoo dividend/split-adjusted close, retaining price gap.
    lucan = pd.read_csv(ROOT/"sources/tqqq_lucan_2010_2025.csv",encoding="utf-8-sig")
    lucan["Date"] = pd.to_datetime(lucan.Date)
    lucan = lucan.set_index("Date")
    later = actual_close.index.intersection(lucan.index)
    later = later[later > full_open.index.max()]
    scale = actual_close.loc[later,"t_close"]/lucan.loc[later,"Price"]
    actual_close.loc[later,"t_open"] = lucan.loc[later,"Open"] * scale
    actual_close.loc[later,"open_origin"] = "actual_lucan_2021_2025_two_decimals"
    # Confirm vendor closes are on the same return basis and dates.
    assert len(later) > 1100
    assert (lucan.loc[later,"Price"] > 0).all()
    return actual_close


def hypothetical_daily_3x_qqq(df, i, j):
    """Daily-reset 3x QQQ proxy from entry OPEN i to exit OPEN j.

    First session uses open->close, subsequent ones close->close and final
    previous close->exit open; the actual ETF's intraday open-to-close factor
    can differ, so this path is *not* an actual TQQQ transaction.
    """
    assert j > i
    steps = [df.iloc[i].adj_close/df.iloc[i].adj_open-1]
    steps.extend(df.iloc[k].adj_close/df.iloc[k-1].adj_close-1 for k in range(i+1,j))
    steps.append(df.iloc[j].adj_open/df.iloc[j-1].adj_close-1)
    gross = float(np.prod([1+3*x for x in steps]))
    return gross*(1-FEE)/(1+FEE)-1


def actual_tqqq_open_or_gap_estimate(t_prices, qqq, idx):
    date = qqq.iloc[idx].Date
    if date not in t_prices.index:
        return np.nan,"NOT_LISTED"
    row = t_prices.loc[date]
    if pd.notna(row.t_open):
        return float(row.t_open),str(row.open_origin)
    if idx < 1:
        return np.nan,"NO_PREV_DAY"
    prior_date = qqq.iloc[idx-1].Date
    if prior_date not in t_prices.index:
        return np.nan,"NO_PREV_TQQQ_CLOSE"
    gap = float(qqq.iloc[idx].adj_open / qqq.iloc[idx-1].adj_close-1)
    return float(t_prices.loc[prior_date,"t_close"]*(1+3*gap)),"ESTIMATED_3X_OVERNIGHT_GAP"


def leverage_trades(df,trades):
    t_prices=tqqq_price_map(df)
    rows=[]
    for t in trades.itertuples(index=False):
        i=int(t.entry_idx);j=int(df.index[df.Date==pd.Timestamp(t.exit_date)][0]); assert j>i
        synth=hypothetical_daily_3x_qqq(df,i,j)
        ent,ek=actual_tqqq_open_or_gap_estimate(t_prices,df,i)
        ext,xk=actual_tqqq_open_or_gap_estimate(t_prices,df,j)
        actual=np.nan if not np.isfinite(ent) or not np.isfinite(ext) else ext/ent*(1-FEE)/(1+FEE)-1
        rows.append({"id":t.id,"entry_date":t.entry_date,"exit_date":t.exit_date,
                     "exit_reason":t.exit_reason,"hold_days":t.hold_days,
                     "qqq_net":t.ret_net,"naive_3x_qqq_net":3*t.ret_net,
                     "synthetic_qqq_daily_3x_net":synth,"tqqq_actual_prices_net":actual,
                     "tqqq_data_quality":"ACTUAL_ADJ_OPEN" if ek.startswith('actual') and xk.startswith('actual') else
                         "OPEN_GAP_ESTIMATED" if ek!='NOT_LISTED' and xk!='NOT_LISTED' else "BEFORE_TQQQ_LAUNCH",
                     "entry_open_quality":ek,"exit_open_quality":xk,
                     "tqqq_minus_qqq_pctpt":(actual-t.ret_net)*100 if pd.notna(actual) else np.nan,
                     "tqqq_minus_naive3_pctpt":(actual-3*t.ret_net)*100 if pd.notna(actual) else np.nan})
    return pd.DataFrame(rows)


def same_signals_account(df, completed, open_position, product, end_cutoff=None):
    """One-account comparison from TQQQ launch, identical A2 signal dates.

    QQQ uses frozen adjusted opens/closes. TQQQ uses historical actual closes
    and vendor opens, except post-Oct2025 entry/exit fills approximated from
    QQQ overnight gaps as explicitly labeled in the trade attribution.
    """
    tqqq=tqqq_price_map(df)
    opening=[(r.entry_date,r.exit_date) for r in completed.itertuples(index=False)]
    if open_position is not None:
        opening.append((open_position['entry_date'],None))
    bmap={d:None for d,_ in opening};smap={e:None for _,e in opening if e is not None}
    cash=100000.;shares=0.;result=[]
    for i,bar in enumerate(df.itertuples(index=False)):
        if bar.Date<pd.Timestamp('2010-02-11'):continue
        if end_cutoff is not None and bar.Date>end_cutoff:break
        d=str(bar.Date.date())
        if product=='QQQ':op=float(bar.adj_open);cl=float(bar.adj_close)
        else:
            op,_=actual_tqqq_open_or_gap_estimate(tqqq,df,i)
            cl=float(tqqq.loc[bar.Date,'t_close'])
        assert np.isfinite(cl)
        if d in smap:
            assert shares>0
            cash=shares*op*(1-FEE);shares=0.
        if d in bmap:
            assert shares==0 and np.isfinite(op)
            shares=cash/(op*(1+FEE));cash=0.
        result.append({'Date':d,'Equity':cash+shares*cl,'Invested':shares>0})
    e=pd.DataFrame(result)
    e['DD']=e.Equity/e.Equity.cummax()-1
    years=(pd.Timestamp(e.Date.iloc[-1])-pd.Timestamp(e.Date.iloc[0])).days/365.25
    return {'asset':product,'start':e.Date.iloc[0],'end':e.Date.iloc[-1],
            'end_equity':e.Equity.iloc[-1],
            'cagr':(e.Equity.iloc[-1]/1e5)**(1/years)-1,'mdd':e.DD.min(),
            'mdd_date':e.loc[e.DD.idxmin(),'Date'],
            'exposure':e.Invested.mean()},e


def n20_events(df, baseline):
    rows=[]
    for tr in baseline.itertuples(index=False):
        if tr.exit_reason!='N20 protection':continue
        sig_i=int(df.index[df.Date==pd.Timestamp(tr.exit_signal_date)][0]);sig=df.iloc[sig_i]
        # Descriptive hindsight labels, never used as a trade signal.
        back=[]
        for window in (2,3,5,10):
            next_bars=df.iloc[sig_i+1:sig_i+window+1]
            back.append(bool((next_bars.ma10_gold).any()))
        # Hold-to-slow-death from original entry, matched counterfactual.
        death_idx=next((k for k in range(sig_i+1,len(df)) if bool(df.iloc[k].death)),None)
        late_ret=np.nan
        if death_idx is not None and death_idx+1<len(df):
            late_ret=float(df.iloc[death_idx+1].adj_open/tr.entry_price*(1-FEE)/(1+FEE)-1)
        rows.append({"id":tr.id,"entry_date":tr.entry_date,"n20_signal":tr.exit_signal_date,
                     "n20_exit":tr.exit_date,"n20_ret_net":tr.ret_net,
                     "cross_prev10":int(sig.cross_prev10),"cross_prev20":int(sig.cross_prev20),
                     "cross_prev30":int(sig.cross_prev30),
                     "gap_atr":float(sig.gap_atr),"gap_pct":float(sig.gap_pct),
                     "ma20_slope5":float(sig.ma20/df.iloc[sig_i-5].ma20-1),
                     "vol_at_signal":float(df.adj_close.pct_change().iloc[sig_i-19:sig_i+1].std()*np.sqrt(252)),
                     "recover2":back[0],"recover3":back[1],"recover5":back[2],"recover10":back[3],
                     "late_20_60_death_ret":late_ret,"late_minus_n20_pp":(late_ret-tr.ret_net)*100})
    return pd.DataFrame(rows)


def simulate(df,hold_days=0, gate="all"):
    """Only N20 exits modified. Watch N sessions for MA10 > MA20 recovery.

    Deadcross on signal day initiates watch. A recovery on a subsequent close
    before/including deadline cancels the watch; otherwise exits at next open.
    MA20/60 death always exits at next open; fresh 20/60 gold required to rebuy.
    gate is computed on signal day, is NOT the forward recovery label.
    """
    cash=100000.;shares=0.;position=None;pending=None;trades=[];equity=[]
    for i,bar in enumerate(df.itertuples(index=False)):
        if bar.Date<frozen.START:continue
        if pending is not None:
            side,si,reason=pending
            if side=='buy':
                assert shares==0
                shares=cash/(float(bar.adj_open)*(1+FEE))
                position={"entry_date":str(bar.Date.date()),"entry_idx":i,
                          "entry_price":float(bar.adj_open),"debit":cash,
                          "watch":None}
                cash=0.
            else:
                assert position is not None
                cash=shares*float(bar.adj_open)*(1-FEE)
                trades.append({"entry_date":position['entry_date'],
                               "exit_date":str(bar.Date.date()),
                               "exit_signal_date":str(df.iloc[si].Date.date()),
                               "exit_reason":reason,"hold_days":i-position['entry_idx'],
                               "ret_net":cash/position['debit']-1})
                position=None;shares=0.
            pending=None
        if position is not None:
            age=i-position['entry_idx'];n20=(1<=age<=20 and bool(bar.protect_cross)
                      and float(bar.adj_close)<position['entry_price'])
            if bool(bar.death):
                pending=('sell',i,'MA20/60 death cross')
            elif position['watch'] is not None:
                w=position['watch']
                if float(bar.ma10)>float(bar.ma20):
                    position['watch']=None
                elif i-w >= hold_days:
                    pending=('sell',i,'N20 delayed confirmation')
                # Repeated crosses after a watch cancellation can open new watches
                # within the original entry-based 20d window.
            elif n20:
                if gate=='all': eligible=True
                elif gate=='prior20_ge2':eligible=bool(df.iloc[i].cross_prev20>=2)
                elif gate=='prior30_ge2':eligible=bool(df.iloc[i].cross_prev30>=2)
                elif gate=='tight_atr025':eligible=bool(df.iloc[i].gap_atr<=0.25)
                elif gate=='tight_atr050':eligible=bool(df.iloc[i].gap_atr<=0.5)
                elif gate=='tight025_prior20ge2':eligible=bool(df.iloc[i].gap_atr<=0.25 and df.iloc[i].cross_prev20>=2)
                else:raise ValueError(gate)
                if hold_days>0 and eligible:
                    position['watch']=i
                else:
                    pending=('sell',i,'N20 protection')
        elif bool(bar.gold):
            pending=('buy',i,'FRESH_MA20_60_GOLD')
        equity.append({'Date':str(bar.Date.date()),'Equity':cash+shares*float(bar.adj_close),
                       'Invested':shares>0})
    t=pd.DataFrame(trades);e=pd.DataFrame(equity)
    e['Drawdown']=e.Equity/e.Equity.cummax()-1
    return t,e


def metrics(name,t,e,baseline):
    wins=t[t.ret_net>0].ret_net;losses=t[t.ret_net<0].ret_net
    years=(pd.Timestamp(e.Date.iloc[-1])-frozen.START).days/365.25
    # Outcome comparisons only for exact same entry dates (some early stop
    # changes prevent a subsequently scheduled baseline fresh-gold trade).
    b=baseline[['entry_date','ret_net']].rename(columns={'ret_net':'baseline_ret'})
    matched=t.merge(b,on='entry_date',how='inner')
    changed=matched[abs(matched.ret_net-matched.baseline_ret)>1e-8]
    return {'variant':name,'trades':len(t),'wins':len(wins),'losses':len(losses),
            'payoff':wins.mean()/(-losses.mean()),'win_rate':len(wins)/len(t),
            'pf':wins.sum()/(-losses.sum()),'mdd':e.Drawdown.min(),
            'cagr':(e.Equity.iloc[-1]/1e5)**(1/years)-1,
            'end_equity':e.Equity.iloc[-1],
            'max_loss':t.ret_net.min(),
            'matched_entry_count':len(matched),'changed_entry_count':len(changed),
            'improved_same_entry':int((changed.ret_net>changed.baseline_ret).sum()),
            'worsened_same_entry':int((changed.ret_net<changed.baseline_ret).sum()),
            'loss_to_win_same_entry':int(((changed.baseline_ret<=0)&(changed.ret_net>0)).sum()),
            'win_to_loss_same_entry':int(((changed.baseline_ret>0)&(changed.ret_net<=0)).sum())}


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    raw,_=frozen.prices()
    df=enrich(raw)
    original,base_equity,base_stats,open_position=frozen.run(raw)
    lev=leverage_trades(df,original)
    account_summary=[]
    for asset in ('QQQ','TQQQ'):
        for cutoff,tag in [(pd.Timestamp('2025-10-28'),'exact_open_period'),(None,'extended_gap_estimate')]:
            s,a=same_signals_account(df,original,open_position,asset,cutoff)
            s['valuation_scope']=tag
            account_summary.append(s)
            a.to_csv(OUT/f'a2_same_signals_{asset}_{tag}_equity.csv',index=False)
    pd.DataFrame(account_summary).to_csv(OUT/'a2_same_signals_account_metrics.csv',index=False)
    ev=n20_events(df,original)
    t0,e0=simulate(df)
    assert len(t0)==len(original)==58
    assert np.allclose(t0.ret_net.values,original.ret_net.values,atol=1e-10)
    assert np.allclose(e0.Equity.values,base_equity.Equity.values,atol=1e-8)
    scenarios=[('A2_N20_original',0,'all')]
    scenarios += [(f'N20_wait{k}_all',k,'all') for k in (1,2,3,5,7,8,10)]
    scenarios += [(f'N20_wait{k}_{gate}',k,gate) for k in (2,3,5,6,7,8,9,10,12)
                  for gate in ('prior20_ge2','prior30_ge2','tight_atr025','tight_atr050','tight025_prior20ge2')]
    summaries=[];all_trades=[]
    for name,k,gate in scenarios:
        t,e=simulate(df,k,gate)
        summaries.append(metrics(name,t,e,original))
        t.insert(0,'variant',name)
        all_trades.append(t)
    stats=pd.DataFrame(summaries)
    full_trades=pd.concat(all_trades,ignore_index=True)
    lev.to_csv(OUT/'a2_leverage_all_trades.csv',index=False)
    ev.to_csv(OUT/'a2_n20_tangle_at_stop.csv',index=False)
    stats.to_csv(OUT/'a2_n20_tangle_delay_metrics.csv',index=False)
    full_trades.to_csv(OUT/'a2_n20_tangle_delay_trades.csv',index=False)
    report={'baseline':base_stats,'leverage':{'winners':int((lev.qqq_net>0).sum()),
       'actual_priced_winners':int(((lev.qqq_net>0)&(lev.tqqq_data_quality=='ACTUAL_ADJ_OPEN')).sum()),
       'approx_priced_winners':int(((lev.qqq_net>0)&(lev.tqqq_data_quality=='OPEN_GAP_ESTIMATED')).sum()),
       'synthetic_winners':int(((lev.qqq_net>0)&(lev.tqqq_data_quality=='BEFORE_TQQQ_LAUNCH')).sum())},
       'n20':{'events':len(ev),'recover2':int(ev.recover2.sum()),'recover3':int(ev.recover3.sum()),
              'recover5':int(ev.recover5.sum()),'recover10':int(ev.recover10.sum())}}
    (OUT/'a2_tqqq_tangle_meta.json').write_text(json.dumps(report,indent=2))
    return lev,ev,stats,full_trades


if __name__=='__main__':
    lev,ev,stats,full_trades=run()
    print('WINNERS TQQQ ACTUAL/ESTIMATED OR PRE-LAUNCH SYNTHETIC')
    print(lev.loc[lev.qqq_net>0,['id','entry_date','exit_date','qqq_net','naive_3x_qqq_net','synthetic_qqq_daily_3x_net','tqqq_actual_prices_net','tqqq_data_quality']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nN20 EVENT DIAGNOSTIC')
    print(ev[['id','n20_signal','n20_ret_net','cross_prev20','gap_atr','recover3','recover5','late_20_60_death_ret']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nSCENARIOS')
    print(stats[['variant','trades','payoff','mdd','cagr','improved_same_entry','worsened_same_entry','loss_to_win_same_entry','win_to_loss_same_entry']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
