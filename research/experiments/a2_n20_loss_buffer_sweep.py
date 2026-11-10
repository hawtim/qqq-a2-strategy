"""A2 N20 single-parameter loss-buffer sweep; original baseline never edited.

N20 exit is signaled only on a fresh MA10/20 bearish cross within the first
20 trading days and at a signal close strictly below entry*(1-buffer).
No other A2 rules are modified. Signals are close-confirmed, next open filled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research"))
import a2_n20 as frozen

BUFFERS = [0,.0025,.005,.0075,.01,.0125,.014,.015,.0154,.0155,.016,
           .0175,.02,.0225,.025,.03,.035,.04,.05]


def simulate(df:pd.DataFrame,loss_buffer:float,warm_start:bool=True):
    assert 0 <= loss_buffer <= .5
    cash, shares, position, pending = 100000.,0.,None,None
    trades,curve=[] , []
    for i,bar in enumerate(df.itertuples(index=False)):
        if bar.Date < frozen.START:
            continue
        if warm_start and bar.Date == frozen.START and df.iloc[i-1].ma20>df.iloc[i-1].ma60:
            pending=("buy",i-1,"WARM_START")

        if pending is not None:
            action,signal_i,reason=pending
            if action=="buy":
                assert position is None and shares==0
                shares=cash/(float(bar.adj_open)*(1+frozen.FEE))
                position={"entry_date":str(bar.Date.date()),
                          "entry_idx":i,"entry_price":float(bar.adj_open),
                          "entry_signal_date":str(df.iloc[signal_i].Date.date()),
                          "entry_origin":reason,"debit":cash}
                cash=0.
            else:
                assert position is not None and shares>0
                proceeds=shares*float(bar.adj_open)*(1-frozen.FEE)
                trades.append({**position,"exit_date":str(bar.Date.date()),
                               "exit_signal_date":str(df.iloc[signal_i].Date.date()),
                               "exit_reason":reason,"ret_net":proceeds/position["debit"]-1,
                               "hold_days":i-position["entry_idx"],
                               "exit_price":float(bar.adj_open)})
                shares,position,cash=0.,None,proceeds
            pending=None

        if position is not None:
            elapsed=i-position["entry_idx"]
            if bool(bar.death):
                pending=("sell",i,"MA20/60 death cross")
            elif (1<=elapsed<=20 and bool(bar.protect_cross) and
                  float(bar.adj_close)<position["entry_price"]*(1-loss_buffer)):
                pending=("sell",i,"N20 protection")
        elif bool(bar.gold):
            pending=("buy",i,"FRESH_GOLD_CROSS")
        curve.append({"Date":str(bar.Date.date()),"Equity":cash+shares*float(bar.adj_close),
                      "Invested":shares>0})

    t,e=pd.DataFrame(trades),pd.DataFrame(curve)
    e["drawdown"]=e.Equity/e.Equity.cummax()-1
    win=t.loc[t.ret_net>0,"ret_net"]
    lose=t.loc[t.ret_net<0,"ret_net"]
    terminal=t.ret_net.to_list()
    if position is not None:
        terminal.append(shares*float(df.iloc[-1].adj_close)*(1-frozen.FEE)/position["debit"]-1)
    common=pd.Series(terminal)
    cw,cl=common[common>0],common[common<0]
    years=(pd.Timestamp(e.Date.iloc[-1])-frozen.START).days/365.25
    s={"buffer":loss_buffer,"k":1-loss_buffer,"completed":len(t),
       "n20_exits":int((t.exit_reason=="N20 protection").sum()),
       "win_rate":len(win)/len(t),"avg_win":win.mean(),"avg_loss":lose.mean(),
       "payoff_completed":win.mean()/abs(lose.mean()),
       "payoff_common":cw.mean()/abs(cl.mean()),
       "pf_common":cw.sum()/abs(cl.sum()),
       "mdd":e.drawdown.min(),"cagr":(e.Equity.iloc[-1]/100000)**(1/years)-1,
       "equity":e.Equity.iloc[-1],"exposure":e.Invested.mean(),
       "worst_trade":t.ret_net.min(),
       "open":position is not None}
    for name,start,end in [("early","2000-01-03","2012-12-31"),
                           ("late","2013-01-02","2026-07-13"),
                           ("gfc","2008-01-01","2009-12-31"),
                           ("bear2022","2022-01-01","2022-12-31")]:
        sub=e[(e.Date>=start)&(e.Date<=end)]
        s[name+"_ret"]=sub.Equity.iloc[-1]/sub.Equity.iloc[0]-1
        s[name+"_mdd"]=(sub.Equity/sub.Equity.cummax()-1).min()
    return t,e,s


def main():
    df,_=frozen.prices()
    orig,eq_orig,baseline,_=frozen.run(df,warm_start=True)
    base_t,base_e,base=simulate(df,0)
    assert len(base_t)==len(orig)==59
    assert base_t.entry_date.tolist()==orig.entry_date.tolist()
    assert base_t.exit_date.tolist()==orig.exit_date.tolist()
    assert np.allclose(base_t.ret_net.to_numpy(),orig.ret_net.to_numpy())
    assert np.allclose(base_e.Equity.to_numpy(),eq_orig.Equity.to_numpy())
    assert np.isclose(base["mdd"],baseline["mdd"])

    print("buffer,k,N20_exits,payoff,PF,MDD,CAGR,ending_equity,2003_ret,2003_exit,late_equity_return,2022_return")
    original_dates=orig.entry_date.tolist()
    for b in BUFFERS:
        t,e,s=simulate(df,b)
        assert t.entry_date.tolist()==original_dates
        saved=t.set_index("entry_date").loc["2003-03-24"]
        print(f"{b:.4f},{1-b:.4f},{s['n20_exits']},{s['payoff_common']:.3f},"
              f"{s['pf_common']:.3f},{s['mdd']*100:.2f}%,{s['cagr']*100:.2f}%,"
              f"{s['equity']:.2f},{saved.ret_net*100:+.2f}%,{saved.exit_date},"
              f"{s['late_ret']*100:+.2f}%,{s['bear2022_ret']*100:+.2f}%")


if __name__=="__main__":
    main()
