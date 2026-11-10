"""A2 N20 two-stage variant: MA10/20 cross arms; daily close checks loss buffer.

Only the N20 protection module is replaced. Entry is still a fresh daily
MA20/60 bullish cross, and normal death-cross exits and re-entry are frozen.
The alert can arm in the first 20 days and remains armed until the trade exits.
Close-confirmed signals always fill at the *next* trading session's open.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"research"))
import a2_n20 as frozen

BUFFERS=[0,.005,.01,.0125,.015,.0155,.0175,.02,.025,.03,.04,.05]


def simulate(df:pd.DataFrame,buffer:float,warm_start:bool=True):
    cash,shares,position,pending=100000.0,0.0,None,None
    trades,curve=[],[]
    alerts=[]
    for i,bar in enumerate(df.itertuples(index=False)):
        if bar.Date < frozen.START:
            continue
        if warm_start and bar.Date==frozen.START and df.iloc[i-1].ma20>df.iloc[i-1].ma60:
            pending=("buy",i-1,"WARM_START")

        if pending is not None:
            action,signal_i,reason=pending
            if action=="buy":
                assert position is None and shares==0
                shares=cash/(float(bar.adj_open)*(1+frozen.FEE))
                position={"entry_date":str(bar.Date.date()),
                          "entry_price":float(bar.adj_open),"entry_idx":i,"capital":cash,
                          "alert":False,"alert_date":None,"alert_idx":None,
                          "alert_close_loss":None}
                cash=0.0
            else:
                assert position is not None and shares>0
                cash=shares*float(bar.adj_open)*(1-frozen.FEE)
                trades.append({**position,
                               "exit_date":str(bar.Date.date()),
                               "exit_signal_date":str(df.iloc[signal_i].Date.date()),
                               "exit_reason":reason,
                               "exit_price":float(bar.adj_open),
                               "ret_net":cash/position["capital"]-1,
                               "hold_days":i-position["entry_idx"]})
                shares,position=0.,None
            pending=None

        if position is not None:
            age=i-position["entry_idx"]
            if 1<=age<=20 and bool(bar.protect_cross) and not position["alert"]:
                position["alert"]=True
                position["alert_date"]=str(bar.Date.date())
                position["alert_idx"]=i
                position["alert_close_loss"]=float(bar.adj_close/position["entry_price"]-1)
                alerts.append({"entry_date":position["entry_date"],
                               "alert_date":str(bar.Date.date()),
                               "close_loss_at_alert":position["alert_close_loss"]})
            if bool(bar.death):
                pending=("sell",i,"MA20/60 death cross")
            elif (position["alert"] and
                  float(bar.adj_close)<position["entry_price"]*(1-buffer)):
                pending=("sell",i,"N20_armed_daily_close")
        elif bool(bar.gold):
            pending=("buy",i,"FRESH_GOLD_CROSS")
        curve.append({"Date":str(bar.Date.date()),
                      "Equity":cash+shares*float(bar.adj_close),
                      "Invested":bool(shares>0)})

    t,e=pd.DataFrame(trades),pd.DataFrame(curve)
    e["dd"]=e.Equity/e.Equity.cummax()-1
    win=t.loc[t.ret_net>0,"ret_net"]
    lose=t.loc[t.ret_net<0,"ret_net"]
    common=t.ret_net.to_list()
    if position is not None:
        common.append(shares*float(df.iloc[-1].adj_close)*(1-frozen.FEE)/position["capital"]-1)
    common=pd.Series(common)
    cw,cl=common[common>0],common[common<0]
    years=(pd.Timestamp(e.Date.iloc[-1])-frozen.START).days/365.25
    stats={"buffer":buffer,"alert_count":len(alerts),
           "completed":len(t),"protect_exits":int((t.exit_reason=="N20_armed_daily_close").sum()),
           "protect_exits_after_day20":int(((t.exit_reason=="N20_armed_daily_close")
                                             &(t.exit_signal_date.map(lambda z:df.index[df.Date==z][0])
                                               -t.entry_idx>20)).sum()),
           "win_rate":len(win)/len(t),"avg_loss":lose.mean(),"avg_win":win.mean(),
           "payoff":cw.mean()/abs(cl.mean()),"pf":cw.sum()/abs(cl.sum()),
           "mdd":e.dd.min(),"cagr":(e.Equity.iloc[-1]/100000)**(1/years)-1,
           "equity":e.Equity.iloc[-1],"exposure":e.Invested.mean(),
           "worst_trade":t.ret_net.min(),"open_at_end":position is not None}
    return t,e,stats,pd.DataFrame(alerts)


def main():
    df,_=frozen.prices()
    base,be,bs,_=frozen.run(df,warm_start=True)
    base_n20=base[base.exit_reason=="N20 protection"]
    print("BASE_A2",bs["completed_trades"],bs["payoff_ratio"],bs["mdd"],bs["cagr"],bs["end_equity"])
    print("buffer,alerts,protect_exits,protect_after_day20,completed,payoff,PF,MDD,CAGR,end_equity,2003_return,2003_exit")
    for b in BUFFERS:
        t,e,s,alerts=simulate(df,b)
        assert t.entry_date.tolist()==base.entry_date.tolist()
        row=t.set_index("entry_date").loc["2003-03-24"]
        print(f"{b:.4f},{s['alert_count']},{s['protect_exits']},{s['protect_exits_after_day20']},"
              f"{s['completed']},{s['payoff']:.3f},{s['pf']:.3f},{s['mdd']*100:.2f}%,"
              f"{s['cagr']*100:.2f}%,{s['equity']:.2f},"
              f"{row.ret_net*100:+.2f}%,{row.exit_date}")
        if np.isclose(b,.0155):
            assert str(row.exit_date)=="2003-04-10" and np.isclose(row.ret_net,-.039118,atol=.00001)
            changed=base[["entry_date","ret_net","exit_date"]].merge(
                t[["entry_date","ret_net","exit_date"]],on="entry_date",validate="one_to_one",
                suffixes=("_A2","_two_stage"))
            changed=changed[(changed.ret_net_A2-changed.ret_net_two_stage).abs()>1e-10].copy()
            changed["delta_pp"]=(changed.ret_net_two_stage-changed.ret_net_A2)*100
            print("Changed entries @1.55%:")
            print(changed.to_string(index=False))


if __name__=="__main__":
    main()
