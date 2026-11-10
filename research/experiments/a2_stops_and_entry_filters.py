"""A2-N20 isolated experiments: profit-armed MA stops and pre-entry gates.

The prior legacy_warm_start reference is held fixed, including its synthetic
2000-01-03 position. An entry gate never filters that synthetic warm start.
Each experiment changes only one isolated module and always waits for a new
MA20/60 golden cross after an exit.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
START = frozen.START
FEE = frozen.FEE


@dataclass(frozen=True)
class Stop:
    ma: int
    factor: float
    mode: str = "intraday"  # previous day MA applied to next day's OHLC
    ratchet: bool = False    # never lower an already armed stop
    arm_at: float = 0.0      # net closing profit needed to activate permanently


@dataclass(frozen=True)
class Gate:
    field: str
    op: str
    threshold: float

    def blocked(self, bar) -> bool:
        x = float(getattr(bar, self.field))
        assert np.isfinite(x)
        return x > self.threshold if self.op == ">" else x < self.threshold


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist200"] = df.adj_close / df.ma200 - 1
    df["ret60"] = df.adj_close / df.adj_close.shift(60) - 1
    df["ret20"] = df.adj_close / df.adj_close.shift(20) - 1
    df["abs_dist200"] = df.dist200.abs()
    return df


def simulate(df: pd.DataFrame, label: str, *, stop: Stop | None = None,
             gate: Gate | None = None, warm_start: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    cash = 100000.0
    shares = 0.0
    trade = None
    pending = None  # ('buy'/'sell', signal_index, reason)
    completed = []
    skipped = []
    curve = []
    for i, bar in enumerate(df.itertuples(index=False)):
        if bar.Date < START:
            continue
        if warm_start and bar.Date == START and bool(df.iloc[i-1].ma20 > df.iloc[i-1].ma60):
            pending = ("buy", i-1, "WARM_START")

        if pending is not None:
            action, sigidx, reason = pending
            if action == "buy":
                assert trade is None and shares == 0.0
                debit = cash
                shares = cash / (bar.adj_open * (1 + FEE))
                cash = 0.0
                trade = {
                    "entry_date": str(bar.Date.date()),
                    "entry_signal_date": str(df.iloc[sigidx].Date.date()),
                    "entry_origin": reason,
                    "entry_idx": i,
                    "entry_price": float(bar.adj_open),
                    "cost_basis": float(debit),
                    "mfe_close": float(bar.adj_close / bar.adj_open - 1),
                    "max_close": float(bar.adj_close),
                    "armed": False,
                    "stop_for_today": None,
                    "stop_armed_date": None,
                    "stop_hit_level": None,
                }
            else:
                assert trade is not None and shares > 0
                fill = float(bar.adj_open)
                proceeds = shares * fill * (1 - FEE)
                done = dict(trade)
                done.update({
                    "exit_date": str(bar.Date.date()),
                    "exit_signal_date": str(df.iloc[sigidx].Date.date()),
                    "exit_reason": reason,
                    "exit_price": fill,
                    "ret_net": float(proceeds / trade["cost_basis"] - 1),
                    "hold_days": i - trade["entry_idx"],
                    "mfe_close": float(trade["max_close"] / trade["entry_price"] - 1),
                    "exit_equity": float(proceeds),
                })
                completed.append(done)
                cash, shares, trade = proceeds, 0.0, None
            pending = None

        # Intraday stop uses exclusively the prior session's completed MA.
        # A pending A2 exit was filled at the open above and takes precedence.
        if trade is not None and stop is not None and stop.mode == "intraday":
            level = trade["stop_for_today"]
            if level is not None:
                if float(bar.adj_open) <= level:
                    fill = float(bar.adj_open)   # opening gap through stop
                    reason = "PROFIT_MA_STOP_GAP"
                elif float(bar.adj_low) <= level:
                    fill = float(level)          # idealized stop-market execution
                    reason = "PROFIT_MA_STOP_TOUCH"
                else:
                    fill = None
                if fill is not None:
                    proceeds = shares * fill * (1-FEE)
                    done = dict(trade)
                    done.update({
                        "exit_date": str(bar.Date.date()),
                        "exit_signal_date": str(bar.Date.date()),
                        "exit_reason": reason,
                        "exit_price": fill,
                        "stop_hit_level": float(level),
                        "ret_net": float(proceeds/trade["cost_basis"] - 1),
                        "hold_days": i-trade["entry_idx"],
                        "mfe_close": float(trade["max_close"]/trade["entry_price"] - 1),
                        "exit_equity": float(proceeds),
                    })
                    completed.append(done)
                    cash, shares, trade = proceeds, 0.0, None

        if trade is not None:
            trade["max_close"] = max(trade["max_close"], float(bar.adj_close))
            elapsed = i - trade["entry_idx"]
            if bool(bar.death):
                pending = ("sell", i, "MA20/60 death cross")
            elif 1 <= elapsed <= 20 and bool(bar.protect_cross) and bar.adj_close < trade["entry_price"]:
                pending = ("sell", i, "N20 protection")
            elif stop is not None and stop.mode == "close" and trade["armed"]:
                candidate = float(getattr(bar, f"ma{stop.ma}")) * stop.factor
                threshold = max(trade["stop_for_today"] or -np.inf, candidate) if stop.ratchet else candidate
                if float(bar.adj_close) <= threshold:
                    pending = ("sell", i, "PROFIT_MA_CLOSE")

            if stop is not None and pending is None:
                # Arm once (irrevocably) after a profitable daily close. The
                # theoretical closing liquidation must cover both fees.
                net_close_profit = float(bar.adj_close) * (1-FEE) / (trade["entry_price"] * (1+FEE)) - 1
                profitable_close = net_close_profit > stop.arm_at
                if profitable_close and not trade["armed"]:
                    trade["armed"] = True
                    trade["stop_armed_date"] = str(bar.Date.date())
                if trade["armed"]:
                    proposed = float(getattr(bar, f"ma{stop.ma}")) * stop.factor
                    trade["stop_for_today"] = (max(trade["stop_for_today"], proposed)
                                              if stop.ratchet and trade["stop_for_today"] is not None
                                              else proposed)
        elif bool(bar.gold):
            if gate is not None and gate.blocked(bar):
                skipped.append({"signal_date": str(bar.Date.date()),
                                "intended_entry_date": str(df.iloc[i+1].Date.date()) if i+1<len(df) else "",
                                "gate_field": gate.field, "gate_value": float(getattr(bar,gate.field)),
                                "threshold": gate.threshold})
            else:
                pending = ("buy", i, "FRESH_GOLD_CROSS")

        curve.append({"Date": str(bar.Date.date()),
                      "Equity": float(cash+shares*bar.adj_close),
                      "Invested": bool(shares > 0)})

    trades = pd.DataFrame(completed)
    eq = pd.DataFrame(curve)
    eq["Peak"] = eq.Equity.cummax()
    eq["Drawdown"] = eq.Equity/eq.Peak - 1
    win = trades[trades.ret_net > 0].ret_net
    loss = trades[trades.ret_net < 0].ret_net
    empty = 0
    longest_cash = 0
    for invested in eq.Invested:
        empty = 0 if invested else empty+1
        longest_cash = max(longest_cash, empty)
    longest_loss = 0
    losses_in_row = 0
    for ret in trades.ret_net:
        losses_in_row = 0 if ret > 0 else losses_in_row+1
        longest_loss = max(longest_loss, losses_in_row)
    year_len = (pd.Timestamp(eq.iloc[-1].Date)-START).days/365.25
    res = {
        "case": label,
        "completed": len(trades), "wins": len(win), "losses": len(loss),
        "win_rate": float(len(win)/len(trades)),
        "avg_win": float(win.mean()), "avg_loss": float(loss.mean()),
        "payoff": float(win.mean()/abs(loss.mean())),
        "pf": float(win.sum()/abs(loss.sum())),
        "mdd": float(eq.Drawdown.min()),
        "cagr": float((eq.Equity.iloc[-1]/100000)**(1/year_len)-1),
        "end_equity":float(eq.Equity.iloc[-1]),
        "exposure": float(eq.Invested.mean()),
        "longest_cash": longest_cash,
        "longest_loss_streak": longest_loss,
        "worst_trade": float(trades.ret_net.min()),
        "trailing_exit_count": int(trades.exit_reason.str.startswith("PROFIT_MA").sum()),
        "gap_stop_count": int((trades.exit_reason=="PROFIT_MA_STOP_GAP").sum()),
        "blocked_signals": len(skipped),
        "open_at_end": trade is not None,
        "open_entry_date": trade["entry_date"] if trade else None,
    }
    # A fair trade-metric comparator closes ALL outstanding positions at the
    # exact same historical cutoff. Preserve completed-only stats above to
    # reproduce the user-supplied 59 closed trades without alteration.
    common_returns = trades.ret_net.to_list()
    if trade is not None:
        terminal_close_return = shares * float(df.iloc[-1].adj_close) * (1-FEE) / trade["cost_basis"] - 1
        common_returns.append(float(terminal_close_return))
    common = pd.Series(common_returns)
    cwin = common[common>0]
    closs = common[common<0]
    res.update({
        "common_cutoff_trades":len(common),
        "common_cutoff_payoff":float(cwin.mean()/abs(closs.mean())),
        "common_cutoff_pf":float(cwin.sum()/abs(closs.sum())),
        "common_cutoff_win_rate":float(len(cwin)/len(common)),
        "end_equity_after_hypothetical_liquidation":float(cash+shares*df.iloc[-1].adj_close*(1-FEE)),
    })
    # Independent market episodes with the same date boundaries in all cases.
    for tag,start,end in [
        ("dotcom","2000-01-03","2002-12-31"),
        ("gfc","2008-01-01","2009-12-31"),
        ("covid","2020-02-01","2020-12-31"),
        ("rates","2022-01-01","2022-12-31"),
        ("early","2000-01-03","2012-12-31"),
        ("late","2013-01-02","2026-07-13"),
    ]:
        sub = eq[(eq.Date>=start)&(eq.Date<=end)]
        res[tag+"_return"] = float(sub.Equity.iloc[-1]/sub.Equity.iloc[0]-1)
        res[tag+"_mdd"] = float((sub.Equity/sub.Equity.cummax()-1).min())
    return trades, eq, res, pd.DataFrame(skipped)


CASES = {
    "BASE_A2_N20": {},
    "S_MA10_0.98_touch": {"stop":Stop(10,.98)},
    "S_MA20_0.98_touch": {"stop":Stop(20,.98)},
    "S_MA10_0.98_ratchet": {"stop":Stop(10,.98,ratchet=True)},
    "S_MA20_0.98_ratchet": {"stop":Stop(20,.98,ratchet=True)},
    "S_MA10_0.98_close": {"stop":Stop(10,.98,mode="close")},
    "S_MA20_0.98_close": {"stop":Stop(20,.98,mode="close")},
    "S_MA20_0.95_touch": {"stop":Stop(20,.95)},
    "S_MA20_0.98_arm5": {"stop":Stop(20,.98,arm_at=.05)},
    "S_MA20_0.98_arm10": {"stop":Stop(20,.98,arm_at=.10)},
    "S_MA10_0.98_arm5": {"stop":Stop(10,.98,arm_at=.05)},
    "F_dist200_gt_08pct": {"gate":Gate("dist200", ">", .08)},
    "F_dist200_gt_10pct": {"gate":Gate("dist200", ">", .10)},
    "F_dist200_gt_12pct": {"gate":Gate("dist200", ">", .12)},
    "F_ret60_gt_05pct": {"gate":Gate("ret60", ">", .05)},
    "F_ret60_gt_08pct": {"gate":Gate("ret60", ">", .08)},
    "F_ret20_gt_10pct": {"gate":Gate("ret20", ">", .10)},
    "F_absdist200_gt_10pct": {"gate":Gate("abs_dist200", ">", .10)},
    "F_below200_05pct": {"gate":Gate("dist200", "<", -.05)},
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, meta = frozen.prices()
    df = prepare(raw)
    baseline_t, baseline_e, baseline_s, _ = frozen.run(raw,warm_start=True)
    best10 = set(baseline_t.nlargest(10,"ret_net").entry_date)
    worst10 = set(baseline_t.nsmallest(10,"ret_net").entry_date)
    base_ret_map = baseline_t.set_index("entry_date").ret_net.to_dict()
    rows = []
    audits = []
    alltrades = []
    skipped_rows = []
    for label, kwargs in CASES.items():
        t, e, s, skipped = simulate(df,label,**kwargs)
        s["module"] = "baseline" if not kwargs else ("trailing_stop" if "stop" in kwargs else "entry_gate")
        existing = set(t.entry_date)
        s["top10_removed"] = int(len(best10-existing))
        s["worst10_removed"] = int(len(worst10-existing))
        s["baseline_top10_profit_retained"] = float(sum(t.loc[t.entry_date.isin(best10),"ret_net"])
                                                    / sum(base_ret_map[x] for x in best10))
        s["total_baseline_wins_removed"] = int(sum((d not in existing and r>0) for d,r in base_ret_map.items()))
        s["total_baseline_losses_removed"] = int(sum((d not in existing and r<0) for d,r in base_ret_map.items()))
        s["base_top10_damaged_20pct"] = int(sum((x in existing and t.set_index("entry_date").at[x,"ret_net"]
                                               < .8*base_ret_map[x]) for x in best10))
        rows.append(s)
        t["case"] = label
        alltrades.append(t)
        if len(skipped):
            skipped["case"] = label
            skipped_rows.append(skipped)
        for x in best10:
            match = t[t.entry_date==x]
            audits.append({"case":label,"baseline_entry":x,"base_return":base_ret_map[x],
                           "new_return":float(match.ret_net.iloc[0]) if len(match) else np.nan,
                           "new_exit_date":str(match.exit_date.iloc[0]) if len(match) else "SKIPPED",
                           "new_exit_reason":str(match.exit_reason.iloc[0]) if len(match) else "SKIPPED"})
        if label == "BASE_A2_N20":
            assert len(t)==len(baseline_t)
            assert (t.entry_date.to_numpy()==baseline_t.entry_date.to_numpy()).all()
            assert (t.exit_date.to_numpy()==baseline_t.exit_date.to_numpy()).all()
            assert np.allclose(t.ret_net,baseline_t.ret_net,atol=1e-12)
            assert np.allclose(e.Equity,baseline_e.Equity,atol=1e-8)
            assert abs(s["payoff"]-baseline_s["payoff_ratio"])<1e-10
            assert abs(s["end_equity"]-baseline_s["end_equity"])<1e-8
    pd.DataFrame(rows).to_csv(OUT/"scenario_metrics.csv",index=False)
    pd.concat(alltrades,ignore_index=True).to_csv(OUT/"all_scenario_trades.csv",index=False)
    pd.DataFrame(audits).to_csv(OUT/"baseline_top10_audit.csv",index=False)
    if skipped_rows:
        pd.concat(skipped_rows,ignore_index=True).to_csv(OUT/"blocked_entry_signals.csv",index=False)
    strict_rows = []
    for label in ("BASE_A2_N20","S_MA10_0.98_touch","S_MA20_0.98_touch",
                  "F_dist200_gt_10pct","F_ret60_gt_08pct"):
        _, _, s, _ = simulate(df,label+"_STRICT",warm_start=False,**CASES[label])
        s["variant"] = "strict_new_cross_no_synthetic_start"
        strict_rows.append(s)
    strict_baseline_t, strict_baseline_e, strict_baseline_s, _ = frozen.run(raw,warm_start=False)
    assert strict_rows[0]["completed"] == len(strict_baseline_t)
    assert abs(strict_rows[0]["payoff"] - strict_baseline_s["payoff_ratio"]) < 1e-10
    assert abs(strict_rows[0]["end_equity"] - strict_baseline_s["end_equity"]) < 1e-7
    pd.DataFrame(strict_rows).to_csv(OUT/"strict_start_sensitivity.csv",index=False)
    (OUT/"methodology.json").write_text(json.dumps({
        "data":meta,
        "benchmark":"A2-N20 warm-start; 2000-01-03 to 2026-07-13",
        "stop_activation":"Once the hypothetical close liquidation meets the configured net-profit threshold (0%, 5%, or 10%), remains armed",
        "stop_intra_day":"Stop at prior completed day's MA*factor; if open below stop, fill open; otherwise if low hits, fill exactly stop; no extra slippage",
        "stop_close":"Stop only after current day's completed close is below MA*factor, sell next open",
        "ratchet":"Armed stop never decreases versus preceding available MA stop",
        "entry_gates":"Frozen at golden-cross signal-day close. Blocked entry remains flat until a fresh MA20/60 golden cross. Warm-start synthetic trade unfiltered.",
        "other_rules":"A2 base exit, 20 trading days after purchase, 0.05% fees per side, original new-cross re-entry, no cash interest",
        "cases":{k: {n:str(v) for n,v in x.items()} for k,x in CASES.items()},
    },ensure_ascii=False,indent=2))
    c=pd.DataFrame(rows)
    pd.set_option("display.max_rows",100)
    print(c[["case","completed","payoff","pf","mdd","cagr","trailing_exit_count","blocked_signals","top10_removed","baseline_top10_profit_retained"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
