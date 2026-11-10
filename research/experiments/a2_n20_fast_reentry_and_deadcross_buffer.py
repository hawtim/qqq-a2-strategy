"""A2-N20: two interpretations of MA10/20 fast re-entry, plus false-death buffering.

Entry/exit evaluation happens at close; next trading day adjusted open is the
execution price. No intraday stops. Cost is 0.05% per side, cash earns 0%.
The original A2 (with the historical synthetic 2000-01-03 warm start) must
match the frozen engine exactly before variants are accepted.

Run: python research/experiments/a2_n20_fast_reentry_and_deadcross_buffer.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen

OUTPUT = ROOT / "research" / "results" / "experiments_2026_11_09"
FEE = frozen.FEE


@dataclass(frozen=True)
class Variant:
    name: str
    fast_reentry: bool = False
    fast_exit_any_ma10_death: bool = False
    reentry_requires_slow_bull: bool = True
    # None preserves the original R1 unlimited-retry behavior.
    fast_retry_limit_per_slow_cycle: int | None = None
    slow_death_buffer_days: int = 0
    slow_death_buffer_pct: float = 0.0
    # Optional, orthogonal profit-protection overlay. Empty rule preserves A2.
    profit_arm_gain: float = 0.20
    profit_scope: str = "fast_only"  # fast_only or all
    profit_rule: str = ""  # down_days, peak_drawdown, below_ma10_atr, above_ma10_atr
    profit_rule_parameter: float = 0.0
    profit_lookback_days: int = 0  # rapid decline lookback (2,3,5 trading sessions)


VARIANTS = [
    Variant("A2_N20_original"),
    Variant("R1_N20_exit_fast_rebuy_reset_N20", fast_reentry=True),
    Variant("R2_N20_exit_fast_rebuy_all_MA10_deaths", fast_reentry=True,
            fast_exit_any_ma10_death=True),
    Variant("R1_unrestricted_fast_rebuy_sensitivity", fast_reentry=True,
            reentry_requires_slow_bull=False),
    *[Variant(f"D{days}_down{int(pct*100)}pct_slow_death_buffer",
              slow_death_buffer_days=days, slow_death_buffer_pct=pct)
      for days in (1, 2, 3, 5) for pct in (0.01, 0.02, 0.03)]
]


def enriched(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ma10_gold"] = (x.ma10 > x.ma20) & (x.ma10.shift(1) <= x.ma20.shift(1))
    previous_close = x.adj_close.shift(1)
    true_range = pd.concat([
        x.adj_high - x.adj_low,
        (x.adj_high - previous_close).abs(),
        (x.adj_low - previous_close).abs(),
    ], axis=1).max(axis=1)
    x["atr14"] = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return x


def simulate(df: pd.DataFrame, cfg: Variant, warm_start: bool = True):
    cash, shares, position, pending = 100000.0, 0.0, None, None
    flat_state = "wait_slow"
    trades, equity, actions = [], [], []
    fast_uses_in_cycle, slow_cycle_id = 0, 0

    for i, bar in enumerate(df.itertuples(index=False)):
        if bar.Date < frozen.START:
            continue
        if warm_start and bar.Date == frozen.START and df.iloc[i - 1].ma20 > df.iloc[i - 1].ma60:
            pending = ("buy", i - 1, "WARM_START")

        if pending is not None:
            side, signal_i, reason = pending
            if side == "buy":
                assert shares == 0 and position is None
                if reason in ("WARM_START", "FRESH_MA20_60_GOLD"):
                    # The only event that re-arms the fast-retry allowance is
                    # a truly fresh MA20/60 entry (or the legacy warm start).
                    fast_uses_in_cycle = 0
                    slow_cycle_id += 1
                elif reason == "FAST_MA10_GOLD":
                    assert cfg.fast_reentry
                    assert (cfg.fast_retry_limit_per_slow_cycle is None or
                            fast_uses_in_cycle < cfg.fast_retry_limit_per_slow_cycle)
                    fast_uses_in_cycle += 1
                shares = cash / (float(bar.adj_open) * (1 + FEE))
                position = {
                    "entry_date": str(bar.Date.date()),
                    "entry_signal_date": str(df.iloc[signal_i].Date.date()),
                    "entry_idx": i,
                    "entry_origin": reason,
                    "slow_cycle_id": slow_cycle_id,
                    "fast_uses_at_entry": fast_uses_in_cycle,
                    "entry_price": float(bar.adj_open),
                    "capital_before_buy": cash,
                    "max_close": float(bar.adj_close),
                    "max_close_idx": i,
                    "min_close": float(bar.adj_close),
                    "down_close_streak": 0,
                    "death_watch": None,
                }
                actions.append({"date": str(bar.Date.date()), "type": "buy",
                                "origin": reason, "price": float(bar.adj_open)})
                cash = 0.0
                flat_state = "invested"
            else:
                assert shares > 0 and position is not None
                proceeds = shares * float(bar.adj_open) * (1 - FEE)
                trade = {
                    "entry_date": position["entry_date"],
                    "entry_signal_date": position["entry_signal_date"],
                    "entry_origin": position["entry_origin"],
                    "slow_cycle_id": position["slow_cycle_id"],
                    "fast_uses_at_entry": position["fast_uses_at_entry"],
                    "entry_price": position["entry_price"],
                    "exit_date": str(bar.Date.date()),
                    "exit_signal_date": str(df.iloc[signal_i].Date.date()),
                    "exit_price": float(bar.adj_open),
                    "exit_reason": reason,
                    "hold_days": i - position["entry_idx"],
                    "ret_net": proceeds / position["capital_before_buy"] - 1,
                    "mfe_close": position["max_close"] / position["entry_price"] - 1,
                    "mae_close": position["min_close"] / position["entry_price"] - 1,
                    "equity_after_trade": proceeds,
                }
                trades.append(trade)
                actions.append({"date": str(bar.Date.date()), "type": "sell",
                                "origin": reason, "price": float(bar.adj_open)})
                # A fresh fast-reentry branch is *only* opened by a protected
                # N20 exit or by an MA10 death of a previous fast re-entry.
                eligible_retry = (
                    cfg.fast_reentry and
                    reason in ("N20 protection", "FAST_MA10_DEATH") and
                    (cfg.fast_retry_limit_per_slow_cycle is None or
                     fast_uses_in_cycle < cfg.fast_retry_limit_per_slow_cycle)
                )
                flat_state = "wait_fast" if eligible_retry else "wait_slow"
                cash, shares, position = proceeds, 0.0, None
            pending = None

        if position is not None:
            if float(bar.adj_close) > position["max_close"]:
                position["max_close"] = float(bar.adj_close)
                position["max_close_idx"] = i
            position["min_close"] = min(position["min_close"], float(bar.adj_close))
            age = i - position["entry_idx"]
            if age >= 1 and float(bar.adj_close) < float(df.iloc[i - 1].adj_close):
                position["down_close_streak"] += 1
            else:
                position["down_close_streak"] = 0
            original_n20 = (1 <= age <= 20 and bool(bar.protect_cross)
                            and float(bar.adj_close) < position["entry_price"])

            # With an unbuffered regular MA20/60 death, the frozen A2 regular
            # exit takes priority over N20 (which also can cross that day).
            if cfg.slow_death_buffer_days == 0:
                if bool(bar.death):
                    pending = ("sell", i, "MA20/60 death cross")
            else:
                watch = position["death_watch"]
                if watch is None and bool(bar.death):
                    position["death_watch"] = {"idx": i, "close": float(bar.adj_close)}
                elif watch is not None and i > watch["idx"]:
                    # A protective price break wins if it occurs at the same
                    # close as a recovered MA20/60 spread.
                    if float(bar.adj_close) <= watch["close"] * (1 - cfg.slow_death_buffer_pct):
                        pending = ("sell", i, "BUFFER_PRICE_BREAK")
                    elif float(bar.ma20) > float(bar.ma60):
                        position["death_watch"] = None
                    elif i - watch["idx"] >= cfg.slow_death_buffer_days:
                        pending = ("sell", i, "BUFFER_TIME_EXPIRED")

            if pending is None and original_n20:
                pending = ("sell", i, "N20 protection")
            if (pending is None and cfg.fast_exit_any_ma10_death and
                    position["entry_origin"] == "FAST_MA10_GOLD" and
                    age >= 1 and bool(bar.protect_cross)):
                pending = ("sell", i, "FAST_MA10_DEATH")
            if pending is None and cfg.profit_rule:
                applicable = (cfg.profit_scope == "all" or
                              (cfg.profit_scope == "fast_only" and
                               position["entry_origin"] == "FAST_MA10_GOLD"))
                armed = (position["max_close"] / position["entry_price"] - 1 >=
                         cfg.profit_arm_gain)
                trigger = False
                if applicable and armed:
                    if cfg.profit_rule == "down_days":
                        trigger = position["down_close_streak"] >= int(cfg.profit_rule_parameter)
                    elif cfg.profit_rule == "peak_drawdown":
                        trigger = (float(bar.adj_close) <= position["max_close"] *
                                   (1 - cfg.profit_rule_parameter))
                    elif cfg.profit_rule == "rapid_rolling_peak_dd":
                        assert cfg.profit_lookback_days >= 1
                        recent_start = max(position["entry_idx"], i - cfg.profit_lookback_days)
                        rolling_peak = float(df.iloc[recent_start:i + 1].adj_close.max())
                        trigger = (float(bar.adj_close) <= rolling_peak *
                                   (1 - cfg.profit_rule_parameter))
                    elif cfg.profit_rule == "rapid_alltime_peak_dd":
                        assert cfg.profit_lookback_days >= 1
                        trigger = (
                            i - position["max_close_idx"] <= cfg.profit_lookback_days
                            and float(bar.adj_close) <= position["max_close"] *
                                (1 - cfg.profit_rule_parameter)
                        )
                    elif cfg.profit_rule == "below_ma10_atr":
                        trigger = (float(bar.adj_close) <
                                   float(bar.ma10) - cfg.profit_rule_parameter * float(bar.atr14))
                    elif cfg.profit_rule == "above_ma10_atr":
                        trigger = (float(bar.adj_close) >
                                   float(bar.ma10) + cfg.profit_rule_parameter * float(bar.atr14))
                    else:
                        raise ValueError(f"Unsupported profit rule: {cfg.profit_rule}")
                if trigger:
                    pending = ("sell", i, f"PROFIT_{cfg.profit_rule}_{cfg.profit_rule_parameter:g}")
        else:
            if bool(bar.gold):
                pending = ("buy", i, "FRESH_MA20_60_GOLD")
            elif flat_state == "wait_fast" and cfg.fast_reentry:
                if (cfg.reentry_requires_slow_bull and
                        not (float(bar.ma20) > float(bar.ma60))):
                    flat_state = "wait_slow"
                elif bool(bar.ma10_gold):
                    pending = ("buy", i, "FAST_MA10_GOLD")

        equity.append({"Date": str(bar.Date.date()),
                       "Equity": float(cash + shares * float(bar.adj_close)),
                       "Invested": bool(shares > 0),
                       "Pending": pending[0] if pending else ""})

    t = pd.DataFrame(trades)
    e = pd.DataFrame(equity)
    a = pd.DataFrame(actions)
    e["Drawdown"] = e.Equity / e.Equity.cummax() - 1
    open_return = None
    if position is not None:
        open_return = (shares * float(df.iloc[-1].adj_close) * (1 - FEE) /
                       position["capital_before_buy"] - 1)
    return t, e, a, open_return


def max_consecutive_losses(returns: pd.Series) -> int:
    streak = longest = 0
    for ret in returns:
        streak = streak + 1 if ret < 0 else 0
        longest = max(longest, streak)
    return longest


def longest_uninvested(e: pd.DataFrame) -> int:
    streak = longest = 0
    for invested in e.Invested:
        streak = 0 if invested else streak + 1
        longest = max(longest, streak)
    return longest


def summary(name: str, t: pd.DataFrame, e: pd.DataFrame,
            actions: pd.DataFrame, open_return: float | None):
    close = t.ret_net
    common = (pd.concat([close, pd.Series([open_return])], ignore_index=True)
              if open_return is not None else close)
    wins, losses = common[common > 0], common[common < 0]
    years = (pd.Timestamp(e.Date.iloc[-1]) - frozen.START).days / 365.25
    end = float(e.Equity.iloc[-1])
    record = {
        "scenario": name, "closed": len(close), "winners": int((close > 0).sum()),
        "losers": int((close < 0).sum()),
        "n20_exits": int((t.exit_reason == "N20 protection").sum()),
        "fast_reentries": int((t.entry_origin == "FAST_MA10_GOLD").sum()),
        "fast_ma10_exits": int((t.exit_reason == "FAST_MA10_DEATH").sum()),
        "common_payoff": float(wins.mean() / -losses.mean()),
        "common_pf": float(wins.sum() / -losses.sum()),
        "win_rate": float((common > 0).mean()),
        "mean_win": float(wins.mean()), "mean_loss": float(losses.mean()),
        "median_closed": float(close.median()),
        "std_closed": float(close.std()),
        "q10_closed": float(close.quantile(0.10)),
        "q25_closed": float(close.quantile(0.25)),
        "q75_closed": float(close.quantile(0.75)),
        "q90_closed": float(close.quantile(0.90)),
        "worst_closed": float(close.min()),
        "five_worst_mean": float(close.nsmallest(min(5, len(close))).mean()),
        "loss_over_5pct": int((close <= -0.05).sum()),
        "loss_over_8pct": int((close <= -0.08).sum()),
        "gain_over_20pct": int((close >= 0.20).sum()),
        "gain_over_30pct": int((close >= 0.30).sum()),
        "max_consecutive_losses": max_consecutive_losses(close),
        "mdd": float(e.Drawdown.min()), "cagr": float((end / 100000) ** (1 / years) - 1),
        "end_equity": end, "exposure": float(e.Invested.mean()),
        "longest_cash_days": longest_uninvested(e),
        "buys": int((actions.type == "buy").sum()),
        "sells": int((actions.type == "sell").sum()),
        "approx_fee_rate_if_flat": 2 * FEE * len(close),
        "open_position": open_return is not None,
    }
    return record


def histogram(t: pd.DataFrame, name: str):
    groups = [
        ("< -10%", lambda r: r < -0.10),
        ("-10% to -5%", lambda r: (r >= -0.10) & (r < -0.05)),
        ("-5% to -2%", lambda r: (r >= -0.05) & (r < -0.02)),
        ("-2% to 0%", lambda r: (r >= -0.02) & (r < 0)),
        ("0% to 2%", lambda r: (r >= 0) & (r < 0.02)),
        ("2% to 5%", lambda r: (r >= 0.02) & (r < 0.05)),
        ("5% to 10%", lambda r: (r >= 0.05) & (r < 0.10)),
        ("10% to 20%", lambda r: (r >= 0.10) & (r < 0.20)),
        (">= 20%", lambda r: r >= 0.20),
    ]
    return [{"scenario": name, "bucket": label, "count": int(test(t.ret_net).sum())}
            for label, test in groups]


def period_change(e: pd.DataFrame, start: str, stop: str) -> float:
    period = e[e.Date.between(start, stop)]
    previous = e[e.Date < start]
    capital = float(previous.iloc[-1].Equity) if not previous.empty else 100000.0
    return float(period.iloc[-1].Equity / capital - 1)


def fmt_pct(number, digits=2):
    return f"{number*100:+.{digits}f}%"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw, meta = frozen.prices()
    df = enriched(raw)
    old_t, old_e, old_s, old_active = frozen.run(raw, warm_start=True)
    metrics, trades, histograms, samples = [], [], [], []
    for cfg in VARIANTS:
        t, e, a, op = simulate(df, cfg)
        if cfg.name == "A2_N20_original":
            assert t.entry_date.tolist() == old_t.entry_date.tolist()
            assert t.exit_date.tolist() == old_t.exit_date.tolist()
            np.testing.assert_allclose(t.ret_net, old_t.ret_net, atol=1e-12)
            np.testing.assert_allclose(e.Equity, old_e.Equity, atol=1e-6)
            assert abs(e.Drawdown.min() - old_s["mdd"]) < 1e-10
            assert abs(op - old_active["return_mark_to_market_with_hypothetical_exit_fee"]) < 1e-10
        s = summary(cfg.name, t, e, a, op)
        s["period_2000_2002"] = period_change(e, "2000-01-03", "2002-12-31")
        s["period_2003_2004"] = period_change(e, "2003-01-01", "2004-12-31")
        s["period_2020"] = period_change(e, "2020-01-01", "2020-12-31")
        s["period_2022"] = period_change(e, "2022-01-01", "2022-12-31")
        trough = e.Drawdown.idxmin()
        s["mdd_trough_date"] = e.Date.iloc[trough]
        s["mdd_peak_date"] = e.Date.iloc[e.Equity.loc[:trough].idxmax()]
        metrics.append(s)
        histograms.extend(histogram(t, cfg.name))
        t = t.copy()
        t.insert(0, "scenario", cfg.name)
        trades.append(t)
        # For comparable counterfactual analysis around 2003 and 2020.
        if cfg.name in ("A2_N20_original", "R1_N20_exit_fast_rebuy_reset_N20",
                        "R2_N20_exit_fast_rebuy_all_MA10_deaths",
                        "D2_down1pct_slow_death_buffer", "D3_down2pct_slow_death_buffer"):
            focus = t[((t.entry_date >= "2003-01-01") & (t.entry_date <= "2003-12-31")) |
                      ((t.entry_date >= "2020-01-01") & (t.entry_date <= "2021-12-31"))]
            samples.append(focus)

    m = pd.DataFrame(metrics)
    all_t = pd.concat(trades, ignore_index=True)
    bucket = pd.DataFrame(histograms)
    m.to_csv(OUTPUT / "a2_n20_reentry_and_buffer_metrics.csv", index=False)
    all_t.to_csv(OUTPUT / "a2_n20_reentry_and_buffer_trades.csv", index=False)
    bucket.to_csv(OUTPUT / "a2_n20_return_distribution.csv", index=False)
    pd.concat(samples, ignore_index=True).to_csv(
        OUTPUT / "a2_n20_reentry_buffer_focus_2003_2020.csv", index=False)

    # Same-entry comparisons only, *not* a complete strategy-level attribution:
    # skipped or merged trade cycles appear in the summary as missing matches.
    base_t = all_t[all_t.scenario == "A2_N20_original"].set_index("entry_date")
    changes, change_summary = [], []
    for cfg in VARIANTS:
        if not cfg.name.startswith("D"):
            continue
        variant_t = all_t[all_t.scenario == cfg.name].set_index("entry_date")
        paired = (base_t[["ret_net", "exit_date"]]
                  .join(variant_t[["ret_net", "exit_date", "exit_reason"]],
                        lsuffix="_base", rsuffix="_new", how="inner"))
        paired["delta_pp"] = (paired.ret_net_new - paired.ret_net_base) * 100
        changed = paired[paired.delta_pp.abs() > 1e-10].copy()
        changed.insert(0, "scenario", cfg.name)
        changes.append(changed.reset_index())
        change_summary.append({
            "scenario": cfg.name, "same_entry_matches": len(paired),
            "base_unmatched_entries": len(base_t) - len(paired),
            "variant_unmatched_entries": len(variant_t) - len(paired),
            "same_entry_improved": int((changed.delta_pp > 0).sum()),
            "same_entry_worsened": int((changed.delta_pp < 0).sum()),
            "same_entry_equal": int(len(paired) - len(changed)),
            "largest_gain_pp": float(changed.delta_pp.max()),
            "largest_loss_pp": float(changed.delta_pp.min()),
        })
    pd.concat(changes, ignore_index=True).to_csv(
        OUTPUT / "a2_n20_buffer_matched_trades_delta.csv", index=False)
    pd.DataFrame(change_summary).to_csv(
        OUTPUT / "a2_n20_buffer_match_summary.csv", index=False)

    # Write a plain-text report as a directly reproducible product of the study.
    lines = [
        "# A2-N20｜MA10/20 快速重新买入与 MA20/60 假死叉缓冲",
        "", "**日期：2026-11-09。探索性历史回测，非样本外证明。**", "",
        "## 实验定义", "",
        "- A2：MA20/60 新金叉买、死叉卖，首次进场前20交易日若MA10/20新死叉且浮亏则N20卖；任何卖出后等待下一次新20/60金叉。",
        "- R1：**仅N20卖出后**，在MA20仍高于MA60的前提下，下一次MA10上穿MA20次日开盘买回；每笔买回都重启原版N20保护。正常出场仍看MA20/60。",
        "- R2：首次N20止损后按R1快买回；**此后快速买回的仓位**不论盈利与否，只要MA10新下穿MA20即下一日卖出，之后再次等MA10新金叉买回；MA20/60死叉总是优先退出并关闭快买回权限。",
        "- R1_no_gate：取消R1中'MA20高于MA60'的入场限制，仅为了解是否误入下跌趋势，不是建议的规则。",
        "- D_N_P：不改入场、N20及重入；MA20/60死叉当日只记录警戒收盘价，未来N个交易日内若收盘跌破警戒价(1-P)则下日退出；若先金叉恢复则取消警戒；若N日届满仍死叉则下日退出。N20可以独立抢先触发。",
        "- 两种死叉缓冲：价格阈值按**初次20/60死叉信号收盘价**定锚，不按买入成本或持仓高点；仅用收盘检查，无法保证在阈值价格成交。",
        "- QQQ复权价2000-01-03至2026-07-13、单边费用0.05%、无滑点和现金收益，信号收盘确认、次日开盘执行；2000-01-03沿用旧版合成持仓。",
        "- Payoff/PF加入期末未平仓头寸并假设付出退出费，其他逐笔分布只统计完成的交易。期末资产用实测收盘净值。",
        "", "## 核心指标（共同截止口径）", "",
        "| 策略 | 完成交易 | Payoff | PF | MDD | CAGR | 资产终值 | 快买回 | 连亏最大 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for row in m.itertuples():
        lines.append(f"| {row.scenario} | {row.closed} | {row.common_payoff:.2f} | "
                     f"{row.common_pf:.2f} | {fmt_pct(row.mdd,1)} | {fmt_pct(row.cagr,2)} | "
                     f"${row.end_equity:,.0f} | {row.fast_reentries} | {row.max_consecutive_losses} |")
    lines += ["", "## 收益分布与尾部风险（已完成交易）", "",
              "| 策略 | 单笔标准差 | 10%分位 | 中位数 | 最差交易 | 最差5笔均值 | <=−5%笔数 | >=+20%笔数 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in m.itertuples():
        lines.append(f"| {row.scenario} | {fmt_pct(row.std_closed)} | "
                     f"{fmt_pct(row.q10_closed)} | {fmt_pct(row.median_closed)} | "
                     f"{fmt_pct(row.worst_closed)} | {fmt_pct(row.five_worst_mean)} | "
                     f"{row.loss_over_5pct} | {row.gain_over_20pct} |")
    lines += ["", "## 各时期权益的复利变动", "",
              "| 策略 | 2000–2002 | 2003–2004 | 2020 | 2022 | MDD峰值→谷底 |",
              "|---|---:|---:|---:|---:|---|"]
    for row in m.itertuples():
        lines.append(f"| {row.scenario} | {fmt_pct(row.period_2000_2002)} | "
                     f"{fmt_pct(row.period_2003_2004)} | {fmt_pct(row.period_2020)} | "
                     f"{fmt_pct(row.period_2022)} | {row.mdd_peak_date} → {row.mdd_trough_date} |")

    base = m.set_index("scenario").loc["A2_N20_original"]
    r1 = m.set_index("scenario").loc["R1_N20_exit_fast_rebuy_reset_N20"]
    d2 = m.set_index("scenario").loc["D2_down1pct_slow_death_buffer"]
    d3 = m.set_index("scenario").loc["D3_down1pct_slow_death_buffer"]
    lines += ["", "## 归因要点", "",
              "1. **2003-03-24 买入的历史被N20阻断的赢家，R1救了回来**：原交易2003-04-09 "
              "以 -1.50% 止损，R1于2003-04-22重新金叉买回，在2004-03-10卖出，重入交易 +33.66%；"
              "R2同样重入，但2003-07-01早早死叉离场，仅 +11.00%。",
              f"2. **R1最大回撤 {fmt_pct(r1.mdd,1)}，较原始A2 {fmt_pct(base.mdd,1)} 恶化**："
              f"2000–2002权益 {fmt_pct(r1.period_2000_2002)}，原A2为 {fmt_pct(base.period_2000_2002)}；"
              f"R1的亏损超过5%交易从 {int(base.loss_over_5pct)}笔增加到 {int(r1.loss_over_5pct)}笔。"
              "差异不是单纯佣金造成的，而是额外暴露了错误趋势。",
              "3. **D2（死叉警戒后观察2个交易日，跌1%则次日退出）**把2020-05-01交易的"
              "2020-10-08原卖出 +31.75% 延长至2021-03-18卖出 +48.10%，同时跳过了"
              "2020-10-12重新入场后 -6.13% 的亏损交易。",
              f"4. D2的Payoff={d2.common_payoff:.2f}，MDD={fmt_pct(d2.mdd,1)}；"
              f"D3的Payoff={d3.common_payoff:.2f}，MDD={fmt_pct(d3.mdd,1)}。"
              "改善主要是价格缓冲与均线快速恢复共同作用，但仍应逐笔核对损失扩大的案例。",
              "5. 同一入场日的对照不能覆盖**被跳过的再次入场**；完整CAGR与MDD必须使用各自独立的资金曲线。",
              "", "## 死叉缓冲的同一入场日归因", "",
              "| 方案 | 相同入场日 | 减少损失/提高利润 | 损失扩大/利润缩水 | 原始A2未匹配交易 |",
              "|---|---:|---:|---:|---:|"]
    for row in change_summary:
        lines.append(f"| {row['scenario']} | {row['same_entry_matches']} | "
                     f"{row['same_entry_improved']} | {row['same_entry_worsened']} | "
                     f"{row['base_unmatched_entries']} |")
    lines += ["", "## 可检验的解释和局限", "",
              "- 交易次数增加不是免费：除了买卖双方各0.05%费用，还存在滑点、跳空风险与执行摩擦。",
              "- 收益标准差同时包括大的正收益，不能把更低标准差单独解读成风险更小；结合最差5笔、左尾及权益MDD判断。",
              "- 如果R2交易剧增、损失大赢家，则MA10/20并不能直接替代趋势退出；R1的正常退出保留MA20/60更符合A2定位。",
              "- D_N_P两维参数为探索性网格；选中的历史最好参数不能直接视为实盘有效。",
              "- 缓冲的风险：刚出现慢线死叉后的额外N天，可能正是趋势断崖阶段；要逐笔列出避免假死叉和恶化亏损的数量与幅度。",
              "- 与原始A2逐笔交易不一定一一对应；跨策略资产曲线必须从当天状态完整重算，不能简单相加单笔收益。",
              "", "## 文件", "",
              "`a2_n20_reentry_and_buffer_metrics.csv`、`a2_n20_reentry_and_buffer_trades.csv`、"
              "`a2_n20_return_distribution.csv`、`a2_n20_reentry_buffer_focus_2003_2020.csv`、"
              "`a2_n20_buffer_matched_trades_delta.csv`、`a2_n20_buffer_match_summary.csv`。",
              "运行 `python research/experiments/a2_n20_fast_reentry_and_deadcross_buffer.py` 可复现。", ""]
    (OUTPUT / "A2-N20_快速重入与假死叉缓冲_独立回测.md").write_text("\n".join(lines))

    print(m[["scenario", "closed", "common_payoff", "common_pf", "mdd",
             "cagr", "end_equity", "fast_reentries", "max_consecutive_losses",
             "worst_closed", "std_closed", "loss_over_5pct"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
