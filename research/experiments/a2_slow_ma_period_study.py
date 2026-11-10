"""Isolated A2-N20 slow-moving-average sensitivity study.

Only replace the slow trend moving average for *both* entry and regular exit.
All other A2-N20 rules, data, and accounting are inherited from a frozen engine.

Run: python research/experiments/a2_slow_ma_period_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import a2_n20 as frozen

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
SLOW_PERIODS = [40, 45, 50, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 80]


def with_slow_ma(df: pd.DataFrame, slow_period: int) -> pd.DataFrame:
    result = df.copy()
    slow = result.adj_close.rolling(slow_period, min_periods=slow_period).mean()
    result["gold"] = ((result.ma20 > slow) &
                      (result.ma20.shift(1) <= slow.shift(1)))
    result["death"] = ((result.ma20 < slow) &
                       (result.ma20.shift(1) >= slow.shift(1)))
    return result


def completed_plus_open(trades: pd.DataFrame, active: dict | None) -> pd.Series:
    """Common-cutoff trade-level returns: mark any open trade at end, exit fee assumed."""
    returns = trades.ret_net.copy()
    if active is not None:
        returns = pd.concat([
            returns,
            pd.Series([active["return_mark_to_market_with_hypothetical_exit_fee"]])
        ], ignore_index=True)
    return returns


def payoff(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    return float(wins.mean() / -losses.mean())


def period_change(curve: pd.DataFrame, begin: str, end: str) -> float:
    subset = curve[curve.Date.between(begin, end)]
    if subset.empty:
        return float("nan")
    # A period's initial equity is last day before it, not its first day's close.
    previous = curve[curve.Date < begin]
    prior_equity = (float(previous.iloc[-1].Equity) if not previous.empty
                    else 100000.0)
    return float(subset.iloc[-1].Equity / prior_equity - 1)


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def scenario(df: pd.DataFrame, slow: int, warm_start: bool = True):
    adjusted = with_slow_ma(df, slow)
    # The warm-start check inside the frozen engine is MA20/60-specific.
    # Ensure the initial state is the same for the compared slow period.
    prestart = adjusted.loc[adjusted.Date < frozen.START].iloc[-1]
    specific = prestart.ma20 > adjusted.adj_close.rolling(slow).mean().loc[prestart.name]
    baseline = prestart.ma20 > prestart.ma60
    assert bool(specific) == bool(baseline), (slow, "inconsistent warm start")
    trades, curve, raw, active = frozen.run(adjusted, warm_start=warm_start)
    returns = completed_plus_open(trades, active)
    positive, negative = returns[returns > 0], returns[returns < 0]
    trough = curve.Drawdown.idxmin()
    peak = curve.Equity.loc[:trough].idxmax()
    row = {
        "slow_ma": slow,
        "warm_start": warm_start,
        "completed_trades": len(trades),
        "n20_protection_trades": int((trades.exit_reason == "N20 protection").sum()),
        "closed_only_payoff": raw["payoff_ratio"],
        "common_cutoff_payoff": payoff(returns),
        "common_cutoff_pf": float(positive.sum() / -negative.sum()),
        "win_rate_common": float(len(positive) / len(returns)),
        "mean_winner_common": float(positive.mean()),
        "mean_loser_common": float(negative.mean()),
        "mdd": raw["mdd"],
        "mdd_peak": str(curve.loc[peak, "Date"]),
        "mdd_trough": str(curve.loc[trough, "Date"]),
        "cagr": raw["cagr"],
        "end_equity": raw["end_equity"],
        "invested_fraction": raw["invested_fraction"],
        "longest_cash_days": raw["longest_cash_days"],
        "max_single_loss_completed": raw["max_single_loss"],
        "open_position_at_end": raw["active_position_at_end"],
        "dot_com_bust_2000_2002_return": period_change(curve, "2000-01-03", "2002-12-31"),
        "financial_crisis_2008_2009_return": period_change(curve, "2008-01-01", "2009-12-31"),
        "pandemic_2020_return": period_change(curve, "2020-01-01", "2020-12-31"),
        "bear_2022_return": period_change(curve, "2022-01-01", "2022-12-31"),
        "post_2013_return": period_change(curve, "2013-01-01", "2026-07-13"),
    }
    trades = trades.copy()
    trades["slow_ma"] = slow
    trades["exit_reason"] = trades.exit_reason.replace(
        {"MA20/60 death cross": f"MA20/{slow} death cross"}
    )
    return row, trades, curve, active


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, metadata = frozen.prices()
    assert str(df.Date.iloc[-1].date()) == "2026-07-13"

    metrics, main_trades = [], []
    for slow in SLOW_PERIODS:
        row, trades, _, _ = scenario(df, slow)
        metrics.append(row)
        if slow in [50, 60, 61, 62, 65]:
            main_trades.append(trades)
    numbers = pd.DataFrame(metrics)
    numbers.to_csv(OUT / "a2_slow_ma_sweep_metrics.csv", index=False)
    all_trades = pd.concat(main_trades, ignore_index=True)
    all_trades.to_csv(OUT / "a2_slow_ma_selected_trades.csv", index=False)

    strict_rows = [scenario(df, n, warm_start=False)[0]
                   for n in [45, 50, 55, 60, 61, 62, 63, 65, 70]]
    pd.DataFrame(strict_rows).to_csv(OUT / "a2_slow_ma_strict_start_sensitivity.csv", index=False)

    # Key date-level audit: a very narrow threshold crossing in October 2020.
    audit = df[df.Date.between("2020-10-05", "2020-10-12")].copy()
    for slow in [50, 60, 61, 62, 65]:
        audit[f"ma{slow}"] = df.adj_close.rolling(slow).mean().loc[audit.index]
        audit[f"ma20_minus_ma{slow}"] = audit.ma20 - audit[f"ma{slow}"]
    audit[["Date", "adj_close", "ma20", "ma50", "ma60", "ma61", "ma62",
           "ma20_minus_ma60", "ma20_minus_ma61"]].to_csv(
               OUT / "a2_2020_10_false_death_cross_audit.csv", index=False)

    base = numbers[numbers.slow_ma == 60].iloc[0]
    assert int(base.completed_trades) == 59
    assert abs(base.mdd - (-0.2029)) < 0.001
    assert abs(base.cagr - 0.0566) < 0.001
    assert abs(base.closed_only_payoff - 3.95) < 0.04
    assert abs(base.end_equity - 430834) < 40

    headline = numbers[numbers.slow_ma.isin([45, 50, 55, 60, 61, 62, 65, 70])]
    lines = [
        "# A2-N20｜MA20/50 对 MA20/60：单因子替换实验（2026-11-09）",
        "",
        "**研究结论：MA20/50 没有改善盈亏比，且最大回撤和年化收益均明显恶化。**",
        "MA20/61、62 的历史表现有探索价值，但尚不能视为最优参数或实盘规则。",
        "",
        "## 不变的研究前提",
        "",
        "- 数据：2000-01-03 至 2026-07-13；同一份拼接后的复权 QQQ OHLC，原始 CSV 见 `sources/`。",
        "- 买入：MA20 从下向上穿过 MA(N)，收盘确认，下一交易日开盘全仓买入；退出：MA20 从上向下穿过同一 MA(N)，下一交易日开盘全卖。",
        "- N20 保护：入场后前20个交易日，MA10 新下穿 MA20 且当日收盘低于买价，下一交易日开盘全卖。",
        "- 任何出场后只等新 MA20/N 金叉；没有其他重入。单边费用0.05%，无滑点、现金利息。",
        "- **旧回测的合成初始持仓**：2000-01-03 已在多头区间，warm-start=True。严格新金叉启动的结果在独立 CSV 中。",
        "- 收盘标记权益曲线（包含期末仍持有头寸）；Payoff 与 PF 采用所有平仓交易 + 2026-07-13 未平仓交易的假设卖出成本，保障共同截止日可比。",
        "",
        "## 主要回测指标",
        "",
        "| MA20 / 慢线 | 已完成交易 | Payoff（共同截止） | PF（共同截止） | MDD | CAGR | 最终资产 | 最长空仓 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in headline.itertuples():
        lines.append(f"| {r.slow_ma} | {r.completed_trades} | {r.common_cutoff_payoff:.2f} | "
                     f"{r.common_cutoff_pf:.2f} | {fmt_pct(r.mdd)} | "
                     f"{fmt_pct(r.cagr)} | ${r.end_equity:,.0f} | {r.longest_cash_days} 日 |")

    lines += ["", "## 不同市场环境与风险（阶段复利收益）", "",
              "| 慢线 | 2000–02 | 2008–09 | 2020 | 2022 | 2013至截止 | 最差已完成交易 | N20触发 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in headline.itertuples():
        lines.append(f"| {r.slow_ma} | {fmt_pct(r.dot_com_bust_2000_2002_return)} | "
                     f"{fmt_pct(r.financial_crisis_2008_2009_return)} | "
                     f"{fmt_pct(r.pandemic_2020_return)} | {fmt_pct(r.bear_2022_return)} | "
                     f"{fmt_pct(r.post_2013_return)} | {fmt_pct(r.max_single_loss_completed)} | {r.n20_protection_trades} |")

    baseline_winners = all_trades[(all_trades.slow_ma == 60) & (all_trades.ret_net > 0)]
    baseline_losers = all_trades[(all_trades.slow_ma == 60) & (all_trades.ret_net < 0)]
    faster_winners = all_trades[(all_trades.slow_ma == 50) & (all_trades.ret_net > 0)]
    faster_losers = all_trades[(all_trades.slow_ma == 50) & (all_trades.ret_net < 0)]
    lines += ["", "## 交易层面：50 比 60 差在哪里", "",
              "| 已平仓交易 | MA20/50 | MA20/60 |", "|---|---:|---:|",
              f"| 盈利笔数 | {len(faster_winners)} | {len(baseline_winners)} |",
              f"| 平均盈利 | {fmt_pct(faster_winners.ret_net.mean(), 2)} | {fmt_pct(baseline_winners.ret_net.mean(), 2)} |",
              f"| 亏损笔数 | {len(faster_losers)} | {len(baseline_losers)} |",
              f"| 平均亏损 | {fmt_pct(faster_losers.ret_net.mean(), 2)} | {fmt_pct(baseline_losers.ret_net.mean(), 2)} |",
              f"| 最大单笔亏损 | {fmt_pct(faster_losers.ret_net.min(), 2)} | {fmt_pct(baseline_losers.ret_net.min(), 2)} |",
              "", "**MA20/50 的前三大亏损（入场日）**：",
              ""]
    for item in all_trades[all_trades.slow_ma == 50].nsmallest(3, "ret_net").itertuples():
        lines.append(f"- {item.entry_date} → {item.exit_date}：{fmt_pct(item.ret_net, 2)}，{item.exit_reason}")

    lines += ["", "## 机制归因", "",
              "1. **MA20/50 的退出更快，但没有让 A2 更好地抓住长期趋势。** 平仓数量并非显著减少；" 
              "在 2000–02、2022 等震荡熊市区间损失仍明显。其历史最大回撤发生于 "
              "2000-01-03 至 2004-09-27 的累计权益低谷（具体峰谷以 CSV 为准）。",
              "2. **MA20/61 对 2020 反复交叉特别敏感。** 2020-10-07 MA20="
              f"{audit[audit.Date == '2020-10-07'].ma20.iloc[0]:.6f}，"
              "MA60=" f"{audit[audit.Date == '2020-10-07'].ma60.iloc[0]:.6f}，"
              "MA61=" f"{audit[audit.Date == '2020-10-07'].ma61.iloc[0]:.6f}。"
              "MA20 跌破 MA60，但仍高于 MA61；于是 60 触发卖出，61 没有。",
              "3. MA20/60 于 2020-10-08 离场、10-12 重入，随后 10-29 被 N20 提前止损；"
              "MA20/61 持续持有 2020-05 至 2021-03 的上涨段：单笔 +50.07%。不能用这一个精准切点推断 61 日线天然最佳。",
              "4. MA20/62 的最高 Payoff 伴随更少赢家、更大的 MDD；"
              "调慢均线有可能改善单笔平均盈亏，但会更慢确认趋势恶化。",
              "",
              "## 参数稳健性与研究边界", "",
              "- 45、50、55、60、65、70 并没有随参数单调改善；尤其 60→61→62 的跳变，"
              "显示临界交叉日对结果的影响巨大，属于样本内参数敏感性风险。",
              "- 60/61/62 的差别不应在看完历史后直接选最好的；这是一次探索性多重比较。",
              "- MA20/50 本次是假设从入场到正常退出均改成 20/50。若单改进场或单改出场，"
              "属于新实验，不可混同。",
              "- 原 N20 的期末持仓仍未正式退出；若使用只计算已平仓交易的 Payoff，"
              "数值会略不同（基准约3.95而非共同截止3.92）。",
              "- 后续最自然的独立测试是 **MA20/60 死叉连续 2 个收盘日才确认退出**，"
              "保持入场、N20、重新入场机制不变；这是针对 2020 假死叉的机制假设，"
              "**并非** 当前实验的结果。必须检查因此多承受的失败反弹、最大单笔亏损及样本外表现。",
              "", "## 复现", "",
              "运行 `python research/experiments/a2_slow_ma_period_study.py`。输出：",
              "`a2_slow_ma_sweep_metrics.csv`、`a2_slow_ma_selected_trades.csv`、"
              "`a2_slow_ma_strict_start_sensitivity.csv`、`a2_2020_10_false_death_cross_audit.csv`。",
              "原始回测引擎：`research/a2_n20.py`。",
              ""]
    (OUT / "A2-N20_MA20_50_60_61_62_参数研究.md").write_text("\n".join(lines))
    print(numbers[numbers.slow_ma.isin([45, 50, 55, 60, 61, 62, 65, 70])][[
        "slow_ma", "completed_trades", "common_cutoff_payoff", "mdd", "cagr",
        "end_equity", "dot_com_bust_2000_2002_return", "pandemic_2020_return"
    ]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
