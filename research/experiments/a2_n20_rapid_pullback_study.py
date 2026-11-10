"""A2-N20 rapid pullback protection: 5/8% profit arms, 5/8% pullbacks, 2/3/5 days.

Compare rolling local N-day high versus the last *all-time holding* close peak
formed no more than N trading sessions ago. No lookahead; confirmation at
close, next-session adjusted open with 0.05% commission each way.

Run: python research/experiments/a2_n20_rapid_pullback_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))
import a2_n20 as frozen
from a2_n20_fast_reentry_and_deadcross_buffer import (
    Variant, enriched, simulate, summary,
)

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
FAMILIES = [
    ("A2_strict", False, False, "all"),
    ("R1_once_strict", True, False, "fast_only"),
    ("R1_once_legacy", True, True, "fast_only"),
]
DEFINITIONS = ["rapid_alltime_peak_dd", "rapid_rolling_peak_dd"]
ARM_LEVELS = [0.05, 0.08]
PULLBACKS = [0.05, 0.08]
LOOKBACK_DAYS = [2, 3, 5]


def as_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:+.{decimals}f}%"


def config(name: str, fast: bool, scope: str, rule: str = "",
           arm: float = 0.0, pullback: float = 0.0, days: int = 0) -> Variant:
    return Variant(name=name, fast_reentry=fast,
                   fast_retry_limit_per_slow_cycle=1 if fast else None,
                   profit_arm_gain=arm, profit_scope=scope, profit_rule=rule,
                   profit_rule_parameter=pullback, profit_lookback_days=days)


def current_scenario(m: pd.DataFrame, family: str, definition: str,
                     arm: float, pullback: float, days: int) -> pd.Series:
    rows = m[(m.family == family) & (m.definition == definition) &
             (m.profit_arm_gain == arm) & (m.pullback == pullback) &
             (m.window_sessions == days)]
    assert len(rows) == 1
    return rows.iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, meta = frozen.prices()
    bars = enriched(df)
    extension_atr = ((bars.adj_close - bars.ma10) / bars.atr14).dropna()
    over_131_atr = int((extension_atr >= 1.31).sum())
    freeze_t, freeze_e, freeze_stats, freeze_open = frozen.run(df, warm_start=False)
    metrics, focus_trades, baseline = [], [], {}
    for name, fast, warm, scope in FAMILIES:
        experiments = [("none", "", 0.0, 0.0, 0)]
        experiments += [(rule, rule, arm, dd, days)
                        for rule in DEFINITIONS for arm in ARM_LEVELS
                        for dd in PULLBACKS for days in LOOKBACK_DAYS]
        for kind, rule, arm, dd, days in experiments:
            label = f"{name}_{kind}_arm{arm}_dd{dd}_day{days}"
            cfg = config(label, fast, scope, rule, arm, dd, days)
            trades, equity, actions, open_return = simulate(bars, cfg, warm_start=warm)
            if name == "A2_strict" and not rule:
                assert trades.entry_date.tolist() == freeze_t.entry_date.tolist()
                assert trades.exit_date.tolist() == freeze_t.exit_date.tolist()
                np.testing.assert_allclose(trades.ret_net, freeze_t.ret_net, atol=1e-12)
                np.testing.assert_allclose(equity.Equity, freeze_e.Equity, atol=1e-6)
            stat = summary(label, trades, equity, actions, open_return)
            stat.update({"family": name, "definition": kind,
                         "warm_start": warm, "profit_arm_gain": arm,
                         "pullback": dd, "window_sessions": days,
                         "profit_exits": int(trades.exit_reason.str.startswith("PROFIT_").sum())})
            metrics.append(stat)
            if not rule:
                baseline[name] = (trades, equity, open_return)
            if (not rule or
                (rule == "rapid_alltime_peak_dd" and (arm, dd, days) in
                 [(0.05, 0.05, 3), (0.08, 0.05, 3), (0.08, 0.05, 5), (0.08, 0.08, 5)]) or
                (rule == "rapid_rolling_peak_dd" and (arm, dd, days) ==
                 (0.08, 0.05, 5))):
                tr = trades.copy()
                tr.insert(0, "family", name)
                tr.insert(1, "definition", kind)
                tr.insert(2, "profit_arm", arm)
                tr.insert(3, "pullback", dd)
                tr.insert(4, "window_sessions", days)
                focus_trades.append(tr)

    m = pd.DataFrame(metrics)
    t = pd.concat(focus_trades, ignore_index=True)
    m.to_csv(OUT / "a2_rapid_pullback_grid_metrics.csv", index=False)
    t.to_csv(OUT / "a2_rapid_pullback_selected_trades.csv", index=False)

    # Same entry date is an explicit *pairing*, while unmatched fills represent
    # whole-state effects and must be judged from the complete equity curve.
    audits = []
    for name, fast, warm, scope in FAMILIES:
        base_t = baseline[name][0].set_index("entry_date")
        variants = t[(t.family == name) & (t.definition != "none")]
        keys = variants[["definition", "profit_arm", "pullback", "window_sessions"]].drop_duplicates()
        for key in keys.itertuples(index=False):
            other = variants[(variants.definition == key.definition) &
                             (variants.profit_arm == key.profit_arm) &
                             (variants.pullback == key.pullback) &
                             (variants.window_sessions == key.window_sessions)].set_index("entry_date")
            paired = base_t[["ret_net", "exit_date"]].join(
                other[["ret_net", "exit_date", "exit_reason"]],
                how="inner", lsuffix="_base", rsuffix="_test")
            paired["improvement_pp"] = (paired.ret_net_test - paired.ret_net_base) * 100
            for rec in paired[paired.improvement_pp.abs() > 1e-8].reset_index().to_dict("records"):
                audits.append({"family": name, "definition": key.definition,
                               "arm": key.profit_arm, "pullback": key.pullback,
                               "window_sessions": key.window_sessions, **rec})
    pd.DataFrame(audits).to_csv(OUT / "a2_rapid_pullback_trade_attribution.csv", index=False)

    # 2000 diagnostic is only in the historical synthetic warm-start scenario.
    diagnostics = t[(t.family == "R1_once_legacy") &
                    (t.entry_date.isin(["2000-01-21", "2003-04-22", "2019-10-17"]))]
    diagnostics.to_csv(OUT / "a2_rapid_pullback_2000_2003_2019_cases.csv", index=False)

    b = current_scenario(m, "A2_strict", "none", 0, 0, 0)
    assert b.closed == 58 and abs(b.mdd - (-0.2029)) < .001
    assert abs(b.common_payoff - 3.99) < .02
    r = current_scenario(m, "A2_strict", "rapid_alltime_peak_dd", .08, .05, 5)
    assert abs(r.common_payoff - 4.13) < .02
    assert int(r.profit_exits) == 6

    lines = [
        "# A2-N20｜盈利5%/8%后，2/3/5交易日急跌5%/8%止盈：全样本网格",
        "", "**探索性回测｜2026-11-09｜不能代替独立样本外验证。**", "",
        "## 信号与撮合定义", "",
        "- 数据2000-01-03至2026-07-13（同一拼接复权QQQ OHLC），单边手续费0.05%，无额外滑点、现金利息；日线收盘确认，下一交易日开盘卖出。",
        "- 基准A2：日线MA20/60新金叉全仓，前20交易日MA10新下穿MA20且收盘低于买价则N20止损；正常MA20/60死叉退出；下一次全新金叉重入。",
        "- R1_once：仅N20卖后可在MA20仍高于MA60时进行一次MA10/20新金叉快速买回，买回重置N20。止盈只作用于快速买回的仓位。",
        "- 盈利门槛5%或8%，以**曾经达到的最高收盘浮盈**判定并激活，之后持续生效，不能看当天未来走势。",
        "- 回撤门槛5%或8%，用收盘确认，不保证按止损价格成交；单笔成本、跳空、正常N20/MA20/60退出完整计入。",
        "- *持仓历史峰值窗口*: `rapid_alltime_peak_dd`：从自买入以来的最高收盘价回撤x%，且那个历史最高点形成于最近N个交易日内。",
        "- *局部滚动峰值窗口*: `rapid_rolling_peak_dd`：当前收盘价相对过去N个交易日及当天的最高收盘价回撤x%；不要求这个高点也是持仓以来的最高点。",
        "- 另外保留 `none` 作为原始A2或R1不加止盈的对照。严格新金叉是主报告口径；R1历史合成起点单列以审计2000异常。",
        "- Payoff和PF包含截止时仍持仓头寸的假设卖出费；交易分布和卖出原因只使用已完成交易。",
        f"- ATR校验：在全部{len(extension_atr)}个ATR14可计算的日线样本里，"
        f"收盘价高于MA10至少1.31 ATR出现{over_131_atr}次"
        f"（{100*over_131_atr/len(extension_atr):.2f}%的交易日），确实并不罕见；"
        "这不是足够支持全仓止盈的独立条件。",
        "", "## 基本算术：浮盈幅度不等于保本保证", "",
        "- +5%后下跌5%：1.05×0.95−1=−0.25%（不含费用与跳空）。",
        "- +8%后下跌8%：1.08×0.92−1=−0.64%（不含费用与跳空）。",
        "- +8%后下跌5%：1.08×0.95−1=+2.60%（仅阈值正好成交的理论值）。",
        "", "## 原始 A2 严格新金叉：持仓最高价在最近N日内急跌", "",
        "| 曾经浮盈达到 | 急跌阈值 | 2日 Payoff / CAGR / 次数 | 3日 Payoff / CAGR / 次数 | 5日 Payoff / CAGR / 次数 |",
        "|---|---|---:|---:|---:|"]
    for arm in ARM_LEVELS:
        for pullback in PULLBACKS:
            pieces = []
            for days in LOOKBACK_DAYS:
                q = current_scenario(m, "A2_strict", "rapid_alltime_peak_dd", arm, pullback, days)
                pieces.append(f"{q.common_payoff:.2f} / {as_pct(q.cagr)} / {q.profit_exits}")
            lines.append(f"| {as_pct(arm)} | {as_pct(pullback)} | " + " | ".join(pieces) + " |")
    lines += ["", "对应原始A2严格新金叉：Payoff3.99、CAGR5.84%、MDD−20.29%。", "",
              "## 相同网格：过去N日局部峰值急跌（更敏感）", "",
              "| 曾经浮盈达到 | 急跌阈值 | 2日 Payoff / CAGR / 次数 | 3日 Payoff / CAGR / 次数 | 5日 Payoff / CAGR / 次数 |",
              "|---|---|---:|---:|---:|"]
    for arm in ARM_LEVELS:
        for pullback in PULLBACKS:
            pieces = []
            for days in LOOKBACK_DAYS:
                q = current_scenario(m, "A2_strict", "rapid_rolling_peak_dd", arm, pullback, days)
                pieces.append(f"{q.common_payoff:.2f} / {as_pct(q.cagr)} / {q.profit_exits}")
            lines.append(f"| {as_pct(arm)} | {as_pct(pullback)} | " + " | ".join(pieces) + " |")
    lines += ["", "## 重点样本：持仓历史最高收盘价近5日回撤5%、曾浮盈≥8%", "",
              "主策略A2严格新金叉：触发6次（其中2026年一笔本来仍持仓至样本末日）。",
              "| 买入日期 | 提前退出日 | 净收益 | 退出原因 |",
              "|---|---|---:|---|"]
    selected = t[(t.family == "A2_strict") &
                 (t.definition == "rapid_alltime_peak_dd") &
                 (t.profit_arm == 0.08) & (t.pullback == 0.05) &
                 (t.window_sessions == 5) &
                 (t.exit_reason.str.startswith("PROFIT_"))]
    for row in selected.itertuples():
        lines.append(f"| {row.entry_date} | {row.exit_date} | {as_pct(row.ret_net)} | {row.exit_reason} |")
    lines += ["", "## 重要归因与边界", "",
              "1. 持仓历史峰值限时方案比局部滚动峰值方案更少误杀赢家。后者常在长趋势中局部反复下跌时提前出场，尤其会砍掉2009大赢家；见逐笔差异CSV。",
              "2. A2严格样本`arm8、回撤5%、窗口5日`历史Payoff≈4.13、CAGR≈6.61%、MDD≈−20.29%，但这些是看完历史才选出的阈值，不可直接采用。",
              "3. 该方案不是纯粹提前平仓同样笔数：原始A2完成58笔且期末有一笔未平仓，本方案完成59笔且期末空仓；比较用权益曲线而不是把胜负平均简单相加。",
              "4. 另一个风格测试：R1严格一次快重入、止盈仅作用于快速仓位，几乎没有足够的快跌触发；其最差MDD≈−26.09%依旧未变。",
              "5. 2000年1月快速重入大浮盈案例出自旧回测**人为合成初仓**，在严格新金叉回测中并不存在；仅可用作机制诊断，不可当独立证据。",
              "   更重要的是，在旧初仓案例中，8%浮盈激活+持仓新高5日内回撤5%会在**2000-03-15即卖出+10.25%**，"
              "**早于**2000-03-27后来形成的+22.02%最高浮盈。不能先看到后来22%的高点再倒推止盈日，这会产生事后筛选偏差。",
              "6. N天定义：N=2允许相隔最多2个交易日的历史高点；检查当天收盘触发、下个交易日开盘退出。不可能保证保本。",
              "7. 至今没有足够的预先保留的样本外数据证明该模式；后续必须冻结参数、滚动验证和检验跨资产稳定性。",
              "", "## 文件与复现", "",
              "`python research/experiments/a2_n20_rapid_pullback_study.py`。",
              "`a2_rapid_pullback_grid_metrics.csv`、`a2_rapid_pullback_selected_trades.csv`、"
              "`a2_rapid_pullback_trade_attribution.csv`、`a2_rapid_pullback_2000_2003_2019_cases.csv`。", ""]
    (OUT / "A2-N20_盈利5_8_快速回撤2_3_5日_网格回测.md").write_text("\n".join(lines))

    targets = m[((m.family == "A2_strict") & (m.definition != "none")) |
                ((m.family == "R1_once_strict") & (m.definition == "none"))]
    print(targets[["family", "definition", "profit_arm_gain", "pullback",
                   "window_sessions", "common_payoff", "mdd", "cagr",
                   "profit_exits", "closed"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
