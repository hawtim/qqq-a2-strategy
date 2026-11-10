"""Post-20%-gain profit protection: declining closes, peak giveback, MA10/ATR.

Standalone reproducible experiment, using the frozen QQQ A2-N20 engine and
R1 once-only MA10/20 fast-reentry simulator. Close-based signals, next open
fills, 0.05% fees each side, no cash interest or intraday stops.

Run: python research/experiments/a2_n20_profit_guard_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))
import a2_n20 as frozen
from a2_n20_fast_reentry_and_deadcross_buffer import (
    Variant, enriched, simulate, summary,
)

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"
RULES = [
    ("none", "", 0),
    ("down2", "down_days", 2),
    ("down3", "down_days", 3),
    ("down4", "down_days", 4),
    ("peak5", "peak_drawdown", 0.05),
    ("peak6", "peak_drawdown", 0.06),
    ("peak8", "peak_drawdown", 0.08),
    ("peak10", "peak_drawdown", 0.10),
    ("peak12", "peak_drawdown", 0.12),
    ("belowMA10_1ATR", "below_ma10_atr", 1),
    ("belowMA10_2ATR", "below_ma10_atr", 2),
    ("belowMA10_3ATR", "below_ma10_atr", 3),
    ("aboveMA10_2ATR", "above_ma10_atr", 2),
    ("aboveMA10_3ATR", "above_ma10_atr", 3),
]
COHORTS = [
    ("R1_fast_only_legacy", True, True, "fast_only"),
    ("R1_fast_only_strict", False, True, "fast_only"),
    ("R1_all_strict", False, True, "all"),
    ("A2_all_strict", False, False, "all"),
]
FOCUS_ENTRIES = ["2000-01-21", "2003-04-22", "2019-10-17"]


def p(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f}%"


def make_variant(name: str, rule: str, param: float,
                 fast: bool, scope: str, arm: float = 0.20) -> Variant:
    return Variant(name=name, fast_reentry=fast,
                   fast_retry_limit_per_slow_cycle=1 if fast else None,
                   profit_rule=rule, profit_rule_parameter=param,
                   profit_arm_gain=arm, profit_scope=scope)


def study():
    OUT.mkdir(parents=True, exist_ok=True)
    df, meta = frozen.prices()
    bars = enriched(df)
    metrics, trades, focus, arm_test = [], [], [], []
    for cohort, warm, fast, scope in COHORTS:
        for label, rule, param in RULES:
            cfg = make_variant(cohort + "_" + label, rule, param, fast, scope)
            t, e, a, op = simulate(bars, cfg, warm_start=warm)
            stat = summary(cfg.name, t, e, a, op)
            stat.update({"cohort": cohort, "warm_start": warm,
                         "scope": scope, "profit_rule": label,
                         "armed_gain": 0.20,
                         "profit_exit_count": int(t.exit_reason.str.startswith("PROFIT_").sum())})
            metrics.append(stat)
            if label in ("none", "down2", "down3", "peak5", "peak8", "peak10",
                         "belowMA10_1ATR", "belowMA10_2ATR", "belowMA10_3ATR",
                         "aboveMA10_3ATR"):
                t2 = t.copy()
                t2.insert(0, "cohort", cohort)
                t2.insert(1, "profit_rule", label)
                trades.append(t2)
                if cohort == "R1_fast_only_legacy":
                    focus.append(t2[t2.entry_date.isin(FOCUS_ENTRIES)])
        if cohort.startswith("R1_fast_only_"):
            for armed in (0.15, 0.20, 0.25):
                for trail in (0.06, 0.08, 0.10):
                    cfg = make_variant("sensitivity", "peak_drawdown", trail, fast, scope, armed)
                    t, e, a, op = simulate(bars, cfg, warm_start=warm)
                    stats = summary(cfg.name, t, e, a, op)
                    case_2000 = t[t.entry_date == "2000-01-21"]
                    arm_test.append({"cohort": cohort, "profit_arm": armed,
                                     "trail_pct": trail,
                                     "profit_exit_count": int(t.exit_reason.str.startswith("PROFIT_").sum()),
                                     "sample_2000_exit_date": case_2000.exit_date.iloc[0] if not case_2000.empty else "not_held",
                                     "sample_2000_net": float(case_2000.ret_net.iloc[0]) if not case_2000.empty else None,
                                     "payoff": stats["common_payoff"],
                                     "mdd": stats["mdd"],
                                     "cagr": stats["cagr"]})

    m = pd.DataFrame(metrics)
    t_all = pd.concat(trades, ignore_index=True)
    focused = pd.concat(focus, ignore_index=True)
    sensitive = pd.DataFrame(arm_test)
    m.to_csv(OUT / "a2_profit_guard_metrics.csv", index=False)
    t_all.to_csv(OUT / "a2_profit_guard_selected_trades.csv", index=False)
    focused.to_csv(OUT / "a2_profit_guard_2000_2003_2019_cases.csv", index=False)
    sensitive.to_csv(OUT / "a2_profit_guard_arm_threshold_sensitivity.csv", index=False)

    old_warm = m[(m.cohort == "R1_fast_only_legacy") & (m.profit_rule == "none")].iloc[0]
    old_strict = m[(m.cohort == "R1_fast_only_strict") & (m.profit_rule == "none")].iloc[0]
    a2_strict = m[(m.cohort == "A2_all_strict") & (m.profit_rule == "none")].iloc[0]
    assert old_warm.closed == 69 and abs(old_warm.mdd + 0.42) < .001
    assert old_strict.closed == 67 and abs(old_strict.mdd + 0.2609) < .001
    assert a2_strict.closed == 58 and abs(a2_strict.mdd + 0.2029) < .001
    # In the strict initialization the historical synthetic Jan-2000 trade
    # should never exist, so it cannot drive research selection.
    assert "2000-01-21" not in t_all[t_all.cohort == "R1_fast_only_strict"].entry_date.values

    # Daily causal evidence around 2000-03-27 peak.
    daily = bars[bars.Date.between("2000-03-23", "2000-04-07")].copy()
    price_peak = float(daily.loc[daily.Date == "2000-03-27", "adj_close"].iloc[0])
    daily["distance_above_ma10_atr"] = (daily.adj_close - daily.ma10) / daily.atr14
    daily["drawdown_from_2000_03_27_peak"] = daily.adj_close / price_peak - 1
    daily["down_vs_prior_close"] = daily.adj_close < bars.adj_close.shift(1).loc[daily.index]
    daily[["Date", "adj_open", "adj_high", "adj_low", "adj_close", "ma10", "atr14",
           "distance_above_ma10_atr", "drawdown_from_2000_03_27_peak",
           "down_vs_prior_close", "protect_cross"]].to_csv(
               OUT / "a2_profit_guard_2000_peak_daily_path.csv", index=False)

    # Exact outcome of different sale rules on the 2000 synthetic-start sample.
    sample = focused[focused.entry_date == "2000-01-21"].set_index("profit_rule")
    assert abs(sample.loc["none", "ret_net"] - (-.0644075)) < 1e-6
    assert abs(sample.loc["peak8", "ret_net"] - (.1195)) < .0002
    assert sample.loc["belowMA10_3ATR", "exit_date"] == "2000-04-20"

    core = ["none", "down2", "down3", "peak5", "peak8", "peak10",
            "belowMA10_1ATR", "belowMA10_3ATR", "aboveMA10_3ATR"]
    lines = [
        "# A2-N20｜浮盈20%后的利润保护：连续下跌、ATR、最高收盘价回撤",
        "", "**探索性历史回测｜2026-11-09｜尚未进行样本外验证。**", "",
        "## 一次只变更一条退出规则", "",
        "- 冻结数据：复权QQQ OHLC，2000-01-03至2026-07-13；费用单边0.05%，无滑点、现金利息；信号收盘确认，下交易日开盘执行。",
        "- 原始A2：MA20/60新金叉入、死叉出；前20交易日MA10/20新死叉且浮亏则N20卖；严格等待全新MA20/60金叉重入。",
        "- R1_once：只在N20退出后且仍MA20>MA60时允许一次MA10/20金叉快速重入；买回重置N20；二次失败不再快速重入。",
        "- 止盈统一激活条件：持仓期间截至当日已观察到的**最高复权收盘价/买入价−1达到20%**；未达到时不触发。",
        "- 对比三种家族：①连续2/3/4天收盘价递减；②从持仓最高收盘价回撤5/6/8/10/12%；③收盘价低于MA10若干ATR14，或高于MA10若干ATR14。",
        "- ATR14：用复权High/Low/前Close计算True Range，再用Wilder式EMA(alpha=1/14, adjust=False, min_periods=14)。",
        "- 止盈不做盘中成交、不过度理想化；阈值触发后的次日开盘可以发生跳空；止盈出场后仍须等新的MA20/60金叉。",
        "- **两种作用范围**：仅快速重入仓位(fast_only)、全部持仓(all)；并将原始A2与R1严格新金叉初始化分开。",
        "- Payoff含未平仓头寸截至样本末日假设付出退出费的收益；其他交易分布只统计已完成交易。",
        "", "## 2000-01-21 合成首仓后的第一笔快速重入（诊断用，不得当作真实首个金叉）", "",
        "当日复权开盘买入价81.3143。2000-03-27收盘99.2203，最高浮盈+22.02%；"
        "2000-03-29收盘92.69，相对峰值−6.58%，2000-04-05收盘84.37，相对峰值−14.97%。",
        "", "| 止盈信号 | 卖出开盘日期 | 净收益 |", "|---|---|---:|"]
    for name in core:
        r = sample.loc[name]
        lines.append(f"| {name} | {r.exit_date} | {p(r.ret_net)} |")
    lines += ["", "### 重要诊断", "",
              "- 当2000-03-27最大浮盈时，收盘价仅高于MA10约1.31 ATR。至MA10/20于04-05死叉之前，"
              "价格最远只低于MA10约1.67 ATR。因此**MA10±3ATR不是这个样本的提前止盈信号**。",
              "- 两日连跌：2000-03-29收盘确认，03-30开盘卖出，约+11.03%。",
              "- 峰值回撤8%：03-30收盘确认，03-31开盘卖出，约+11.95%。",
              "- 这些利润保护规则并未在最高点卖出；'最大浮盈22%'是事后审计的峰值，不是可事前知道的目标。",
              "", "## 严格新金叉初始化：A2全部仓位增加止盈（全市场交易泛化检验）", "",
              "| 止盈规则 | 已完成交易 | 止盈触发 | Payoff | MDD | CAGR | 盈利>=20%笔数 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in core:
        q = m[(m.cohort == "A2_all_strict") & (m.profit_rule == name)].iloc[0]
        lines.append(f"| {name} | {q.closed} | {q.profit_exit_count} | {q.common_payoff:.2f} | "
                     f"{p(q.mdd)} | {p(q.cagr)} | {q.gain_over_20pct} |")
    lines += ["", "## 限定R1快速重入仓位止盈：消除合成首仓后的真实观察", "",
              "| 止盈规则 | 止盈触发次数 | Payoff | MDD | CAGR | 亏损>5%笔数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in core:
        q = m[(m.cohort == "R1_fast_only_strict") & (m.profit_rule == name)].iloc[0]
        lines.append(f"| {name} | {q.profit_exit_count} | {q.common_payoff:.2f} | "
                     f"{p(q.mdd)} | {p(q.cagr)} | {q.loss_over_5pct} |")
    lines += ["", "## 快速重入的三笔诊断样本", "",
              "| 规则 | 2000（合成起点） | 2003赢家 | 2019—20疫情前趋势 |",
              "|---|---:|---:|---:|"]
    for name in ["none", "down2", "down3", "peak5", "peak8", "peak10", "belowMA10_3ATR"]:
        vals = focused[focused.profit_rule == name].set_index("entry_date")
        lines.append(f"| {name} | {p(vals.loc['2000-01-21','ret_net'])} | "
                     f"{p(vals.loc['2003-04-22','ret_net'])} | "
                     f"{p(vals.loc['2019-10-17','ret_net'])} |")
    lines += ["", "## 15%、20%、25%激活阈值的稳健性", "",
              "以R1的快速重入仓位、峰值回撤6/8/10%网格试验，详细见"
              "`a2_profit_guard_arm_threshold_sensitivity.csv`。",
              "仅提高激活阈值至25%，2000样本即完全无法触发盈利保护，证明20%不是天然临界数字。",
              "", "## 研究判断", "",
              "- **能解释2000不是通用化证明**：该笔源自旧版合成起点，严格新金叉A2中不存在。",
              "- R1严格新金叉fast_only的峰值回撤8%规则，全历史仅触发1笔（2019-10-17开始的快速入场，2020-02-26获利+11.72%卖出）。"
              "这不足以支持通用性；原始快速重入交易后于2020-03-12以−6.13%退出。",
              "- 2003-04-22快速重入的大赢家原本约+33.66%；'连续2日下跌'规则在2003-07-11提前退出，"
              "只剩约+18.29%，属于大赢家误杀。峰值回撤8%没有误杀这笔。",
              "- 应用至原始A2的全部仓位时，连续2日下跌与ATR偏离容易截断赢家；"
              "历史Payoff/CAGR都变差，不能简单升级为全局止盈规则。",
              "- **严格新金叉下R1的MDD约−26.09%不变**，因为最差2000年7月首次快速重入亏损−11.11%，"
              "它从未浮盈到20%，利润保护根本不会启动。",
              "- 因此可以探索'浮盈20%后从最高收盘价回撤8%'作为快速重入仓位的风险子模块，"
              "但不得基于小样本或阈值优化直接实盘；需滚动样本外测试及时间/ATR阈值邻域稳健性。",
              "", "## 复现", "",
              "`python research/experiments/a2_n20_profit_guard_study.py`。CSV产出："
              "`a2_profit_guard_metrics.csv`、`a2_profit_guard_selected_trades.csv`、"
              "`a2_profit_guard_2000_2003_2019_cases.csv`、"
              "`a2_profit_guard_arm_threshold_sensitivity.csv`、"
              "`a2_profit_guard_2000_peak_daily_path.csv`。", ""]
    (OUT / "A2-N20_盈利20后止盈保护_独立回测.md").write_text("\n".join(lines))

    print(m[m.cohort.isin(["R1_fast_only_legacy", "R1_fast_only_strict", "A2_all_strict"]) &
            m.profit_rule.isin(["none", "down2", "down3", "peak5", "peak8", "peak10",
                                "belowMA10_1ATR", "belowMA10_3ATR", "aboveMA10_3ATR"])][
            ["cohort", "profit_rule", "profit_exit_count", "common_payoff", "mdd", "cagr"]
          ].round(4).to_string(index=False))


if __name__ == "__main__":
    study()
