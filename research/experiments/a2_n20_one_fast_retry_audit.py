"""A2-N20: limit MA10/20 re-entry to one attempt per fresh MA20/60 cycle.

Crucially audits both historical synthetic warm-start and strict fresh-cross
initialization. Original frozen engine is used to verify baseline in each mode.

Run: python research/experiments/a2_n20_one_fast_retry_audit.py
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
    Variant, enriched, period_change, simulate, summary,
)

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"


def percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, meta = frozen.prices()
    prices = enriched(df)
    scenarios = [
        Variant("A2_original"),
        Variant("R1_max_one_retry", fast_reentry=True, fast_retry_limit_per_slow_cycle=1),
        Variant("R1_max_two_retries", fast_reentry=True, fast_retry_limit_per_slow_cycle=2),
        Variant("R1_unlimited", fast_reentry=True),
    ]

    rows, all_trades = [], []
    for warm_start in (True, False):
        frozen_t, frozen_e, frozen_s, frozen_active = frozen.run(
            df, warm_start=warm_start
        )
        for cfg in scenarios:
            t, e, actions, open_ret = simulate(prices, cfg, warm_start=warm_start)
            if cfg.name == "A2_original":
                assert t.entry_date.tolist() == frozen_t.entry_date.tolist()
                assert t.exit_date.tolist() == frozen_t.exit_date.tolist()
                np.testing.assert_allclose(t.ret_net, frozen_t.ret_net, atol=1e-12)
                np.testing.assert_allclose(e.Equity, frozen_e.Equity, atol=1e-6)
                assert abs(e.Drawdown.min() - frozen_s["mdd"]) < 1e-10
            stat = summary(cfg.name, t, e, actions, open_ret)
            trough = e.Drawdown.idxmin()
            peak = e.Equity.loc[:trough].idxmax()
            stat.update({
                "warm_start": warm_start,
                "initialization": "historical_synthetic_position" if warm_start else "strict_fresh_cross_only",
                "mdd_peak_date": e.Date.iloc[peak],
                "mdd_trough_date": e.Date.iloc[trough],
                "mdd_peak_equity": float(e.Equity.iloc[peak]),
                "mdd_trough_equity": float(e.Equity.iloc[trough]),
                "return_2000_2002": period_change(e, "2000-01-03", "2002-12-31"),
                "return_2003_2004": period_change(e, "2003-01-01", "2004-12-31"),
                "return_2022": period_change(e, "2022-01-01", "2022-12-31"),
                "cycle_with_second_retry": int((t.fast_uses_at_entry >= 2).sum()),
            })
            rows.append(stat)
            tr = t.copy()
            tr.insert(0, "warm_start", warm_start)
            tr.insert(1, "scenario", cfg.name)
            all_trades.append(tr)

    summary_df = pd.DataFrame(rows)
    trades_df = pd.concat(all_trades, ignore_index=True)
    summary_df.to_csv(OUT / "a2_one_fast_retry_metrics.csv", index=False)
    trades_df.to_csv(OUT / "a2_one_fast_retry_trades.csv", index=False)

    synthetic = summary_df[summary_df.warm_start].set_index("scenario")
    strict = summary_df[~summary_df.warm_start].set_index("scenario")
    assert synthetic.loc["A2_original", "closed"] == 59
    assert strict.loc["A2_original", "closed"] == 58
    assert synthetic.loc["R1_max_one_retry", "fast_reentries"] == 10
    assert synthetic.loc["R1_unlimited", "fast_reentries"] == 11
    assert abs(synthetic.loc["R1_max_one_retry", "mdd"] - (-0.4200)) < 0.001
    assert abs(strict.loc["R1_max_one_retry", "mdd"] - (-0.2609)) < 0.001

    relevant = trades_df[
        ((trades_df.entry_date >= "2000-01-01") & (trades_df.entry_date < "2002-01-01")) |
        (trades_df.entry_date == "2003-04-22") |
        (trades_df.entry_date == "2015-04-17")
    ].copy()
    relevant.to_csv(OUT / "a2_one_fast_retry_causal_trades.csv", index=False)

    lines = [
        "# A2-N20｜每轮 MA20/60 多头周期仅重试一次：因果归因与初始化敏感性",
        "", "**研究时间：2026-11-09，历史探索回测（非样本外结果）。**", "",
        "## 新规则的唯一变化", "",
        "- 原版A2：MA20/60新金叉全仓、MA20/60死叉全卖；前20交易日MA10/20新死叉且浮亏则下一开盘止损；卖出后仅等全新MA20/60金叉。",
        "- R1无限次：N20退出后，若仍处MA20>MA60多头状态，等下一次MA10/20金叉快速买回；买回重置N20，允许再重复。",
        "- **R1最多重试一次**：每次新的MA20/60金叉启动一个周期。首次N20失败可快买回一次；若快速买回的仓位再次被N20止损，**立即停用快速重入，等待下一次新的MA20/60金叉**。正常MA20/60死叉退出也会结束该周期。",
        "- R1最多重试两次作为诊断点；绝不改变仓位、成本、趋势状态判定或止损条件。",
        "- 历史区间2000-01-03至2026-07-13，复权QQQ、单边0.05%费、无额外滑点与现金利息；收盘确认次日开盘执行。",
        "- **必须审计起点**：warm_start=True合成首日仓位，只为精确复现旧版59笔；warm_start=False严格等待首个真正MA20/60金叉，为执行逻辑更忠实的敏感性检验。",
        "- Payoff以已平仓+期末未平仓按假设退出费标记；MDD为日末总资产历史峰谷回撤。",
        "", "## 同样初始化（复现旧版59笔）", "",
        "| 策略 | 完成交易 | 快买回 | Payoff | PF | MDD | CAGR | 最差单笔 | ≥5%亏损交易 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for row in synthetic.itertuples():
        lines.append(f"| {row.Index} | {row.closed} | {row.fast_reentries} | {row.common_payoff:.2f} | "
                     f"{row.common_pf:.2f} | {percent(row.mdd)} | {percent(row.cagr)} | "
                     f"{percent(row.worst_closed)} | {row.loss_over_5pct} |")
    lines += ["", "## 严格新MA20/60金叉初始化（去除合成首仓）", "",
              "| 策略 | 完成交易 | 快买回 | Payoff | PF | MDD | CAGR | 2000–02收益 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in strict.itertuples():
        lines.append(f"| {row.Index} | {row.closed} | {row.fast_reentries} | {row.common_payoff:.2f} | "
                     f"{row.common_pf:.2f} | {percent(row.mdd)} | {percent(row.cagr)} | "
                     f"{percent(row.return_2000_2002)} |")

    lines += ["", "## 为什么限制重试仍不改善最大回撤？", "",
              "1. **全历史重复重试恰好只有1笔**：2015-04-17第二次快速买回，2015-07-10退出，净 +0.86%。"
              "限制一次主要是删掉这笔小盈利，未改变最坏风险阶段。",
              "2. **首次重试就已经足以导致风险**：2000-07-17快速买回，2000-08-01触发N20，"
              "亏损−11.11%；这发生在任何一个新20/60多头周期的第一次重试。",
              "3. **2000年初回撤高度依赖合成首仓**：旧回测2000-01-03凭warm-start虚拟买入，"
              "2000-01-18 N20止损；随后2000-01-21 MA10/20金叉重新买回，"
              "持仓期间曾浮盈约+22.02%，但2000-04-20以−6.44%平仓；"
              "该笔先使权益在2000-03-27创新高，再与后续亏损共同形成旧版−42%峰谷回撤。",
              "4. 去掉这笔合成首仓后，真实规则严格起点下，R1一次重试的MDD约−26.09%；"
              "仍高于严格原版A2的−20.29%，但并非−42%。",
              "5. **2003年的高质量机会保留**：2003-03-24原买入被N20以−1.50%止损后，"
              "2003-04-22第一次快速重入，至2004-03-10赚+33.66%；限制一次没有误杀它。",
              "", "## 结论和下一步", "",
              "- '次数上限=1'只能减少重复接飞刀，不能修复首次错误重入、快速获利回吐或整体熊市敞口。",
              "- 不能从『单次重试』的直觉推导MDD必然大幅降低；本次完整回测证伪了这个假设。",
              "- 未来若继续探索，应该先检查**第一次快速重入时趋势处于什么状态**，"
              "而不是继续微调重试次数；仍须维持A2原版作为冻结的对照组，并滚动样本外验证。",
              "- 这些样本内结果并不能证明新规则未来有效；单次亏损的成交仍可能因滑点/跳空恶化。",
              "", "## 复现", "",
              "`python research/experiments/a2_n20_one_fast_retry_audit.py`",
              "（依赖同目录 `a2_n20_fast_reentry_and_deadcross_buffer.py` 与冻结引擎 `research/a2_n20.py`）。",
              "数据文件：`a2_one_fast_retry_metrics.csv`、`a2_one_fast_retry_trades.csv`、"
              "`a2_one_fast_retry_causal_trades.csv`。", ""
    ]
    (OUT / "A2-N20_每轮只重试一次_真实金叉敏感性.md").write_text("\n".join(lines))
    print(summary_df[["initialization", "scenario", "closed", "fast_reentries",
                      "common_payoff", "mdd", "cagr", "return_2000_2002",
                      "loss_over_5pct"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
