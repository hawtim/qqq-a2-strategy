"""A2-N20 pre-entry quality audit. Keeps entry/exit/re-entry fixed.

Reproduce: python research/experiments/a2_n20_entry_quality_study.py
Only information available at the gold-cross *closing signal* can enter a gate.
Gated entries are skipped, and the system waits for a fresh MA20/60 gold cross.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))
import a2_n20 as frozen
from a2_stops_and_entry_filters import Gate, prepare, simulate

OUT = ROOT / "research" / "results" / "experiments_2026_11_09"


@dataclass(frozen=True)
class Test:
    label: str
    family: str
    field: str
    op: str
    cutoff: float

    def condition(self, df: pd.DataFrame) -> pd.Series:
        return df[self.field].gt(self.cutoff) if self.op == ">" else df[self.field].lt(self.cutoff)


TESTS = [
    Test("MA60过去10日下降", "MA60斜率", "ma60_slope10", "<", 0.0),
    Test("MA60过去10日下降超过0.5%", "MA60斜率", "ma60_slope10", "<", -.005),
    Test("MA60过去10日几乎水平(±0.25%)", "MA60斜率", "ma60_abs_slope10", "<", .0025),
    Test("MA200过去20日下降", "MA200方向", "ma200_slope20", "<", 0.0),
    Test("MA200过去20日下降超过1%", "MA200方向", "ma200_slope20", "<", -.01),
    Test("过去63日出现至少2次均线交叉", "反复交叉", "cross_count63", ">", 1.5),
    Test("过去63日出现至少3次均线交叉", "反复交叉", "cross_count63", ">", 2.5),
    Test("过去126日出现至少3次均线交叉", "反复交叉", "cross_count126", ">", 2.5),
    Test("过去126日出现至少4次均线交叉", "反复交叉", "cross_count126", ">", 3.5),
    Test("过去20日年化波动率超过25%", "波动率", "vol20_ann", ">", .25),
    Test("过去20日年化波动率超过35%", "波动率", "vol20_ann", ">", .35),
    Test("ATR14占收盘价超过2%", "波动率", "atr14_pct", ">", .02),
    Test("ATR14占收盘价超过3%", "波动率", "atr14_pct", ">", .03),
    Test("股价高于MA200超过8%", "过度延伸", "dist200", ">", .08),
    Test("股价高于MA200超过10%", "过度延伸", "dist200", ">", .10),
    Test("股价高于MA200超过12%", "过度延伸", "dist200", ">", .12),
    Test("过去60日上涨超过5%", "近期涨幅", "ret60", ">", .05),
    Test("过去60日上涨超过8%", "近期涨幅", "ret60", ">", .08),
    Test("过去20日上涨超过10%", "近期涨幅", "ret20", ">", .10),
    Test("金叉时相比过去60日低点已涨15%", "底部反弹幅度", "rebound60", ">", .15),
    Test("金叉时相比过去60日低点已涨20%", "底部反弹幅度", "rebound60", ">", .20),
    Test("MA200下降且MA60下降", "交互假设", "both_ma_down", ">", .5),
    Test("股价低于MA200且MA60下降", "交互假设", "below200_ma60_down", ">", .5),
]


def enriched_prices(prices: pd.DataFrame) -> pd.DataFrame:
    d = prepare(prices)
    d["ma60_slope10"] = d.ma60 / d.ma60.shift(10) - 1
    d["ma60_abs_slope10"] = d.ma60_slope10.abs()
    d["ma200_slope20"] = d.ma200 / d.ma200.shift(20) - 1
    crosses = (d.gold | d.death).astype(float)
    # Exclude current golden cross, so this is an actual prior-market feature.
    d["cross_count63"] = crosses.shift(1).rolling(63, min_periods=63).sum()
    d["cross_count126"] = crosses.shift(1).rolling(126, min_periods=126).sum()
    d["vol20_ann"] = d.adj_close.pct_change().rolling(20).std() * np.sqrt(252)
    previous = d.adj_close.shift(1)
    tr = pd.concat([d.adj_high - d.adj_low,
                    (d.adj_high - previous).abs(),
                    (d.adj_low - previous).abs()], axis=1).max(axis=1)
    d["atr14_pct"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / d.adj_close
    d["rebound60"] = d.adj_close / d.adj_close.rolling(60, min_periods=60).min() - 1
    d["both_ma_down"] = ((d.ma200_slope20 < 0) & (d.ma60_slope10 < 0)).astype(float)
    d["below200_ma60_down"] = ((d.dist200 < 0) & (d.ma60_slope10 < 0)).astype(float)
    return d


def tab_medians(trades: pd.DataFrame):
    cols = ["rebound60", "ret20", "ret60", "dist200", "ma60_slope10",
            "ma200_slope20", "cross_count63", "cross_count126", "vol20_ann", "atr14_pct"]
    slices = [
        ("全部完成交易", trades),
        ("盈利交易", trades[trades.ret_net > 0]),
        ("亏损交易", trades[trades.ret_net < 0]),
        ("N20早期保护", trades[trades.exit_reason == "N20 protection"]),
        ("大赢家≥20%", trades[trades.ret_net >= .20]),
    ]
    return pd.DataFrame([{"cohort": label, "n": len(part),
                          **{key + "_median": part[key].median() for key in cols}}
                         for label, part in slices])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, metadata = frozen.prices()
    d = enriched_prices(df)
    t_ref, e_ref, refstats, open_ref = frozen.run(df, warm_start=False)
    t0, e0, base_metrics, base_skips = simulate(d, "A2_strict", warm_start=False)
    assert len(t_ref) == len(t0) == 58
    assert t_ref.entry_date.tolist() == t0.entry_date.tolist()
    assert t_ref.exit_date.tolist() == t0.exit_date.tolist()
    np.testing.assert_allclose(t_ref.ret_net.to_numpy(), t0.ret_net.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(e_ref.Equity.to_numpy(), e0.Equity.to_numpy(), atol=1e-6)
    assert not base_skips.shape[0]

    # Every row is what would have been visible at the close triggering
    # the original A2 entry. Completed-only outcomes are known later.
    baseline = t_ref.join(d.loc[t_ref.signal_idx, [
        "dist200", "ret60", "ret20", "ma60_slope10", "ma60_abs_slope10",
        "ma200_slope20", "cross_count63", "cross_count126", "vol20_ann",
        "atr14_pct", "rebound60", "both_ma_down", "below200_ma60_down",
    ]].reset_index(drop=True), rsuffix="_audit")
    assert baseline.signal_idx.notna().all()
    assert baseline["cross_count126"].notna().all()
    baseline["is_win"] = baseline.ret_net > 0
    baseline["is_n20"] = baseline.exit_reason.eq("N20 protection")
    baseline["is_big_win"] = baseline.ret_net >= .20
    baseline["is_loss5"] = baseline.ret_net <= -.05
    baseline["entry_era"] = np.where(baseline.entry_date < "2013-01-01", "2000-2012", "2013-2026H1")
    baseline.to_csv(OUT / "a2_entry_quality_baseline_signal_features.csv", index=False)

    cohort = tab_medians(baseline)
    cohort.to_csv(OUT / "a2_entry_quality_cohort_medians.csv", index=False)
    # AUC is descriptive, not a defensible independent-signal test: gold
    # crosses in the same market regime are serially correlated.
    feature_tests = []
    for feature in ["rebound60", "ret20", "ret60", "dist200", "ma60_slope10",
                    "ma200_slope20", "cross_count63", "cross_count126", "vol20_ann",
                    "atr14_pct"]:
        losses = baseline.loc[~baseline.is_win, feature].dropna()
        wins = baseline.loc[baseline.is_win, feature].dropna()
        u, p = mannwhitneyu(losses, wins, alternative="two-sided")
        feature_tests.append({"feature": feature, "loss_count": len(losses),
                              "win_count": len(wins), "loss_direction_auc": u / (len(losses)*len(wins)),
                              "raw_p_exploratory": p})
    univariate = pd.DataFrame(feature_tests)
    # Benjamini-Hochberg: control the multiple comparisons within this
    # descriptive feature panel (not a substitute for time-series validation).
    order = np.argsort(univariate.raw_p_exploratory.to_numpy())
    pvals = univariate.raw_p_exploratory.to_numpy()[order]
    corrected = np.minimum.accumulate((pvals * len(pvals) / np.arange(1, len(pvals)+1))[::-1])[::-1]
    qvals = np.empty_like(corrected)
    qvals[order] = np.minimum(corrected, 1)
    univariate["bh_q_exploratory"] = qvals
    univariate.to_csv(OUT / "a2_entry_quality_feature_discrimination.csv", index=False)
    stats_rows, excluded_rows, eras = [], [], []
    results = {"A2严格基准": base_metrics}
    for item in TESTS:
        masked = item.condition(baseline)
        caught = baseline[masked]
        missed = baseline[~masked]
        assert len(caught) + len(missed) == len(baseline)
        assert not caught[item.field].isna().any()
        scenario, curve, stats, blocked = simulate(d, item.label,
            gate=Gate(item.field, item.op, item.cutoff), warm_start=False)
        results[item.label] = stats
        # No future price or outcome is used to filter actual trades.
        assert all(pd.Timestamp(row.signal_date) < pd.Timestamp(row.intended_entry_date)
                   for row in blocked.itertuples() if row.intended_entry_date)
        stats_rows.append({
            "label": item.label, "family": item.family,
            "field": item.field, "op": item.op, "threshold": item.cutoff,
            "excluded_original_completed": len(caught),
            "excluded_original_losers": int((caught.ret_net < 0).sum()),
            "excluded_original_winners": int((caught.ret_net > 0).sum()),
            "excluded_original_n20": int(caught.is_n20.sum()),
            "excluded_original_loss5": int(caught.is_loss5.sum()),
            "excluded_original_big_winners": int(caught.is_big_win.sum()),
            "excluded_original_winner_returns_sum": caught.loc[caught.is_win, "ret_net"].sum(),
            "excluded_original_loss_returns_sum": caught.loc[~caught.is_win, "ret_net"].sum(),
            "conditional_loss_rate": float((caught.ret_net < 0).mean()) if len(caught) else np.nan,
            "not_flagged_loss_rate": float((missed.ret_net < 0).mean()) if len(missed) else np.nan,
            "new_completed": stats["completed"],
            "new_common_payoff": stats["common_cutoff_payoff"],
            "new_common_pf": stats["common_cutoff_pf"],
            "new_mdd": stats["mdd"],
            "new_cagr": stats["cagr"],
            "new_end_equity": stats["end_equity"],
            "new_exposure": stats["exposure"],
            "new_worst_closed": stats["worst_trade"],
            "new_longest_loss_streak": stats["longest_loss_streak"],
            "new_longest_cash_days": stats["longest_cash"],
            "simulation_skipped_signals": stats["blocked_signals"],
            "new_early_total": stats["early_return"],
            "new_late_total": stats["late_return"],
            "base_early_total": base_metrics["early_return"],
            "base_late_total": base_metrics["late_return"],
        })
        for tx in caught.itertuples():
            excluded_rows.append({"test": item.label, "entry_date": tx.entry_date,
                "entry_signal_date": tx.signal_date, "ret_net": tx.ret_net,
                "exit_reason": tx.exit_reason, "hold_days": tx.hold_days,
                "is_big_win": tx.is_big_win, "feature": float(getattr(tx, item.field)),
                "family": item.family})
        for era, sub in baseline.groupby("entry_era"):
            selected = item.condition(sub)
            chosen = sub[selected]
            eras.append({"label": item.label, "family": item.family,
                         "era": era, "completed_in_era": len(sub),
                         "excluded": len(chosen),
                         "excluded_losers": int((chosen.ret_net < 0).sum()),
                         "excluded_winners": int((chosen.ret_net > 0).sum()),
                         "excluded_big_winners": int(chosen.is_big_win.sum())})
    summary = pd.DataFrame(stats_rows)
    summary["delta_payoff"] = summary.new_common_payoff - base_metrics["common_cutoff_payoff"]
    summary["delta_mdd_pp"] = (summary.new_mdd - base_metrics["mdd"]) * 100
    summary["delta_cagr_pp"] = (summary.new_cagr - base_metrics["cagr"]) * 100
    summary.to_csv(OUT / "a2_entry_quality_single_factor_filters.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv(OUT / "a2_entry_quality_excluded_original_trades.csv", index=False)
    pd.DataFrame(eras).to_csv(OUT / "a2_entry_quality_era_split.csv", index=False)

    # The 20%+ winner audit is the main anti-overfit guardrail.
    top = baseline[baseline.is_big_win].sort_values("ret_net", ascending=False)
    impact = []
    for row in top.itertuples():
        # op==< is strict too; equality must not count.
        masked = [item.label for item in TESTS if
                  (float(d.loc[row.signal_idx, item.field]) > item.cutoff if item.op == ">"
                   else float(d.loc[row.signal_idx, item.field]) < item.cutoff)]
        impact.append({"entry_date": row.entry_date, "net_return": row.ret_net,
                       "rebound60": row.rebound60, "ma60_slope10": row.ma60_slope10,
                       "ma200_slope20": row.ma200_slope20, "vol20_ann": row.vol20_ann,
                       "flagged_by_n_rules": len(masked), "flagged_by": "; ".join(masked)})
    pd.DataFrame(impact).to_csv(OUT / "a2_entry_quality_big_winner_misclassification.csv", index=False)

    # Counterfactual invalidity check: some new trades differ from original
    # history after skipping prior entries; use scenario equity for evaluation.
    shortlist = summary.loc[summary.excluded_original_completed.ge(3)].sort_values(
        "delta_cagr_pp", ascending=False).head(10)
    print("BASE", {k: base_metrics[k] for k in
          ["completed", "common_cutoff_payoff", "mdd", "cagr", "end_equity"]})
    print(summary[["label", "excluded_original_completed", "excluded_original_losers",
                   "excluded_original_winners", "excluded_original_big_winners",
                   "new_common_payoff", "new_mdd", "new_cagr", "delta_mdd_pp"]].round(4).to_string(index=False))
    print("TOP_CAGR", shortlist[["label", "delta_cagr_pp"]].round(3).to_dict("records"))

    # Generate a transparent, compact audit memo.
    fmt = lambda value: f"{value*100:+.2f}%"
    lines = [
        "# QQQ A2-N20｜金叉入场质量：四类事前特征与单因素过滤实验", "",
        "**探索性回测；信号日2000年至2026-07-13。不是样本外验证。**", "",
        "## 结论先行", "",
        "1. 23种事前入场过滤均未能在统一样本下同时提高Payoff和CAGR、降低MDD。",
        "2. MA60/MA200斜率、过去交叉次数、波动率、近期反弹幅度，单独使用均会误杀重要赢家；",
        "   对连续指标的十项探索性Mann-Whitney分布比较，BH校正后的q值均大于0.95。",
        "3. 从用户最重视的Payoff/低频/MDD看，MA200上方8%拒买是值得留作候选的方案（Payoff4.61、完成41笔、MDD-18.88%）；",
        "   但它令CAGR从5.84%降至5.28%，还会错过2020年的+31.75%赢家。",
        "   相邻10%/12%参数虽保留该赢家，却分别出现'MDD变差'、'MDD不改善'，不能直接推广。",
        "4. 原始58笔完成交易中N20早期保护触发25次，其中24笔最终亏损，平均约-2.57%；这部分风险已被原始A2限制。",
        "5. 假设的风险机理与历史收益关联，不等于在买入当天对未来收益存在稳定可交易的预测能力。",
        "", 
        "## 研究问题与冻结口径", "",
        "A2严格版：MA20/60新金叉次日开盘全仓，前20交易日MA10/20死叉且浮亏止损，"
        "其余等MA20/60新死叉退出；所有退出只等待下一个新金叉。",
        "采用同一拼接复权QQQ OHLC、单边费用0.05%、无滑点/现金利息。"
        "对照组58笔已完成+1笔期末持仓，不使用2000年旧版合成初仓。",
        "各类信号仅可使用当日收盘以及更早数据，所有买卖在下个交易日开盘。",
        f"基准统一口径Payoff={base_metrics['common_cutoff_payoff']:.2f}，"
        f"MDD={fmt(base_metrics['mdd'])}，CAGR={fmt(base_metrics['cagr'])}。",
        "对尚未完成交易，Payoff统一按期末假设卖出（计卖出费），"
        "MDD/CAGR按逐日账户权益，不算现金利息。", "",
        "## 入场时已知信息：赢家/输家中位数对照", "",
        "|分组|交易数|过去60日低点反弹|过去20日涨幅|MA60/10日斜率|MA200/20日斜率|20日年化波动率|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for z in cohort.itertuples():
        lines.append(f"|{z.cohort}|{z.n}|{fmt(z.rebound60_median)}|"
                     f"{fmt(z.ret20_median)}|{fmt(z.ma60_slope10_median)}|"
                     f"{fmt(z.ma200_slope20_median)}|{fmt(z.vol20_ann_median)}|")
    lines += ["", "## 单变量区分能力：有无可重复的赢家/输家差异？", "",
              "|事前特征|AUC(值更大预测亏损)|Mann-Whitney原始p|BH校正q|",
              "|---|---:|---:|---:|"]
    for z in univariate.itertuples():
        lines.append(f"|{z.feature}|{z.loss_direction_auc:.3f}|{z.raw_p_exploratory:.3f}|{z.bh_q_exploratory:.3f}|")
    lines += ["", "注：AUC=0.5代表该特征没有单向区分能力（小于0.5意味着值更低更偏向亏损）。"
              "p/q仅为探索性分布比较，不能对相关的连续市场信号构成严格显著性保证。",
              "", "## 独立拒买规则（每次只增加一条）", "",
              "**排除亏/赢**：指原始A2已完成交易中若应用规则会被拒买的交易；"
              "**新结果**：指从头重新模拟实际可能交易的账户结果，不能简单相加排除笔数。",
              "| 过滤条件 | 原策略中拒买亏/赢 | 误杀20%+大赢家 | 完成交易数 | 新Payoff | 新MDD | 新CAGR |",
              "|---|---:|---:|---:|---:|---:|---:|",]
    for row in summary.itertuples():
        lines.append(f"|{row.label}|{row.excluded_original_losers}/{row.excluded_original_winners}|"
                     f"{row.excluded_original_big_winners}|{row.new_completed}|{row.new_common_payoff:.2f}|"
                     f"{fmt(row.new_mdd)}|{fmt(row.new_cagr)}|")
    lines += ["", "## 年代拆分：防止只对单一熊市有效", "",
              "|条件|2000—2012拒买亏/赢|2013—2026H1拒买亏/赢|",
              "|---|---:|---:|"]
    split = pd.DataFrame(eras)
    for item in TESTS:
        q = split[split.label == item.label].set_index("era")
        try:
            old, new = q.loc["2000-2012"], q.loc["2013-2026H1"]
            lines.append(f"|{item.label}|{old.excluded_losers}/{old.excluded_winners}|"
                         f"{new.excluded_losers}/{new.excluded_winners}|")
        except KeyError:
            pass
    lines += ["", "## 事前特征与限制", "",
              "- 均线斜率以历史MA的10日/20日变化率衡量，交叉次数采用上一个交易日及更早63/126交易日的MA20/60金死叉数（不包含今天金叉）。",
              "- 高波动率不天然等于坏入场，历史大赢家往往也来自高波动、深跌复苏期；简单过滤会误伤。",
              "- `低于MA200+MA60向下`可能标出部分2022熊市信号，却同样会过滤其他年份盈利交易。",
              "- 各阈值并非事前注册，已经对全历史做过多轮探索；2013—2026只是年代敏感性检查，不可宣称真实样本外。",
              "- 收益为已实现策略交易回报，不是单一信号的固定20日/60日前瞻回报；N20出场与正常退出均计入。",
              "- 对58笔交易测试二十余候选会产生多重比较偏差；小样本的不稳定‘最佳值’不能直接实盘。",
              "- 账户最大回撤包括已持仓期间的未实现跌幅，即使后来止损低于门槛也可能发生更大MDD。",
              "- 不使用新闻、市场事后命名、未来K线或后验收益作为入场输入。",
              "", "## 复现文件", "",
              "`python research/experiments/a2_n20_entry_quality_study.py`。",
              "逐笔原信号特征、按组中位数、单过滤器完整账户结果、被拒买交易、年代拆分、20%+赢家误杀清单均有CSV。", ""]
    (OUT / "A2-N20_金叉入场质量_事前特征与独立过滤.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
