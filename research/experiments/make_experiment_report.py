"""Render the audited A2 isolated experiment tables and event details."""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2] / "research" / "results" / "experiments_2026_11_09"
s = pd.read_csv(BASE / "scenario_metrics.csv").set_index("case")
t = pd.read_csv(BASE / "all_scenario_trades.csv")
strict = pd.read_csv(BASE / "strict_start_sensitivity.csv").set_index("case")

def p(x, n=1): return f"{100*x:+.{n}f}%"
def tb(head, rows):
    return "\n".join(["|"+"|".join(head)+"|","|"+"|".join("---" for _ in head)+"|"] +
                     ["|"+"|".join(map(str,row))+"|" for row in rows])

stops = [
    ("BASE_A2_N20","原始A2"),("S_MA10_0.98_touch","MA10×0.98盘中"),
    ("S_MA20_0.98_touch","MA20×0.98盘中"),("S_MA10_0.98_ratchet","MA10×0.98单向跟踪"),
    ("S_MA20_0.98_ratchet","MA20×0.98单向跟踪"),("S_MA10_0.98_close","MA10×0.98收盘确认"),
    ("S_MA20_0.98_close","MA20×0.98收盘确认"),("S_MA20_0.95_touch","MA20×0.95盘中"),
    ("S_MA20_0.98_arm5","MA20×0.98浮盈5%启动"),("S_MA20_0.98_arm10","MA20×0.98浮盈10%启动"),
    ("S_MA10_0.98_arm5","MA10×0.98浮盈5%启动"),
]
gates = [
    ("BASE_A2_N20","原始A2"),("F_dist200_gt_08pct","高出MA200 >8%拒买"),
    ("F_dist200_gt_10pct","高出MA200 >10%拒买"),("F_dist200_gt_12pct","高出MA200 >12%拒买"),
    ("F_ret60_gt_05pct","60日上涨 >5%拒买"),("F_ret60_gt_08pct","60日上涨 >8%拒买"),
    ("F_ret20_gt_10pct","20日上涨 >10%拒买"),("F_absdist200_gt_10pct","MA200偏离绝对值 >10%拒买"),
    ("F_below200_05pct","低于MA200 >5%拒买"),
]

rows1 = [[label,int(s.loc[k].completed),f"{s.loc[k].common_cutoff_payoff:.2f}",
          f"{s.loc[k].common_cutoff_pf:.2f}",p(s.loc[k].mdd),p(s.loc[k].cagr,2),
          p(s.loc[k].exposure),int(s.loc[k].trailing_exit_count),p(s.loc[k].baseline_top10_profit_retained)]
          for k,label in stops]
rows2 = [[label,int(s.loc[k].completed),int(s.loc[k].total_baseline_losses_removed),
          int(s.loc[k].total_baseline_wins_removed),int(s.loc[k].top10_removed),
          f"{s.loc[k].common_cutoff_payoff:.2f}",p(s.loc[k].mdd),p(s.loc[k].cagr,2)]
          for k,label in gates]

orig = t[t.case=="BASE_A2_N20"].set_index("entry_date")
best = orig.nlargest(10,"ret_net")
fast = t[t.case=="S_MA10_0.98_touch"].set_index("entry_date")
slow = t[t.case=="S_MA20_0.98_touch"].set_index("entry_date")
top_rows = [[date,p(r.ret_net,2),p(fast.at[date,"ret_net"],2),p(slow.at[date,"ret_net"],2),int(r.hold_days)]
            for date,r in best.iterrows()]

def removed(k):
    remaining=set(t[t.case==k].entry_date)
    excluded=orig.loc[~orig.index.isin(remaining)]
    return tb(["原买入日","原卖出日","原净收益","原退出类型"],
              [[d,r.exit_date,p(r.ret_net,2),r.exit_reason] for d,r in excluded.iterrows()])

def giveback(k):
    x=t[(t.case==k)&(t.ret_net>0)]
    return (x.mfe_close-x.ret_net).median()

stage_rows=[]
for k,label in [("BASE_A2_N20","原始A2"),("S_MA10_0.98_touch","MA10×0.98"),
                ("S_MA20_0.98_touch","MA20×0.98"),("F_dist200_gt_08pct","MA200高8%过滤"),
                ("F_dist200_gt_10pct","MA200高10%过滤"),("F_ret60_gt_08pct","60日涨8%过滤")]:
    q=s.loc[k]
    stage_rows.append([label,p(q.dotcom_return),p(q.gfc_return),p(q.covid_return),p(q.rates_return),
                       p(q.rates_mdd),p(q.early_return),p(q.late_return)])

strict_rows=[]
for k,label in [("BASE_A2_N20_STRICT","仅新金叉基准"),("S_MA10_0.98_touch_STRICT","MA10×0.98"),
                ("S_MA20_0.98_touch_STRICT","MA20×0.98"),
                ("F_dist200_gt_10pct_STRICT","MA200高10%过滤"),
                ("F_ret60_gt_08pct_STRICT","60日涨8%过滤")]:
    q=strict.loc[k]
    strict_rows.append([label,int(q.completed),f"{q.common_cutoff_payoff:.2f}",p(q.mdd),p(q.cagr,2)])

body = [
    "# QQQ A2-N20｜盈利后MA止损与入场过滤独立实验",
    "",
    "> 历史价格截止 2026-07-13；2026-11-09 研究。原始A2的买入、20日保护、MA20/60常规退出、新金叉重入均不改变。所有附加实验独立运行，尚未组合，也非样本外结果。",
    "",
    "## 一、概览",
    "",
    "- MA10×0.98盘中止损：MDD −17.9%（原−20.3%），但统一截止Payoff 2.88（原3.92）、CAGR 2.85%（原5.66%），原十大赢家收益率合计仅保留34.8%。",
    "- MA20×0.98盘中止损：MDD −15.5%、Payoff 3.37、CAGR 4.09%，原十大赢家收益率合计保留39.3%。这项变体没有同时改善高Payoff和低MDD。",
    "- 金叉时价格高于MA200超过10%拒买：历史过滤7亏2盈，Payoff 4.41、CAGR 6.27%，但MDD恶化至−21.8%。未满足三个目标同时改进。",
    "",
    "## 二、止损计算与成交协议",
    "",
    "1. 只有在一笔仓位于已完成日K收盘产生过扣双边0.05%交易费后的浮盈，才启动跟随MA止损；启动后不可撤销。另测浮盈5%/10%才启动。",
    "2. 盘中止损位由前一已完成交易日的MA10或MA20乘倍率确定，下一交易日若开盘跳空穿越止损位则以开盘价卖出，若当日Low触及则按止损价卖出；后者不加额外滑点，偏乐观。另测MA止损位只升不降，以及收盘确认、次日开盘卖出。",
    "3. 任何止损后仍必须等待新MA20/60金叉。新金叉以当日收盘确认、下一开盘买入。传统N20保护与慢死叉出场优先于新收盘跟随规则；历史起点2000-01-03按旧引擎既存多头初始化，以便对齐59笔已平仓档案。",
    "4. 对于原始A2在2026-07-13仍持仓的交易，Payoff/PF统一假设当日收盘强制平仓并计卖出费（原历史3.95仍是59笔已平仓、未计最后持仓）。MDD/CAGR沿原每日收盘权益曲线，未额外计终点卖出费。",
    "",
    "真实stop-market的成交价不受触发价保证，跳空和流动性会使回测止损表现偏乐观。参见[美国SEC Investor.gov的止损单说明](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-15)。动态MA止损还要求交易系统每天更新委托，并不等于一次性券商百分比跟踪止损单。",
    "",
    "## 三、Selection1：所有盈利MA止损独立变体",
    "",
    tb(["策略","平仓数","统一Payoff","统一PF","MDD","CAGR","持仓占比","跟随止损次数","Top10合计利润保留"],rows1),
    "",
    f"盈利交易的持仓期峰值浮盈减去最终净收益，中位数原A2为{p(giveback('BASE_A2_N20'))}，MA10×0.98为{p(giveback('S_MA10_0.98_touch'))}，MA20×0.98为{p(giveback('S_MA20_0.98_touch'))}。但各变体峰值来自各自缩短的持仓区间，此指标无法反映被卖出后又继续上涨的机会成本，必须结合下面同一笔入场对照。",
    "",
    "### 原始十大赢家逐笔收益对照",
    "",
    tb(["原买入日","原A2收益","MA10×0.98","MA20×0.98","原持有日"],top_rows),
    "",
    "主要问题：2009 +34.94%的趋势被MA20×0.98提前退出至约+3%，2023 +26.51%的趋势至约+1%，2013 +27.67%变成约−2%。跟随止损减少了持仓内回吐，却切断了随后继续上涨的行情，而原始新金叉重入规则不会很快买回来。",
    "",
    "MA20×0.95虽然保留约93.9%的原十大赢家收益，但MDD恶化至−25.5%。单向跟踪、提高启动盈利门槛或改为收盘确认，也没有使Payoff/MDD/CAGR同时胜出。",
    "",
    "## 四、Selection2：入场前单因素拒买测试",
    "",
    "所有门槛只看金叉信号日收盘时的历史指标，拒买后等下一个金叉。2000年非自然金叉的初始化交易不参与过滤；被拒买的最后一次未平仓信号可能不计入下表'删除盈亏'两列。",
    "",
    tb(["独立过滤器","完成交易","删亏","误删赢","误删Top10","统一Payoff","MDD","CAGR"],rows2),
    "",
    "### '高出MA200 >10%'实际排除的9笔",
    "",
    removed("F_dist200_gt_10pct"),
    "",
    "### '60日上涨 >8%'实际排除的3笔",
    "",
    removed("F_ret60_gt_08pct"),
    "",
    "### '20日上涨 >10%'：反例与误杀赢家",
    "",
    removed("F_ret20_gt_10pct"),
    "",
    "阈值稳健性：高出MA200超过8%虽同时改善Payoff/MDD，但会误删2020 +31.75%的大赢家与2023年11月+8.08%的赢家，CAGR下降。超过10%则保留2020大赢家但MDD扩大，超过12%回撤近基线。不能只选最顺眼的阈值。60日涨幅>8%只影响3笔且都亏损，样本太少；降到5%会误删盈利。20日涨幅>10%会删2009、2020、2025的大赢家，应作负对照。",
    "",
    "## 五、不同市场环境与起点敏感性",
    "",
    "下表为指定日期的权益首末收益，2000—12与2013—26/07未严格等长且所有阈值已查看过全历史，**不可称为样本外验证**。",
    "",
    tb(["策略","2000—02","2008—09","2020","2022","2022 MDD","2000—12累计","2013—26/07累计"],stage_rows),
    "",
    "使用更严格的2000年起点（不做人工多头初始化）做方向复核：",
    "",
    tb(["策略","已完成","统一Payoff","MDD","CAGR"],strict_rows),
    "",
    "## 六、下一阶段保留问题",
    "",
    "- 不将MA10×0.98、MA20×0.98直接并入A2主策略。它们改善MDD却削弱大赢家和Payoff。之后可以独立研究'利润峰值回吐达到某阈值'的收益保护，而不是把它和本次实验组合后宣称更优。",
    "- MA200的正向偏离8/10/12%保留为候选入场否决因素；还需固定阈值的时间前向检验、误杀率、分年代检查和抽样不确定性。",
    "- 60日涨幅>8%只是3笔的探索巧合；不能据此形成交易许可。短期20日涨幅>10%明显误杀大赢家，不应简单用于拒买。",
    "- 本轮没有加入周线、ATR、单日/3日急跌熔断、更快重入或复合过滤；数据只覆盖至2026-07-13。使用复权日OHLC回测止损只是近似，不保证真实逐笔成交。",
    "",
    "## 七、可复算材料",
    "",
    "[1999—2022 QQQ CSV](https://github.com/jtang25/Stock-Market-Index-Daily-Return-Distribution-/blob/main/QQQ.csv)；[2013—2026 QQQ CSV](https://github.com/jameswdonoghue/QQQ-Data/blob/main/QQQ.csv)。程序为`research/experiments/a2_stops_and_entry_filters.py`，校验为`validate_experiments.py`。输出`scenario_metrics.csv`、`all_scenario_trades.csv`、`baseline_top10_audit.csv`、`blocked_entry_signals.csv`、`strict_start_sensitivity.csv`。",
    "",
]
dest=BASE/"A2-N20_盈利MA止损与入场过滤_独立测试.md"
dest.write_text("\n".join(body),encoding="utf-8")
print(dest)
