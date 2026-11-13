# H3 — 当前主展示的趋势研究策略（冻结参考规则）

> **版本：`H3_FROZEN` / `H3_NEGATIVE_ACCELERATING_MONTH_EMA5_FAST`。状态：研究及纸面回测，不是实盘自动交易授权。**
>
> 本文固定截至 **2026-11-11** 的研究行为和对照口径；不是声明2026-11-14当前实时持仓。早期 SMA V1、EMA V2/V3、2026-07-13 H3 研究快照继续保留，不能覆写其结果。

## 策略目标

基于日 EMA20/60 捕捉中期趋势，使用 EMA10/20 的 FAST 恢复信号尝试更早跟随趋势，借助**上一完整月 EMA5 的斜率**控制入场风险，通过 N20 和亏损保护限制不利走势。策略**只做多或持有现金**，不做空、不使用杠杆。

### H3 = V3 主风险状态机 + 两个经过版本化的 FAST 候选资格

唯一原H3复算入口：[a2_v3_1_h3_trend_spread_quick_reentry.py](../research/experiments/a2_v3_1_h3_trend_spread_quick_reentry.py) 中 `replay(signals, "H3_FROZEN")`；内核复用 [a2_month_ema5_slope_regime.py](../research/experiments/a2_month_ema5_slope_regime.py)，已冻结 V3 的 `ENTRY_GATE` 和 `V3_PARAMETERS` 来自 [a2_v3_selected.py](../research/experiments/a2_v3_selected.py)。其中 H3 特有参数：

```python
replay_gated(
    signals,
    ENTRY_GATE,                       # "up_slow_veto_1_abs"
    config=config,
    **V3_PARAMETERS,                  # 原 V3 N20、day0、net7% 收盘止损
    defer_rejected_fast=True,         # 被拒绝的 FAST 金叉保持待处理资格
    fast_negative_month_mode="accelerating",
    trend_spread_min_gain=.10,        # 无 spread exit 时不额外触发策略信号
)
```

**完成收盘判断，下一常规交易日开盘成交。** 日 EMA10/20/60 使用统一复权日价；月 EMA5 使用截至当日最后**已经完成**的月份，绝不偷用本月尚未结束的最终月收盘数据。

## 买入逻辑

1. **常规 REGULAR**：日 EMA20 由下向上穿越 EMA60；如果最新已完成月 EMA5 **斜率仍正，但低于更早一个已完成月的斜率**，`up_slow_veto_1_abs` 暂缓入场。当慢线依然处于多头排列、月线 gate 以后解除时，允许延期补买；再次慢线死叉清除延期资格。
2. **FAST**：日 EMA10/20 新金叉；原 H3 必须空仓、FAST 状态机允许、全局月线 gate 允许，并满足FAST月线条件：正常情形为上一完整月 EMA5 斜率为正；H3 另外允许连续两个月均为负斜率、且最新完整月下降斜率更负（`fast_negative_month_mode="accelerating"`）的特殊历史反弹资格。
3. **延期 FAST**：被原月线条件阻挡的 EMA10/20 新金叉可以保留资格；仍须 EMA10>EMA20、原 FAST 资格有效、相关月线条件满足才可入场；新的 EMA10/20 死叉会清除此资格。FAST **不是**可无条件绕过月线 gate 的信号。
4. 常规和 FAST 同日获准时，以**常规 REGULAR** 为入场分类优先。

## 卖出 / 风险规则

- **全程慢趋势退出**：EMA20/60 新死叉，下一常规开盘卖出。
- **全程 7% 净亏损收盘保护**：当日完成收盘按交易费用计的账面净亏触发后，下一常规开盘退出。不是保证最大亏损7%的盘中限价/止损单；跳空可能亏更多。
- **N20**：买入当日（age=0）起开始保护。需出现**新鲜 EMA10/20 死叉**且当日收盘价低于本笔真实入场开盘价才产生下一开盘退出。常规入场默认 `age=0..20`，在第20日如果近5个收盘至少2次低于入场价且EMA10/20相对距离在5日内收窄，则条件延至age=30；FAST在慢线确认之前持续保护、确认后仍保护20个完整交易日。具体各交易日年龄和执行优先级以代码与 [V3冻结合同](STRATEGY_SELECTED_V3.md) 为准。
- **入场首日亏损3%**：买入当天收盘低于买入开盘价3%（价格跌幅、不含佣金）则下一开盘退出；如果当天还有更高优先级的7%亏损、慢死叉或N20退出信号，原优先级优先，不能生成重复订单。
- 不采用额外 `trend_spread_exit`、盈利追踪停损、局部减仓等实验标志作为 H3 原始规则。

## A/B 利润保护 = 另一个模块，不是改写 H3

冻结 A/B 是同一 H3 **影子持仓**上额外的**研究候选**，不是已经合并进 H3 的默认止盈规则。无叠加、A、B 必须可独立对照。

- **A**：原H3影子仍持仓、曾最高完整收盘浮盈≥10%、当前仍盈利、EMA20>EMA60，出现新鲜 EMA5/20 死叉，额外提前卖出。
- **B**：A 的全部条件 **再加** EMA10>EMA20 且近三完整收盘日跌幅 / ATR20≥1.5。
- 二者均以**新 EMA5/10 金叉且 EMA20>EMA60、原影子H3依然持仓**作为一次特别买回资格，最多每个原H3窗口额外卖出一次、特别买回一次；不强制回买价低于上次卖价。最低持仓天数门槛已取消（age0）。
- A/B 实际卖出后，**影子H3的原N20、月线gate和原入场退出日期不重置**；原H3正常退出强制优先。

唯一可复核合同：[A/B v1.0规则](STRATEGY_H3_PROFIT_GUARD_AB_V1.md) · [冻结参数JSON](../research/config/h3_profit_guard_ab_v1.json) · [固定校验运行入口](../research/experiments/h3_profit_guard_ab_v1_frozen.py)。

## 固定 QQQ 研究快照

样本 QQQ，2000-01-03～2026-11-11，初始$100,000，单边手续费0.05%，零现金利息，未计税/额外滑点，完整日K确认、次日常规开盘成交。

| 版本 | CAGR | MDD | 期末模拟权益 | H3已完成交易/额外卖出+买回 |
|---|---:|---:|---:|---:|
| QQQ 买入持有 | 8.31% | −82.96% | 约$852,247 | — |
| **H3（无止盈叠加）** | **9.89%** | **−24.30%** | 约$1,257,934 | 60 / 0+0 |
| **H3+A v1.0（候选）** | **10.69%** | **−23.94%** | 约$1,528,788 | 60 / 28+18 |
| **H3+B v1.0（候选）** | **11.01%** | **−24.30%** | 约$1,652,616 | 60 / 22+12 |

数字是**单一已反复研究历史区间的样本内结果**，不是已验证的独立样本外优势。历史数据上B年化较高，A历史MDD略低，**B未降低H3整体MDD**；B的12次特别买回相对不买回都削弱了原交易收益，研究中没有默默关闭买回开关。

- [H3最新交易地图、60笔成交和2026年7～11月细节](../assets/a2_ema_v3_1_h3_trade_map_2000_2026_11.svg)
- [H3 2026-11-11完整派生交易账本](../research/results/v3_1_h3_2026_11_11/h3_trades.csv) · [H3元数据与样本快照](../research/results/v3_1_h3_2026_11_11/summary.json)
- [最新 H3 历史信号与成交审计](research/V3_1_H3_2026年7至11月补样与进出场审计_2026-11-12.md)
- [A/B 完整账户及负面归因](research/H3_QQQ_AB_利润保护完整对比_2026-11-13.md)

## 复现、日期、数据权利

```bash
python -m pip install -e '.[dev]'

# 自备有权使用的、1999年底以前足够预热且统一复权的QQQ OHLCV
python research/experiments/a2_v3_1_h3_trend_spread_quick_reentry.py \
  --symbol QQQ \
  --csv sources/qqq_adjusted_with_warmup.csv \
  --end 2026-11-11 \
  --outdir research/results/h3_local_audit

# A/B 冻结专用四组验证：按研究脚本封存的数据源取得路径执行
python research/experiments/h3_profit_guard_ab_v1_frozen.py \
  --outdir research/results/h3_profit_guard_ab_v1_local_audit

# 不含第三方行情的静态回归测试
python -m unittest discover -s tests -v
```

A/B冻结脚本复用研究数据加载器，复算时必须能按 [数据来源及复权规范](DATA_SOURCES.md) 取得合法完整历史；如来源变动、缺数据或2026-11-11当日数据不能校准，应失败或标记历史不可复现，不能默默更改规则以匹配结果。图表与账本的日期只是固定研究快照；**不能把2026-11-11的 FLAT 状态读成今天的实盘/实时信号**。此仓库不向券商发送订单。
