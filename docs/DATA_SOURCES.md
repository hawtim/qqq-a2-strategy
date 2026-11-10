# 数据来源、可再分发性与复现规范

最后核查：2026-11-10。此文件用于让其他人**理解、获取、校验、独立复算**原研究，而不是默默要求他们相信某份行情。

## 1. 原研究的数据从哪里来

| 用途 | 原来源与可核查路径 | 公开仓库的处理 |
|---|---|---|
| QQQ 较早一段日 OHLCV（1999–2022） | [jtang25/Stock-Market-Index-Daily-Return-Distribution-/QQQ.csv](https://github.com/jtang25/Stock-Market-Index-Daily-Return-Distribution-/blob/main/QQQ.csv)，上游仓库在2026-11-10显示 **MIT** 许可 | **只提供链接和校验逻辑**；不混同下列来源的授权 |
| QQQ 较新一段日 OHLCV（2013—2026-07-13 有效完整日） | [jameswdonoghue/QQQ-Data/QQQ.csv](https://github.com/jameswdonoghue/QQQ-Data/blob/main/QQQ.csv)；上游仓库截至核查日**未声明 LICENSE** | **不再分发原始 CSV**。使用者需自行确认数据权利后下载 |
| **2026-07-14 至 2026-11-09 增量 QQQ 日 OHLCV** | [StockQuote.io QQQ 历史行情](https://www.stockquote.io/stocks/QQQ/history?page=1) 的连续 4 页完整历史，逐日包含 O/H/L/C、Adj Close、Volume；2026-11-09 收盘价另与 [StockAnalysis（来源：S&P Global）](https://stockanalysis.com/etf/qqq/history/) 的 $679.35 对照 | **只提交复权一致性审计与派生交易/账户表，不提交 StockQuote.io 原始 CSV**，版权/再分发授权未确认 |
| 原 QQQ 回测讨论及分钟级资料 | 用户此前取得的 FirstRate Data（约从2000开始的 QQQ 1m/5m/30m/1h） | 仅提供研究方法；**不提供付费数据本体或自动镜像**。A2 V1.0 在本仓库仅需要**日 K** |
| TQQQ 曾用于独立杠杆对照的开盘数据 | [Sunethan/backtesting/data/TQQQ.csv](https://github.com/Sunethan/backtesting/blob/main/data/TQQQ.csv)（2010–2021），[LucanLAM/VS_trading/TQQQ_Historical_Data.csv](https://github.com/LucanLAM/VS_trading/blob/main/TQQQ_Historical_Data.csv)（至2025） | **不分发原始文件**，单独读取各来源许可和质量 |
| TQQQ 对照调整基准 | [holochart daily.json](https://github.com/holochart/holochart/blob/main/examples/demos/tqqq-soxl/data/daily.json) | 仅归档已生成的统计/实验记录，不分发原始序列 |
| A2 空仓期间美元现金收益的历史参照 | [FRED DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)：美联储 H.15 的 **3 个月期恒定到期美国国债收益率（investment basis）**，每个交易日报价、单位年化% | **只提交模型、派生账户权益与研究统计；不复制/再分发 FRED 原始序列**。研究本地快照见下方 |
| **2026-10-26 至 2026-11-06 最新利率** | [Federal Reserve Board H.15 官方数据下载](https://www.federalreserve.gov/datadownload/Download.aspx?rel=H15&series=2dbf56a690d90872e08d4edbf48df0a4&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package)；关键点 10/26 4.22%、11/06 4.19% | 单独本地增量，仅公开合并后的模拟资产结果 |

**版权和研究数据隔离**：公众可看实验脚本、方法、交易回报等**派生结果**，但不得从本仓库推断拥有任一上游原始数据的再分发权。`sources/*.csv`、`sources/*.json` 被 Git 忽略。研究使用数据与开源代码许可证分别处理。

## 2. 原始 QQQ 日 K 的复现步骤

1. 从上述两个仓库**自行**取得相应 `QQQ.csv`，放到 `sources/qqq_jtang.csv` 和 `sources/qqq_james.csv`。确保文件头为 `Date,Open,High,Low,Close,Adj Close,Volume` 或兼容的 CSV 列名；需按时间排序、日期唯一。
2. 新来源较晚期覆盖到 **2026-07-13** 的完整日 K。2026-07-14 之后某份研究快照有无效/不完整记录，不能把它当完整交易日参与回测。
3. 两来源共有 **2439** 条重叠交易日，其未复权 Close 在研究时完全一致；第一份历史 `Adj Close` 相对第二份的比例中位数约 **0.975881155**，重叠比值标准差约 **1.81×10⁻⁷**。
4. 按第二来源最早完整日 **2013-01-02** 分段：此前用较早源，并乘上述重基准因子；此后使用较新源。每一天的调整因子为 `Adj Close / Close`，再同比例调整当日 Open/High/Low/Close。连续拼接得到 **6877** 个完整交易日。
5. 用调整后 **Close** 计算简单移动平均 SMA10/20/60/200。金叉信号日在**收盘后**成立，成交在**下一常规交易日调整后开盘价**；不会用未来行情、盘中猜测的 K 线填单。
6. 初始 100,000 美元，每次把模型仓位全投入 QQQ，单边费用 0.05%，现金不生息，滑点与税收设为0。**这是实验假设**，不是实际账户仓位建议。

```
qqq-a2 --old sources/qqq_jtang.csv --new sources/qqq_james.csv \
  --start 2000-01-03 --end 2026-07-13 --out .a2-output
python research/verify_frozen_baseline.py \
  --old sources/qqq_jtang.csv --new sources/qqq_james.csv
```

期望得到 **58 笔已平仓、24胜34负、25 N20 保护、2026-07-13仍有1笔未平仓**。历史指标用 `research/results/` 中冻结的 CSV 和 `A2_reproduction_metadata.json` 对照；若源在更新中被修订，应记录下载日期、内容哈希和不同数据，而不是强行改规则去拟合既有指标。

## 3. 不同数据源间尤其容易犯的错误

| 问题 | 正确做法 |
|---|---|
| QQQ 分红/拆分导致旧历史价改变 | 使用同一日期的 `Adj Close / Close` 比例调整 OHLC；从**同一复权基准**比较收盘与实际入场价 |
| 两家来源的 Adj Close 复权基准不同 | 先重基准并核对重叠期比例稳定，不要直接拼接产生虚假的金叉 |
| 盘前/当天半日 K 与完整日 K 混淆 | 只用美东当天已完整收盘的数据；遇到空字段中止计算 |
| 两次数据拉取不一致 | 保留来源、时间戳、哈希和差异；不得静默替换数据 |
| 早期长均线没有足够 warmup | SMA60 未建立前不得产生信号。开始交易日期2000-01-03并不代表历史数据从那天才开始 |
| **冻结基准**截至2026-07-13，后续需实盘跟踪 | 已做一次补齐至2026-11-09；以后仍必须更新至最新完整交易日，**重新重放状态机**，不能假定目前没有仓位 |
| 不同券商收费、盘中开盘拥堵 | 实际纸面跟踪单独测滑点/点差。市价开盘不保证在历史开盘价成交 |
| 2020 TQQQ 的跨月收益 | TQQQ是每日复位杠杆产品，不能把 QQQ 单笔收益乘3；其历史对照有独立行情源和不同可用区间 |

## 4. 数据授权与审计清单

提交新行情源或回测报告时至少记录：`provider`, `source_url`, `license_or_terms_url`, `downloaded_at_utc`, `file_sha256`, `coverage_start/end`, `rows`, `adjustment`, `missing_count`, `duplicate_count`, `overlap_validation`, `sample_cutoff`。

没有明确可再分发许可的原始行情文件不得提交 Pull Request；若数据源要求登录/付费，应让每名使用者在本地自行配置。贡献者可以提交自动**校验器**和仅使用本地已授权数据的**适配器**。

## 5. 研究结果的限制

本仓库的所有回测均为历史**样本内**，多轮参数搜索会提高过拟合风险；同一市场暴跌周期内的多个事件高度相关。这里展示的盈亏比、最大回撤、CAGR 和未来走势统计都不是对未来交易的保证，也不是针对当前市场的实时交易建议。

### QQQ 同起点长期买入持有基准和税收遗漏

独立比较器 `src/qqq_a2/benchmarks.py` 和 `research/experiments/a2_buy_hold_benchmark.py` **不会下载或再分发原始行情**。它们读取与 A2 相同的已合法取得的复权 QQQ 日线，从 **2000-01-03 首个常规交易日复权开盘价**买入，付一次 0.05% 买入费，持仓每日以复权收盘估值，截止日不卖，因此**没有计入末期卖出佣金**。统计采用日度权益（而非月末抽样），截止日期为冻结 **2026-07-13** 与扩展 **2026-11-09**；生成结果见 [独立基准报告](research/A2_QQQ买入持有基准_回撤与税务口径_2026-11-10.md)、[结果JSON](../research/results/a2_vs_buy_hold_2026_07_13_and_11_09.json)、[月末账户曲线CSV](../research/results/a2_vs_buy_hold_monthly_equity.csv)。

使用 `Adj Close / Close` 调整 OHLC 所得到的是**税前毛分红再投资代理**，并非实际收取每笔 QQQ 分红、美国预扣税后再投资的账户净值。股息入账通常与股票出售价差分别计税/预扣；中国居民、新加坡个人投资者及美国非居民的所得税规则不同。**税务居民身份、年度成本基础、股息日期与真实分红流水未输入模型，因此没有合规意义的税后 CAGR。** 这些限制及国税总局、IRAS、IRS 原文均列在独立报告中。

## 6. README 的交易总览图

README 现在使用 `assets/a2_trade_map_facets_2000_2026.svg` 这张**分七段四年区间的静态 SVG**，无需 HTML/JS。早先的单张 26 年/月 K 方案 `assets/a2_trade_map_2000_2026.svg` 仍可保留作为实验档案，但密度太高，不再展示在 README。图表以本地已授权的 QQQ 复权日线和冻结 A2-N20 V1.0 生成，当前展示至2026-11-09：

```bash
python -m pip install -e '.[charts]'
python scripts/render_trade_chart_facets.py \
  --csv sources/qqq_merged_updated_local.csv --end 2026-11-09
```

图上**所有实际买卖日期和执行价格都来自日线模型**。背景蜡烛图为**周 K**，每个四年区间采用独立价格刻度以方便辨认；绝不能用周 K 推算策略信号。入场为蓝色上三角，N20 保护卖出为橙色下三角，MA20/60 趋势卖出为紫色下三角，期末仍持仓的最后一个买点为星形。精确成交价及退出原因以完整交易 CSV 为准。

公开 SVG 是对历史样本的**图形化派生展示**，没有附带任何第三方原始 OHLCV 下载包；上游许可与可能的数据修订限制仍适用。

## 6A. 2026-11-09 前推补样（不修改冻结基准）

本次从 `2026-07-13` 的原始 **6877** 行完整交易日序列，追加 2026-07-14 至 2026-11-09 的 **84** 个完整交易日，合计 **6961** 行；已排除纽约市场休市的 2026-09-07（Labor Day），不存在其他解释不了的工作日缺口。2026-11-10 北京时间上午，美东当天尚未开盘，故 **11月9日** 是最近已完成的日 K。

不能直接把旧 Adj Close 和新 Adj Close 拼在一起：两次快照期间 QQQ 已有新的分红，**新快照会按更晚的分红重算所有旧日的 Adj Close**。本次在旧截止日附近对比 **10 个重叠完整交易日**，未复权 Close 的最大差异 **$0.000029**；新快照与旧快照的复权比值中位数为 **0.9989609457128912**，比值最大偏差 **8.50×10⁻⁶**。因此把旧全历史复权 OHLC 同比缩放该因子，再接上新源已复权 OHLC。重算后**原58笔交易的所有信号日、成交日、原因、收益率和逐日账户权益均保持不变**；历史价格数字按新分红基准等比例变化属于预期。

增量第三方源本次本地 CSV 的 SHA-256 为 `2cef4c6567a0e44295033d3d8bd84d26e95942840b3f8683fd1e7a290613e248`。如供应商日后更新数据，可比对该哈希；仓库无法凭哈希还原未授权数据。增量价格源完整交易日的量、OHLC 由 StockQuote.io 提供；最新收盘另经独立页面交叉核验；**这不是两家供应商全历史逐日对账**。

自行取得合法的新增日线后，依次执行：

```bash
python research/experiments/a2_forward_update.py \
  --old sources/qqq_jtang.csv --new sources/qqq_james.csv \
  --update sources/qqq_stockquote_2026_11_09.csv --asof 2026-11-09
python scripts/render_trade_chart_facets.py \
  --csv sources/qqq_merged_updated_local.csv --end 2026-11-09
```

该脚本生成**只本地保存**的 `sources/qqq_merged_updated_local.csv`，公开仅提交 `research/results/A2_forward_*`、`A2_current_trades_to_2026_11_09.csv`、`A2_newly_completed_trades_2026_07_14_to_11_09.csv`。旧 58 笔结果是冻结对照，不被覆盖。当前结果见 [补样报告](research/A2_2026_07_14到11_09_样本扩展复核.md)。

## 7. FRED DGS3MO 的来源、复现与可交易性限制

2026-11-10 进行空仓现金收益独立研究时，以 FRED 官方指标 [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO) 作为来源说明。实验机器使用的历史下载镜像快照为 `rasmuspeterk-ctrl/recession/v5/data/raw/2026-10-27/DGS3MO.csv`，文件 SHA-256 为 `5f795620cdc28b3a14d9e509ba1f3dc65704aa1aa45754a6d3fa2ba0279237e1`，记录覆盖 **1999-11-12 至 2026-10-23**。这不是对镜像仓库授权政策的背书；回测作者还与独立公开镜像的历史观测做了抽样比对。建议复现者优先从 FRED 自行获取原始 `DGS3MO.csv`，并保存下载日期、文件哈希和来源。

`DGS3MO` 是参考**年化收益率报价**，不是银行/券商实际美元现金账户的到账收益，也不是某只 ETF 的真实总回报。以旧交易日可用的上次报价计息，排除未来利率窥视；最长回填间隔为十个日历日。动态短债收益减 `15bp` 的情景仅为模拟费用/摩擦假设，并不代表 SGOV、货币市场基金、券商账户或实际滚动国库券的历史费率。SGOV 直到2020年才推出，不能直接替代 2000 年起的现金仓位。

生成曲线需要在本地准备 `sources/DGS3MO.csv`；执行：

```bash
python research/experiments/a2_cash_yield_backtest.py \
  --old sources/qqq_jtang.csv --new sources/qqq_james.csv \
  --yield-csv sources/DGS3MO.csv --end 2026-07-13 \
  --outdir research/results --chart assets/a2_cash_yield_equity.svg
```

其**公开派生文件**分别为 `research/results/a2_cash_yield_scenarios.csv`（六种现金利率情景）与 `research/results/a2_cash_yield_monthly_equity.csv`（按月末抽样的模拟账户资产，不含第三方原始利率或 OHLCV 行情）。详细现金计息公式、MDD 反向案例及情景局限见 [独立研究报告](research/A2_空仓美债现金收益_独立回测_2026-11-10.md)。

**2026-11-09 更新版：**原历史镜像利率序列截至 10月23日；从美联储 H.15 **官方**下载补齐 10月26日至11月6日十个有报价交易日，追加并拒绝相互矛盾的重叠记录。11月9日 DGS3MO 尚未公开，因此模型按前一已知 **11月6日 4.19%** 的利率计息；不能把当日尚未公布的报价提前使用。新数据只进本地 `sources/`，新派生权益发布到 `research/results/forward_2026_11_09/`；旧冻结现金实验仍留在 `research/results/` 顶层。不应把这些年化收益率数据解读为真实可交易国债组合的逐日净值。
