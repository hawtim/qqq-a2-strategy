# 本地行情输入（不随仓库分发）

本目录只存放你自己有权获取、使用的 QQQ/TQQQ 日线行情。

建议文件名：

- `qqq_daily.csv`：用于每日只读信号跟踪的最新 QQQ CSV；
- `qqq_jtang.csv` 与 `qqq_james.csv`：仅供对照复现历史研究的两份原始导出；
- `tqqq_daily.csv`：仅供另行授权的 TQQQ 实验使用，**不是 A2 V1.0 必需输入**。

全部 CSV 被根目录 `.gitignore` 忽略。请勿把私人行情、付费数据、账号凭据、订单或账户资产上传至 GitHub。

来源链接、各版本价格复权和拼接方法及其已知限制见 [DATA_SOURCES.md](../docs/DATA_SOURCES.md)。
