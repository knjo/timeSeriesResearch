# fh_neg_rejection 更新（2026-10-14）

實際回測截止 **2026-10-13 收盤**。執行查核時 10/14 尚未收盤，當日 tickData、tickFeature、preMarket、tickBar、TXF 五種來源尚未產出；沒有將盤中資料或缺資料補零納入。

沿用原 D/G/B selector、short stop108-or-close、每筆一張、逐股 capacity、19.3 bp 成本及無模型門檻的 static backtest。控制版 `T1L1S1` 與先前凍結的擴量候選 `T1L0S0`（保留 ToRef，移除 L1-rate/spread-elapsed）設定不變。保留全部 8 格既有 gate factorial 供核對，未依本次結果重選。

狀態維持 `RESEARCH_SHADOW_ONLY`。本次為研究回測更新，不是實盤；selector-only extension 未重作完整 factorial lineage audit。候選在 9/21 結果後選定，其 shadow 日曆從 9/22 開始；更早期間含候選形成資料。Residual 是另外的市場因子診斷，不是實際可領取的損益。

## 本次結果

新增 9/24–10/13 的 11 個完整交易日，控制版 944 筆、成本後 -37.89 萬元；擴量候選 2,289 筆、-110.53 萬元。候選同期比控制版少 72.64 萬元。從候選凍結後 9/22 開始至 10/13，控制版 -2.33 萬元、候選 -50.59 萬元；維持原觀察狀態。

## 9 月與 10 月

10 月只計至 10/13，共 8 個交易日；9 月為完整 20 個交易日。單位：萬元。

| 月份 | 版本 | 交易日 | 筆數 | 淨損益 | Residual 淨損益 |
|---|---|---:|---:|---:|---:|
| 202609 | T1L0S0 | 20 | 4443 | +251.86 | +174.79 |
| 202609 | T1L1S1 | 20 | 1799 | +146.68 | +99.49 |
| 202610 | T1L0S0 | 8 | 1722 | -32.76 | +24.32 |
| 202610 | T1L1S1 | 8 | 782 | -7.58 | +21.69 |

## 期間比較

| 期間 | 版本 | 交易日 | 筆數 | Net BP | 淨損益（萬） | Residual 淨損益（萬） | Sharpe | MDD（萬） |
|---|---|---|---|---|---|---|---|---|
| 新增 9/24–10/13 | T1L1S1 | 11 | 944 | -24.235 | -37.89 | -18.27 | -3.778 | -66.02 |
| 候選 shadow 9/22–10/13 | T1L1S1 | 13 | 1164 | -1.179 | -2.33 | -28.12 | -0.177 | -75.44 |
| 控制版日曆 8/28–10/13 | T1L1S1 | 30 | 2827 | +21.199 | +104.51 | +107.91 | 1.664 | -90.38 |
| 2026 YTD | T1L1S1 | 187 | 27511 | +28.377 | +1,107.43 | +801.66 | 2.267 | -158.29 |
| 新增 9/24–10/13 | T1L0S0 | 11 | 2289 | -32.481 | -110.53 | -82.39 | -5.306 | -146.84 |
| 候選 shadow 9/22–10/13 | T1L0S0 | 13 | 2693 | -11.821 | -50.59 | -122.06 | -1.888 | -174.05 |
| 控制版日曆 8/28–10/13 | T1L0S0 | 30 | 6726 | +14.688 | +160.85 | +180.54 | 1.227 | -214.21 |
| 2026 YTD | T1L0S0 | 187 | 61818 | +14.716 | +1,207.33 | +566.98 | 1.279 | -466.40 |

## 新增 11 個完整交易日

| 日期 | 版本 | 筆數 | 股票數 | 淨損益（萬） | Residual 淨損益（萬） |
|---|---|---|---|---|---|
| 20260924 | T1L0S0 | 140 | 18 | -27.22 | -20.46 |
| 20260924 | T1L1S1 | 44 | 11 | -9.41 | -7.13 |
| 20260929 | T1L0S0 | 257 | 24 | -16.45 | -34.21 |
| 20260929 | T1L1S1 | 52 | 15 | -3.98 | -9.04 |
| 20260930 | T1L0S0 | 170 | 34 | -34.10 | -52.05 |
| 20260930 | T1L1S1 | 66 | 19 | -16.92 | -23.78 |
| 20261001 | T1L0S0 | 139 | 19 | -53.97 | -25.48 |
| 20261001 | T1L1S1 | 55 | 12 | -23.76 | -10.86 |
| 20261002 | T1L0S0 | 323 | 26 | -42.31 | -33.51 |
| 20261002 | T1L1S1 | 165 | 21 | -21.37 | -17.21 |
| 20261005 | T1L0S0 | 375 | 31 | +48.57 | +47.77 |
| 20261005 | T1L1S1 | 196 | 26 | +24.29 | +22.58 |
| 20261006 | T1L0S0 | 117 | 18 | +19.06 | +29.77 |
| 20261006 | T1L1S1 | 65 | 16 | +12.21 | +20.75 |
| 20261007 | T1L0S0 | 180 | 31 | -12.10 | -11.55 |
| 20261007 | T1L1S1 | 68 | 18 | +5.73 | +6.36 |
| 20261008 | T1L0S0 | 248 | 14 | -12.04 | -19.91 |
| 20261008 | T1L1S1 | 81 | 11 | -5.88 | -8.20 |
| 20261012 | T1L0S0 | 199 | 20 | +15.33 | +37.68 |
| 20261012 | T1L1S1 | 82 | 13 | +2.43 | +11.92 |
| 20261013 | T1L0S0 | 141 | 8 | +4.69 | -0.45 |
| 20261013 | T1L1S1 | 70 | 7 | -1.23 | -3.64 |

## 驗證與產出

- 8/25 frozen parity、9/23 以前逐筆鍵／報酬及逐日損益均與前版一致。
- 全 8 版本 variant-event 無重複，逐筆加總與逐日損益一致；成本不重複扣除。
- Sharpe 使用含無交易日的完整來源交易日日曆；無交易日只有確認來源完整後才保留零損益。
- 既有四項 static contract 測試通過，詳見 verification.json；來源檔案與時間涵蓋見 source_coverage.json。

![回測比較](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/backtest_2026_t1l0s0_vs_control.png)

- [run_manifest.json](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/run_manifest.json)
- [verification.json](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/verification.json)
- [accepted_signals.parquet](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/accepted_signals.parquet)
- [daily_summary.parquet](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/daily_summary.parquet)
- [shadow_period_summary.parquet](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/shadow_period_summary.parquet)
- [shadow_monthly_summary.parquet](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/shadow_monthly_summary.parquet)
- [shadow_daily_20260922_20261013.parquet](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/shadow_daily_20260922_20261013.parquet)

## 與 negFill 最近 DD 的比較

後續已完成 [2024 起 negFill notebook 重播與市場歸因](../20261014_negfill_market_attribution/REPORT.md)。該報告解釋控制版定義，並區分 9 月選擇效果轉弱與 10 月共同市場逆風。FH／negFill 出場和容量不同，不能把金額差當成受控替代策略效果；本文件的 FH 設定與結果不變。
