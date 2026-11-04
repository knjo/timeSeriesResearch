# fh_neg_rejection static BackTest 更新（2026-10-14）

實際截止 **2026-10-13 收盤**，新增 11 個完整交易日。10/14 尚未收盤且五種來源未齊，不列入績效。

固定 short / T1L1S1 / no-model gate / 19.3 bp 成本，狀態維持 `RESEARCH_SHADOW_ONLY`。

| 期間 | 版本 | 交易日 | 筆數 | Net BP | 淨損益（萬） | Residual 淨損益（萬） | Sharpe | MDD（萬） |
|---|---|---|---|---|---|---|---|---|
| 新增 9/24–10/13 | T1L1S1 | 11 | 944 | -24.235 | -37.89 | -18.27 | -3.778 | -66.02 |
| 候選 shadow 9/22–10/13 | T1L1S1 | 13 | 1164 | -1.179 | -2.33 | -28.12 | -0.177 | -75.44 |
| 控制版日曆 8/28–10/13 | T1L1S1 | 30 | 2827 | +21.199 | +104.51 | +107.91 | 1.664 | -90.38 |
| 2026 YTD | T1L1S1 | 187 | 27511 | +28.377 | +1,107.43 | +801.66 | 2.267 | -158.29 |

[控制版與既有擴量候選的完整比較](../20260921_fh_neg_rejection_execution_gate_ablation/EXTENSION_20261013.md)。

8/25 frozen parity 及 9/23 以前逐筆、逐日結果與前版一致，未修改正式策略或模型。Selector-only extension 的 lineage 限制沿用原報告。

![2026 回測](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/backtest_2026_extended.png)

- [run_manifest.json](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/run_manifest.json)
- [verification.json](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/verification.json)
- [source_coverage.json](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/source_coverage.json)
- [accepted_signals_extension.parquet](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/accepted_signals_extension.parquet)
- [daily_combined.parquet](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/daily_combined.parquet)
- [monthly_combined.parquet](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20261013_v1/monthly_combined.parquet)
