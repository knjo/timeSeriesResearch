# `fh_neg_rejection` static BackTest extension through 2026-09-23

Status: `RESEARCH_SHADOW_ONLY` (static control; not deployment evidence)  
Direction: short  
Variant: `T1L1S1` conservative control  
Frozen baseline: `20250211–20260825`  
Cost: `19.3bp` per accepted event

## 更新結果

沿用固定 D/G/B selector、short stop108-or-close label、no-model-gate static execution
與逐股 capacity，已延伸到完整的 9/23 收盤。9/22 為 139 筆、23 檔、成本後
`+28.23 萬`；9/23 為 81 筆、11 檔、`+7.33 萬`。兩日合計 220 筆、`+35.56 萬`、
net `+86.803bp`。

| Period | Sessions | Trades | Net BP | Net PnL（萬） | Sharpe | MDD（萬） |
|---|---:|---:|---:|---:|---:|---:|
| 日曆 first-live `20260828–20260923` | 19 | 1,883 | 42.298 | +142.40 | 2.959 | -51.08 |
| 2026 YTD `20260102–20260923` | 176 | 26,567 | 30.573 | +1,145.32 | 2.428 | -158.29 |
| 全窗 `20250211–20260923` | 398 | 73,654 | 65.444 | +5,876.85 | 6.500 | -161.59 |

8/25 frozen parity 143/143 keys 與 PnL 一致；9/21 以前與前版逐日一致。新 artifact
另存 `static_backtest_extension_20260923_v1`，沒有覆寫 frozen 或 9/21 版本。

本次也同步追蹤保留 ToRef、移除 L1-rate 與 spread-elapsed 的 `T1L0S0` 擴量候選；
候選自 9/22 才開始自己的 forward-shadow clock。雙版本比較、residual 診斷與驗證詳見
[execution-gate shadow update](../20260921_fh_neg_rejection_execution_gate_ablation/EXTENSION_20260923.md)。

![2026 extended BackTest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260923_v1/backtest_2026_extended.png)

## Artifacts

- [run manifest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260923_v1/run_manifest.json)
- [accepted event ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260923_v1/accepted_signals_extension.parquet)
- [daily extension](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260923_v1/daily_extension.parquet)
- [combined daily ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260923_v1/daily_combined.parquet)
- [full chart](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_extension_20260923_v1/backtest_full_extended.png)
