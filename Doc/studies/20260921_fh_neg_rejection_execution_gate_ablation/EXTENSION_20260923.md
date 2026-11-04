# `fh_neg_rejection` execution-gate shadow update through 2026-09-23

Status: `RESEARCH_SHADOW_ONLY`  
Candidate: `T1L0S0` (keep positive ToRef; drop L1-rate and spread-elapsed gates)  
Control: `T1L1S1`  
Candidate selection frozen after: `20260921`  
Candidate forward-shadow window: `20260922–20260923`  
Direction / label: short / `negfill.short_taker_stop108.v1`  
Cost: `19.3bp` per accepted event

## 結論

9/22、9/23 五種來源完整，已用相同 D/G/B condition、short stop108-or-close
label、19.3bp 成本、其他 static gates 與逐股 capacity 更新兩個版本。`T1L0S0`
是看完 9/21 以前結果後選定，因此只有 9/22、9/23 能列為這個候選的 forward-shadow；
8/28～9/21 只能保留為候選形成期的歷史診斷。

兩個新交易日，擴量候選共 404 筆、成本後 `+59.94 萬`、net `+68.341bp`；
保守 control 共 220 筆、`+35.56 萬`、net `+86.803bp`。候選多 184 筆並多賺
`+24.38 萬`，但 edge density 較低。只有兩日，不足以升級狀態。

另外，兩日 residual net 分別為候選 `-39.67 萬`、control `-9.84 萬`；raw short
PnL 為正，但去掉 prior beta × TXF 成分後為負，表示這兩天的獲利含有顯著市場方向
貢獻，不能解讀成已證明穩定的 idiosyncratic edge。

## 新增日期

| Date | Variant | Trades | Stocks | Net BP | Net PnL（萬） | Residual net（萬） |
|---|---|---:|---:|---:|---:|---:|
| 2026-09-22 | `T1L1S1` control | 139 | 23 | +91.946 | +28.23 | -12.51 |
| 2026-09-22 | `T1L0S0` candidate | 223 | 35 | +91.927 | +53.40 | -31.56 |
| 2026-09-23 | `T1L1S1` control | 81 | 11 | +71.415 | +7.33 | +2.66 |
| 2026-09-23 | `T1L0S0` candidate | 181 | 21 | +22.085 | +6.54 | -8.11 |

9/22 候選增加的量沒有稀釋 raw net BP，且 PnL 明顯增加；9/23 則相反，候選雖多
100 筆，總 PnL 反而比 control 少約 `0.79 萬`。這正是需要並行 forward shadow、
不能用兩日直接刪 gate 的原因。

## 更新後歷史視圖

下表包含候選形成期，不是 `T1L0S0` 的純 prospective estimate。

| Period | Variant | Sessions | Trades | Net BP | Net PnL（萬） | Sharpe | MDD（萬） |
|---|---|---:|---:|---:|---:|---:|---:|
| 8/28–9/23 | `T1L1S1` | 19 | 1,883 | 42.298 | +142.40 | 2.959 | -51.08 |
| 8/28–9/23 | `T1L0S0` | 19 | 4,437 | 35.952 | +271.38 | 2.707 | -104.14 |
| 2026 YTD | `T1L1S1` | 176 | 26,567 | 30.573 | +1,145.32 | 2.428 | -158.29 |
| 2026 YTD | `T1L0S0` | 176 | 59,529 | 16.759 | +1,317.86 | 1.447 | -466.40 |

因此 `T1L0S0` 仍適合做擴量 shadow，但 control 的單位 edge、Sharpe 與 MDD 明顯較好。
目前不修改 production，也不把候選取代 control。

![2026 comparison](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/backtest_2026_t1l0s0_vs_control.png)

## Lineage 與驗證

- 9/22、9/23 StockTick 到 13:33、tickBar 到 13:19:59、TXF 到 13:45；來源完整。
- 兩日 TXF `QuoteSource` 均為 `nas_txf_only_snapshot`。
- 8/25 frozen parity：143/143 keys 一致，PnL 一致。
- `T1L1S1` 與 frozen + extension reference：73,654/73,654 keys，missing/unexpected 均 0。
- 全 8 variants 共 1,114,224 ledger rows，無重複 variant-event key。
- 逐筆與逐日 PnL 最大差 `1.06e-6` 元，在 `atol=2e-6, rtol=1e-12` 下通過。
- 9/21 以前 136 個 variant-day rows 與前版逐日一致；最大 PnL 差低於 `1e-9` 元，交易筆數差 0。
- 四個 frozen static contract unit tests 通過。

## Artifacts

- [dated runner](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/run_gate_ablation_20260923.py)
- [run manifest](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/run_manifest.json)
- [verification](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/verification.json)
- [candidate manifest](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/shadow_candidate_manifest.json)
- [candidate/control daily ledger](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/shadow_daily_20260922_20260923.parquet)
- [candidate/control summary](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/shadow_summary_20260922_20260923.parquet)
- [full factorial period summary](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/period_summary.parquet)
- [full factorial accepted ledger](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20260923_v1/accepted_signals.parquet)
