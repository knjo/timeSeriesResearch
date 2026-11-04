# `fh_neg_rejection` inherited execution-gate ablation

Status: `RESEARCH_DIAGNOSTIC_ONLY`  
Window: `20250211–20260921`  
Direction / label: short / `negfill.short_taker_stop108.v1`  
Cost: `19.3bp` per accepted event

Latest calendar update: [EXTENSION_20260923.md](EXTENSION_20260923.md)

## 直接結論

這三條都不是 `fh_neg_rejection` 的 D/G/B event condition，而是從舊 negFill
回測繼承的 execution gates。先前把 static baseline 簡稱為「直接全下」不夠精確；
它只是 **no model gate**，並不是 condition events 全部下單。

完整 `2 x 2 x 2` factorial 顯示：三條全部關閉後，2025、凍結前 2026 與
8/28～9/21 都仍為正，因此沒有任何一條是「策略要有正 PnL」的必要條件。不過現行
三條全開版本在三段都提供較高的每單位名目 net BP，並通常有較高 Sharpe 與較小 MDD。
它們的角色是犧牲量以換取品質和風險控制，不是創造 `fh_neg_rejection` bias。

現階段不應直接刪除三條。若目標是擴量，最合理的新 shadow candidate 是只拿掉
`SpreadPairElapsed > 0.1`，因為三個觀察窗的總 PnL 都增加；但其 first-live-calendar
Sharpe `2.358 → 1.795`、MDD `-51.08 → -81.83 萬`，且回測沒有 latency／impact，
所以必須另開版本向前追蹤，不能回寫現行 baseline。

## Gate 定義

- `T`：positive `ToRef` bucket，實際為 `0 < ToRef <= 5%`。因 robustness pool 已有
  `-1.5% < ToRef < 5%`，關閉 T 只會新增 `-1.5% < ToRef <= 0`。
- `L`：`OTC OR MD_L1Rate_30_re > 0.25`；上市股票要求最近 30 個行情封包有足夠成交比例。
- `S`：`SpreadPairElapsed > 0.1` 秒，避免立即反應於存活不足 100ms 的 spread pair。

所有 variant 都保留時間 bucket、成交額排名、價格上限、前日 maker-spread gate、
一張／event、前日一檔深度上限及單股每日 200 萬元上限；capacity 逐 variant 重新計算。

## 從現行版本逐條拿掉

`T1L1S1` 是目前 static baseline；其餘列一次只拿掉一條，`T0L0S0` 則三條全拿掉。
PnL／MDD 單位為萬元。

### 2025 retrospective

| Variant | Trades | Net BP | Net PnL | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|
| `T1L1S1` current | 47,087 | 90.404 | +4,731.53 | 11.921 | -40.15 |
| `T0L1S1` no ToRef | 72,143 | 45.609 | +3,707.18 | 7.466 | -74.49 |
| `T1L0S1` no L1 rate | 64,931 | 66.446 | +4,675.18 | 9.274 | -46.73 |
| `T1L1S0` no spread elapsed | 70,070 | 79.734 | **+6,059.45** | 11.396 | -72.17 |
| `T0L0S0` all three off | 137,225 | 30.330 | +4,409.87 | 4.974 | -216.48 |

### 2026 pre-freeze (`20260102–20260825`)

| Variant | Trades | Net BP | Net PnL | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|
| `T1L1S1` current | 24,570 | **32.086** | +1,083.56 | 2.588 | **-158.29** |
| `T0L1S1` no ToRef | 37,038 | 30.286 | **+1,526.51** | **2.613** | -221.53 |
| `T1L0S1` no L1 rate | 40,363 | 21.514 | +1,170.09 | 1.901 | -310.67 |
| `T1L1S0` no spread elapsed | 38,835 | 24.131 | +1,270.43 | 2.031 | -327.81 |
| `T0L0S0` all three off | 78,179 | 15.911 | +1,577.49 | 1.458 | -616.85 |

### Baseline calendar first-live (`20260828–20260921`)

對現行 `T1L1S1` 而言這 17 日是既有 calendar first-live；其他 variant 是今天才查看的
post-hoc counterfactual，不能稱為各 variant 自己的 prospective first-live。

| Variant | Trades | Net BP | Net PnL | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|
| `T1L1S1` current | 1,663 | **36.132** | +106.84 | **2.358** | **-51.08** |
| `T0L1S1` no ToRef | 2,775 | 24.395 | +116.35 | 1.551 | -77.34 |
| `T1L0S1` no L1 rate | 3,072 | 29.115 | +155.42 | 2.088 | -84.35 |
| `T1L1S0` no spread elapsed | 2,611 | 27.888 | +128.37 | 1.795 | -81.83 |
| `T0L0S0` all three off | 6,624 | 23.964 | **+246.41** | 1.733 | -130.03 |

三條全關會把這 17 日交易量放大到 `3.98x`、總 PnL 增加 `139.57 萬`，但 Net BP
下降約三分之一、MDD 變成 `2.55x`。因此「更多交易仍有正邊際」與「現行 gates
提升品質」可以同時成立。

9/21 單日也顯示量的來源：現行 62 筆；no-ToRef 103、no-L1 101、no-spread 94、
三條全關 228。單日 PnL 分別為 `+0.58、+9.61、-4.34、-1.51、+6.92 萬`，只作
funnel 說明，不拿單日選 gate。

## 分別判讀

- **ToRef**：比較像價格位置／state partition，不是成交可執行性。2025 的品質差異很大，
  但 2026 拿掉後總 PnL增加且 pre-freeze Sharpe持平，顯示它可能有 regime drift。
  研究框架應把 `ToRef <= 0` 與 `> 0` 分層報告，不應把正 ToRef 偷塞回 event condition。
- **L1 rate**：是三者中最有 execution/liquidity 理由的一條。拿掉雖增加近期 PnL，
  但 2026 pre-freeze Sharpe `2.588 → 1.901`、MDD近乎翻倍；而 replay 沒有 slippage、
  latency 或 fill-failure，暫時保留較安全。
- **Spread elapsed**：總 PnL在三段都因放寬而增加，是最值得獨立 forward-shadow 的
  擴量候選；但品質與 MDD都變差，且 100ms 本來就是防 fleeting quote，不能用理想化
  same-tick fill 直接證明它多餘。

## Artifacts 與限制

現行 `T1L1S1` 73,434 個 accepted keys 與 frozen snapshot 加 9/21 extension 完全一致，
missing／unexpected keys 都是 0。八組使用同一個 D/G/B cohort、short stop108-or-close
label、19.3bp 成本與逐股 capacity。

這是看過結果後的 sensitivity，不是 gate selection 或 deployment evidence；沒有 borrow、
impact、latency、同時曝險或 portfolio cap。若要改 gate，需先凍結候選與風險 objective，
再讓下一批完整交易日成為真正 first-live。

- [study contract](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/study_spec.json)
- [run manifest](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_v1/run_manifest.json)
- [period summary](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_v1/period_summary.parquet)
- [daily summary](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_v1/daily_summary.parquet)
- [funnel summary](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_v1/funnel_summary.parquet)
- [accepted-signal trial ledger](../../../test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_v1/accepted_signals.parquet)
