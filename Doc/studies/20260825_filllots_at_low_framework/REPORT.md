# FillLots_atLow framework validation

## 2026-08-26 研究目標更正

本報告的 26-candidate registry 從 formal-trade state update、epoch entry 與 depth
threshold 中挑選候選，最後依股票內 timing 證據凍結 `depth_gt_pos8_state_update`。
它適合保存 event-shaping 與 matched 歸因結果，但**不是**目前「找出類似 negFill
的長持倉 bias distribution」之主要 positive-control 判定。

目前長持倉探索以 shared-pool 為 primary estimand。第一關應直接測事前已知的完整
state condition `FillLots_atLow < 0`，使用既有方向別 barrier-or-close label；不得
先加 formal-trade update、epoch、depth threshold 或 cooldown。本報告沒有把這個
原始 state condition 列入固定 26 candidates，因此不能用 `+8 state_update` 的
matched 結果取代 recovery，也不能以 matched 反號否決負向 state 的 shared bias。

後續正確流程見
[DISCOVERY_PROTOCOL.md](../../../docs/protocols/DISCOVERY_PROTOCOL.md)。以下內容保留為
原研究問題與數值紀錄；其中「最終候選」只代表該 timing-oriented candidate sweep，
不是目前長持倉 discovery 的總冠軍。

## 結論

可以用這個框架找到 `FillLots_atLow` 的 bias event。這次把它當成新的 event
family，而不是把 negFill 條件改名；最終候選是
`depth_gt_pos8_state_update`：正式成交更新狀態後，
`FillLots_atLow / D-1 top-of-book depth > 8`。

最可靠的結果不是跨股票的 shared-pool alpha，而是同股票、同 5-minute 時段內的
short-label timing bias。376-session dense check 的 same-stock-time 平均
`Δz = +0.035138`，HAC `t = 43.130`，375 個 active days 中有 374 日為正，
19 / 19 個月為正；相對地，shared-pool 在 2026 轉成 `Δz = -0.046489`。
因此這個條件應解讀成深正向累積 flow 後的股票內 timing pattern；它與
耗竭／反轉假說一致，但 label 沒有直接辨識該機制。它不能解讀成任意股票組合上的
普遍 alpha，也還不是可直接交易的單次 entry event。

## 固定研究契約

| 項目 | 契約 |
|---|---|
| Event family | `filllots_at_low`, version `3` |
| 狀態 | `F = cumsum(FillLots × InOut × (TrialMatch == 0))` over `(QuoteCode, RecordLow)` |
| 深度尺度 | `D = avg_bidLots1 + avg_askLots1`，來自 decision-date preMarket 的 D-1 統計 |
| Formal trade | `marketOpen`, `TrialMatch == 0`, `FillLots > 0`, `InOut ∈ {-1, 1}` |
| Causal order | 在完整 raw tick stream 依 `(QuoteCode, ChannelSeq)` 排序並重算 F、lag、epoch，再與 eligible pool 交集 |
| Label | `negfill.short_taker_stop108.v1`；fingerprint `bc3c3adef1cb3dc7b4e43d27c203ca8b077c56c6f1f0a7882c18fcd19639a3cd` |
| Pool | `tw_equity_daytrade_rank150_090030_123000_bidref_v1`, version `1` |
| Control | 完整 shared eligible labeled pool；`control_sample_size=None` |
| Sparse screen | 26 個 code-frozen candidates，2025-02-11 至 2026-08-24，每 5 個 available sessions 取 1 日，共 76 日 |
| Dense check | 同期間全部 376 個 available sessions；2025 為 222 日，2026 為 154 日 |
| Inference | equal-day Student t、Newey-West Bartlett HAC lag 5（依 null removal 後的 ordered active observations）；sparse full-period 對 shared、same-stock、same-stock-time 各做一個 26-candidate BH family，不合併成 78 tests |

正的 delta z 表示對本次 short label 較有利，不等於股價正報酬。

## 三種 estimand

| Estimand | 計算 | 能回答的問題 |
|---|---|---|
| shared-pool | condition mean z − 完整 eligible pool mean z | condition 是否切出任何可見 bias；包含股票與時段組成 |
| same-stock | 每日先算每個 active QuoteCode 的 condition − 該股 full-pool mean，再等權股票 | 移除股票日組成後是否仍有 timing 差異 |
| same-stock-time | 在 QuoteCode × 5-minute bucket 內比較，先等權該股 active buckets，再等權股票 | 再移除粗粒度 clock-time 組成後是否仍有事件差異 |

這三者不能互相替代。shared-pool 顯著而 matched 為零或反號，應解讀成
exposure/composition bias，不應宣稱找到日內 timing event。
Matched baseline 是包含 condition 本身的無條件完整 eligible pool，不是
event-vs-non-event causal control；self-inclusion 會把高 event-share cell 的差異
往零壓。在單一 matched cell 內，它只以 `(1 - event share)` 衰減
event-vs-non-event contrast，不會翻號；跨 cell 等權後因各 cell 衰減率不同，若
cell effects 本來異號，aggregate 方向仍可能和 disjoint-control estimand 不同。

## 候選形成與版本修正

初始 registry 對正負方向、state update／epoch entry、深度倍率
`1/2/4/8` 做對稱掃描。稀疏掃描後凍結 `depth_gt_pos8_state_update`，理由是正向
dose ladder 中它有最大的 same-stock effect、最低事件密度，而且 2025、2026
同向。它在加入 5-minute match 前已凍結，因此 clock-time 檢查是 falsification，
不是用同一資料重新挑 threshold。

審查期間發現舊 profile 先 inner join tickFeature 才算 lag，會漏掉 tickFeature
沒有保存的 trial rows。family v3 改由完整 tickData 依 feature source-of-truth 公式
重算 F，並明列第一個 `marketOpen` row 為 epoch start。76 日逐鍵比較顯示：

- `depth_gt_pos8_state_update` raw index 新增 1,599 / 28,603,709 keys；和研究 pool
  交集後只新增 5 / 7,932,560 keys。
- `depth_lt_neg1_epoch_entry` raw 與 eligible keys 都完全不變。

因此修正解決了框架語意缺口，沒有實質改寫本次候選結論。
精確計數與 audit 限制保存在
[AUDIT_SUMMARY.json](../../../test/20260825_filllots_at_low_framework/AUDIT_SUMMARY.json)；這是 ad-hoc
key-level audit 的 retained summary，沒有保留數千萬筆中間 key frames。

另有一項只改描述、不改 selector 的 erratum：現存 v3 artifact manifests 是在這次
文件審查前產生，depth `epoch_entry` description 仍寫成 `first enters`；實際 v3
selector 一直是「epoch start 或同 epoch 每次 crossing／re-entry」，所以 keys 與
數值不變；後續由現行 source 產生的 manifests 會使用正確語意。

## 最終候選：正向深狀態 update

`depth_gt_pos8_state_update` 不是「第一次越過 +8」，而是 F/D 已在 +8 以上時，
每次正式成交造成 F 改變的 state update。這種定義適合辨識有 bias 的狀態列，
但同一個 RecordLow epoch 會重複觸發，尚不是可直接下單的 episode entry。

稀疏 76-session screen 的正向 depth ladder 顯示：門檻越深，same-stock 與
same-stock-time effect 越大，同時事件密度下降。這是一條 descriptive monotone
threshold ladder，與 dose-response 假說一致，不是只靠 `+8` 單點成立；但各門檻
的 active stocks/cells 會改變，尚未做同一單位內的 causal dose test：

| candidate | events / active day | shared pooled Δz | shared equal-day Δz / HAC-BH q | same-stock Δz | same-stock-time Δz | time HAC t | time BH q | time-positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `depth_gt_pos1_state_update` | 198,255.4 | +0.026827 | +0.039279 / 0.124353 | +0.141775 | +0.026581 | 35.514 | 1.963e-275 | 19 / 19 |
| `depth_gt_pos2_state_update` | 173,617.0 | +0.033315 | +0.047010 / 0.078597 | +0.171882 | +0.029388 | 52.517 | <1e-300 | 19 / 19 |
| `depth_gt_pos4_state_update` | 141,715.2 | +0.038049 | +0.052150 / 0.078597 | +0.212236 | +0.033942 | 29.468 | 3.835e-190 | 19 / 19 |
| `depth_gt_pos8_state_update` | 104,375.8 | +0.039603 | +0.063432 / 0.078597 | +0.267851 | +0.036794 | 21.380 | 5.963e-101 | 19 / 19 |

shared-pool 的 `+8` equal-day HAC raw `p = 0.031409`，但在固定 26 candidates 內
`BH q = 0.078597`，沒有通過 5% FDR；matched families 則通過。這個差異本身就是
結論的一部分：候選主要是股票內的 timing effect，而不是穩定的 pool-level
composition effect。

凍結 `+8 state_update` 後，全部 376 sessions 的 dense check 為：

| period | sessions / active | event rows | shared pooled Δz | shared equal-day Δz / HAC t | same-stock Δz / HAC t | same-stock-time Δz / HAC t | time-positive days | time-positive months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 222 / 221 | 19,870,081 | +0.061774 | +0.109261 / 3.935 | +0.279471 / 41.584 | +0.033363 / 32.678 | 220 / 221 | 11 / 11 |
| 2026 | 154 / 154 | 18,173,221 | -0.046489 | -0.010764 / -0.534 | +0.260545 / 26.875 | +0.037686 / 32.197 | 154 / 154 | 8 / 8 |
| full | 376 / 375 | 38,043,302 | +0.010186 | +0.059971 / 3.039 | +0.271699 / 47.323 | +0.035138 / 43.130 | 374 / 375 | 19 / 19 |

2025-04-07 沒有 eligible pool，因此 376 sessions 中有 375 active days。2026 的
shared-pool 反號而兩個 matched estimands 維持正向，是區分 composition 與 timing
最重要的 falsification。

狀態 phase 也不能省略。稀疏 full-period 中，所有 `F > 0` 的 state update 在
same-stock-time 為 `+0.013514`（HAC `t = 15.318`），但 RecordLow epoch 起點為正，
或同 epoch 從非正再次跨入正值的 `positive_epoch_entry` 是 `-0.077114`
（HAC `t = -48.857`，0 / 19 time-positive months）。這個 selector 可在同一 epoch
反覆跨零時重複觸發，不能稱為 first-ever entry。結果只支持「sign crossing／
re-entry 較像延續，累積到深正向狀態後則對 short label 有利」的 phase 差異。
它本身不識別耗竭機制。

## 負向 epoch entry：是 exposure bias，不是 timing event

早期 shared-pool screen 的 `depth_lt_neg1_epoch_entry` 很強，但 corrected
same-stock estimand 反號。family v3 的 selector 和 v2 在 76 日逐鍵完全相同；
它選的是 epoch 起點已低於 -1，或同 epoch 跨入／再次跨入 -1 以下，可能重複，
不是每個 epoch 僅第一筆。v3 sparse matched 結果如下：

| period | sessions | event rows | shared pooled Δz | shared equal-day Δz / HAC t | same-stock Δz / HAC t | same-stock-time Δz / HAC t | time BH q (full only) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 45 | 28,072 | +0.150592 | +0.157807 / 3.736 | -0.170282 / -23.712 | -0.015780 / -7.429 | — |
| 2026 | 31 | 14,813 | +0.068536 | +0.100406 / 2.121 | -0.186887 / -10.304 | -0.031130 / -8.931 | — |
| full | 76 | 42,885 | +0.126947 | +0.134393 / 4.141 | -0.177055 / -19.789 | -0.022042 / -7.819 | 6.609e-15 |

它有 17 / 19 個 shared-positive months，卻只有 4 / 76 個 time-positive days、
0 / 19 個 time-positive months。基於 76 sparse 日的逐鍵 v2-v3 equivalence audit，
舊 v2 全日 dense run只作 supporting robustness，尚未完成負向候選全部 376 日的
v3 replication；該 run 得到
shared pooled `Δz = +0.122264`、equal-day `Δz = +0.099706`、HAC `t = 8.261`，
但 same-stock
`Δz = -0.179328`、HAC `t = -43.070`。所以它能證明 shared-pool screen 找到了
bias condition，卻不能當作對 short label 有利的股票內 timing event。

## 與 negFill 的關係

依共同 canonical F 定義，正向最終候選要求 `F > 0`，而 negFill reference 要求
`F < 0`，理論上互斥。但目前前者從 raw tickData 重算 F，後者讀 tickFeature，
runtime 尚未設 cross-source parity guard；9 個跨期 sparse 抽查日、3,198,102 個
positive candidate keys 未見 F 漂移或 negFill exact overlap。抽樣規則是 ordered
sparse sessions 每 10 日取 1 日再補最後一日，確切日期與計數見
[AUDIT_SUMMARY.json](../../../test/20260825_filllots_at_low_framework/AUDIT_SUMMARY.json)。另對負向
`depth_lt_neg1_epoch_entry` 做 76 日 exact partition，確認它的 shared-pool bias
也不只是重播 negFill：

| cohort | eligible rows | pooled Δz | equal-day Δz | daily t | positive months |
|---|---:|---:|---:|---:|---:|
| candidate | 42,885 | +0.126947 | +0.134393 | 3.691 | 17 / 19 |
| exact overlap negFill | 16,701 | +0.105395 | +0.119081 | 2.893 | 16 / 19 |
| exact disjoint negFill | 26,184 | +0.140694 | +0.150347 | 4.180 | 18 / 19 |

Raw candidate keys 和 negFill 的 exact overlap 為 48.742%；stock-day overlap 為
99.910%，說明 exact-disjoint 不是相同 tick，但大多仍落在同一批股票日。
Overlap runner 只提供 classical daily t，不能把上表寫成 HAC/BH 證據。

本文的 time-positive month 是先在每個 calendar month 對 active-day
`stock_time_equal_delta_z` 等權平均，再檢查月均值是否大於零；不是把 event rows
視為獨立觀測。

## 研究限制與下一關

- 2025、2026 都參與候選探索；dense rerun 也重用相同日期。2026 是 regime
  stability diagnostic，不是 untouched holdout。
- Sparse report 的 BH 只修正這次固定 26-candidate registry 的 full-period
  p-values；shared、same-stock、same-stock-time 是三個獨立的 26-candidate
  families，不是聯合 78-test correction。這不能消除 registry 設計與候選凍結
  之前的 researcher degrees of freedom。Dense report 只有凍結後的一個
  candidate，其 q 等於 raw p，不當作 26-candidate 或 selection-adjusted 證據。
- state-update 會在同一 epoch 重複觸發。daily HAC 不等於 stock/epoch cluster
  inference，也不能把數百萬 event rows 當獨立樣本。
- 5-minute match 移除了粗粒度時段組成，但尚未匹配更細的 order-book state。
- condition 與 BidPrice1 entry 位於同 tick；交易化前必須做 next-tick／latency
  sensitivity、episode cooldown 與 executable fill 驗證。
- Main sweep runner 現已把每日 `tick_data`、`pre_market`、`txf_tick` 的 resolved path、
  size、mtime 綁進 cache 與 run manifest；同路徑來源被替換、或舊 cache 缺 identity
  都會 fail closed。現存 final v3 數值是在這項 cache schema 強化前完成；新 runner
  接續它們時需用 `--force` 將舊 daily cache quarantine 後重建，不能靜默沿用。
- Overlap runner 套用相同契約，另包含它實際讀取的 `tick_feature` identity。
- 下一次確認應凍結 selector/profile/pool/label hashes，使用本報告之後的新日期。

## Reproduce

以下刻意使用新的 output directories，避免覆蓋本報告 artifacts，也避免碰到
cache schema 強化前的 daily files。

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/run_sweep.py \
  --start 20250211 --end 20260824 --step 5 \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_sparse_v3_sourceid
```

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/report_sweep.py \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_sparse_v3_sourceid \
  --quiet
```

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/run_sweep.py \
  --start 20250211 --end 20251231 --step 1 \
  --candidate depth_gt_pos8_state_update \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_dense2025_pos8_v3_sourceid
```

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/run_sweep.py \
  --start 20260101 --end 20260824 --step 1 \
  --candidate depth_gt_pos8_state_update \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_dense2026_pos8_v3_sourceid
```

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/report_sweep.py \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_dense2025_pos8_v3_sourceid \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_dense2026_pos8_v3_sourceid \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_dense_pos8_v3_sourceid/report \
  --quiet
```

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/run_negfill_overlap.py \
  --start 20250211 --end 20260824 --step 5 \
  --candidate depth_lt_neg1_epoch_entry \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/reproduce_overlap_sparse_v3_sourceid
```

Artifact source map：

- sparse/dense Student、HAC、BH 數值：
  `src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/<run>/report/period_summary.parquet`；
- time-positive day/month：各 `<run>/daily_stats.parquet`，依本文定義聚合；
- negFill overlap partition：
  `src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/overlap_sparse_v3/summary.parquet`；
- v2-v3 selector diff 與 9-day source parity：
  [AUDIT_SUMMARY.json](../../../test/20260825_filllots_at_low_framework/AUDIT_SUMMARY.json)。

Runner 的 `summary.parquet` 只是 quick descriptive output，不是 final inference table。
