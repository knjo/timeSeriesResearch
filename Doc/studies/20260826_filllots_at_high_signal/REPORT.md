# FillLots_atHigh 空方訊號與 BackTest

狀態：`scoped`／research-only  
研究日期：2026-08-26  
資料區間：2025-02-11 至 2026-08-25  
方向：short `fh_neg_rejection`  
主要結果：2025 development 選出的 `timing_only` gate，在 2026 YTD gate-only
prequential evaluation 扣 19.3bp 後仍為正

## 結論

`fh_neg_rejection` 可以做成有正絕對 PnL 的一口 taker short prototype，但目前尚未證明
它是 negFill-independent 的第二本書，也沒有證據顯示 development 選出的 timing
model gate 在 2026 增加價值。

- 2026 YTD gate-variant prequential evaluation：17,739 trades、155/155 active days、PnL
  `+653.91萬`，資金加權 gross/net 為 `43.558/24.258bp`，日 Sharpe `2.149`，MDD
  `-136.94萬`。19.3bp 成本後仍為正，25bp 壓力測試仍有 `+18.558bp`。
- Full model era：41,545 trades、PnL `+3,363.48萬`、net `57.923bp`、calendar-filled
  日 Sharpe `6.127`；exact BackTest active-day Sharpe 為 `6.161`。
- 但是 2026 的 static-only condition 是 `31.728bp / Sharpe 2.550`，優於事前由 2025
  選出的 timing-only `24.258bp / 2.149`。因此接受的是 condition 的 standalone
  tradingized result，不接受「timing model 帶來 holdout incremental alpha」。
- 只有 `1.34%` selected ticks 與既有 negFill raw index 完全相同；排除這些 ticks 後
  2026 仍有 `22.898bp / Sharpe 2.078`。但 `98.13%` selected stock-days 位於廣義
  negFill event stock-day domain；只保留 stock-day disjoint 後，2026 變成
  `-187.003bp / Sharpe -3.895`。因此 exact trigger 明顯不同，但這個 broad-domain
  disjoint 診斷沒有建立 incremental portfolio alpha。

建議用途是保留在 `test` 作 bearish universe/filter 與後續 incremental portfolio
研究，不發布到 production。

## Raw edge 口徑釐清

前一份 bias study 的 `+12.080bp` 是 `fh_neg_rejection` 相對完整 shared pool 的
equal-day **uplift**，不是 condition 自己的絕對 raw edge。固定同一個 short
barrier-or-close label，尚未套模型、execution filter、容量或成本時：

| condition | absolute raw equal-day | absolute raw row-pooled | 與 19.3bp 比較 |
|---|---:|---:|---|
| `fh_neg_rejection` current model-input cohort | +28.887bp | +43.057bp | 兩種口徑皆高於成本 |
| 裸 `FillLots_atLow < 0` state | +18.065bp | +23.378bp | equal-day 低於、row-pooled 高於成本 |

第二列只是未改寫的 signed state，不等於正式 negFill selector，也不是可成交策略。它的
兩種答案不同，是因 row-pooled 會讓高事件量日期與 persistent same-price states 得到較大
權重；不能只用 `23.378 > 19.3` 宣稱通過成本。這也是本研究要求 fresh sampling、再以
容量與 BackTest 檢查絕對 PnL 的原因。

前一份 852,800-row bias cohort 的 absolute 值為 28.899/43.071bp；本表改用 current
852,863-row model input 的 28.887/43.057bp。63-row cohort 差異不改變成本判斷，但
避免把上一份 artifact 的數字誤稱為本 run 直接產物。

交易化後 selected 2026 gross `43.558bp`、扣 19.3bp 後 net `24.258bp`，才是本研究
最接近「是否高於成本」的答案。

## Event 與避免重複取樣

`FillLots_atHigh` 在完整 raw tick stream 上重建為同一 `RecordHigh` epoch 內的累積
signed formal flow：

```text
F_high = cumsum(FillLots × InOut × (TrialMatch == 0))
         over (QuoteCode, RecordHigh), ordered by ChannelSeq
```

`fh_neg_rejection` 要求：

1. `F_high / (D-1 avg_bidLots1 + avg_askLots1) <= -8`；
2. 最近有效成交價（`ToHigh` 的 base price）離 record high 至少 4 ticks；
3. `B1_A1B1 <= .35`、`MicroMomentum_30 <= -.10bp`、
   `LOB_NetPressure_30 < 0` 三項至少兩項成立；
4. trigger 必須是目前 high epoch 中，B1/A1 至少移動半 tick 後的整數價對 first visit，
   或 formal `FillLots >= max(20, D-1 same-side large-fill reference)`。

因此持續存在的負 `F_high` 不會讓每個 unchanged-price 小成交都重複成為樣本。12 個
deterministic audit dates 共查 194,164 events，194,164 全部為 fresh；其中 151,953
有 MD price change、50,627 達 large-fill gate（兩類可重疊），重建與 upstream
`FillLots_atHigh` 的 mismatch 為 0。

Event index 有 6,443,810 個唯一 raw fresh keys、377 sessions。套 eligible universe 後
event dataset 為 855,594 rows；label-valid model input 為 852,863 rows。2025-04-07 是
唯一零列 dataset partition，仍保留在 manifest。

Dataset 到 model input 排除 2,731 rows（0.319%）：`txf_to_1330_bp` 缺值 2,674、
`txf_residual_vol_to_1330` 缺值 58，兩者重疊 1；沒有任何 row 因
`abs(label_raw_bp) >= 10,000` 被裁掉。因此結果不是 outcome-tail trimming 產物。

## Model、threshold 與 execution contract

流程以 `time_series_research_model.ipynb` 為形狀參考，但所有契約在 study-local code
顯式凍結：

- FeatureManifest：`filllots_at_high_short_context@1`，38 個固定 event-time features，
  hash `09a29ca300424620a1cbf6331e585072d91a9a6fa2c8dd7ea4e4f1d6e6559468`；不做
  feature screening，也不含 Close、FutureHigh 或 TXF future return。
- timing model：`label_z_within_stock_day`、最多 60 prior event-active dates；selection
  model：`label_z_within_day`、最多 20 prior event-active dates。兩者皆 Ridge alpha `0.1`、
  `causal_v2` strictly-prior median preprocessing、單股 training sample cap 500萬。
- 雖然 rolling engine 可在第 2 個 event date 產生分數，本 study 額外 fail closed：滿
  60 個 prior event-active dates 前不得讓分數進 threshold 或交易。2025-04-07 是空
  partition，所以 model era 從 2025-05-13 開始；再累積 prior-threshold history 後，
  第一筆 selected trade 為 2025-05-16。
- threshold 使用 RemainSeconds × positive ToRef 的 4×4 cells，只用前 3 個 calendar
  sessions，最少 3 sessions／30 observations；timing q=`.30`、selection q=`.10`。
- 只在 2025 development 比較 `timing_only`、`selection_only`、`dual`，要求至少 60
  active days／1,000 trades，以扣 19.3bp 的 calendar-filled daily Sharpe 選型；2026
  沒有參與 gate 選擇。

Execution 是同 tick `BidPrice1` 一口 taker short。若 `FutureHigh > 1.08×RefPrice`，以
`1.08×RefPrice + 2 ticks` 回補，否則以 Close 出場；所以這是 barrier-or-close，不是
無條件 hold-to-close。其他 gate 為：

- OTC 或 `MD_L1Rate_30 > .25`、`SpreadPairElapsed > .1`；
- positive ToRef 0–5% 且在指定時間 bucket；
- D-1 `day_amount_rank <=100` 或 causal completed-slot
  `AmountRank_canDayTrade <=100`；
- price `<=1000`、prior HFT maker spread `>-70bp` 或缺值；
- 一口累積數嚴格小於 D-1 bid+ask top depth，單股當日累積名目嚴格小於 200萬；
- 固定 taker fee 19.3bp。

## BackTest 結果

2025-05-13 至年底只負責選 gate variant；2026 YTD 是選型後不改 gate 規則的
prequential evaluation。
以下 Sharpe 均用完整交易日補零；holdout 每天都有交易，因此和 BackTest 的 active-day
Sharpe 相同。

| period | calendar / active days | trades | PnL | gross bp | net bp | residual net bp | daily Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025 development | 161 / 158 | 23,806 | +2,709.57萬 | 106.390 | 87.090 | 74.132 | 12.142 | -28.54萬 |
| 2026 YTD holdout | 155 / 155 | 17,739 | +653.91萬 | 43.558 | 24.258 | 15.545 | 2.149 | -136.94萬 |
| full model era | 316 / 313 | 41,545 | +3,363.48萬 | 77.223 | 57.923 | 46.936 | 6.127 | -136.94萬 |

`residual net bp` 是扣除 prior beta × TXF component 後再扣 fee 的研究診斷，不是另一份
可直接成交的 PnL。

### Gate selection 與 model incremental value

Development model-gated Sharpe 排名為 timing-only `12.142`、dual `11.868`、
selection-only `10.921`，所以按 frozen rule 選 timing-only。Static-only 不在模型 gate
選型集合內，只作 diagnostic。

| variant | 2025 dev trades | dev net bp / Sharpe | 2026 trades | 2026 net bp / Sharpe | 2026 PnL |
|---|---:|---:|---:|---:|---:|
| static-only | 34,030 | 86.116 / 11.122 | 24,685 | 31.728 / 2.550 | +1,076.12萬 |
| timing-only（selected） | 23,806 | 87.090 / 12.142 | 17,739 | 24.258 / 2.149 | +653.91萬 |
| selection-only | 29,222 | 89.620 / 10.921 | 21,998 | 31.542 / 2.590 | +972.70萬 |
| dual | 21,562 | 89.142 / 11.868 | 16,437 | 25.139 / 2.195 | +636.42萬 |

Timing gate 的 development Sharpe 略升，但 2026 的 PnL、net bp 與 Sharpe 都低於
static-only；selection-only 也優於 timing-only。不能把 selected curve 解讀為 model
增加了 OOS alpha。

### 2026 月度與成本壓力

2025 model-era 8/8 月為正；2026 YTD 5/8 月為正，負月份為 1、2、5 月。

| month | trades | PnL | net bp |
|---|---:|---:|---:|
| 2026-01 | 3,606 | -53.70萬 | -12.591 |
| 2026-02 | 1,194 | -5.65萬 | -2.996 |
| 2026-03 | 2,917 | +224.69萬 | +58.666 |
| 2026-04 | 2,342 | +174.99萬 | +42.827 |
| 2026-05 | 2,491 | -41.92萬 | -9.943 |
| 2026-06 | 2,214 | +138.63萬 | +38.734 |
| 2026-07 | 1,615 | +89.19萬 | +33.232 |
| 2026-08 至 25 日 | 1,360 | +127.68萬 | +52.980 |

| assumed fee | 2026 net bp | PnL | daily Sharpe |
|---:|---:|---:|---:|
| 0bp | 43.558 | +1,174.16萬 | 3.802 |
| 10bp | 33.558 | +904.60萬 | 2.952 |
| 19.3bp | 24.258 | +653.91萬 | 2.149 |
| 25bp | 18.558 | +500.26萬 | 1.651 |

## negFill overlap 與獨立性

Reference 是 read-only 的廣義 `src/research/timeSeries/data_index.parquet` event index，
不是正式 negFill 最終成交 signals；本節只能回答事件域重疊，不能直接等同 portfolio
correlation。

- Exact tick overlap：558 / 41,545 = `1.343%`。
- Stock-day overlap：6,700 / 6,828 = `98.125%`。

| segment | full trades | full net bp / Sharpe | 2026 trades | 2026 net bp / Sharpe |
|---|---:|---:|---:|---:|
| all selected | 41,545 | 57.923 / 6.127 | 17,739 | 24.258 / 2.149 |
| exact-tick disjoint | 40,987 | 56.609 / 6.066 | 17,520 | 22.898 / 2.078 |
| stock-day disjoint | 658 | -87.580 / -2.971 | 208 | -187.003 / -3.895 |

所以低 exact-tick overlap 的結論成立；「獨立 stock-day book」則尚未建立。這份
negFill reference 很廣，stock-day disjoint 只剩 full 658／holdout 208 trades，切片本身
也有嚴重 composition confounding。觀察到的正績效幾乎都活在也曾出現 negFill event 的
股票日中，但這不等於已證明兩個最終交易帳簿相關或互相依賴；仍需拿正式 negFill
executed ledger 做 marginal PnL、covariance 與共同容量回測。

## 圖與 position 語意

- [完整 model era BackTest 圖](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/backtest_full.png)
- [2026 YTD holdout BackTest 圖](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/backtest_2026_holdout.png)

圖由專案 `BackTest` 的原始兩 panel `_plot` 產生。下 panel 與報表中的平均／最大部位
是每天所有一口 signals 的名目金額加總；它不是逐時點的 concurrent peak exposure。
Full/holdout 平均每日加總名目分別為 1,855.22萬／1,739.09萬，最大均為 4,798.89萬。

## 驗證、lineage 與限制

自動 verification 20/20 checks 通過：event/dataset/model key 唯一性、固定 38 features、
prior-only thresholds、60-event-date warmup、static/cap gates、development-only selection、
逐筆 PnL 對帳、25bp cost、sampling、overlap、407 個 artifact hashes、12 個 implementation/
environment source hashes。Study unit
tests 4/4、相關 framework regression tests 47/47 通過。

- Study spec SHA-256：
  `8c353ab3296bf17477e3a38e65e28ec6b43c50a30fbdf9aae8a97596251da02e`。
- Label：`negfill.short_taker_stop108.v1`，hash
  `bc3c3adef1cb3dc7b4e43d27c203ca8b077c56c6f1f0a7882c18fcd19639a3cd`。
- Dataset：377 partitions、`schema_policy=union`、tickBar
  `causal_completed_slot`、strict cardinality。
- 2026 只是不參與 gate variant 選擇的 prequential evaluation。`fh_neg_rejection` 本身曾由
  2025–2026 retrospective bias study promotion，threshold q/模型形狀也承襲既有
  notebook，因此不是 condition discovery 或所有 hyperparameters 的 pristine OOS。
- 2026 中 rolling models 會用 strictly-prior event dates 進行 causal online refit；不是
  把 2025 年底的一組係數永久凍結。
- 19.3bp 只涵蓋固定 taker fee 假設；沒有 market impact、borrow availability/cost、
  next-tick latency、fill uncertainty 或盤中 portfolio concurrency simulation。
- Full/2026 分別有 8.16%／12.07% rows 觸發 stop108，但 BackTest 只根據 FutureHigh
  existence 套固定 `1.08×Ref+2 ticks` exit，沒有 first-hit gap fill、stop 後 lockout、
  部位釋放／重進場或 account-level gross cap。容量是單股當日累積且不釋放；這是
  idealized independent one-lot label aggregation，不是 stateful execution simulator。
- Entry 使用 event tick 的 Bid，且 label 使用 FutureHigh/Close 判定 exit；outcome 欄位未
  進入 features、threshold、execution gate。
- 前一份 shared-pool bias 的 same-stock 與 same-stock×5m attribution 反號，本次 broad
  stock-day disjoint 也失敗；這與 composition/universe effect 的解讀一致。
- 沒有修改 `build_index.py`、`data_loader.py`、正式 `data_index.parquet` 或 production
  modelParams。
- Raw source manifests 的 identity 契約是 path/size/mtime_ns，不是每個 raw input 的
  content SHA；selector、framework/model/BackTest source、uv lock 與所有 study-local
  event/dataset outputs 則另有 content hashes。

## Artifacts

- [frozen study spec](../../../test/20260826_filllots_at_high_signal/study_spec.json)
- [study code](../../../test/20260826_filllots_at_high_signal/study.py) ·
  [dataset builder](../../../test/20260826_filllots_at_high_signal/build_dataset.py) ·
  [model/backtest runner](../../../test/20260826_filllots_at_high_signal/run_model_backtest.py) ·
  [verification code](../../../test/20260826_filllots_at_high_signal/verify_run.py)
- [run manifest](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/run_manifest.json) ·
  [selection decision](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/selection_decision.json) ·
  [variant summary](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/variant_period_summary.parquet) ·
  [fee sensitivity](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/fee_sensitivity.parquet) ·
  [monthly summary](../../../test/20260826_filllots_at_high_signal/artifacts/backtest_v1/monthly_summary.parquet)
- [sampling audit](../../../test/20260826_filllots_at_high_signal/artifacts/sampling_audit_v1/audit.json) ·
  [negFill overlap](../../../test/20260826_filllots_at_high_signal/artifacts/negfill_overlap_v1/overlap.json) ·
  [verification result](../../../test/20260826_filllots_at_high_signal/artifacts/verification_v1/verification.json)

## Reproduce

所有 Python 執行使用 `uv run`。完整重建需要本機 raw data；`--force` 會重建
study-local artifacts，不觸碰 production index。

```bash
UV_CACHE_DIR=/tmp/hft_uv_cache uv run python \
  src/research/timeSeries/test/20260826_filllots_at_high_signal/build_dataset.py \
  --stage all --force

UV_CACHE_DIR=/tmp/hft_uv_cache uv run python \
  src/research/timeSeries/test/20260826_filllots_at_high_signal/audit_sampling.py

UV_CACHE_DIR=/tmp/hft_uv_cache MPLCONFIGDIR=/tmp/hft_mpl_config uv run python \
  src/research/timeSeries/test/20260826_filllots_at_high_signal/run_model_backtest.py \
  --n-jobs 8 --force-input --force-model --force-thresholds

UV_CACHE_DIR=/tmp/hft_uv_cache uv run python \
  src/research/timeSeries/test/20260826_filllots_at_high_signal/audit_negfill_overlap.py

UV_CACHE_DIR=/tmp/hft_uv_cache MPLCONFIGDIR=/tmp/hft_mpl_config uv run python \
  src/research/timeSeries/test/20260826_filllots_at_high_signal/run_model_backtest.py \
  --n-jobs 8 --manifest-only

UV_CACHE_DIR=/tmp/hft_uv_cache uv run python \
  src/research/timeSeries/test/20260826_filllots_at_high_signal/verify_run.py
```
