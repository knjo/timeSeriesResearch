# `fh_neg_rejection` static BackTest snapshot

Status: `RESEARCH_SNAPSHOT_ONLY`  
Classification: `scoped companion`  
Parent decision: `RESEARCH_SHADOW_ONLY`  
Direction: short  
Evidence: outcome-informed retrospective diagnostic  
Window: `20250211–20260825`

後續完整交易日的獨立版本延伸見 [EXTENSION_20260917.md](EXTENSION_20260917.md)（前版 [EXTENSION_20260916.md](EXTENSION_20260916.md)、[EXTENSION_20260914.md](EXTENSION_20260914.md)）；
本凍結快照與下列歷史結論不改寫。

## 結論

已把 2026-08-27 robustness study 的 conservative `D+G+B` anchor 做成可重現、
有圖且保留逐筆 accepted ledger 的 static BackTest 快照。主線完全不使用 rolling model
score 或 threshold；每個通過 execution/capacity 的 event 放空一張，成本固定
`19.3bp`。

完整歷史窗共接受 71,657 筆，成本後 PnL `+5,815.09萬`、capital-weighted net
`+67.533bp`、calendar-filled daily Sharpe `6.854`、MDD `-161.59萬`。但 2026
明顯較弱：net 降至 `+32.086bp`、Sharpe `2.588`、日勝率 `49.68%`，而
2026-01、2026-05 是 19 個月中唯二虧損月。

這證明的是固定規則在理想化歷史 replay 下有正 PnL，不是 prospective OOS，也沒有
處理組合總資金、同時曝險、borrow、impact 或 latency。結果不改變 parent
`RESEARCH_SHADOW_ONLY`，更不構成 deployment handoff。

![Full retrospective BackTest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/backtest_full.png)

## 期間結果

表內 PnL 與 MDD 單位為新台幣萬元；Sharpe 對 requested calendar 補零後計算。

| Period | Sessions | Trades | Stock-days | Gross BP | Net BP | Net PnL | Daily Sharpe | Daily win | MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full retrospective `20250211–20260825` | 377 | 71,657 | 8,751 | 86.833 | 67.533 | +5,815.09 | 6.854 | 69.23% | -161.59 |
| 2025 `20250211–20251231` | 222 | 47,087 | 5,353 | 109.704 | 90.404 | +4,731.53 | 11.921 | 82.88% | -40.15 |
| 2026 historical `20260102–20260825` | 155 | 24,570 | 3,398 | 51.386 | 32.086 | +1,083.56 | 2.588 | 49.68% | -158.29 |
| Old model-era comparable `20250513–20260825` | 316 | 58,518 | 7,366 | 80.479 | 61.179 | +4,464.36 | 5.984 | 67.72% | -161.59 |

`full_retrospective` 是新的 no-model-gate execution contract。舊報告的
`static_only` 雖然沒有 score threshold，仍隱含兩個 rolling predictions 有效以及
60-event-day warmup，從 `20250513` 才開始。因此只有最後一列能和舊 static aggregate
直接比較；不能把前 61 個 requested sessions 稱為舊回測的延長版 OOS。

2026 淨值圖清楚顯示年初曾落到負值，之後以階梯狀方式回升；不能只看期末正 PnL
忽略中間長平台與回撤。

![2026 historical BackTest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/backtest_2026_historical.png)

## 每天、每檔怎麼配置

固定 sizing 是：每個 accepted event 一張，不依 score 放大。

1. 先依 `Date, QuoteCode, TransTime, ChannelSeq` 排序。
2. 同檔當日累積張數必須小於 D-1 `avg_bidLots1 + avg_askLots1`。
3. 同檔當日累積進場名目必須小於 `200萬`。
4. 沒有跨檔 portfolio cap，也沒有把同檔多筆 event 淨額化；所以圖下半部是當日所有
   一張訊號的「名目加總」，不是任一時刻真正 concurrent gross exposure。

完整窗每日名目加總平均 `2,284.02萬`、中位數 `2,200.73萬`、P90
`3,559.31萬`、最大 `6,529.38萬`；每日平均 190.1 筆，最大 650 筆。這些數字回答
目前 replay 怎麼配，但還不是可部署的 account sizing。

## Cohort 與 label parity

不能直接拿 2026-08-26 舊 static aggregate 畫圖，因為它的 row ledger 沒有保存，而且
舊圖其實是 development-selected `timing_only`。本 snapshot 從頭重建 static ledger：

```text
conservative_dgb pre-label membership                 6,422,686
old model_input ∩ conservative membership               850,439
套 current rounded Bid/Ref strict boundary 後             850,376
static execution gate 前                                 102,408
full-window capacity 後                                    71,657
20250513+ comparable capacity 後                           58,518
```

63 筆差異來自 robustness pool 的
`round((BidPrice1-RefPrice)/RefPrice, 6)` strict boundary。capacity 必須在移除後整批
重算，不能只從舊 accepted signals 刪 row，因為被釋放的同檔容量會讓後續合法事件補進。

另一個更隱蔽的差異是：舊 model input 使用 tickFeature `TickSize`，目前 robustness pool
依 `BidPrice1` 重算合法 tick。兩者 label profile hash 相同，但在跨 tick 級距附近的
兩-tick adverse fill 數值不一定相同。新 runner 重算 exit/raw/residual/z；377 日
`condition_n_z` 與 parent 的 raw/residual/z/z² sums 全部相符，最大絕對誤差低於
`1e-6`。

關鍵 canonical key hash：

```text
labeled anchor     eff1e68b5017e82a2dc044262e8747bb5513012b4239d8528eb43616a58fc993
full accepted      09d4d9b75c5669d752c7541c06467479e26dfb4bc6eae9d78577fbc37f264ce1
20250513+ accepted d81e26ebb2457b27d20d3e1317dfefcd134068899a5489d83cc0f13d947015d6
```

## Execution gate

anchor 後固定套用既有 static execution gate：

- inherited `RemainSeconds` 與正 `ToRef` buckets 非空；
- `OTC` 或 `MD_L1Rate_30_re > .25`；
- `SpreadPairElapsed > .1`；
- `AmountRank_canDayTrade <= 100` 或 D-1 `day_amount_rank <= 100`；
- `BidPrice1 <= 1000`；
- D-1 `hft_strick_makerSpreadBP` 為 null 或 `> -70`。

這次只有一個固定 static variant，沒有用本次 PnL 選 gate 或 threshold，因此沒有新增
winner's curse trial。但這組 execution gate 已在先前研究看過 outcome，也沒有像 selector
的 D/G/B 一樣完成完整 local-neighborhood sensitivity；它仍是 outcome-informed 歷史
snapshot。若下一步要把 execution gate 當部署策略，就要另凍結其 sensitivity 或 rolling
first-live contract，不能拿 selector plateau 代替。

## 成本敏感度

下表只改每筆線性成本，signals 與容量完全固定。PnL/MDD 單位為萬元。

| Cost | Net BP | Net PnL | Daily Sharpe | MDD |
|---:|---:|---:|---:|---:|
| 0bp | 86.833 | +7,476.97 | 8.569 | -132.56 |
| 10bp | 76.833 | +6,615.89 | 7.694 | -145.89 |
| 19.3bp | 67.533 | +5,815.09 | 6.854 | -161.59 |
| 25bp | 61.833 | +5,324.28 | 6.326 | -184.19 |
| 35bp | 51.833 | +4,463.21 | 5.376 | -243.36 |

這不是 impact curve；固定每筆扣更多 BP 仍未模擬 signals 同時搶量、排隊、滑價尾部或
borrow fail。

## Overfit 與可解讀邊界

- 本 snapshot 沒有新增 grid、ranking 或 winner selection；`new_variants_compared=1`。
- 它完整繼承 `fh_neg_rejection` 過去 condition discovery 的歷史 selection exposure。
- 2025/2026 level 差異仍大：net BP `90.404 → 32.086`，與 parent bias decay 同方向。
- 19 個月中 17 個正、2 個負，不能當成 prospective monotonicity 或參數預測力測試。
- parent 的 27-cell plateau 支持 selector 周圍不尖銳，不代表本 execution gate、成本或
  account sizing 已通過 robustness。

因此本結果適合保存為模型前的 fixed static benchmark；之後模型是否有 lift，應相對
同一 outer first-live static cohort 比，而不是只和零交易比較。

## Materialization、驗證與 artifacts

新 runner 若偵測到既有 `run_manifest.json` 會拒絕覆寫；要重物化應建立新 artifact
version，保留這份 snapshot。下方第一個指令是本次初次 materialization 的完整命令；
目前對同一 v1 再執行會依設計 fail closed。第二個指令仍可直接驗證新加入的純函數契約。

```bash
MPLCONFIGDIR=/tmp/matplotlib-hft UV_CACHE_DIR=/tmp/uv-cache-hft \
uv run python \
  src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness/run_static_backtest.py

MPLCONFIGDIR=/tmp/matplotlib-hft UV_CACHE_DIR=/tmp/uv-cache-hft \
uv run python -m unittest \
  src/research/intradaySignal/experiments/20260827_fh_neg_rejection_robustness/test_static_backtest.py
```

- [snapshot contract](../../../test/20260827_fh_neg_rejection_robustness/static_backtest_contract.json)
- [machine decision](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/snapshot_decision.json)
- [verification](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/verification.json)
- [run manifest](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/run_manifest.json)
- [period summary](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/period_summary.parquet)
- [fee sensitivity](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/fee_sensitivity.parquet)
- [accepted signal ledger](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/accepted_signals.parquet)

`verification.json` 為全數通過；run manifest SHA-256：
`618842f2c3af9c311406eee34dbc365a0ae913328f25c0d79df55de8e4fab382`。
整個 robustness study 現為 93/93 tests 通過，其中 static snapshot 新增 4 項合成測試。
