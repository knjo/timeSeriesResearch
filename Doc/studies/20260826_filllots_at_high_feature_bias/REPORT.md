# FillLots_atHigh 搭配 tickFeature 的收盤方向 bias

狀態：`scoped`  
研究日期：2026-08-26  
資料區間：2025-02-11 至 2026-08-25  
Primary estimand：direction-specific barrier-or-close label 的 shared-pool equal-day `Δz`

## 結論

在避免相同價格小樣本重複取樣後，`FillLots_atHigh` 確實存在兩個穩定但性質不同的
方向分布：

- 空方 `fh_neg_rejection`：高點 epoch 內累積賣超已達前一日 top depth 的 8 倍、
  現價離高點至少 4 ticks，且 3 個 bearish tickFeatures 至少 2 個成立。377-session
  dense 結果為 shared `Δz=+0.110343`，HAC(5) `t=7.889`、`p=3.06e-15`、
  95% CI `[+0.082927,+0.137759]`，19/19 月為正。raw／residual equal-day uplift
  分別為 `+12.080bp (p=3.16e-6)`、`+18.010bp (p=1.50e-12)`。這是一個明確的
  risk-adjusted 利空分布。
- 多方 `fh_pos_breakout`：累積買超至少為前一日 top depth 的 2 倍、仍在高點下方
  1.5 ticks 內，且 3 個 bullish tickFeatures 至少 2 個成立。dense shared
  `Δz=+0.084155`，HAC(5) `t=5.498`、`p=3.83e-8`、95% CI
  `[+0.054157,+0.114153]`，18/19 月為正。可是 full-period raw uplift 是
  `-1.946bp (p=.6223)`，只有 residual uplift 為 `+7.842bp (p=.0372)`；因此它是
  風險調整後的 bullish bias，尚不是股票本身抱至出場就有正報酬的多單規則。

兩個候選都通過事前凍結的 dense acceptance，2025、2026 也同號；排除 sparse
篩選日後的 301-session 內部 calendar complement 仍重現。但 same-stock 與
same-stock-by-5-minute 的 `Δz` 全部大幅反號，且候選所在 stock-day 與既有 negFill
高度重疊。最可靠的解讀是：這是穩定的股票／股票日／時段 composition bias，
不是已識別的單股局部進場 timing alpha，也還不是可直接上線的交易策略。

## 研究問題與 Stage 0

本研究在既有 discovery protocol 下回答：當下可觀察的 `FillLots_atHigh` 狀態，
搭配因果 tickFeature，是否能把 long／short barrier-or-close label 的分布穩定地推向
有利方向。Primary contrast 是每個 active day：

```text
mean(label_z | condition) - mean(label_z | complete eligible shared pool)
```

再對日期等權，以 Bartlett Newey-West HAC lag 5 推論；`Δz>0` 表示對該方向 label
較有利。same-stock 與 same-stock × 5m 是來源歸因，不是這個 long-holding objective
的 primary estimand。

事前要求的 Stage 0 positive control 已先完成：未改寫的 `FillLots_atLow < 0` 在 short
label 找回 `Δz=+0.023279`、HAC(5) `p=.01923`。詳見
[raw-state recovery report](../20260826_filllots_at_low_state_bias/REPORT.md)。

## Condition 與去重取樣

`FillLots_atHigh` 不是「只計算成交價剛好等於最高價」；它是同一個 RecordHigh epoch
內的累積 signed flow：

```text
F_high = cumsum(FillLots × InOut × (TrialMatch == 0))
         over (QuoteCode, RecordHigh), ordered by ChannelSeq

R = F_high / (D-1 avg_bidLots1 + D-1 avg_askLots1)
gap_ticks = ToHigh × RefPrice / TickSize
```

`F_high`、B1/A1 變化與 first-visit 都先在完整 raw tick stream 上建立，再與
tickFeature 及 eligible label pool 交集。每個候選必須滿足以下 fresh trigger 的聯集：

1. `md_first_visit`：B1 或 A1 至少移動半個現行 tick，而且整數 tick lattice 上的
   `(B1, A1)` pair 是目前 RecordHigh epoch 第一次出現；回到曾見過的同一價對不再取樣。
2. `large_fill`：market-open formal trade，且同 tick `FillLots >= max(20,
   D-1 same-side big-lot q95)`。這允許同價位真的出現一筆有經濟量級的新成交時再次取樣。

D-1 big-lot threshold 缺失時 fail closed。單元測試也直接驗證：沒有 MD first-visit、
又未達 large-fill threshold 的相同價格小成交，不會進入任何候選。因此 persistent
`F_high` 狀態本身不會讓每個 unchanged-price quote 都被重複當成新樣本。

## 非純值 tickFeature interaction

本次沒有只 sweep `F_high` 純值，而是使用以下當下可觀察狀態：

- depth normalization：`R` 使用 D-1 top-of-book 平均掛單量；
- high location：`near <= 1.5 ticks`、`far >= 4 ticks`；
- bullish score：`B1_A1B1 >= .65`、`MicroMomentum_30 >= +.10bp`、
  `LOB_NetPressure_30 > 0`，至少 2/3；
- bearish score：`B1_A1B1 <= .35`、`MicroMomentum_30 <= -.10bp`、
  `LOB_NetPressure_30 < 0`，至少 2/3。

12 個候選及門檻在 full sparse outcomes 之前凍結：

| 類別 | candidates | 定義摘要 |
|---|---|---|
| sign baseline | `fh_pos_sample`, `fh_neg_sample` | fresh trigger 且 `F_high` 正／負 |
| depth | `fh_pos_depth2`, `fh_neg_depth8` | fresh trigger 且 `R>=2`／`R<=-8` |
| high location | `fh_pos_near`, `fh_pos_far`, `fh_neg_near`, `fh_neg_far` | sign 加 near／far |
| interaction | `fh_pos_breakout`, `fh_pos_exhaustion` | positive depth + near + bull／bear score |
| interaction | `fh_neg_rejection`, `fh_neg_absorption` | negative depth + far + bear，或 negative depth + near + bull |

Close、FutureHigh/FutureLow、future profit、`SpreadNarrowOrderTime` 與
`SpreadNarrowSide` 都未用於 condition。

## Sparse screen 與 promotion

Sparse screen 使用固定 session step 5，共 76 sessions；每個方向對全部 12 候選做
within-direction BH。Promotion gate 要求 coverage `>=.8`、BH `q<=.1`、full effect
為正、至少 20 events/active day、正月份比例 `>=.65`，且 2025／2026 effect 都為正；
每個方向最多 promotion 1 個，以最低 q、再以最大 `Δz` 排名。

| label | candidate | full Δz | HAC p | BH q | 24-test joint Holm p | 正月份 | 2025 / 2026 Δz | promotion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| short | `fh_neg_sample` | +0.022096 | .06837 | .10255 | .82042 | 12/19 | +.038372 / -.001530 | fail |
| short | `fh_neg_depth8` | +0.086294 | .000721 | .00432 | .01658 | 17/19 | +.101844 / +.063722 | pass，未入選 |
| short | `fh_neg_rejection` | +0.122311 | 6.41e-6 | 7.70e-5 | .000154 | 18/19 | +.146676 / +.086942 | promoted |
| long | `fh_pos_sample` | +0.027581 | .07741 | .13365 | .85155 | 11/19 | +.019799 / +.038879 | fail |
| long | `fh_pos_near` | +0.038715 | .00924 | .03695 | .16626 | 12/19 | +.035558 / +.043297 | fail：正月比不足 |
| long | `fh_pos_breakout` | +0.078228 | .00786 | .03695 | .14931 | 15/19 | +.082353 / +.072241 | promoted |

這個比較顯示 interaction 有實質增量：純 sign baseline 弱且未過 gate；空方先由
depth normalization 放大，再由 far-from-high + bearish book state 集中；多方則要
同時 near-high 與 bullish book state 才穩定。Sparse 中 `fh_neg_rejection` 的 long
effect 為 `-0.084426`，`fh_pos_breakout` 的 short effect 為 `-0.074605`，方向鏡像也
一致。不過 long breakout 的 sparse 24-test joint Holm `p=.14931` 未過 5%，後續
dense 結果應視為凍結 promotion 流程內的確認，而非忽略 discovery multiplicity。

## Dense 主結果

Dense check 只重跑每方向 promotion 的一個候選。377 requested sessions 中 376 active；
2025-04-07 的 eligible pool 為 0，保留為零樣本 lineage 列。

| label / candidate | event rows | events / active day | 平均 selection rate | shared Δz (HAC 95% CI) | HAC p | 正 days | 正 months | raw uplift | residual uplift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short / `fh_neg_rejection` | 852,800 | 2,268.1 | .09273% | +.110343 `[+.082927,+.137759]` | 3.06e-15 | 253/376 | 19/19 | +12.080bp | +18.010bp |
| long / `fh_pos_breakout` | 143,383 | 381.3 | .01593% | +.084155 `[+.054157,+.114153]` | 3.83e-8 | 246/376 | 18/19 | -1.946bp | +7.842bp |

HAC lag 0/1/3/5/10 全部同號；short 的最大 p 為 `5.30e-13`，long 的最大 p 為
`3.83e-8`。因此主結論不是 lag 5 的偶然選擇。

### 跨年與 BP 分解

| label | period | shared Δz | HAC p | 正月份 | raw uplift (p) | residual uplift (p) |
|---|---|---:|---:|---:|---:|---:|
| short rejection | 2025 | +.147725 | 4.04e-15 | 11/11 | +19.142bp (5.04e-11) | +24.254bp (6.60e-16) |
| short rejection | 2026 | +.057044 | .000151 | 8/8 | +2.011bp (.5927) | +9.107bp (.01362) |
| long breakout | 2025 | +.060201 | .00117 | 10/11 | -7.468bp (.08485) | -1.283bp (.7511) |
| long breakout | 2026 | +.118309 | 1.41e-6 | 8/8 | +5.927bp (.3993) | +20.852bp (.000799) |

空方的 z、raw、residual 在 full period 一致，但 2026 raw effect 已弱化。多方的 z
跨年穩定，raw BP 卻由 2025 負轉 2026 正，且兩年 raw p 都不顯著；不能把
`fh_pos_breakout` 寫成 outright-price bullish alpha。

### Daily bias distribution

以下是 active-day 的 condition-minus-shared-pool `Δz` 分布，不是把數十萬 event rows
當成獨立交易：

| label / candidate | q05 | q25 | median | q75 | q95 | positive-day ratio |
|---|---:|---:|---:|---:|---:|---:|
| short / `fh_neg_rejection` | -.213692 | -.049656 | +.100973 | +.234850 | +.474808 | 67.29% |
| long / `fh_pos_breakout` | -.319258 | -.076598 | +.058541 | +.229795 | +.542623 | 65.43% |

兩者是明顯的分布平移，但不是每日必勝；各自仍約有三分之一 active days 的 `Δz<=0`。

## Calendar complement robustness

為檢查結果是否只由 76 個 sparse 篩選日驅動，另取 dense calendar 中不屬於
`dense_sessions[::5]` 的 301 sessions；2025-04-07 無 pool，所以 300 active。

| label / candidate | complement Δz | HAC p | 2025 / 2026 Δz | 正月份 | raw / residual uplift |
|---|---:|---:|---:|---:|---:|
| short / `fh_neg_rejection` | +.107311 | 2.85e-13 | +.147993 / +.049570 | 19/19 | +11.377 / +17.764bp |
| long / `fh_pos_breakout` | +.085657 | 3.42e-7 | +.054538 / +.129825 | 19/19 | -3.763 / +6.229bp |

這證明 dense 結果不是 sparse 日期的機械重述；但 complement 仍位於同一段 retrospective
calendar，沒有未來資料，也不是另行事前登記的 acceptance test，所以只標為 internal
robustness，不稱為 future OOS。Complement 的 long raw／residual p 分別為 `.404`、
`.146`，所以這裡的 `directionally stable` 只適用 primary shared-pool z。

## Shared 與 matched 歸因

| label / candidate | shared Δz | same-stock Δz (HAC t) | same-stock × 5m Δz (HAC t) |
|---|---:|---:|---:|
| short / `fh_neg_rejection` | +.110343 | -.229948 (-37.33) | -.055897 (-69.62) |
| long / `fh_pos_breakout` | +.084155 | -.271915 (-37.12) | -.071146 (-35.44) |

兩個 matched estimands 都穩定反號，2025、2026 亦同。也就是 condition 挑到的
股票／股票日／時段相對完整 shared pool 有利，但進入相同股票或相同股票 × 5m cell
後，condition rows 反而較差。Shared-pool bias 對目前 objective 仍是真實 primary
distribution shift；matched reversal 則否定「同一檔股票看到 condition 就有較佳進場
時機」的因果／交易化解讀。

## 與 negFill / FillLots_atLow 的重疊

在固定每 7 個 sparse sessions 取一日的 12-session raw-key audit 中，以 pinned
`negfill_spread_bid_move_v1` 為 reference：

| candidate | candidate ticks 與 negFill 重疊 | tick Jaccard | candidate stock-days 含 negFill | stock-day Jaccard |
|---|---:|---:|---:|---:|
| `fh_neg_rejection` | 32.486% | 2.827% | 98.928% | 29.243% |
| `fh_pos_breakout` | .882% | .0185% | 90.169% | 24.327% |

兩者通常不是同一 tick，尤其 breakout 幾乎完全不同；但幾乎都發生在也有 negFill 的
同一批 stock-days。76 個 sparse dates 的 daily shared effect correlation 方面，short
rejection 對 `fh_neg_sample` 為 `.582`、對 `FillLots_atLow < 0` 為 `.699`；long
breakout 對 `fh_pos_sample` 為 `.647`、對 `FillLots_atLow < 0` 僅 `.254`。

這個 overlap 是 raw selector-key audit，沒有套正式 shared eligible-label pool；其 tick
counts 不能和 dense event rows 直接比較。Long dense condition 的 pooled z 本身為負，
正 `Δz` 表示它比更負的 shared control 有利，不表示 condition 的 absolute z 大於零。

因此 short rejection 很可能是既有 bearish stock-day domain 的更精確切片，而非獨立
第二本書；long breakout 的日級 effect 較不同，但仍受相同 stock-day composition
支配。若要交易化，應先當 universe／feature filter，再做 disjoint stock-day、容量與
incremental portfolio 測試，不能只用低 tick overlap 宣稱分散。

## 「至收盤」label 的實際邊界

本 study 沿用目前凍結的 direction-specific barrier-or-close labels，不是忽略途中
barrier 的 unconditional pure-close return：

- short：同 tick Bid taker 進場；若 `FutureHigh > 1.08 × RefPrice`，以
  `1.08 × RefPrice + 2 ticks` 回補，否則以 13:30 Close 出場；
- long：同 tick Ask taker 進場；若 `FutureHigh > 1.09 × RefPrice`，以 1.09 barrier
  出場，否則以 13:30 Close 出場；
- label 再扣 prior beta × event-to-close TXF component，並用 prior residual volatility
  標準化成 `z`。

Long／short labels 與 valid pools 獨立，故本研究兩邊都實際計算，沒有以一邊負號代替
另一邊。若研究問題必須是「無條件持有到 13:30、途中完全不出場」，應另建 versioned
pure-close labels；不能把本報告的結果直接改名。

## Lineage 與資料完整性

- Candidate registry：`filllots-at-high-feature-v1`；family
  `filllots_at_high_feature/v1`。
- Frozen study spec SHA-256：
  `feff1e289ef73fdd9c1b8c1c5d474baa69f5d76ceeb79e4a08d140ce45b60641`。
- Profile source SHA-256：
  `6565b129961536a2b6414d8daf2be2db31bc65fcdbed1f8bf39fbd5ff72ab600`。
- Pool：`tw_equity_daytrade_rank150_090030_123000_bidref_v1`；short control
  944,359,617 rows，long control 944,352,126 rows。
- Short label：`negfill.short_taker_stop108.v1`，hash
  `bc3c3adef1cb3dc7b4e43d27c203ca8b077c56c6f1f0a7882c18fcd19639a3cd`。
- Long label：`benchmark.long_taker_tp109.v1`，hash
  `a7d70c7eac9e3ac55bdb963b7aea2f335e145afe422bcca08cff713e9f79b3c8`。
- Sparse manifests 為 76 sessions；dense manifests 為 377 sessions；四份
  `failures=[]`，current report lineage 已驗證 exact candidate registry、label/pool
  fingerprint 與四種 daily source identity。

Source identity 契約使用 path、size、mtime，不是 raw-file content SHA；上列 study/profile
SHA 則是研究規格與 selector source 的內容 fingerprint。

Dense 執行途中，2026-08-24 preMarket source 被外部程序改寫，檔案大小由 1,081,981
變成 1,141,161 bytes。Runner 的 source-identity guard 在 375/377 時 fail closed；
舊的 2026-08-24 sparse/dense daily caches 與舊報表已移至 study-local
`quarantine/source_drift_20260824/`，再以目前來源只重建該日並重驗全部 manifests。
Promotion 沒有改變，current overlap audit 也已包含重建後的 2026-08-24。這些 quarantine
檔可回復，沒有靜默覆寫來源版本。

## 限制與下一步

- 這是 retrospective discovery + dense validation；dense full 包含用來 promotion 的
  76 個 sparse dates，沒有 future OOS。Long sparse promotion 未過 24-test joint
  Holm，尤其需要新日期確認。2026 切片只到 2026-08-25，是 YTD，不是完整年度。
- Shared pool 包含 condition rows，是 unconditional distribution contrast，不是
  disjoint causal control；condition 的稀疏度會讓 self-inclusion 影響很小，但不為零。
- 多個 event horizon 高度重疊。推論已提升至 equal-day HAC，但未做 stock-day two-way
  cluster；數十萬 rows 不是獨立樣本。
- 沒有 next-tick latency、queue/fill、cooldown、同股持倉合併、fees、capacity、
  portfolio risk 或 incremental Sharpe。Condition rows 不是已成交訂單。
- Matched reversal 與 stock-day overlap 表示兩者不能直接當 negFill-independent timing
  book；應先做 disjoint stock-day incremental test。
- `fh_pos_breakout` 的 full raw BP 為負，需在 pure-close label 與可成交 long model 下
  另驗，不能由正 `Δz` 推論 outright long PnL。

建議下一個 version：固定這兩個 selector，不再調門檻；用新增未見日期與 pure-close
long/short labels 做 OOS，並加入 next-tick、每股 cooldown／單一持倉與 disjoint
stock-day attribution。Short rejection 可先作 bearish universe/filter；long breakout
只先作 model feature，不直接發布交易規則。

## Artifacts

- [frozen study_spec.json](../../../test/20260826_filllots_at_high_feature_bias/study_spec.json)
- [profile implementation](../../../framework/profiles/filllots_at_high.py)
- Sparse：[report.json](../../../test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/report.json)、
  [period_summary.parquet](../../../test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/period_summary.parquet)、
  [promotion.json](../../../test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/promotion.json)
- Dense：[report.json](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_report_v1/report.json)、
  [decision.json](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_report_v1/decision.json)、
  [daily_effects.parquet](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_report_v1/daily_effects.parquet)、
  [distribution_stats.parquet](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_report_v1/distribution_stats.parquet)
- [calendar complement decision](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_complement_report_v1/complement_decision.json)
- [overlap audit](../../../test/20260826_filllots_at_high_feature_bias/artifacts/overlap_audit_v1/overlap_report.json)
- Dense short [manifest](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_short_neg_rejection_v1/run_manifest.json)／
  dense long [manifest](../../../test/20260826_filllots_at_high_feature_bias/artifacts/dense_long_pos_breakout_v1/run_manifest.json)

## Reproduce

```bash
# 1. Frozen 12-candidate sparse screen
uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/run_sweep.py \
  --start 20250211 --end 20260825 --step 5 \
  --label-profile short-taker-stop108-v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_short_v1

uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/run_sweep.py \
  --start 20250211 --end 20260825 --step 5 \
  --label-profile long-taker-tp109-v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_long_v1

uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/report_sweep.py \
  --short-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_short_v1 \
  --long-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_long_v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1

uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/select_promotions.py \
  --period-summary src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/period_summary.parquet \
  --output src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/promotion.json

# 2. Only the preregistered promoted candidate in each direction
uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/run_sweep.py \
  --start 20250211 --end 20260825 --step 1 --candidate fh_neg_rejection \
  --label-profile short-taker-stop108-v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_short_neg_rejection_v1

uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/run_sweep.py \
  --start 20250211 --end 20260825 --step 1 --candidate fh_pos_breakout \
  --label-profile long-taker-tp109-v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_long_pos_breakout_v1

uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/report_dense.py \
  --promotion-json src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/promotion.json \
  --short-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_short_neg_rejection_v1 \
  --long-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_long_pos_breakout_v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_report_v1

# 3. Internal robustness and overlap attribution
uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/report_dense_complement.py \
  --promotion-json src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/promotion.json \
  --dense-short-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_short_neg_rejection_v1 \
  --dense-long-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_long_pos_breakout_v1 \
  --sparse-short-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_short_v1 \
  --sparse-long-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_long_v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/dense_complement_report_v1

uv run python src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/overlap_audit.py \
  --promotion-json src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1/promotion.json \
  --sparse-manifest src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_short_v1/run_manifest.json \
  --sparse-report-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/sparse_report_v1 \
  --short-state-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/state_recovery_short_dense_v1 \
  --long-state-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/state_recovery_long_dense_v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_high_feature_bias/artifacts/overlap_audit_v1 \
  --date-step 7

# 4. Contract and reporter regression suite
uv run python -m unittest \
  src.research.timeSeries.tests.test_filllots_at_high_profile \
  src.research.timeSeries.tests.test_filllots_at_high_report \
  src.research.timeSeries.tests.test_filllots_at_high_dense_report \
  src.research.timeSeries.tests.test_filllots_at_high_dense_complement_report -v
```
