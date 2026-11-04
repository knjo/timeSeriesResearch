# FillLots_atLow 原始狀態的多空 bias distribution

研究日期：2026-08-26  
資料區間：2025-02-11 至 2026-08-25

## 結論

`FillLots_atLow < 0` 找回一個**小幅、跨年同號的風險調整偏空
shared-pool bias**：

- short label 的 equal-day `Δz=+0.023279`，HAC(5) `t=2.341`、
  `p=.01923`、95% CI `[+.00379,+.04277]`；
- long label 的 equal-day `Δz=-0.016808`，HAC(5) `t=-1.741`、
  `p=.08174`、95% CI `[-.03573,+.00212]`；
- short 與 long 在 2025、2026 都維持上述偏空方向，但各年度估計本身不夠精確。

因此，依 discovery protocol 事前指定的單一 `negative_state × short`
positive-control，可判定為**弱度 recovery**。若把兩個 state × 兩個方向視為四個
exploratory primary tests，最佳格的 joint Holm-adjusted `p=.07691`，未過 5%；
不能宣稱已找到 family-adjusted 的新 alpha。

`FillLots_atLow > 0` 的點估計呈鏡像利多方向（short `Δz<0`、long
`Δz>0`），但兩邊都不顯著；目前**沒有可靠的正 state 利多 shared bias**。

這個偏空結論只存在於 beta／TXF 調整後的 residual 與 z：負 state 的 equal-day
residual uplift 為 short `+5.745bp (p=.00121)`、long
`-3.949bp (p=.02437)`；raw BP 則為 short `+1.246bp (p=.477)`、long
`+0.550bp (p=.750)`，不形成一致的 outright price bearish return。

最後，same-stock 與 same-stock-by-5-minute 的結果全部反號。負 state 在 shared
pool 看似偏空，但同股同時段內反而偏 long；這是很強的 Simpson／composition
pattern，不是 `FillLots_atLow < 0` 發生後即可執行的 short timing alpha。

## 本次實際測的問題

原始狀態定義為：

```text
F = cumsum(FillLots × InOut × (TrialMatch == 0))
    over (QuoteCode, RecordLow), ordered by ChannelSeq

negative_state := F < 0
positive_state := F > 0
```

F 先在完整 raw tick stream 上 causal 重建，再與 eligible labeled pool 交集。本次
沒有加入 formal-trade-only、state update、onset、epoch entry、深度門檻或 cooldown；
因此每個符合狀態的 eligible tick 都是一個 hypothetical entry row，而不是獨立訂單。

Primary estimand 是每個 active day 的：

```text
mean(label_z | state) - mean(label_z | complete eligible pool)
```

再對 active days 等權平均，以 Bartlett Newey-West HAC lag 5 推論。完整 pool 包含
condition rows，這是 unconditional shared contrast，不是 disjoint causal control；
self-inclusion 會把差異往 0 壓縮。same-stock 與 same-stock-by-5-minute 只作來源歸因。

`Δz>0` 一律表示「對該方向 label 較有利」，所以偏空的 coherent pattern 是
short `Δz>0` 且 long `Δz<0`；偏多則相反。

## 「至收盤」label 的邊界

依 timeSeries 目前凍結的研究契約，本次不是 unconditional pure-close return，而是
direction-specific **barrier-or-close**：

- short：以同 tick `BidPrice1` taker 進場；若 `FutureHigh > 1.08 × RefPrice`，
  以 `1.08 × RefPrice + 2 ticks` 回補，否則以 13:30 `Close` 出場；
- long：以同 tick `AskPrice1` taker 進場；若 `FutureHigh > 1.09 × RefPrice`，
  以 `1.09 × RefPrice` 停利，否則以 13:30 `Close` 出場；
- 兩者都扣 prior stock beta × event-to-13:30 TXF component，再除以 prior residual
  volatility 得到 `label_z`。

long／short barrier 不對稱，且各有自己的 label-valid pool，所以不能用一邊的負號
代替另一方向的實測。若問題要的是「忽略途中 barrier、無條件抱到 Close」，需要另建
一組 versioned pure-close labels，不能把本結果直接改名。

## 樣本與 lineage

- Requested sessions：377；active sessions：376。2025-04-07 的 eligible pool 為 0，
  仍保留為完整性稽核列。
- 2025：222 sessions／221 active；2026：155／155 active。
- short label-valid pool：944,359,617 rows；long：944,352,126 rows。
- 兩方向的 raw input identities 完全相同，run manifests 均 `failures=[]`；label-valid
  pool 在 109 日有微小差異，總差 7,491 rows，單日最大 427 rows。
- short pooled state partition：negative 283,223,181 rows（29.9910%）、positive
  658,818,471（69.7635%）、non-signed 2,317,965（0.2455%）。按日等權的平均
  selection rate 分別為 30.4167%、69.3332%，non-signed 為 0.2501%。

兩個 sign state 幾乎覆蓋整個 pool，因此它們相對同一 full control 的效果會機械性
接近反向；這不是兩個獨立發現，也不能把鏡像結果重複計算為兩份證據。

## Full-period 主結果

95% CI 使用 HAC standard error 的常態近似。`within-label BH` 是各方向內兩個 state
的探索校正；`joint Holm` 是四個 direction × state shared tests 的共同校正。事前
指定的 `negative_state × short` recovery gate 仍以 raw HAC effect／p 判定。

| state | label | equal-day Δz | HAC(5) 95% CI | HAC p | within-label BH q | joint Holm p | 方向解讀 |
|---|---|---:|---:|---:|---:|---:|---|
| negative | short | +0.023279 | [+0.003790, +0.042768] | .01923 | .03846 | .07691 | 對 short 有利 |
| negative | long | -0.016808 | [-0.035734, +0.002118] | .08174 | .16349 | .24523 | 對 long 不利 |
| positive | short | -0.006885 | [-0.015461, +0.001690] | .11557 | .11557 | .24523 | 點估計偏多，不顯著 |
| positive | long | +0.004541 | [-0.003473, +0.012555] | .26674 | .26674 | .26674 | 點估計偏多，不顯著 |

Row-weighted pooled `Δz` 較大：negative short `+0.034699`、negative long
`-0.030716`、positive short `-0.015081`、positive long `+0.013309`。數億 event
rows 不是獨立樣本；正式推論仍以上表的日等權 HAC 為準。

## 跨年穩定性

| state | label | 2025 equal-day Δz | HAC p | 2026 equal-day Δz | HAC p |
|---|---|---:|---:|---:|---:|
| negative | short | +0.018768 | .12564 | +0.029710 | .07337 |
| negative | long | -0.011657 | .29421 | -0.024152 | .15924 |
| positive | short | -0.004920 | .41555 | -0.009687 | .11731 |
| positive | long | +0.002500 | .64029 | +0.007451 | .23840 |

四格都跨年同號，沒有明顯 regime sign flip；但任何單一年份都沒有 5% 顯著性，
所以應解讀為方向穩定、估計仍弱，而不是把 2026 當成 untouched holdout。

## Daily bias distribution

以下是 active-day 的 condition-minus-full-pool `Δz` 分布；不是 event-level tail
distribution。`positive days/months` 表示 `Δz>0`，不等同股票價格上漲。

| state | label | q05 | q25 | median | q75 | q95 | positive days | positive months |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| negative | short | -0.216394 | -0.076733 | +0.010722 | +0.107785 | +0.268961 | 202/376 | 14/19 |
| negative | long | -0.277066 | -0.100771 | -0.006913 | +0.084668 | +0.223242 | 184/376 | 9/19 |
| positive | short | -0.119055 | -0.046086 | -0.005506 | +0.034393 | +0.097104 | 174/376 | 10/19 |
| positive | long | -0.102064 | -0.036045 | +0.003812 | +0.044902 | +0.112456 | 194/376 | 8/19 |

負 state 的 short 分布中位數雖為正，但 q05 到 q95 很寬；只有 53.7% active days
為正。long 則有 192/376 days、10/19 months 為負。這符合「小幅 distribution
shift」，不符合「幾乎每天都出現」的強狀態訊號。

## Raw BP 與 residual BP

`pooled uplift` 是 row-weighted 描述；`equal-day uplift` 與 HAC p 才是日級推論。

| state | label | condition raw pooled | pool raw pooled | raw pooled uplift | raw equal-day uplift (p) | residual equal-day uplift (p) |
|---|---|---:|---:|---:|---:|---:|
| negative | short | +23.378bp | +14.303bp | +9.075bp | +1.246bp (.4771) | +5.745bp (.00121) |
| negative | long | -49.853bp | -42.484bp | -7.368bp | +0.550bp (.7498) | -3.949bp (.02437) |
| positive | short | +10.383bp | +14.303bp | -3.920bp | +0.369bp (.5884) | -1.820bp (.00548) |
| positive | long | -39.287bp | -42.484bp | +3.198bp | -1.137bp (.0942) | +1.052bp (.1051) |

負 state 的 raw equal-day long uplift 甚至是小幅正值；只有 residual／z 呈 coherent
bearish。因此最精確的措辭是「相對市場、風險調整後偏空」，不是「股票本身必然跌到
收盤」。condition 的 absolute raw BP 也不是扣費、可成交的策略損益。

## Shared 與 matched 歸因

| state | label | shared equal-day Δz | same-stock Δz | same-stock × 5m Δz |
|---|---|---:|---:|---:|
| negative | short | +0.023279 | -0.221894 | -0.046546 |
| negative | long | -0.016808 | +0.232606 | +0.047749 |
| positive | short | -0.006885 | +0.077513 | +0.015568 |
| positive | long | +0.004541 | -0.076189 | -0.015374 |

所有 matched signs 都與 shared 相反，而且日級 HAC 絕對 t 值約 39～59。這表示
`FillLots_atLow` sign 在不同股票／股票日／時段的分布帶來 shared bias；控制股票與
5 分鐘 bucket 後，局部 timing 關係反轉。shared 是本 objective 的 primary estimand，
所以 condition distribution 確實有偏移；matched 則否定把它解讀成同股可交易 entry
timing。反轉也極穩定：negative short 的 same-stock 與 same-stock-time 都是
0/376 positive days、0/19 positive months；negative long 則都是 376/376、19/19。
positive state 幾乎完全鏡像，但因兩個 sign states 近互補，不能把這些穩定鏡像當成
多份獨立證據。

## HAC lag sensitivity

lag 5 是事前固定的主規格；其餘只作 robustness，不能事後挑選。

| label / negative_state | lag 0 p | lag 5 p | lag 10 p | lag 20 p |
|---|---:|---:|---:|---:|
| short | .01081 | .01923 | .02019 | .00874 |
| long | .06109 | .08174 | .08624 | .05770 |

short-negative 單格的方向與 raw p 對 lag 穩健；但四格 joint Holm p 在 lag
0/5/10/20 分別為 .0432/.0769/.0808/.0350，會跨過 5% 邊界。正式結論因此保留
lag 5 的「positive-control recovery／joint borderline」判讀。

## 研究限制

- persistent state 會讓同一狀態下的每個 tick 重複成為 entry row；event rows 不是
  獨立交易，且 payoff horizons 大量重疊。推論單位已提升到 day，但未做 stock／epoch
  cluster inference。
- condition 與 entry 在同 tick；未做 next-tick、latency、queue／fill、cooldown、
  fees、capacity 或 portfolio risk。
- shared control 包含 condition，且 shared effect 主要來自 composition；不能當成
  causal event effect。
- 2025、2026 都是 retrospective sample，沒有 untouched holdout。
- additive daily artifacts 保存 pooled moments與 daily effects，沒有 event-level
  quantiles／histogram；本報告的「distribution」指 daily conditional mean shift。
- 若要回答 unconditional pure-close、可執行事件或 event-level tails，必須另開
  versioned study，而不是從本結果外推。

## Artifacts

- [`period_summary.parquet`](../../../test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1/period_summary.parquet)：
  full／2025／2026 的 shared、matched、raw、residual、分位數與 multiplicity。
- [`daily_effects.parquet`](../../../test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1/daily_effects.parquet)：
  1,508 個 direction × date × state 的可重算 daily distribution。
- [`monthly_stats.parquet`](../../../test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1/monthly_stats.parquet)：完整月度統計。
- [`daily_state_partition.parquet`](../../../test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1/daily_state_partition.parquet)：
  每日 sign partition 與 non-signed complement。
- [`hac_sensitivity.parquet`](../../../test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1/hac_sensitivity.parquet)：
  lag 0/5/10/20 robustness。
- [`report.json`](../../../test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1/report.json)：報表契約、lineage 驗證與完整摘要。
- short run：
  [`run_manifest.json`](../../../test/20260825_filllots_at_low_framework/artifacts/state_recovery_short_dense_v1/run_manifest.json)、
  [`daily_stats.parquet`](../../../test/20260825_filllots_at_low_framework/artifacts/state_recovery_short_dense_v1/daily_stats.parquet)。
- long run：
  [`run_manifest.json`](../../../test/20260825_filllots_at_low_framework/artifacts/state_recovery_long_dense_v1/run_manifest.json)、
  [`daily_stats.parquet`](../../../test/20260825_filllots_at_low_framework/artifacts/state_recovery_long_dense_v1/daily_stats.parquet)。

## Reproduce

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/run_sweep.py \
  --candidate-registry state --start 20250211 --end 20260825 --step 1 \
  --label-profile short-taker-stop108-v1 \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/state_recovery_short_dense_v1

UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/run_sweep.py \
  --candidate-registry state --start 20250211 --end 20260825 --step 1 \
  --label-profile long-taker-tp109-v1 \
  --output-dir src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/state_recovery_long_dense_v1

UV_CACHE_DIR=/tmp/hft-uv-cache uv run python \
  src/research/timeSeries/test/20260826_filllots_at_low_state_bias/report_state_bias.py \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/state_recovery_short_dense_v1 \
  src/research/timeSeries/test/20260825_filllots_at_low_framework/artifacts/state_recovery_long_dense_v1 \
  --output-dir src/research/timeSeries/test/20260826_filllots_at_low_state_bias/artifacts/dense_report_v1
```
