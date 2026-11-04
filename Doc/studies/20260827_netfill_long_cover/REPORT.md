# Signed `FillLots_atLow/atHigh` long-cover BackTest

Status: `RESEARCH_DIAGNOSTIC_ONLY`  
Classification: `scoped`  
Decision: `NO_RETROSPECTIVE_STANDALONE_OR_COVER_SCREEN_PASS`  
Direction: taker long，並與既有 `fh_neg_rejection` static short 配對  
Evidence: outcome-informed retrospective diagnostic  
Window: `20250211–20260825`（377 requested sessions）

## 結論先講

這次沒有找到接近損益兩平、也沒有找到能改善 `fh_neg_rejection` 整體風險報酬的固定
多單。六個事前凍結的 `FillLots_atLow/atHigh` candidate 在 **0bp 成本時就全部是負
edge**；19.3bp 後 full-period net edge 落在 `-65.770bp` 到 `-114.370bp`。
相對最好的是 `fh_pos_breakout`，但仍虧 `2,301.17萬`、Sharpe `-7.525`，2025／2026
也分別是 `-84.371bp`／`-34.097bp`，不能稱為 break-even long。

另一方面，它們並非完全沒有 cover 形狀：六條 long daily PnL 與 short 的 Pearson／
Spearman 都是負相關，而且在 short 固定最差 10% 日期的平均 long PnL 都是正數；六條
worst-tail effect 也通過 current-six moving-block max-t retrospective screen。不過 long
本身的負 carry 太大：預先指定的 native `h=0.5` 下，六個組合的 Sharpe 與 MDD 全部
比 short-only 差，因此沒有 cover pass。

最小的 `h=0.25` 也沒有解決問題：六條全都改善 ES5，但沒有任何一條改善 Sharpe；
`depth_gt_pos4_epoch_entry` 雖同時讓 MDD 改善 `1.29萬`、ES5 改善 `2.11萬`，卻付出
`604.60萬` total carry，Sharpe 下降 `0.405`。這是有 tail payoff 的 **paid hedge**，
不是可自償的 long cover。

因此本次不選 winner、不 promotion，也不準備 deployment handoff。v2 的逐筆 ledger、
daily series、費用 grid、cover frontier、12 張個別淨值圖與 2 張比較圖均已保留，並由
manifest guard 禁止原地覆寫。

![Six-candidate full-period comparison](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/comparison_full.png)

![Native h=0.5 cover comparison](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/cover_comparison.png)

## `netFillLots` 在這裡的確切語意

使用者口語上的 `netFillLots_atLow/atHigh` 對應 repository canonical 欄位
`FillLots_atLow/FillLots_atHigh`。欄位本身已是 signed net flow，不再做第二次 netting：

```text
cumsum(FillLots * InOut * (TrialMatch == 0))
within QuoteCode and the current RecordLow / RecordHigh epoch
```

`atLow` 的 `epoch_entry` 不是「每個 epoch 只取第一筆」。它接受新 RecordLow epoch
一開始就高於門檻的 formal trade，也接受同一 epoch 後續由 `<= threshold` 再 cross／
re-entry 到 `> threshold` 的事件；離開後重新進入可再次取樣。

## Frozen candidate family

本次只跑下列六條，沒有看完結果再新增或刪除 loser：

| Family | Candidate | Frozen definition |
|---|---|---|
| `FillLots_atHigh` | `fh_pos_sample` | fresh MD first-visit 或相對 strictly-prior big-lot reference 的 large fill，且 `FillLots_atHigh > 0` |
| `FillLots_atHigh` | `fh_pos_depth2` | `fh_pos_sample` 再要求 `FillLots_atHigh / D-1 top depth >= 2` |
| `FillLots_atHigh` | `fh_pos_breakout` | `fh_pos_depth2`、距 RecordHigh 1.5 ticks 內，且 strong bid share／positive micro-momentum／positive LOB pressure 至少 2/3 |
| `FillLots_atLow` | `depth_gt_pos1_epoch_entry` | RecordLow epoch entry／cross／re-entry 的 normalized signed depth ratio `> +1` |
| `FillLots_atLow` | `depth_gt_pos2_epoch_entry` | 同上，門檻 `> +2` |
| `FillLots_atLow` | `depth_gt_pos4_epoch_entry` | 同上，門檻 `> +4` |

fresh high event 的目的，是避免相同價格狀態被逐 tick 重複抽樣；large-fill branch 使用
prior-day side-specific big-lot reference，而不是看累積 FillLots 增長後才取樣。完整 selector
expression、禁止欄位與 runtime constants 以
[frozen contract](../../../test/20260827_netfill_long_cover/study_contract.json) 為準。

## Standalone BackTest

共同 label 是 `benchmark.long_taker_tp109.v1`：同 tick `AskPrice1` taker long；若
`FutureHigh > RefPrice * 1.09`，以 `RefPrice * 1.09` 出場，否則以 13:30 Close 出場。
表內成本固定 19.3bp；PnL／MDD 單位為新台幣萬元。Sharpe 使用完整 377-session calendar，
zero-signal day 補零。

| Candidate | Trades | Gross BP | Net BP | 2025 net BP | 2026 net BP | Net PnL | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fh_pos_sample` | 177,255 | -57.795 | -77.095 | -89.529 | -55.876 | -15,783.73 | -10.308 | -15,853.63 |
| `fh_pos_depth2` | 66,546 | -46.844 | -66.144 | -87.512 | -36.068 | -4,828.68 | -8.259 | -4,850.37 |
| `fh_pos_breakout` | 23,815 | -46.470 | -65.770 | -84.371 | -34.097 | -2,301.17 | -7.525 | -2,313.11 |
| `depth_gt_pos1_epoch_entry` | 18,360 | -95.070 | -114.370 | -133.540 | -81.582 | -4,282.97 | -11.339 | -4,284.14 |
| `depth_gt_pos2_epoch_entry` | 14,902 | -85.719 | -105.019 | -124.470 | -73.220 | -3,261.75 | -9.917 | -3,264.95 |
| `depth_gt_pos4_epoch_entry` | 11,953 | -77.970 | -97.270 | -122.084 | -60.222 | -2,418.39 | -9.540 | -2,420.26 |

2026 的 level 雖比 2025 少虧，仍不是廣泛的月度穩定：19 個月中，各 candidate 只有
0–2 個正月；所有 candidate 在觀察到的 11 個 2025 月份都為負，而 2026-03 到
2026-08 又全部逐月為負。raw label 是 primary；accepted trades 的 residual diagnostic
coverage 為 `99.6003%–99.8076%`，own-denominator residual net BP 在每個
candidate × period 仍全負，所以 diagnostic missingness 不能解釋虧損。

代表性的 `fh_pos_breakout` full BackTest 如下；其餘五條 full 與六條 2026 圖均保存在
同一 finalized artifact 目錄。

![fh_pos_breakout full BackTest](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/backtest_full_fh_pos_breakout.png)

### 成本敏感度

下表只改固定線性成本，signals 與容量不變；數字是 full-period capital-weighted net BP。

| Candidate | 0bp | 10bp | 19.3bp | 25bp | 35bp |
|---|---:|---:|---:|---:|---:|
| `fh_pos_sample` | -57.795 | -67.795 | -77.095 | -82.795 | -92.795 |
| `fh_pos_depth2` | -46.844 | -56.844 | -66.144 | -71.844 | -81.844 |
| `fh_pos_breakout` | -46.470 | -56.470 | -65.770 | -71.470 | -81.470 |
| `depth_gt_pos1_epoch_entry` | -95.070 | -105.070 | -114.370 | -120.070 | -130.070 |
| `depth_gt_pos2_epoch_entry` | -85.719 | -95.719 | -105.019 | -110.719 | -120.719 |
| `depth_gt_pos4_epoch_entry` | -77.970 | -87.970 | -97.270 | -102.970 | -112.970 |

所以問題不是「19.3bp 太高」：即使交易成本設成零，也沒有一條達到 break-even。

## 與 `fh_neg_rejection` short 的 cover

short-only snapshot 在相同 377-session calendar 的基準是：PnL `+5,815.09萬`、
Sharpe `6.854`、MDD `-161.59萬`、ES5 `-45.43萬`。primary accounting 是 native cash：

```text
combined_daily_pnl(h) = short_net_pnl + h * long_net_pnl
h in {0, 0.25, 0.5, 1.0}
```

`h` 是 long event lots 的算術 multiplier，不是資金配置比例。預先指定的 formal screen
使用 `h=0.5`；下表 delta 大於零才表示改善，金額單位均為萬元。

| Long candidate | P/S corr | Worst-short-day long mean | Combined PnL | Sharpe Δ | MDD Δ | ES5 Δ |
|---|---:|---:|---:|---:|---:|---:|
| `fh_pos_sample` | -.716 / -.756 | +30.79 | -2,076.77 | -10.241 | -1,965.84 | -8.22 |
| `fh_pos_depth2` | -.636 / -.677 | +15.18 | +3,400.75 | -1.995 | -85.19 | +4.35 |
| `fh_pos_breakout` | -.584 / -.632 | +7.40 | +4,664.51 | -.790 | -40.51 | +3.83 |
| `depth_gt_pos1_epoch_entry` | -.770 / -.777 | +5.93 | +3,673.61 | -1.703 | -61.20 | +4.29 |
| `depth_gt_pos2_epoch_entry` | -.711 / -.689 | +5.82 | +4,184.22 | -1.204 | -35.08 | +3.42 |
| `depth_gt_pos4_epoch_entry` | -.661 / -.675 | +5.00 | +4,605.90 | -.877 | -19.97 | +3.52 |

`Worst-short-day long mean` 是未乘 `h` 的 native long，在 short-only 事先固定最差 38 日
的條件平均；不能把它當作每天可賺的 cover PnL。雖然 tail max-t adjusted p 在主
5-session block 下為 `.0032–.0468`，且 3/5/10/20-session sensitivity 都符合預先指定
的 `.10` retrospective threshold，formal cover 還同時要求 combined Sharpe 與 MDD
不惡化；六條都在這兩項失敗。

`h=0.25` 的較小部位也沒有 joint improvement：

| Long candidate | Incremental total PnL | Sharpe Δ | MDD Δ | ES5 Δ |
|---|---:|---:|---:|---:|
| `fh_pos_sample` | -3,945.93 | -3.904 | -129.98 | +3.33 |
| `fh_pos_depth2` | -1,207.17 | -.822 | -29.37 | +3.78 |
| `fh_pos_breakout` | -575.29 | -.354 | -2.40 | +2.34 |
| `depth_gt_pos1_epoch_entry` | -1,070.74 | -.756 | -12.75 | +2.95 |
| `depth_gt_pos2_epoch_entry` | -815.44 | -.542 | +0.01 | +2.37 |
| `depth_gt_pos4_epoch_entry` | -604.60 | -.405 | +1.29 | +2.11 |

`depth_gt_pos2` 的 `+0.01萬` MDD delta 只有約 139 元，實質上可視為不變；
`depth_gt_pos4` 的小幅 MDD／ES improvement 也遠小於其 carry cost。secondary
position-normalized frontier 已完整保存，但 daily `position` 是累積 entry notional，
不是 concurrent capital，故不拿來覆蓋 primary native-cash 結論。

## 參數穩定與 overfit 邊界

### Low threshold ladder

`+1 → +2 → +4` 在 full、2025、2026 都呈單調「少虧」：full net BP
`-114.370 → -105.019 → -97.270`，trades 則
`18,360 → 14,902 → 11,953`。這代表 filtering 方向平滑，但它是穩定的負 ladder，
不是 profitable plateau；machine status 是 `LOW_THRESHOLD_LADDER_REPORTED`，不是
plateau pass。

### High nested ablation

`sample → depth2 → breakout` 的 full net BP 為
`-77.095 → -66.144 → -65.770`。depth gate 的改善在 2025 只有 `+2.016bp`，在 2026
卻是 `+19.808bp`；breakout 再砍掉 depth2 約 64.2% trades，full 只多改善
`0.374bp`。這可以當 nested ablation 描述，不能當多參數 neighborhood 穩定性；三條
high candidate 的 plateau status 均為 `NOT_EVALUATED`。

### Selection-bias accounting

- current family raw trials 為 6；daily-PnL correlation 的 eigenvalue participation ratio
  給 effective trials `1.6691`；
- standalone positive-PnL moving-block max-t adjusted p 六條皆為 `1.0`，block length
  3/5/10/20 也不變；
- 可重建的 historical selector exposure 下限為 28 組；6-trial 與 28-trial approximate
  DSR 均保存，但只作 sensitivity diagnostic，沒有假裝完整消除更早 search exposure；
- trial ledger 保留六個 selector fingerprints、未入選組合與零樣本期，`selected=false`
  且 `promotion_authorized=false`。

本次結果本來就全面為負，因此 multiplicity correction 不會把它救成正 edge；對 tail
complement 的正結果則有做 current-six max-t。仍然不能忽略兩個限制：本 family 是在
過去 long-side 研究看過 outcome 後才凍結，且 high multi-parameter local plateau 沒做。

## Execution、容量與部位口徑

所有 candidate 使用與 kept short snapshot 相同的 static execution gate，沒有模型 score：

- inherited `RemainSeconds`／positive Bid-to-Ref buckets；
- OTC 或 `MD_L1Rate_30 > .25`；`SpreadPairElapsed > .1`；
- current 或 D-1 day-trade amount rank `<= 100`；
- `AskPrice1 <= 1000`；D-1 maker spread BP 為 null 或 `> -70`。

每個 accepted event 一張，依 `candidate, Date, QuoteCode, TransTime, ChannelSeq` 排序；
同檔每日累積張數嚴格低於 D-1 `avg_bidLots1 + avg_askLots1`，同檔每日 Ask entry
notional 嚴格低於 `200萬`。沒有 portfolio cap、同檔 inventory netting、cooldown、
latency、impact 或 fill model。

| Candidate | Active / calendar | Mean daily entry notional | Max daily entry notional |
|---|---:|---:|---:|
| `fh_pos_sample` | 376 / 377 | 5,430.55萬 | 9,627.20萬 |
| `fh_pos_depth2` | 376 / 377 | 1,936.39萬 | 4,635.29萬 |
| `fh_pos_breakout` | 375 / 377 | 928.06萬 | 2,596.04萬 |
| `depth_gt_pos1_epoch_entry` | 375 / 377 | 993.33萬 | 2,190.09萬 |
| `depth_gt_pos2_epoch_entry` | 375 / 377 | 823.83萬 | 1,946.01萬 |
| `depth_gt_pos4_epoch_entry` | 375 / 377 | 659.49萬 | 1,588.02萬 |

這些 position 數字是每日累積進場名目，偏 turnover-like；不能解讀成任一時點需要的
concurrent gross capital。

## Freeze chronology、lineage 與驗證

v1 daily materialization 開始後、任何 aggregate candidate PnL、summary、圖或 winner
結果被讀取前，pre-result audit 發現三件事：cache 沒綁 materialization implementation、
atLow prose 沒寫清 repeat re-entry、raw-PnL eligibility 錯誤依賴 residual-z diagnostic。
因此 v1 明確作廢且不是 evidence；v2 從 raw sources 重建。這次 amendment 沒改六個
candidate、threshold、execution gate、capacity、fee grid、calendar 或 cover multipliers。

kept v2 identities：

```text
contract                 netfill_long_cover/retrospective_v2
contract SHA-256         3e33e539185af217ed27867dcbc4667bca440e61fd0e1996b8166db669e7f659
materialization SHA-256  815082e6bc9aa0b0ed1848d4a1654456fbd641e7cc8c2af3195ddae89dc2c1b2
label profile SHA-256    a7d70c7eac9e3ac55bdb963b7aea2f335e145afe422bcca08cff713e9f79b3c8
run manifest SHA-256     5a00bcb2bf24122971e5ceaebbd734237931d271e9fd4cb02250090aa5d5bb01
```

materialization 掃描 377 日、34,881,468 candidate-indexed rows，產生 312,831 個
candidate accepted rows；candidate 彼此 nested／overlap，不能把這個加總稱為 unique
market events。final verification 為 38/38 checks 全通過，包含 dense calendar、逐日與
candidate reconciliation、成本公式、cover 全 grid、short `h=0` parity、MDD initial-zero、
source／implementation／short snapshot identities。另有 24/24 synthetic unit tests 通過。

runner 偵測到 finalized `run_manifest.json` 會在任何寫入前 fail closed；這是 runner-level
immutability，不是 OS immutable bit。原始 raw source binding 使用 path／size／mtime，
不是全部逐檔 content hash，仍是 lineage 限制。

## Artifacts 與 reproduce

- [study README](../../../test/20260827_netfill_long_cover/README.md)
- [frozen contract](../../../test/20260827_netfill_long_cover/study_contract.json)
- [machine decision](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/decision.json)
- [verification](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/verification.json)
- [run manifest](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/run_manifest.json)
- [period summary](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/period_summary.parquet)
- [fee sensitivity](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/fee_sensitivity.parquet)
- [cover frontier](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/cover_summary.parquet)
- [family-bias adjustment](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/family_bias_adjustment.json)
- [trial ledger](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/trial_ledger.parquet)
- [accepted signal ledger](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/accepted_signals.parquet)
- [dense daily series](../../../test/20260827_netfill_long_cover/artifacts/static_long_cover_v2/daily.parquet)

本次初次 materialization／finalization 使用：

```bash
MPLCONFIGDIR=/tmp/matplotlib-hft \
UV_CACHE_DIR=/tmp/hft-uv-cache \
uv run python \
  src/research/timeSeries/test/20260827_netfill_long_cover/run_backtest.py
```

目前因 manifest 已 finalized，同一指令會依設計拒絕覆寫。若要獨立重建，應建立新的
artifact version 並保留 v2，不可刪掉 manifest 後原地重跑。純函數與 runner contract
測試可重跑：

```bash
MPLCONFIGDIR=/tmp/matplotlib-hft-test \
UV_CACHE_DIR=/tmp/hft-uv-cache \
uv run python -m unittest \
  src/research/timeSeries/test/20260827_netfill_long_cover/test_analysis_utils.py \
  src/research/timeSeries/test/20260827_netfill_long_cover/test_run_backtest.py
```

## 限制與下一步邊界

這是 retrospective、historically outcome-informed replay，不是 prospective first-live。
即使把 long 當 insurance，它仍缺少 joint portfolio cap、concurrent exposure、inventory
netting、latency、impact、borrow／fill 等實盤層。現在沒有理由從六條中挑一條固定掛上；
若未來要繼續，合理的新 objective 是在 **事前凍結的 tail-risk budget** 下研究可接受的
insurance premium，或把 long condition 當樣本交給 strictly-prior model，再用下一期
first-live 驗證。不能拿這次負相關的事後觀察，回頭調 `h` 或 threshold 後稱為 OOS。
