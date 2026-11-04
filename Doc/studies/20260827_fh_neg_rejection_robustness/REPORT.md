# `fh_neg_rejection` robustness re-study

Status: `RESEARCH_SHADOW_ONLY`  
Classification: `canonical`  
Study: `20260827_fh_neg_rejection_robustness`  
Direction: short  
Primary estimand: shared-pool equal-active-session `delta_z`  
Retrospective window: `20250211–20260825`

固定 conservative anchor 的後續交易化快照、淨值圖、每日名目與成本敏感度另見
[STATIC_BACKTEST.md](STATIC_BACKTEST.md)。它是 outcome-informed retrospective
companion，不改變本報告的 `RESEARCH_SHADOW_ONLY` decision。

## 結論先講

`fh_neg_rejection` 在既有歷史窗內確實切出穩定的空方 shared-pool bias，而且
`D/G/B` 八格 ablation、5/10/20-session block inference，以及事後凍結的 27 格
threshold neighborhood 都沒有顯示「只有精確門檻才成立」的孤立尖峰。這部分的
retrospective diagnostic 結果是正面的。

但目前不能部署，亦不能準備 deployment handoff，理由不是樣本不足，而是證據等級：

- 所有主要正結果都來自已參與條件發現的 `Date <= 20260825` 歷史資料；
- `20260826` 只有一個 non-pristine artifact-OOS session，而且當日 anchor 反號；
- `Date > 20260827` 的 prospective first-live artifact 尚不存在；
- 2025 到 2026 的 anchor `delta_z` 從 `+.147385` 降至 `+.055341`，衰退
  `62.45%`；固定三等分的 T1 到 T3 亦下降 `74.25%`；
- current-family max-stat 有處理本次 8 格與 27 格的 multiplicity，但更早的 condition
  search 只能重建下限，不能把歷史資料轉成 pristine OOS。

對「三個條件是否都必要」的直接回答是：依凍結的統計 rule，`D/G/B` 三者都通過；
依效果量與 sample shaping，`D`、`B` 有實質必要性，`G` 只呈現很小、但穩定為正的
邊際增量，尚未建立經濟必要性。若要拿掉 `G`，應建立新的 selector version 與新的
prospective clock，不能由本次結果直接換線。

## 證據等級不能混用

| 日期 | 證據類別 | 本報告允許的解讀 | 可否用於部署 |
|---|---|---|---:|
| `<= 20260825` | retrospective diagnostic | 機制、support、時間衰退、parsimony、threshold fragility、selection exposure | 否 |
| `20260826` | artifact-OOS, non-pristine | 單日 descriptive spot check | 否 |
| `20260827` | freeze day excluded | 不納入 first-live | 否 |
| `> 20260827` | prospective first-live | 累積足量後才可評估 G1/G2/G3/G6 | 尚無資料 |

全文中的「pass」若未特別註明，都只代表 retrospective diagnostic pass，不代表
deployment pass。

### Freeze chronology

原始 selector、八格 family、objective 與 gates 先凍結。methods amendment 1 是在
`20250408`、`20260825`、`20260826` 三個 smoke-session outputs 已可見、但 full-period
aggregate 尚未產生時補上 circular block bootstrap、外部 status calendar 與證據類別
邊界；amendment 2 是 daily materialization 已開始、但 aggregate/evaluator outcome 尚未
檢視時補齊 HAC、raw/residual BP denominator 與 temporal bound。兩次都沒有改 selector、
候選、threshold、hypothesis 或 gate，但也不包裝成完全 untouched preregistration。

27 格 plateau contract 更晚，明確是在 Stage 1 八格 outcome 已看過後凍結，因此只屬
outcome-informed retrospective diagnostic。

## Condition 的具體邏輯

共同 base 是：

```text
fresh_event AND FillLots_atHigh < 0
```

`fresh_event` 為兩種因果 trigger 的聯集：

1. `md_price_change`：在目前 `(QuoteCode, RecordHigh)` epoch 內，整數 B1/A1 組合是
   第一次造訪，而且 B1 或 A1 相對上一筆 raw row 至少變動半個合法 tick；
2. `large_fill`：目前 row 是 formal trade，且 `FillLots >= max(20,
   strictly-prior same-side big_buy_lots/big_sell_lots)`。

若 prior `big_lots` 或平均深度 reference 不能證明有效，large-fill trigger fail closed；
仍可保留有效的 MD trigger。這避免同一個未變價格被反覆取樣，也把使用者要求的
「相當張數 FillLots 才看」具體化。branch 固定標成 `md_only`、`large_fill_only`、
`md_and_large_fill`，但本研究不從 branch 績效挑 winner。

在共同 base 上再套三個 gate：

```text
D: FillLots_atHigh /
   (strictly-prior avg_bidLots1 + avg_askLots1) <= -8

G: filllots_at_high_gap_ticks >= 4

B: 以下三項至少 2/3 成立
   B1_A1B1 <= 0.35
   MicroMomentum_30 <= -0.10 bp
   LOB_NetPressure_30 < 0
```

完整 anchor 為 `fresh_event AND FillLots_atHigh < 0 AND D AND G AND B`。`D` 是
normalized ratio、`G` 是合法 tick、`B` 是 bounded integer vote，因此目前沒有觀察到
非做 prior-percentile 不可的尺度問題；仍須逐期報告 support 與 selection prevalence。

## Pool、label 與 estimand

| 項目 | 定義 |
|---|---|
| canonical sample key | `(Date, QuoteCode, ChannelSeq)`，每鍵唯一 |
| eligible pool | `tw_equity_daytrade_rank150_090030_123000_bidref_v1` |
| label | `negfill.short_taker_stop108.v1` |
| entry / exit | Bid taker；先遇 1.08 adverse barrier 則含兩 tick slippage 回補，否則 13:30 Close |
| primary label | strictly-prior beta residual return，再以 causal event-to-exit volatility 標準化 |
| primary daily estimand | candidate 當日平均 `label_z` 減完整 shared eligible pool 當日平均 `label_z` |
| period aggregation | active session 等權；zero-event day 保留 support 0，不補成 effect 0 |

這是 candidate-universe distribution study，不是交易簿回測。850,376 個 events 不是
850,376 個同時部位；本研究沒有設定每日／每檔資金、最大持倉、訊號合併或成交容量。
本 contract 只驗證事前指定的 short direction，long mirror 不在這個 family 內，不能由
本結果推論。

## Lineage 與 support

主 contract SHA-256 為
`625cda6b85b68dacf83dee38708278a17b176482215cabfcfcc29dcabd40bc7e`，八格 design
fingerprint 為
`74072ad8f2399ba858e0bd83813743598fd22b13ea02cd78695afc44b1a4eae4`。

| 項目 | Retrospective 結果 |
|---|---:|
| requested sessions | 377 |
| evaluable / anchor-active sessions | 376 / 376 |
| active months | 19 |
| zero-control sessions | 1（`20250407`） |
| missing-source / failed sessions | 0 / 0 |
| anchor effect-valid events | 850,376 |
| anchor mean selection rate | `0.0925035%` |
| duplicate canonical keys | 0 |

在 label/pool filtering 之前，conservative fresh negative base 有 42,959,749 events；
branch 組成如下：

| Fresh branch | Events | Base 占比 |
|---|---:|---:|
| `md_only` | 33,099,511 | 77.05% |
| `large_fill_only` | 8,520,580 | 19.83% |
| `md_and_large_fill` | 1,339,658 | 3.12% |

完整 D/G/B 的 pre-label membership 為 6,422,686；經 eligible/label-valid filtering 後，
primary effect 使用 850,376 events。兩種 support 不應混寫。

reference audit 涵蓋 52,094 個 eligible stock-sessions，其中 356 個曾出現 stored-one
ambiguous reference，299 個判為 invalid 或 mismatch。conservative fail-closed 相對 legacy
base 移除 185,796 events（`0.4306%`），相對 legacy D/G/B 移除 21,124
（`0.3278%`）。selector 靜態 audit 讀到 22 個 raw inputs、52 個 expression columns，
未發現 forbidden future/outcome input 或明顯 forward syntax。

G0 因此是 `PASS_WITH_DISCLOSED_LIMITATIONS`，不是無條件 lineage pass：目前 calendar
不是 exchange-authoritative；raw source identity 綁 path/size/mtime 而非 content hash；
原 runner manifest 未綁 selector code bytes；AST 不能證明上游 tickFeature producer 的
完整因果語意；reference reconstruction 只深入 stored-one ambiguity；membership support
又早於 label-valid filtering。

## Positive control：`FillLots_atLow < 0`

相同 pool、short label、beta/vol 與 shared-control pipeline 回收到既有方向：

| Events | Active sessions | Equal-day `delta_z` | HAC(5) two-sided p | Raw uplift | Residual uplift | 2025 / 2026 |
|---:|---:|---:|---:|---:|---:|---:|
| 283,223,181 | 376 | `+.023279` | `.01923` | `+1.246bp` | `+5.745bp` | 兩年皆正 |

這是 weak positive-control recovery，只校準 shared-pool pipeline；matched estimands 反號，
且 raw uplift `1.246bp` 遠低於 `19.3bp` 成本。舊 raw files 後來刷新，雖然
`20260825` 的 stored control moments 完全一致，仍不能聲稱兩次 run 是 byte-identical。

## 八格 D/G/B retrospective factorial

下表只使用 `20250211–20260825`。`LCB` 與 adjusted p 是 8-cell synchronous circular
moving-block max-stat，主 block length 10；`Events` 是 effect-valid `condition_n_z` support。
所有格都保留，沒有用 winner-only reporting。

| Cell | D | G | B | Events | Selection rate | Equal-day `delta_z` | 90% simultaneous LCB | Adjusted p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `d0_g0_b0` | 0 | 0 | 0 | 5,350,574 | .578059% | +.020270 | +.008300 | .008198 |
| `d0_g0_b1` | 0 | 0 | 1 | 2,668,721 | .291490% | +.060512 | +.046429 | .000200 |
| `d0_g1_b0` | 0 | 1 | 0 | 4,115,328 | .438850% | +.041705 | +.026278 | .000400 |
| `d0_g1_b1` | 0 | 1 | 1 | 2,076,650 | .224736% | +.075371 | +.058897 | .000200 |
| `d1_g0_b0` | 1 | 0 | 0 | 1,843,775 | .197719% | +.066821 | +.040529 | .000400 |
| `d1_g0_b1` | 1 | 0 | 1 | 861,455 | .093684% | +.106186 | +.078854 | .000200 |
| `d1_g1_b0` | 1 | 1 | 0 | 1,799,330 | .192869% | +.072917 | +.047203 | .000200 |
| `d1_g1_b1` | 1 | 1 | 1 | 850,376 | .092503% | +.109441 | +.082169 | .000200 |

block length 5、10、20 下，八格的 simultaneous LCB 全部為正。anchor 自身的 HAC
lag 1/5/10/20 one-sided 90% LCB 分別為 `+.093442`、`+.091441`、`+.089688`、
`+.087073`，所以結論不是單一 HAC lag 造成。

### Anchor 分布與成本量級

| 指標 | 結果 |
|---|---:|
| equal-day `delta_z` | `+.109441` |
| positive active days | 67.82% |
| positive months | 19 / 19 |
| equal-active raw uplift | `+11.891bp` |
| equal-active beta-residual uplift | `+17.822bp` |
| cost reference | `19.3bp` |
| raw / residual 相對成本差 | `-7.409bp` / `-1.478bp` |

所以 condition 本身沒有高於 19.3bp 成本；而且這些數字是 condition 相對 shared pool
的 uplift，不是已含 sizing、同檔訊號淨額、容量與實際成交的 strategy PnL。研究 contract
本來就沒有要求 condition 單獨過成本，獲利改善可以留給後接 prior-only model，但模型
不能用來拯救一個未通過穩定性或 prospective gate 的 selector。

## 時間穩定性：同號，但有明顯衰退

三個區塊邊界只按 requested calendar 等分，不看 event count 或 outcome。表內 interval
是主 block length 10 的 one-sided 90% bounds。

| Block | 日期 | Requested / active sessions | Events | `delta_z` | 90% interval |
|---|---|---:|---:|---:|---:|
| T1 | `20250211–20250812` | 126 / 125 | 205,277 | +.182254 | `[+.140932, +.220608]` |
| T2 | `20250813–20260223` | 126 / 126 | 305,267 | +.099218 | `[+.083014, +.114986]` |
| T3 | `20260224–20260825` | 125 / 125 | 339,832 | +.046933 | `[+.025387, +.067118]` |

三段都為正，因此凍結的 retrospective G3 rule 通過；但 T3 比 T1 低 `74.25%`，不能只
報「3/3 positive」而忽略 level decay。按年度看也一致：

| Cell | 2025 `delta_z` | 2026 `delta_z` | 解讀 |
|---|---:|---:|---|
| base `d0_g0_b0` | +.035549 | -.001516 | common base 在 2026 已近零且微幅反號 |
| anchor `d1_g1_b1` | +.147385 | +.055341 | 仍為正，但降低 62.45% |

selection prevalence 同期沒有等比例崩塌：center 從 2025 的 `.090635%` 到 2026 的
`.095168%`，反而增加約 `5.00%`。因此目前較像 condition-to-outcome relationship 或
市場 regime 衰退，不是 raw threshold 在不同年份因 feature 尺度發散而突然取到不同
比例。

## D/G/B 是否必要

G4 使用 anchor active-session calendar 上的 paired `full - leave-one-gate-out` daily
increment。這是 nested sample-composition contrast，不是 causal treatment effect。

| Gate | Full − ablation mean `delta_z` | Median | Bootstrap 90% CI | `P(>0)` | Gate 帶來的 support 減少 | 2025 / 2026 increment |
|---|---:|---:|---:|---:|---:|---:|
| D | +.034070 | +.027943 | `[+.017198,+.054053]` | .9998 | 59.05% | +.042109 / +.022609 |
| G | +.003255 | +.000527 | `[+.001882,+.004840]` | 1.0000 | 1.286% | +.004277 / +.001798 |
| B | +.036524 | +.029497 | `[+.028918,+.044378]` | 1.0000 | 52.74% | +.040171 / +.031324 |

三個 gate 都符合事前規則：median 為正且 block-bootstrap `P(increment > 0) >= .80`；
12 個 conditional main increments 亦全部為正。但效果量不是同一級：

- `D` 與 `B` 都把 support 約砍半，換來約 `+.034～+.037 delta_z`，且跨年仍有明顯
  增量，具實質 sample-shaping 理由；
- `G` 只排除約 1.3% 的 reduced-cell support，平均增量僅 `+.0033`、median 僅
  `+.0005`，所以是統計上穩定的微增量，不足以宣稱經濟上必要；
- `DG` 與 `GB` interaction 為負，代表 gate 間有 diminishing returns，不是三條件
  完全獨立相加。

因此本版保留三者是尊重 frozen rule；若研究目標轉為簡約 selector，最值得另開版本
檢驗的是 `D+B`、不含 `G` 的版本，而不是在看過本結果後直接改目前 anchor。

## Threshold 周圍敏感度

### 證據性質

27 格 grid 是在八格 Stage 1 outcome 已看過之後、plateau outcome 尚未看過時另行凍結：

```text
D depth ratio: -6, -8, -10
G gap ticks:     3,  4,  5
B bear votes:    1,  2,  3
center:          (-8, 4, 2)
```

所以它是 outcome-informed retrospective robustness diagnostic，不能用來挑新 winner，
也不能建立 independent OOS。contract hash 為
`40ae9d0dc156cca2b1235660661fb0b5ba5d59c45894b29fda83904adcde500f`，design
fingerprint 為
`226f16c8a26f9535bc6bc6462e8da5a16c8ae1f1dc0f409971a36bc8073e51ee`。

### 結果

377/377 sessions 完成 materialization；27/27 cells 都有 376 active sessions。最小格
`d10_g05_b3` 仍有 138,079 effect-valid events，沒有 support collapse。center event keys
與八格 Stage 1 anchor 完全一致。

| Center 的 6 個 axis neighbor | Effect | Neighbor / center |
|---|---:|---:|
| D 放寬至 -6 | +.104955 | .9590 |
| D 收緊至 -10 | +.112339 | 1.0265 |
| G 放寬至 3 ticks | +.107917 | .9861 |
| G 收緊至 5 ticks | +.110167 | 1.0066 |
| B 放寬至 1 vote | +.089963 | .8220 |
| B 收緊至 3 votes | +.118938 | 1.0868 |

6/6 neighbor 都為正且至少是 center 的 50%，median neighbor/center 為 `.9964`，高於
事前 `.75` 門檻。2025 與 2026 也各自為 6/6，median ratio 分別 `.9958`、`.9986`。
center 在 27-family max-stat 的 simultaneous LCB，block 5/10/20 分別為
`+.086724`、`+.085033`、`+.082974`；三者皆正。

將其餘兩因子平均後，三個 factor 的 full-period level mean 都隨 strictness
non-decreasing：

| Factor levels | Level mean `delta_z` |
|---|---|
| D `-6 / -8 / -10` | `.101253 / .105852 / .108277` |
| G `3 / 4 / 5` | `.103348 / .105384 / .106652` |
| B `1 / 2 / 3` | `.089124 / .108647 / .117612` |

2025、2026 的 factor-level means 亦全部 non-decreasing；但 162 個 slice transitions 中
仍有 4 個反轉，全部發生在 2026，量級為 `-.00039～-.00091`：三個是 B=3 時
D 從 -8 收緊到 -10，另一個是 D=-6、B=3 時 G 從 4 收緊到 5。這些不是隱藏失敗，
而是顯示 plateau 大方向平滑、局部不保證嚴格 monotonic。

plateau 的正確結論是：center 不是 retrospective grid 的孤立尖峰，附近參數有穩定
同號與相近尺度；錯誤結論則是「因此 -8/4/2 已被 OOS 證明」或「三條 gate 都有相同
經濟必要性」。

## Overfit 與 selection exposure

| Family / exposure | Raw trials | Dependence-adjusted effective trials | 可用 correction |
|---|---:|---:|---|
| 本次 D/G/B factorial | 8 | 1.635 | 8-cell session block max-stat |
| 本次 threshold plateau | 27 | 1.116 | 27-cell session block max-stat |
| 可重建歷史 selector configurations | 至少 28 | 不可合併估計 | disclosure lower bound |
| 可重建歷史 direction tests | 至少 46 | 不可合併估計 | disclosure lower bound |
| 可重建歷史 label/outcome tests | 至少 51 | 不可合併估計 | disclosure lower bound |

歷史 families 使用不同 selector grammar、sampler、pool、direction、label 或 objective，
不能假裝成一個 exchangeable max-stat family；上表歷史數量也只是不完全下限。本次
adjusted p/LCB 只修正本次凍結 family，無法消除 `fh_neg_rejection` 被先前歷史研究挑出
的 selection bias。

本研究 objective 是 `delta_z` condition effect，不是 daily return Sharpe，因此不套用
Deflated Sharpe Ratio；相容方法是目前的 blocked max-stat/multiple-testing audit。若後續
rolling model 或 execution family 以 net Sharpe 挑選，才應另建包含 zero-trade days 的
DSR-like、CSCV-PBO 或 blocked family audit。

目前最重要的 overfit 警訊不是鄰近 threshold 翻號，而是時間 level decay 與缺少
prospective first-live。plateau pass 降低局部參數過度配適疑慮，但沒有消除歷史條件選擇
偏誤或 regime decay。

## 與 negFill 的相似度及 matched 歸因

目前 anchor 的 same-stock 與 same-stock × 5m attribution 皆反號，而且 2025、2026
一致：

| Period | Shared `delta_z` | Same-stock `delta_z` | Same-stock × 5m `delta_z` |
|---|---:|---:|---:|
| full | +.109441 | 約 -.2300 | 約 -.0560 |
| 2025 | +.147385 | -.250976 | -.057187 |
| 2026 | +.055341 | 約 -.2002 | 約 -.0544 |

也就是 shared-pool distribution shift 為正，但在同一檔、同一檔同 5 分鐘內，condition
rows 並沒有較佳 entry timing。這支持把 condition 當候選 universe／regime slice，
不支持把它直接解讀成同股進場 alpha。

沿用先前對相同 raw selector 的 12-session sparse-key overlap audit：

| 指標 | `fh_neg_rejection` vs negFill |
|---|---:|
| candidate ticks 落在 negFill | 32.486% |
| tick Jaccard | 2.827% |
| candidate stock-days 含 negFill | 98.928% |
| stock-day Jaccard | 29.243% |
| daily shared-effect corr vs `fh_neg_sample` | .582 |
| daily shared-effect corr vs `FillLots_atLow < 0` | .699 |

這個 overlap 是 prior raw-key diagnostic，不是本次 850,376 effect-valid events 的重新
配對；但它已足以否定「低 exact-tick overlap 就是獨立第二本書」。`fh_neg_rejection`
更像既有 bearish/negFill stock-day domain 的精細切片，部署前仍需 disjoint stock-day、
incremental portfolio、容量與重複吃單測試。

## `20260826`：non-pristine descriptive only

這一天不進任何 gate，不能和 retrospective pooled，也不能稱 prospective：

| 項目 | 結果 |
|---|---:|
| anchor events | 1,569 |
| shared `delta_z` | -.061691 |
| raw / residual uplift | `-55.107bp / -11.047bp` |
| D full-minus-LOO | -.152173 |
| G full-minus-LOO | -.005152 |
| B full-minus-LOO | -.019680 |

單日反號不推翻 376-session retrospective distribution，也不能被忽略成「只是一天」後
繼續部署。它的正確角色是提醒：真正 first-live 可能和歷史均值差很大，必須累積完整
prospective evidence。

## Gate-by-gate decision

| Gate | Retrospective status | Prospective / deployment status | 解讀 |
|---|---|---|---|
| G0 lineage + positive control | `PASS_WITH_DISCLOSED_LIMITATIONS` | 非 prospective effect gate | pipeline recovery 與 conservative lineage 可用，但限制仍在 |
| G1 support | `PASS_DIAGNOSTIC_ONLY` | `UNAVAILABLE_NOT_EVALUATED` | 歷史 support 足夠，未來 support 尚無 |
| G2 primary bias | `PASS_DIAGNOSTIC_ONLY` | `UNAVAILABLE_NOT_EVALUATED` | 歷史 simultaneous LCB 與 residual uplift 為正，不可升格 |
| G3 temporal stability | `PASS_DIAGNOSTIC_ONLY` | `UNAVAILABLE_NOT_EVALUATED` | 3/3 同號，但 level decay 明顯 |
| G4 parsimony | `PASS_FROZEN_RULE` | 經濟 materiality 未建立 | D/B 實質；G 邊際很小 |
| G5 selection disclosure | `PASS_DISCLOSURE_LOWER_BOUND_ONLY` | 歷史上界未知 | current max-stat 不是全歷史 correction |
| G6 evidence class | n/a | `FAIL_CLOSED_NO_PROSPECTIVE_FIRST_LIVE_ARTIFACT` | 沒有 `Date > 20260827` qualifying artifact |
| G7 threshold plateau | `diagnostic_plateau_pass` | `NOT_PROSPECTIVE_BY_CONTRACT` | 周圍穩定，但 outcome-informed retrospective |

Final machine decision：`RESEARCH_SHADOW_ONLY`；`deployment_authorized=false`、
`deployment_handoff_authorized=false`、`production_promotion_authorized=false`。

## 下一個可證偽步驟

1. 不改 selector semantics、threshold 或 sampler，從 `Date > 20260827` 累積完整
   first-live sessions；缺資料、fingerprint drift 或 non-finite input 一律 fail closed。
2. prospective window 至少達 60 active sessions、3 active months、1,000 events 後，
   重新跑 shared-pool max-stat、三個 outcome-blind calendar blocks、raw/residual BP、
   prevalence 與 D/G/B paired increments；不能用本次歷史資料補 support。
3. 若要測不含 G 的簡化版，先建立新 selector version、凍結 acceptance，再開始新的
   prospective clock；不得把本次 `D+B` ablation 當部署 winner。
4. condition 穩定後才在其樣本內接 prior-only rolling model，另報 outer first-live
   model lift；實盤部位、每檔上限、同檔 event 合併、成本與容量是新的策略契約。
5. 因 stock-day 與 negFill 高度重疊，部署評估需加入 disjoint stock-day 與 incremental
   combined-book test，不能只看 tick Jaccard。

## Reproduce 與驗證

完整 raw materialization 會讀取 377 個 session；已有 artifacts 時，以下兩個指令會重新
驗證所有 frozen lineage／摘要回接與 fail-closed machine decision：

```bash
uv run python -m unittest discover \
  -s src/research/timeSeries/test/20260827_fh_neg_rejection_robustness \
  -p 'test*.py' -v

uv run python \
  src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/build_decision_artifact.py
```

本次驗證為 89/89 tests 通過；decision builder 連續執行的輸出 SHA-256 相同：

```text
decision.json    eaa0dfaf56907dd314faa54a28222884855fe5edc4f1ec8481e9161d6f1943ee
report_data.json e8839a6fe5fab151243c202c4751949b81f5267c77aa9356caa61f44698436d0
```

若從 frozen daily aggregates 重建兩個 inference family，使用：

```bash
uv run python \
  src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/evaluate_robustness.py \
  --daily-aggregates src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/dgb_factorial_v1/daily_stats.parquet \
  --requested-calendar src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/calendar_retrospective_v1/requested_calendar.parquet \
  --output-dir src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/inference_retrospective_v1

uv run python \
  src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/evaluate_threshold_plateau.py \
  --daily-aggregates src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/threshold_plateau_v1/daily_stats.parquet \
  --plateau-run-manifest src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/threshold_plateau_v1/run_manifest.json \
  --requested-calendar src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/calendar_retrospective_v1/requested_calendar.parquet \
  --requested-calendar-manifest src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/calendar_retrospective_v1/requested_calendar.json \
  --output-dir src/research/timeSeries/test/20260827_fh_neg_rejection_robustness/artifacts/threshold_plateau_evaluation_v1
```

## Canonical artifacts

- [final decision](../../../test/20260827_fh_neg_rejection_robustness/artifacts/final_decision_v1/decision.json)
- [machine-readable report data](../../../test/20260827_fh_neg_rejection_robustness/artifacts/final_decision_v1/report_data.json)
- [study contract](../../../test/20260827_fh_neg_rejection_robustness/study_contract.json)
- [methods amendment 1](../../../test/20260827_fh_neg_rejection_robustness/study_contract_amendment_1.json)
- [methods amendment 2](../../../test/20260827_fh_neg_rejection_robustness/study_contract_amendment_2.json)
- [factorial inference](../../../test/20260827_fh_neg_rejection_robustness/artifacts/inference_retrospective_v1/)
- [lineage support](../../../test/20260827_fh_neg_rejection_robustness/artifacts/lineage_retrospective_v1/lineage_support_summary.json)
- [positive-control reference](../../../test/20260827_fh_neg_rejection_robustness/positive_control_reference.json)
- [historical trial exposure](../../../test/20260827_fh_neg_rejection_robustness/historical_trial_exposure.json)
- [threshold plateau contract](../../../test/20260827_fh_neg_rejection_robustness/threshold_plateau_contract.json)
- [threshold plateau decision](../../../test/20260827_fh_neg_rejection_robustness/artifacts/threshold_plateau_evaluation_v1/plateau_decision.json)
- [static BackTest companion](STATIC_BACKTEST.md)
- [static BackTest verification](../../../test/20260827_fh_neg_rejection_robustness/artifacts/static_backtest_retrospective_v1/verification.json)
- [prior overlap attribution report](../20260826_filllots_at_high_feature_bias/REPORT.md)

研究程式與 tests 位於 paired
[study directory](../../../test/20260827_fh_neg_rejection_robustness/)。本 study 沒有修改
正式 `timeSeries` point pipeline、production strategy 或 sizing contract。
