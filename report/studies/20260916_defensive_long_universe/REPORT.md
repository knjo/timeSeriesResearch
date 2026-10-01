# 抗跌商品清單作為日內多單 universe

狀態：`scoped`（retrospective universe diagnostic，無 prospective first-live）  
研究日期：2026-09-16  
資料區間：2025-02-11 至 2026-08-10（365 sessions；受 `theme_strength` 產物 2026-08-10 截止限制）  
方向：long only  
Primary estimand：stock-day condition 對 long taker eligible pool 的 shared-pool equal-day `Delta`

## 結論先講

把 [`DEFENSIVE_STOCK_SELECTION.md`](../../../DEFENSIVE_STOCK_SELECTION.md) 的
D-1 top-20% 族群清單（point-in-time 重建，無 reliability gate）當成 stock-day
condition 加在多方 taker pool 上：

1. **bp 口徑有穩定的正向分佈平移，但 z 口徑幾乎為零。** 365 日 shared-pool
   equal-day `Delta raw = +15.4bp`（HAC(5) t=2.86，13/19 月正）、stock-equal
   `+22.3bp`（t=4.60）、residual 同量級；2025／2026 同號。`Delta z` 只有
   `+0.024～+0.028`（t 0.9～1.3）。依本模組「切割力必用 residual z」的驗收，這不算
   condition 切割力；它是「低殘差波動、少跌」的組成效果，不是 risk-adjusted alpha。
2. **絕對水位仍深負。** defensive 群的 taker-long raw 為 `-29.6bp`（pool
   `-44.9bp`），2026 `-33.9bp`；離扣費損益兩平還差 ~50bp。單靠 universe 不能成書。
3. **方向分數有超出「有新聞題材」的增量，但只在強的一端。** defensive − middle
   `+17.8bp`（t=2.79）、defensive − weak `+12.7bp`（t=1.68）、weak − middle 只有
   `+5.1bp`（t=0.89）；bottom quintile 並不比中段差。沒有題材的 `none` 群最差
   （`-56.7bp`，2025 貢獻全部，2026 已不顯著）。
4. **與 `fh_pos_breakout` 疊加是唯一浮出水面的組合。** 在 `20260827_netfill_long_cover`
   的六條 static taker-long ledger 裡，defensive 群在六條全部改善淨 bp；
   `fh_pos_breakout x in-top-list` 3,417 筆／810 stock-days：full gross `-10.0` /
   net `-29.3bp`，2025 net `-51.6`，**2026 net `+12.3bp`（1,263 筆，Sharpe 0.62，6/8 月正）**；
   高 `defensive_weight` 三分位 1,138 筆：full net `-7.6bp`（幾乎損益兩平），2026
   net `+46.9bp`（383 筆，Sharpe 1.71，5/8 月正）。這條線每日只有 ~9 筆、平均
   進場名目 ~120 萬，2026 的正值主要由 2026-01 貢獻。

判定：**universe 有效但不足；可作 model-after-condition 的 universe／feature，不能
單獨或直接當交易規則。** 沒有 promotion、沒有 deployment handoff。

## 問題與事前口徑

- 問題：在既有多方 objective（direction-specific barrier-or-close label、shared-pool
  primary estimand）下，D-1 抗跌清單能否把多方分佈推到有利方向、以及能否把已知
  的 `fh_pos_breakout` 多單 book 推過費用。
- 事前 hypothesis：清單本身是 D open-to-close 的相對報酬證據（long-minus-market
  `+14.93bp/日`），預期在日內 taker pool 上出現同量級 bp 平移；是否能出現 z 平移
  與絕對正報酬則未知。
- Condition 是 stock-day 層（`(decision_date, quote_code)`），不是 tick event，因此
  沒有 same-stock／same-stock-time matched 診斷；取而代之的是 `middle` 對照組，用來
  分離「方向分數」與「有題材」兩個效應。
- 群組：`defensive`（top quintile 族群成員）、`weak`（bottom quintile）、`both`
  （透過不同族群同時在兩端）、`middle`（只在中段族群）、`none`（當日無有效族群）。

## Lineage

- Universe 來源：`src/research/news_collector/artifacts/theme_strength/`
  （`theme_daily` / `membership_daily` / `stock_daily`）。清單由
  [`defensive_list.py`](../../../test/20260916_defensive_long_universe/defensive_list.py)
  依 spec §4 重建；與凍結報告的差異只在不以 D outcome 缺值剔除股票（point-in-time），
  2025 起 387 日的 top-side capacity-weighted 日報酬對帳平均 `-11.52` vs 報告
  `-11.47bp`，最大單日差 13.5bp 來自 25 個無 D OHLC 的 stock-day。
- Pool／label：`tw_equity_daytrade_rank150_090030_123000_bidref_v1` x
  `benchmark.long_taker_tp109.v1`（同 tick Ask taker 進場、`1.09 x RefPrice` 停利否則
  13:30 Close、prior beta x TXF 對沖、prior residual vol 標準化），由
  `framework.condition_study.build_eligible_labeled_pool` 逐日物化，只保留 stock-day
  sums（[`run_pool_stockday.py`](../../../test/20260916_defensive_long_universe/run_pool_stockday.py)）。
  365/366 requested sessions 有效（2025-04-07 pool 為 0）。
- Book：`20260827_netfill_long_cover/artifacts/static_long_cover_v2/accepted_signals.parquet`
  （contract `3e33e539...`），限制到清單覆蓋日曆後 306,630 筆。

## Pool 層主結果

覆蓋（每日平均）：defensive 15.0 檔／13.5% rows，weak 24.8 檔／25.4%，middle 48.8 檔／
36.3%，none 38.2 檔／19.7%，both 3.9 檔／5.4%。

Shared-pool `Delta`（群減完整 pool，日內 row-weighted，日等權，HAC lag 5）：

| group | period | days | Δraw bp (t) | Δresid bp (t) | Δz (t) | Δz 正月 |
|---|---|---:|---:|---:|---:|---:|
| defensive | full | 365 | +15.39 (2.86) | +15.03 (3.14) | +0.024 (0.91) | 10/19 |
| defensive | 2025 | 221 | +18.41 (3.50) | +18.05 (3.47) | +0.023 (0.65) | 6/11 |
| defensive | 2026 | 144 | +10.75 (0.99) | +10.39 (1.15) | +0.026 (0.65) | 4/8 |
| both | full | 344 | +11.17 (1.92) | +13.89 (2.57) | -0.031 (-0.86) | 8/19 |
| weak | full | 365 | +2.64 (0.65) | +1.83 (0.49) | +0.012 (0.58) | 11/19 |
| middle | full | 365 | -2.44 (-0.81) | -1.78 (-0.60) | +0.011 (0.70) | 11/19 |
| none | full | 365 | -11.76 (-3.17) | -12.19 (-3.33) | -0.045 (-2.85) | 4/19 |
| none | 2026 | 144 | -0.89 (-0.13) | -1.60 (-0.23) | +0.013 (0.54) | 4/8 |

Stock-equal 版本（每 stock-day 等權）defensive full `+22.26bp`（t=4.60）、2025
`+21.60`（5.82）、2026 `+23.28`（2.14）；z 仍只有 `+0.028`（1.30）。

絕對水位（日等權 row-weighted raw bp）：defensive `-29.6`、both `-34.7`、weak `-42.3`、
pool `-44.9`、middle `-47.4`、none `-56.7`；2026 分別為 `-33.9 / -30.2 / -47.8 / -44.7 /
-54.5 / -45.6`。z 水位全群 `-0.13～-0.24`。

Pairwise（同日兩群差）：

| pair | full Δraw (t) | 2025 | 2026 | full Δz (t) |
|---|---:|---:|---:|---:|
| defensive − middle | +17.8 (2.79) | +16.0 (2.26) | +20.6 (1.72) | +0.013 (0.38) |
| defensive − weak | +12.7 (1.68) | +12.0 (1.44) | +13.9 (0.97) | +0.012 (0.32) |
| weak − middle | +5.1 (0.89) | +4.0 (0.69) | +6.7 (0.59) | +0.001 (0.03) |
| middle − none | +9.3 (1.76) | +21.2 (3.99) | -9.0 (-0.95) | +0.056 (2.28) |
| defensive − none | +27.1 (3.78) | +37.2 (6.30) | +11.6 (0.76) | +0.069 (2.15) |

`defensive_weight` 三分位（pooled cutpoints）：w_high Δraw `+20.6`（t=2.76；2025 +16.8 /
2026 +26.7，絕對 `-23.3bp`），w_low `+11.4`（1.53；2026 翻負），w_mid `+4.0`（0.59）。
高權重＝小而強的族群中的高容量股（中位價 80 vs 68、rank 32 vs 48、OTC 6% vs 12%）。

## Book 層（六條 static taker-long ledger）

費用 19.3bp、一張／事件、原 ledger 的每檔 cap；日曆 365 日補零。

| candidate | all net bp | defensive | both | weak | middle | none | 日級 def−rest Δnet bp (t) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fh_pos_breakout` | -65.1 | **-30.3** | -26.2 | -61.5 | -69.4 | -83.2 | +38.7 (3.17) |
| `fh_pos_depth2` | -66.3 | -40.5 | -39.3 | -57.7 | -66.9 | -91.0 | +24.9 (2.34) |
| `fh_pos_sample` | -77.2 | -58.8 | -33.6 | -63.3 | -77.3 | -103.3 | +18.2 (1.80) |
| `depth_gt_pos1` | -114.5 | -105.8 | -62.7 | -86.8 | -126.8 | -125.5 | +35.0 (2.56) |
| `depth_gt_pos2` | -105.4 | -99.1 | -21.2 | -95.2 | -113.0 | -110.7 | +36.3 (2.48) |
| `depth_gt_pos4` | -97.2 | -94.4 | -44.8 | -84.6 | -97.7 | -112.6 | +39.7 (2.94) |

六條都是 defensive > middle > none；`fh_*` 家族另有 defensive > weak，`depth_gt_*`
家族則 weak > defensive。日級 Δz（defensive − rest）只有 `fh_pos_breakout` `+0.12`
（t=1.74）與 `depth_gt_pos1/4` `+0.15～0.17`（t≈2.8）為正，其餘 ≈ 0。

`fh_pos_breakout` 在 top-list（defensive ∪ both）的年度切片：

| slice | trades | stock-days | gross bp | net bp | mean z | net PnL 萬 | Sharpe | MDD 萬 | 正月 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 3,417 | 810 | -10.0 | -29.3 | +0.13 | -129.6 | -1.92 | -158.2 | 7/19 |
| 2025 | 2,154 | 526 | -32.3 | -51.6 | +0.02 | -148.6 | -4.15 | -158.2 | 1/11 |
| 2026 | 1,263 | 284 | +31.6 | +12.3 | +0.32 | +19.0 | +0.62 | -34.2 | 6/8 |
| w_high full | 1,138 | 267 | +11.7 | -7.6 | -0.03 | -12.1 | -0.39 | -38.2 | 7/19 |
| w_high 2026 | 383 | 87 | +66.2 | +46.9 | +0.08 | +23.7 | +1.71 | -7.5 | 5/8 |

2026 逐月（defensive 群 net bp）：01 `+102`、02 `-24`、03 `-36`、04 `+3`、05 `+6`、
06 `+42`、07 `-22`、08 `-86`（45 筆）。正值集中在一月，之後在零附近擺動。

## 解讀

- 這份清單對多方 pool 的作用符合它自己的定位：相對強弱 prior，平均少跌 15～22bp，
  但不是把 z 推正的 timing／selection alpha。多方的結構坑（taker 付 spread＋池內日內
  下漂 ~ -45bp）它只填了三分之一。
- 疊加到 `fh_pos_breakout`（唯一通過 gate 的多方 condition）後，兩個 composition
  bias 大致可加：book 淨 bp 由 -65 拉到 -29，2026 由 -30 拉到 +12。這是首次看到多方
  static book 有一個年度切片扣費為正，但樣本小（~9 筆／日）、以單月為主、且是
  outcome-informed 的事後切片，不能當 OOS。
- 高 `defensive_weight` 三分位在 pool 與 book 兩層都最穩，且已接近 taker 損益兩平；
  這暗示如果進場改 maker（省 spread）並用多方語義特徵做 prior-only 排序，這條線
  有機會過費用。但 w_high 同時偏大價位、低 rank、少 OTC，需排除「只是流動性效應」。

## 限制

- 全部 retrospective；清單 discovery 用 2023-2024、validation 2025 起，但本研究的
  疊加切片（2026、w_high）是在看過結果後選的，multiplicity 未校正。
- 清單只到 2026-08-10；2026-08-11 之後未評估。
- Pool 層 `Delta` 是 unconditional shared contrast（control 含 condition rows），
  stock-day condition 沒有 matched 對照；`middle` 是替代的 attribution。
- Book 層沿用空方鷹架的 execution gate 與 taker 進場；沒有 cooldown、latency、
  同檔持倉合併、portfolio cap、impact。
- 每日 pool 只有 ~15 檔 defensive 股票、`fh_pos_breakout x top-list` 每日 ~9 筆、
  ~120 萬名目，容量是硬上限。

## 下一個可證偽步驟

1. 凍結 `fh_pos_breakout` selector 與 `in-top-list` universe（不調 quintile、不調
   weight cutpoint），改 maker label（B1 進場、保留 1.09 停利、配 fill-rate 折扣），
   在 2026-08-11 之後的新日期做 first-live 觀察。
2. 在同一 universe 內以多方語義 tickFeature 做 strictly-prior 排序，報 all-condition
   baseline 之上的 lift；門檻用 prior percentile。
3. 用 `defensive_weight`、價位、rank、OTC 做 stratified 對照，確認 w_high 的增量不是
   單純流動性／大價位效應。

## Artifacts 與 reproduce

- [study README](../../../test/20260916_defensive_long_universe/README.md)
- `artifacts/v1/defensive_stock_daily.parquet`：point-in-time 清單（top／bottom／middle）。
- `artifacts/v1/pool_stockday/*.parquet`：365 日 stock-day sums。
- `artifacts/v1/pool_bias/`：`shared_pool_contrast.csv`、`absolute_levels.csv`、
  `pairwise_contrast.csv`、`defensive_weight_tiers.csv`、`coverage.csv`、
  `daily_group_effects.parquet`。
- `artifacts/v1/ledger_split/`：`summary.csv`、`day_contrast.csv`、`monthly.csv`、
  `defensive_weight_tiers.csv`。

```bash
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/defensive_list.py
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/run_pool_stockday.py --start 20250211 --end 20260810
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/report_pool_bias.py
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/run_ledger_split.py
```
