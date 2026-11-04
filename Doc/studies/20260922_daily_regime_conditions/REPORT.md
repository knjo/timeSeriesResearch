# Daily regime conditions：日損益 vs D 盤前可知的日級條件

- Status: `scoped` / `RESEARCH_DIAGNOSTIC_ONLY`
- Study code: [`test/20260922_daily_regime_conditions/`](../../../test/20260922_daily_regime_conditions/)
- Input: `signalwithRisk_df.parquet`（2025-10-15 ～ 2026-09-21，228 天）；日損益照 notebook cell 38：
  每列 1 張、`pnl = Σ BidPrice1 × (TakerSell_CloseBP − 19.3) / 1e4 / 10` 萬、部位 = `Σ BidPrice1 / 10` 萬、`ret_bp = pnl / 部位`。
- 全期：總損益 11,250 萬、日均 49.3 萬、日均 57.8bp（日標準差 206bp）、勝率 0.618。
  月別 day_ret_bp：2025-10 +57、11 +102、12 +68、2026-01 +28、02 +20、03 +73、04 +50、05 +39、06 +100、07 +140、**08 −13、09 −9**。
- 決策口徑：D 盤前決定今天打不打；開低比例／開盤漲跌幅用 D 開盤價（策略開盤 30 秒後才進場），其餘都是 D-1 以前的資料。

## 條件

top-100 = `preMarket(D).day_amount_rank < 100`（D-1 成交金額前 100）。

| 條件 | 定義 |
|---|---|
| `top_open_low_share` / `top_open_ret_mean` | top-100 中 D 開盤 < 參考價的比例 / 平均開盤漲跌幅 |
| `top_dt_buy_count` | top-100 中 D-1 隔日沖分點淨買 > 0 的檔數（brenchFeature(D) `dt_major_net_buy_amount`） |
| `top_dt_net_amount` / `all_dt_net_amount` | 隔日沖分點 D-1 買超−賣超金額合計（top-100 / 全部） |
| `limit_touch_rate_lag1`（及 `_ma5`） | D-1 全市場普通股 high ≥ 漲停價 的比例（及 D-5～D-1 五日均） |
| `limit_close_over_touch_lag1`（及 `_ma5`） | D-1 收盤 ≥ 漲停價 檔數 ÷ 摸漲停檔數 |
| `top_limit_touch_rate_lag1`（及 `_ma5`） | 同上，universe = 當時 top-100 |
| `top_large_cap_share` / `top_mega_cap_share` | top-100 中市值 ≥ 1000 億 / 5000 億 的比例 |
| `top_cap100_share` | top-100 中同時是市值前 100（普通股內排名）的比例 |

三種分析：(A) trailing qcut10：對每個 D 用嚴格在 D 之前 120 個交易日（≥60）的條件值算十分位；
(B) in-window qcut10：228 天自身十分位（描述用）；(C) MA cross：條件序列的 5MA vs 20MA、5MA vs 60MA，
黃金 = 短均 > 長均，死亡 = 短均 < 長均，fresh = 交叉後 ≤5 天。每分位約 23 天，分位均值標準誤 ≈ 43bp。

## A. trailing-120 日 qcut10：仍無不利分位

- 水位漂移比一年視窗小但仍在：隔日沖／漲停率／市值佔比類條件有 86～186 天落在 q8–9（228 天）。
  `top_cap100_share`、`limit_close_over_touch`、開盤類的分位是平衡的。
- 所有條件 Spearman(條件, 日報酬) |ρ| ≤ 0.09；n ≥ 10 的最差分位全部 |t| < 1，沒有分位顯著為負。
  `top_cap100_share` 最差分位（q6，40 天）+0.5bp、t 0.02；in-window 十分位同樣沒有單調關係。
- 一年視窗版（前一次結果）看似顯著好的低分位，全部集中在 2025-10～12，是期間效果（`monthly_lo_decile_share.csv`）。

## C. MA cross：只有隔日沖分點族群有一致方向

`ma_cross_overview.csv`，按 Welch t 排序的前段：

| 條件 | pair | 黃金 n / bp / 萬 | 死亡 n / bp / 萬 | 黃金−死亡 bp | Welch t | 段數 黃金/死亡 | 死亡段為正比例 |
|---|---|---|---|---|---|---|---|
| `top_dt_buy_count` | 5/20 | 121 / 81.0 / 8918 | 107 / 31.5 / 2331 | +49.5 | 1.82 | 12 / 15 | 11/15 |
| `top_dt_buy_count` | 5/60 | 132 / 78.4 / 8940 | 96 / 29.4 / 2310 | +49.0 | 1.83 | 9 / 9 | 6/9 |
| `all_dt_net_amount` | 5/20 | 129 / 74.2 / 8930 | 99 / 36.5 / 2320 | +37.7 | 1.38 | 13 / 11 | 8/11 |
| `top_dt_net_amount` | 5/60 | 128 / 73.8 / 8349 | 100 / 37.4 / 2900 | +36.4 | 1.36 | — | — |
| `top_dt_net_amount` | 5/20 | 131 / 73.0 / 8926 | 97 / 37.3 / 2324 | +35.7 | 1.30 | 11 / 11 | 8/11 |
| `top_mega_cap_share` | 5/20 | 156 / 71.3 / 9121 | 70 / 30.3 / 2170 | +41.0 | 1.38 | — | — |

其餘 16 組 |t| ≤ 1.2，其中 `top_cap100_share` 5/20 與 5/60 方向相反（−24 / +16bp），`top_large_cap_share`、
`top_limit_touch_rate_lag1`、`top_open_ret_mean` 也都是死亡 ≥ 黃金，無訊號。

隔日沖族群六組方向一致（分點買進活動上升 → 空單較好），但：
- 日層級 t 最高 1.8，且是 22 組條件×均線組合裡挑出來的，經多重比較後不顯著。
- 段層級：`top_dt_buy_count` 5/20 死亡 15 段平均 +26bp，其中 11 段仍為正；死亡段在 2025-11～12 是 +120／+91bp，
  在 2026-08～09 是 −90／−53bp——狀態抓到「分點退潮」但不能穩定指出不利日。
- 唯一接近 0 的 pocket：`top_dt_buy_count` 5/20 **死亡已確立（≥6 天）** 39 天：−0.9bp、部位 −14bp、勝率 0.49、−363 萬
  （fresh 死亡 68 天仍 +50bp）。5/60 的對應 pocket 為 +28bp、t 0.96，不重現。

## 結論

- 六組（含市值前 100 佔比）條件的 120 日 trailing qcut10 都挑不出不利交易日；最差分位期望 ≈ 0bp，|t| < 1。
- 均線交叉只有隔日沖分點活動有方向（黃金 +73～81bp vs 死亡 +29～37bp），日層級 t 1.3～1.8、段層級死亡段多數仍為正，
  只能當「降部位候選」，不能當「不打」的開關；若要進一步，用 2026-08 以後的資料做 forward shadow，
  並先凍結 `top_dt_buy_count` 5/20、established ≥6 天這一組，不再挑組合。
- 真正不利的期間（2026-01～02、08～09）在這些條件裡沒有極端水位，開盤前的日級條件在此樣本無法預警。

## 限制

- 228 天、每分位 23 天，只能偵測 ≥ 90bp 的分位差；MA cross 的有效樣本是 9～15 段而非日數。
- 條件與日報酬都未做市場（TXF）調整。
- 22 組條件×均線組合一次比較，未做選擇偏誤校正；隔日沖族群的結果應視為候選假設。
- 成交清單是事後篩選（notebook cell 32），不是實盤成交。

## Reproduce

```bash
uv run python src/research/timeSeries/test/20260922_daily_regime_conditions/run_daily_regime.py
uv run python src/research/timeSeries/test/20260922_daily_regime_conditions/run_daily_regime.py --window-tdays 250   # 一年視窗
```

Artifacts：`test/20260922_daily_regime_conditions/artifacts/`（`daily_pnl.csv`、`conditions.csv`、`decile_panel.csv`、
`decile_<cond>.csv`、`decile_inwindow_<cond>.csv`、`overview.csv`、`condition_drift.csv`、`ma_cross_overview.csv`、
`ma_cross_<cond>_<pair>.csv`、`ma_cross_episodes_<cond>_<pair>.csv`、`monthly_lo_decile_share.csv`、`summary.md`）。
