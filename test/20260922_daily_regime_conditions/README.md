# Daily regime conditions（日損益 vs D 盤前可知的日級條件）

把 `signalwithRisk_df.parquet` 依 Date 組成日損益（cell 38 口徑：每列 1 張、扣 19.3bp），
對每個交易日算 D 盤前（開盤類為開盤價）可知的條件，做三種分析：

- **A. trailing qcut10**：嚴格在 D 之前 120 個交易日的條件值十分位（`--window-tdays`）。
- **B. in-window qcut10**：損益窗內自身十分位（描述用）。
- **C. MA cross**：條件序列 5MA vs 20MA、5MA vs 60MA 的黃金／死亡狀態（`--ma-pairs`），
  分 fresh（交叉後 ≤5 天）／established，並以連續同狀態區段（episode）為單位重看。

條件（top-100 = `preMarket(D).day_amount_rank < 100`，即 D-1 成交金額前 100）：

| 條件 | 定義 | 來源 |
|---|---|---|
| `top_open_low_share` / `top_open_ret_mean` | top-100 中 D 開盤 < 參考價的比例 / 平均開盤漲跌幅 | marketData(D) |
| `top_dt_buy_count` | top-100 中 D-1 隔日沖分點淨買 > 0 的檔數 | brenchFeature(D) `dt_major_net_buy_amount` |
| `top_dt_net_amount` / `all_dt_net_amount` | 隔日沖分點 D-1 買超−賣超金額合計（top-100 / 全部） | brenchFeature(D) |
| `limit_touch_rate_lag1`（及 `_ma5`） | D-1 全市場普通股摸漲停（high ≥ 漲停價）比例 | marketData |
| `limit_close_over_touch_lag1`（及 `_ma5`） | D-1 實際漲停（close ≥ 漲停價）÷ 摸漲停檔數 | marketData |
| `top_limit_touch_rate_lag1`（及 `_ma5`） | 同上，universe = 當時 top-100 | marketData |
| `top_large_cap_share` / `top_mega_cap_share` | top-100 中市值 ≥ 1000 億 / 5000 億 的比例 | preMarket(D) `MarketCap` |
| `top_cap100_share` | top-100 中同時是市值前 100（普通股內排名）的比例 | preMarket(D) |

```bash
uv run python src/research/timeSeries/test/20260922_daily_regime_conditions/run_daily_regime.py
uv run python src/research/timeSeries/test/20260922_daily_regime_conditions/run_daily_regime.py --window-tdays 250 --ma-pairs 5:20,10:60
```

Artifacts（不進版控）：`artifacts/`（`daily_pnl.csv`、`conditions.csv`、`decile_panel.csv`、`decile_*.csv`、
`overview.csv`、`condition_drift.csv`、`ma_cross_overview.csv`、`ma_cross_*.csv`、`ma_cross_episodes_*.csv`、
`monthly_lo_decile_share.csv`、`summary.md`）。

結論見 [REPORT.md](../../Doc/studies/20260922_daily_regime_conditions/REPORT.md)。
