# Beta / residual vol lookback sensitivity（60／20／5 日）

把 preMarket 的 `txf_beta_60d`、`txf_residual_vol_0900_1330_60d` 改用 20、5 日窗口重算，
看 beta 與 vol 對窗口長度有多敏感。直接沿用正式的 `estimate_txf_beta` 與
`residual_vol_time_scale`，只換 `MarketBetaConfig`。不動 preMarket、`data/`、
`data_index.parquet`。

| 窗口 | lookback | min_observations | prior_strength | 滿窗 shrink 權重 |
|---|---|---|---|---|
| 60 | 60 | 20 | 20 | 0.75（正式設定） |
| 20 | 20 | 7 | 20/3 | 0.75 |
| 5 | 5 | 4 | 5/3 | 0.75 |

短窗口讓 prior 等比例縮小，滿窗時 shrink 權重和正式設定一樣，比較的就只是窗口長度。
5 日的下限提高到 4 筆，確保 residual std 至少有 3 個自由度。日期契約和 preMarket
相同：決策日 T 用來源日 S（T 的前一個交易日）往前、含 S 的最後 N 個 marketData 檔。

## 欄位

`artifacts/v1/lookback_beta_vol.parquet`：每個 (Date=決策日, QuoteCode) 一列，
QuoteCode 取自 timeSeries `data/` 當天出現的股票。

- `txf_beta_{N}`：applied beta（shrink、clip，缺值補 1.0，與正式相同）
- `txf_beta_raw_{N}`、`txf_corr_{N}`、`txf_obs_{N}`：未 shrink beta、相關係數、有效觀測數
- `txf_residual_vol_0900_1330_{N}`：完整時段 residual vol
- `window_full_{N}`：來源日前是否已有 N 個 marketData 檔（2023 初為 False）

`attach_lookback_variants` 把表接到事件上，加上 `txf_beta_{N}` 與依
`RemainSeconds_1330` 縮放後的 `txf_residual_vol_to_1330_{N}`。輸入可以是 pandas 或
polars，回傳同型別、列順序不變：

```python
import sys
sys.path.insert(0, "test/20260923_beta_vol_lookback")  # notebook cwd = src/research/timeSeries
from lookback_features import attach_lookback_variants

signal_df = attach_lookback_variants(signal_df)
# 新增 txf_beta_60/20/5 與 txf_residual_vol_to_1330_60/20/5
```

`txf_beta_60`、`txf_residual_vol_to_1330_60` 與既有的 `txf_beta_60d`、
`txf_residual_vol_to_1330` 逐值相同（parity 檢查見 report）。

## 重跑

```bash
uv run python src/research/timeSeries/test/20260923_beta_vol_lookback/run_lookback_sensitivity.py
uv run python src/research/timeSeries/test/20260923_beta_vol_lookback/run_lookback_sensitivity.py --rebuild  # data/ 有新日期時
uv run python src/research/timeSeries/test/20260923_beta_vol_lookback/run_lookback_sensitivity.py --start 20250101 --end 20251231
```

表已存在時只重算評估；`--rebuild` 重建整張表（約 1.5 分鐘）。

Artifacts（不進版控）：`artifacts/v1/` 下的 `lookback_beta_vol.parquet`、
`parity_by_date.csv`、`agreement_by_date.csv`、`nextday_by_date.csv`、
`event_by_date.csv`、`event_by_bucket.csv`、`summary.md`。

結論見 [report](../../Doc/studies/20260923_beta_vol_lookback/REPORT.md)。
