# Stop cascade risk (盤中被嘎檔數 → 後續新空單 / 既有部位)

輸入 `src/research/timeSeries/signalwithRisk_df.parquet`（notebook 風控後成交清單，
每列 1 張以 `BidPrice1` 賣出的空單），用 tickData 還原每檔當日第一次停損時間
`T_k`（首次進場後第一筆成交價 ≥ `RefPrice × stop_ratio`；出場價 = 停損價 + 2 tick）。

- **Q1**：進場前已被嘎 `k` 檔的新空單損益（window：第 k～k+1 檔之間；cumulative：≥k）。
  有 K 檔停損的交易日在 day panel 貢獻 k=1..K 各一筆。
- **Q2**：每個 `T_k` 當下仍在手上的部位，以 `T_k` 的 as-of mid 對收盤的損益
  （`bp_close`），另附之後被嘎者改以停損價結算的 `bp_settle`。
- 每張表同時給逐筆平均 BP、部位 BP（名目加權）、等日平均與總損益（萬，Q1 扣費）。

```bash
uv run python src/research/timeSeries/test/20260922_stop_cascade_risk/run_stop_cascade.py
uv run python src/research/timeSeries/test/20260922_stop_cascade_risk/run_stop_cascade.py --stop-ratio 1.09
```

Artifacts（不進版控）：`artifacts/stop<ratio>/`：`stops.parquet`、`q1_rows.parquet`、
`q2_rows.parquet`、`q1_window_summary.csv`、`q1_cumulative_summary.csv`、`q1_reference.csv`、
`q1_day_panel.csv`、`q2_summary.csv`、`q2_day_panel.csv`、`summary.md`。

結論見 [REPORT.md](../../Doc/studies/20260922_stop_cascade_risk/REPORT.md)。
