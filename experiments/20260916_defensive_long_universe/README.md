# 20260916 defensive stock-day list as a long universe

Question: does restricting the intraday taker-long side to the products picked by
`DEFENSIVE_STOCK_SELECTION.md` (D-1 theme-strength top-20% themes, capacity
weighted) lift the long label enough to matter?  Two views, both retrospective:

1. **Pool level** - the pinned long eligible pool
   (`tw_equity_daytrade_rank150_090030_123000_bidref_v1` x
   `benchmark.long_taker_tp109.v1`) aggregated per stock-day; the defensive list
   is treated as a stock-day condition and contrasted against the full pool the
   same way the framework's shared-pool primary estimand is computed.
2. **Book level** - the kept six-candidate taker-long ledger from
   `20260827_netfill_long_cover` split by the same stock-day groups.

Stock-day groups: `defensive` (top quintile themes), `weak` (bottom quintile),
`both` (top and bottom through different themes), `middle` (only non-quintile
themes), `none` (no valid theme that day).  `middle` separates the directional
score from plain theme-membership.

Conclusions and numbers live in
[`report/studies/20260916_defensive_long_universe/REPORT.md`](../../report/studies/20260916_defensive_long_universe/REPORT.md).

## Files

| File | Role |
|---|---|
| `defensive_list.py` | Rebuild the point-in-time stock-day list from the frozen `news_collector/artifacts/theme_strength` panels (no D outcome used, no reliability gate). |
| `run_pool_stockday.py` | Materialize the long eligible pool day by day and keep stock-day sums (`artifacts/v1/pool_stockday/`). |
| `report_pool_bias.py` | Shared-pool / pairwise / weight-tier contrasts with HAC(5) (`artifacts/v1/pool_bias/`). |
| `run_ledger_split.py` | Split the six-candidate static long book by group (`artifacts/v1/ledger_split/`). |
| `study_stats.py` | HAC mean test, calendar Sharpe, max drawdown. |

## Reproduce

```bash
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/defensive_list.py
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/run_pool_stockday.py --start 20250211 --end 20260810
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/report_pool_bias.py
uv run python src/research/intradaySignal/experiments/20260916_defensive_long_universe/run_ledger_split.py
```

Artifacts are study-local and rebuildable; nothing here touches `data_index.parquet`,
`data/`, `baseline/` or the `news_collector` artifacts.
