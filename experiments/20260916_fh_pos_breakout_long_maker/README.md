# 20260916 fh_pos_breakout maker-long signal (defensive universe)

Tradingized long follow-up to
[`20260916_defensive_long_universe`](../20260916_defensive_long_universe/README.md):
the frozen `fh_pos_breakout` selector, a **maker** Bid1 entry gated by the
`makerFill` queue estimate, prior-only rolling models, 16-cell prior thresholds,
and a pure-long BackTest.  It is a fork of
[`20260826_filllots_at_high_signal`](../20260826_filllots_at_high_signal/) with the
direction, entry, fill model and universe swapped; everything else is inherited
so the two lines stay comparable.

Conclusions live in
[`report/studies/20260916_fh_pos_breakout_long_maker/REPORT.md`](../../report/studies/20260916_fh_pos_breakout_long_maker/REPORT.md).

## Contract summary

| Layer | Choice |
|---|---|
| Event | `fh_pos_breakout` (family v1, thresholds frozen in `framework/profiles/filllots_at_high.py`) |
| Universe | daytrade-allowed, rank <= 150, 09:00:30-12:30, Bid-to-Ref in (-1.5%, 5%) |
| Label | `fhl.long_maker_tp109.v1`: buy at BidPrice1, exit `1.09 x RefPrice` if FutureHigh crosses, else 13:30 Close; beta x TXF hedge and residual-vol z only for model targets |
| Fill gate | order placed 100ms after the event; `Bid1_FillSeconds` (or `Bid2_FillSeconds` if the bid became level 2; 0s if undercut) must be `< 600s`; unknown = no trade. Training and execution use the same gated population |
| Features | short-line manifest (38) + `in_top_list`, `defensive_weight` from the D-1 defensive list |
| Models | causal-v2 rolling Ridge: timing (60d, within-stock-day z) and selection (20d, within-day z), alpha 0.1, position-capped training sample |
| Thresholds | prior 3-session quantile per (time bucket x ToRef bucket); q = 0.30 / 0.10 (inherited, not tuned) |
| Variants | {static_only, timing_only, selection_only, dual} x {all, top_list}; selection on development 2025 only |
| Execution | shared static gate (OTC or MD_L1Rate>0.25, SpreadPairElapsed>0.1, rank<=100, price<=1000, maker spread>-70), per-stock one-lot depth cap and 200 (10k TWD) cap |
| BackTest | `BackTest(side='long', entry_price_col='BidPrice1', bp_col='label_raw_bp', fee=19.3)`, no hedge |
| Calendar | 2025-02-11 to 2026-08-10 (defensive list coverage); development ends 2025-12-31 |

## Files

| File | Role |
|---|---|
| `study.py` | frozen contracts, feature manifest, fill-seconds builder, defensive join, gates |
| `build_dataset.py` | event index -> causal dataset -> `artifacts/maker_fill_v1/` |
| `run_model_backtest.py` | model input, models, thresholds, 8 variants, fee and fill-horizon sensitivity, BackTest plots, run manifest |

## Reproduce

```bash
uv run python src/research/intradaySignal/experiments/20260916_fh_pos_breakout_long_maker/build_dataset.py --stage all
uv run python src/research/intradaySignal/experiments/20260916_fh_pos_breakout_long_maker/run_model_backtest.py --n-jobs 8
```

Requires the defensive list from
`20260916_defensive_long_universe/artifacts/v1/defensive_stock_daily.parquet`
(`uv run python .../20260916_defensive_long_universe/defensive_list.py`).
