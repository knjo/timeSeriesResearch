# Fixed-entry TP/SL versus TP/no-SL/borrow comparison

Research diagnostic for the existing timeSeries short notebook. The two fixed
policies preserve every accepted entry, its price, and its one-lot size. No
selector, model, position cap, or cascade entry gate is optimized or rebuilt.

- Baseline: notebook 2-sigma TP and SL, SL capped at RefPrice * 1.08. TP is
  floored to the entry TickSize plus 0.001; SL is ceiled and exits with one
  additional entry tick. The original strict post-entry FillPrice touch engine
  is reused. A tie goes to SL. Neither hit means Close, rebuilt from prices.
- Alternative: retain TP, remove SL. Any subsequent TP hit wins, including a
  TP after the discarded SL. Otherwise Close > RefPrice * 1.09 uses collected
  nextDayOpen * 1.03; Close <= that threshold uses Close.
- Both pay 19.3 bp on entry notional. The 3% surcharge is already in the
  alternative effective exit price and is not charged a second time. This is
  the user's explicit hypothetical fee/exit rule, not a regulatory assertion.
- Only borrow rows require nextDayOpen. Validate it against the immediate next
  trading session's collected marketData.open_price; use that source if the
  existing label is missing. Missing/invalid opens and conflicting labels fail.
- Daily PnL is assigned to the entry date, as in BackTest. This is an exit-label
  comparison, not an overnight funding, cash settlement or mark-to-market replay.
- Two fixed candidates, one paired comparison, no parameter selection. Report
  all/month/year PnL, Sharpe, drawdown, transition attribution, borrow costs,
  worst stock-days, and a five-day paired block-bootstrap interval.

Run from project root (choose a new output path to repeat):

```bash
uv run --cache-dir /tmp/hft-uv-cache --no-sync python \
  src/research/timeSeries/test/20261007_stop_vs_borrow/run_comparison.py \
  --input src/research/timeSeries/signalwithRisk_df.parquet \
  --output src/research/timeSeries/test/20261007_stop_vs_borrow/artifacts/v1

uv run --cache-dir /tmp/hft-uv-cache --no-sync python -m unittest discover \
  -s src/research/timeSeries/test/20261007_stop_vs_borrow -p 'test_*.py' -v
```

The notebook has an independent pandas comparison cell after its main backtest
cell. It copies the current `signalwithRisk_df`, matches the original BackTest's
`Date > 20260101` window, and recomputes touch times on those exact rows. Never
assign the original `t.low_time` by integer index after sorting, filtering or
resetting the accepted ledger: the flags then belong to different signals.
The cell reports disagreements with cached `tp_hit`, the original versus rebuilt
baseline PnL, and missing next-open counts by date. Run this comparison cell
directly with the current accepted ledger; no model rerun is needed.

The diagnostic preserves the saved comparison's 1.09 close/reference ratio to
isolate the alignment fix. Setting `borrow_ref_ratio = 1.0` implements the user's
later `Close > RefPrice` variant. Missing required `nextDayOpen` rows are excluded
from both policies and listed with their original stop PnL; historical missing
labels must not be described as solely an upcoming session. No automatic open
label filling occurs in the notebook. Dates without signals are not added to
its calendar. The standalone runner and frozen report remain independently
reproducible under their original 1.09 contract.

`ledger.parquet` contains both exit prices, reasons, gross bp, net TWD PnL,
borrow fee, overnight gap PnL and per-row differences. CSV summaries and a PNG
are alongside it. `manifest.json` records the input, source code, output and
next-session market-file SHA256s; large tick sources have size/mtime identities.

Results: [REPORT.md](../../Doc/studies/20261007_stop_vs_borrow/REPORT.md).
