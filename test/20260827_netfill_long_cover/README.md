# Signed-FillLots long-cover study

This versioned study answers whether the repository's canonical signed
`FillLots_atLow` or `FillLots_atHigh` state can support a fixed taker-long
sleeve near break-even after `19.3bp`, and whether that sleeve improves the
loss-day or tail behavior of the kept `fh_neg_rejection` short snapshot.

Canonical interpretation and kept BackTest figures are in
[`REPORT.md`](../../report/studies/20260827_netfill_long_cover/REPORT.md).

`netFillLots_atLow/atHigh` are user-facing aliases only.  The persisted
canonical fields already contain net signed flow:

```text
cumsum(FillLots * InOut * (TrialMatch == 0))
over QuoteCode and the current RecordLow / RecordHigh epoch
```

The exact six-candidate family, long label, static execution gates, capacity,
cost grid, familywise inference, fixed cover frontier, and research screens
are frozen in `study_contract.json`.  A pre-result audit invalidated the
incomplete v1 materialization before any aggregate candidate PnL was read;
the kept v2 rebuild fixes lineage and label-eligibility semantics without
changing the candidate family or thresholds.
This remains outcome-informed retrospective evidence because related long-side
bias and legacy entry outcomes were inspected historically.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/matplotlib-hft \
UV_CACHE_DIR=/tmp/hft-uv-cache \
uv run python \
  src/research/timeSeries/test/20260827_netfill_long_cover/run_backtest.py

MPLCONFIGDIR=/tmp/matplotlib-hft-test \
UV_CACHE_DIR=/tmp/hft-uv-cache \
uv run python -m unittest \
  src/research/timeSeries/test/20260827_netfill_long_cover/test_analysis_utils.py \
  src/research/timeSeries/test/20260827_netfill_long_cover/test_run_backtest.py
```

The runner resumes only daily caches whose contract hash, raw-source metadata,
and materialization implementation fingerprint all match.  It also verifies
the short input against the hashes in that snapshot's manifest.  Once
`run_manifest.json` exists it refuses to replace the kept artifact; a changed
experiment must use a new artifact version.

Raw stock-label validity is the only label eligibility requirement.  TXF hedge,
residual, volatility, and z fields remain diagnostics and may be null.  The
atLow `epoch_entry` profiles admit every threshold cross/re-entry within an
epoch, not only the first cross.

Primary cover accounting is native cash `short + h * long`, with every frozen
`h` reported and no retrospective winner selection.  The aggregate-position
normalized frontier is secondary only: daily `position` is cumulative entry
notional, not concurrent portfolio capital.

The current-six max-stat and tail-cover max-stat are retrospective familywise
screens.  DSR-like values for 6 and 28 trials are sensitivity diagnostics, not
hard gates; historical search exposure is only lower-bounded and therefore is
not claimed to be fully corrected.
