# negFill historical market attribution

Scope: scoped, RESEARCH_DIAGNOSTIC_ONLY. Current notebook replay, 2024-01-02
through 2026-10-13, with 2023-09-01 onward training input. No production changes.

Canonical report: [REPORT.md](../../Doc/studies/20261014_negfill_market_attribution/REPORT.md).
Accepted trades, daily/monthly accounting, market diagnostics, source fingerprints,
and verification are in `artifacts/v1/`. Large reconstructible mother pools and
temporary caches are intentionally omitted. Scripts are the archived analysis
used in this study, not a new production entry point.

Reproduce from `/home/kevin/Project/HFT`, with the notebook/module fingerprints
matching `artifacts/v1/source_fingerprints.json` and the original local data:

```bash
export UV_CACHE_DIR=/tmp/uv-fh-update
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/fh-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
study_scripts=src/research/timeSeries/test/20261014_negfill_market_attribution/audit_scripts
uv run python "$study_scripts/negfill_recent_replay.py"
uv run python "$study_scripts/negfill_history_replay.py"
uv run python "$study_scripts/negfill_market_history.py"
uv run python "$study_scripts/negfill_native_factor.py"
uv run python "$study_scripts/negfill_history_analysis.py"
uv run python "$study_scripts/negfill_across_stock.py"
uv run python "$study_scripts/negfill_verify.py"
uv run python "$study_scripts/negfill_finalize.py"
```

Outputs are staged under `/tmp/negfill_recent_20261014` and
`/tmp/negfill_history_20261014`; the commands do not overwrite production models
or automatically publish new results. The first two steps rerun original rolling
training and need substantial RAM and local tick history. Input inventory records
path, size and mtime, not immutable content hashes; changed market data may alter
results. The notebook snapshot retains the used source cells, including original
selection, label and risk definitions; it does not reconstruct the entire runtime.
Read the report's legacy preprocessing / retrospective-rule limitations before
interpreting this as historical OOS.
