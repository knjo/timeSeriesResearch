# `fh_neg_rejection` inherited execution-gate ablation

This study runs the complete `2 x 2 x 2` factorial for the inherited positive
`ToRef`, listed-stock `MD_L1Rate_30`, and `SpreadPairElapsed` gates. The
conservative D/G/B condition, short label, other static execution gates,
19.3bp cost, and per-stock capacity are fixed.

The run is diagnostic and study-local. It does not modify the frozen static
snapshot, the 2026-09-21 extension, or production artifacts.

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache MPLCONFIGDIR=/tmp/matplotlib-hft \
uv run --no-sync python \
  src/research/timeSeries/test/20260921_fh_neg_rejection_execution_gate_ablation/run_gate_ablation.py
```

Results are documented in
[REPORT.md](../../Doc/studies/20260921_fh_neg_rejection_execution_gate_ablation/REPORT.md).
