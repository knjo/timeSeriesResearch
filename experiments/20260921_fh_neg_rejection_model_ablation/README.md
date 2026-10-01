# `fh_neg_rejection` model ablation

This versioned diagnostic compares the frozen no-model-gate baseline, the
original Ridge model gates, and one direct ElasticNet replacement. It reads the
2026-08-26 study's frozen model input and Ridge artifacts without modifying
them. All new output belongs under `artifacts/model_family_v1/`.

The 38-feature manifest is unchanged and `feature_screening=False` for both
model families. ElasticNet uses `l1_ratio=0.5` and a per-fit alpha conversion
that preserves the original Ridge L2 coefficient despite sklearn's different
objective normalization.

See the paired
[REPORT.md](../../report/studies/20260921_fh_neg_rejection_model_ablation/REPORT.md)
for results and limitations.

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache MPLCONFIGDIR=/tmp/matplotlib-hft \
uv run --no-sync python \
  src/research/intradaySignal/experiments/20260921_fh_neg_rejection_model_ablation/run_comparison.py \
  --n-jobs 8
```
