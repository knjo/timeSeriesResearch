"""Compact per-period t-stat printout from daily IC parquet files."""
from __future__ import annotations
import argparse, calendar, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vwap_ic_lib import summarize_ic

PERIODS = {"AprJul": ("2026-04-01", "2026-07-31"), "AugSep": ("2026-08-01", "2026-09-30"), "Apr+": ("2026-04-01", "2026-12-31"), "JanMar": ("2026-01-01", "2026-03-31")}
MONTHS = {f"m{m:02d}": (f"2026-{m:02d}-01", f"2026-{m:02d}-{calendar.monthrange(2026, m)[1]:02d}") for m in range(1, 10)}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--glob", default="daily_ic_*.parquet")
ap.add_argument("--labels", default=""); ap.add_argument("--n_base", type=int, default=6); ap.add_argument("--n_new", type=int, default=16)
ap.add_argument("--months", default="m04,m05,m06,m07,m08,m09")
a = ap.parse_args()
files = sorted(Path(a.dir).glob(a.glob))
ic = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
if "side" not in ic.columns:
    ic = ic.with_columns(pl.lit("").alias("side"))
ic = ic.unique(["Date", "feature", "label", "side"])
print("days:", ic["Date"].n_unique(), ic["Date"].min(), ic["Date"].max())
months = a.months.split(",")
for (side, lab), sub in sorted(ic.group_by(["side", "label"]), key=lambda x: x[0]):
    if a.labels and lab not in a.labels.split(","):
        continue
    s = summarize_ic(sub, periods={**PERIODS, **{m: MONTHS[m] for m in months}})
    t = s.pivot(on="period", index="feature", values="t")
    icm = s.filter(pl.col("period") == "Apr+").select(["feature", "ic_mean", "days"])
    t = t.join(icm, on="feature")
    def row(r):
        cells = " ".join(f"{(r.get(m) or 0):+5.1f}" for m in months)
        return f"  {r['feature']:26s} ic={r['ic_mean']:+.3f} d={r['days']:3d}  AprJul={r.get('AprJul') or 0:+6.2f} AugSep={r.get('AugSep') or 0:+6.2f} Apr+={r.get('Apr+') or 0:+6.2f} | {cells}"
    print(f"\n### {side} {lab}   (t by period; monthly t: {' '.join(months)})")
    base = t.filter(~pl.col("feature").str.starts_with("VW")).sort(pl.col("Apr+").abs(), descending=True, nulls_last=True)
    new = t.filter(pl.col("feature").str.starts_with("VW")).sort(pl.col("Apr+").abs(), descending=True, nulls_last=True)
    print("  -- baseline --")
    for r in base.head(a.n_base).iter_rows(named=True): print(row(r))
    print("  -- vwap --")
    for r in new.head(a.n_new).iter_rows(named=True): print(row(r))
