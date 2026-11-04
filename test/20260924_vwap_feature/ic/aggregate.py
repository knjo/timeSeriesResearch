"""Aggregate daily IC parquet files (from negfill_ic / maker_ic runs) into period tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vwap_ic_lib import summarize_ic  # noqa: E402

PERIODS = {"JanMar": ("2026-01-01", "2026-03-31"), "AprJul": ("2026-04-01", "2026-07-31"),
           "AugSep": ("2026-08-01", "2026-09-30"), "Apr+": ("2026-04-01", "2026-12-31"), "all": ("2026-01-01", "2026-12-31")}
import calendar
MONTHS = {f"m{m:02d}": (f"2026-{m:02d}-01", f"2026-{m:02d}-{calendar.monthrange(2026, m)[1]:02d}") for m in range(1, 10)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="daily_ic_*.parquet")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min_days", type=int, default=15)
    ap.add_argument("--features", default="", help="comma list to show in stability table (default: auto top)")
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob(a.glob))
    ic = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed").unique(["Date", "feature", "label"] + (["side"] if "side" in pl.read_parquet(files[0]).columns else []))
    if "side" not in ic.columns:
        ic = ic.with_columns(pl.lit("").alias("side"))
    print("days:", ic["Date"].n_unique(), ic["Date"].min(), ic["Date"].max())
    outs = []
    for (side, lab), sub in ic.group_by(["side", "label"]):
        s = summarize_ic(sub, periods={**PERIODS, **MONTHS}).with_columns(pl.lit(side).alias("side"), pl.lit(lab).alias("label"))
        outs.append(s)
    summ = pl.concat(outs)
    summ.write_csv(Path(a.dir) / "summary_agg.csv")
    wide = summ.pivot(on="period", index=["side", "label", "feature"], values="t").with_columns(
        summ.pivot(on="period", index=["side", "label", "feature"], values="ic_mean").select(pl.col("all").alias("ic_all"))["ic_all"])
    cols = ["side", "label", "feature", "ic_all", "all", "JanMar", "AprJul", "AugSep"] + [m for m in MONTHS if m in wide.columns]
    wide = wide.select([c for c in cols if c in wide.columns])
    with pl.Config(tbl_rows=400, tbl_cols=20, fmt_str_lengths=34, tbl_width_chars=260, float_precision=2):
        for (side, lab), sub in wide.group_by(["side", "label"], maintain_order=True):
            days_ok = summ.filter((pl.col("side") == side) & (pl.col("label") == lab) & (pl.col("period") == "Apr+") & (pl.col("days") >= a.min_days))["feature"]
            sub = sub.filter(pl.col("feature").is_in(days_ok.implode()))
            base = sub.filter(~pl.col("feature").str.starts_with("VW"))
            new = sub.filter(pl.col("feature").str.starts_with("VW"))
            print(f"\n##### side={side} label={lab} : t-stat by period (ic_all = mean daily IC over all days) #####")
            print("--- baseline features (sorted |t Apr+|) ---")
            avail = [c for c in ["AprJul", "AugSep"] if c in sub.columns] or ["all"]
            key = sum((pl.col(c).abs().fill_null(0) for c in avail), pl.lit(0.0))
            print(base.sort(key, descending=True, nulls_last=True).head(a.top))
            print("--- new vwap features (sorted |t Apr+|) ---")
            if a.features:
                new = new.filter(pl.col("feature").is_in(a.features.split(",")))
            print(new.sort(key, descending=True, nulls_last=True).head(a.top))


if __name__ == "__main__":
    main()
