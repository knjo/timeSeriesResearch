"""Diagnostic reproduction of the supplied 2.5-sigma notebook excerpt.

Unlike the primary replay, this uses strict touches and barrier-price fills,
adding two entry ticks only to the SL. This is a convention check, not a trial
selected for live execution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
from src.research.tool import first_touch_times


def reproduce(output: Path):
    signal=pl.read_parquet(output/"input_signals.parquet").filter(pl.col("Date") >= "20260101")
    work=signal.with_columns(
        (pl.col("BidPrice1")*(1-2.5*pl.col("txf_residual_vol_to_1330"))).alias("tpPrice"),
        pl.min_horizontal(pl.col("BidPrice1")*(1+2.5*pl.col("txf_residual_vol_to_1330")),pl.col("RefPrice")*1.08).alias("slPrice"))
    # The existing notebook helper owns a pandas API; all study-side transforms
    # and aggregation stay in polars.
    times=first_touch_times(work.select("Date","QuoteCode","TransTime","tpPrice","slPrice").to_pandas(),
                            low_col="tpPrice",high_col="slPrice")
    work=work.hstack(pl.from_pandas(times))
    tp=pl.col("low_time").is_not_null() & (pl.col("high_time").is_null() | (pl.col("low_time")<pl.col("high_time")))
    sl=pl.col("high_time").is_not_null() & ~tp
    work=work.with_columns(pl.when(tp).then((pl.col("BidPrice1")-pl.col("tpPrice"))/pl.col("BidPrice1")*1e4)
                          .when(sl).then((pl.col("BidPrice1")-pl.col("slPrice")-2*pl.col("TickSize"))/pl.col("BidPrice1")*1e4)
                          .otherwise(pl.col("TakerSell_CloseBP")).alias("notebook_bp"))
    daily=work.group_by("Date").agg(
        (pl.col("BidPrice1")*1000*(pl.col("TakerSell_CloseBP")-19.3)/1e4).sum().alias("original_label_pnl"),
        (pl.col("BidPrice1")*1000*(pl.col("notebook_bp")-19.3)/1e4).sum().alias("notebook_vol_pnl"))
    daily=daily.sort("Date").with_columns((pl.col("notebook_vol_pnl")-pl.col("original_label_pnl")).alias("delta_twd"))
    daily.write_csv(output/"primary_2026"/"notebook_literal_daily.csv")
    result={"n_rows":work.height,"n_days":daily.height,"span":2.5,
            "original_label_pnl_wan":daily["original_label_pnl"].sum()/1e4,
            "notebook_vol_pnl_wan":daily["notebook_vol_pnl"].sum()/1e4,
            "delta_wan":daily["delta_twd"].sum()/1e4,
            "convention":"exact supplied snippet, first_touch_times strict default, TP barrier, SL barrier + 2 TickSize",
            "limitation":"given current exported signal ledger; interactive notebook state may differ"}
    (output/"primary_2026"/"notebook_literal.json").write_text(json.dumps(result,indent=2))
    print(result)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    reproduce(p.parse_args().output)
