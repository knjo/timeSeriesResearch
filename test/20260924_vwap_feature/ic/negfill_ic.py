"""Daily Spearman IC of vwap features vs negFill labels (net / netM), replicating modelUpdate.py sample + labels."""
from __future__ import annotations

import argparse
import sys
from datetime import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vwap_ic_lib import VWAP_DIR, daily_ic, summarize_ic, vwap_cols  # noqa: E402

POINT_DIR = Path("/home/kevin/Project/HFT/src/research/timeSeries/data")
OUT_DIR = Path(__file__).resolve().parent / "out_negfill"

BASE_FEATS = ["ToLow", "ToHigh", "Low_High", "ToRef", "ToOpen", "B1_A1B1", "B1_B1B5", "A1_A1A5",
              "B45_AB45", "B1_B12", "A1_A12", "RemainSeconds", "MD_ElaspeTime_30_re", "MD_L1Rate_30"]
RAW_COLS = ["Date", "QuoteCode", "ChannelSeq", "TransTime", "TrialMatch", "BidPrice1", "AskPrice1", "RefPrice", "Open",
            "Close", "FutureHigh", "TickSize", "txf_beta_60d", "txf_to_1330_bp", "txf_residual_vol_to_1330",
            "day_amount_rank", "AmountRank_canDayTrade", "ToLow", "ToHigh", "Low_High", "B1_A1B1", "B1_B1B5",
            "A1_A1A5", "BidLots1", "BidLots2", "AskLots1", "AskLots2", "BidLots4", "BidLots5", "AskLots4",
            "AskLots5", "RemainSeconds", "MD_ElaspeTime_30", "MD_L1Rate_30"]
TIME_W = ["1s", "30s", "60s", "300s"]
VOL_W = ["v01", "v05", "v1", "v3", "v5"]


def load_points(date_compact: str) -> pl.DataFrame:
    df = pl.read_parquet(POINT_DIR / f"{date_compact}.parquet", columns=RAW_COLS)
    b1 = pl.col("BidPrice1")
    df = df.filter(b1 > 0).with_columns([
        ((b1 - pl.col("RefPrice")) / pl.col("RefPrice")).round(6).alias("ToRef"),
        ((b1 - pl.col("Open")) / pl.col("Open")).round(6).alias("ToOpen"),
    ])
    stop = pl.col("FutureHigh") > pl.col("RefPrice") * 1.08
    label = pl.when(stop).then((b1 - pl.col("RefPrice") * 1.08 - pl.col("TickSize") * 2) / b1 * 1e4) \
              .otherwise((b1 - pl.col("Close")) / b1 * 1e4)
    df = df.with_columns(label.alias("TakerSell_CloseBP"))
    df = df.filter(
        (pl.col("ToRef") > -0.015) & (pl.col("ToRef") < 0.05)
        & ((pl.col("day_amount_rank") <= 150) | (pl.col("AmountRank_canDayTrade") <= 150))
        & (pl.col("TransTime").dt.time() < time(12, 30)) & (pl.col("TransTime").dt.time() > time(9, 0, 30))
        & (pl.col("TrialMatch") == 0) & (pl.col("TakerSell_CloseBP").abs() < 1e4)
    )
    resid = (pl.col("TakerSell_CloseBP") + pl.col("txf_beta_60d") * pl.col("txf_to_1330_bp")) / (pl.col("txf_residual_vol_to_1330") * 1e4)
    df = df.with_columns(resid.alias("residual")).with_columns([
        (pl.col("residual") - pl.col("residual").mean().over(["Date", "QuoteCode"])).alias("net"),
        (pl.col("residual") - pl.col("residual").mean().over("Date")).alias("netM"),
        ((pl.col("BidLots4") + pl.col("BidLots5")) / (pl.col("AskLots4") + pl.col("AskLots5") + pl.col("BidLots4") + pl.col("BidLots5"))).alias("B45_AB45"),
        (pl.col("BidLots1") / (pl.col("BidLots1") + pl.col("BidLots2"))).alias("B1_B12"),
        (pl.col("AskLots1") / (pl.col("AskLots1") + pl.col("AskLots2"))).alias("A1_A12"),
        (pl.col("MD_ElaspeTime_30") + 1).log().alias("MD_ElaspeTime_30_re"),
    ])
    return df


def add_vwap(df: pl.DataFrame, date_compact: str) -> pl.DataFrame:
    f = VWAP_DIR / f"{date_compact}_vwapFeature.parquet"
    cols = ["PrevDayLots"] + vwap_cols()
    v = (pl.scan_parquet(f).select(["QuoteCode", "ChannelSeq"] + cols)
           .join(df.select(["QuoteCode", "ChannelSeq"]).unique().lazy(), on=["QuoteCode", "ChannelSeq"], how="inner").collect())
    df = df.join(v, on=["QuoteCode", "ChannelSeq"], how="left")
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2
    prev = pl.col("PrevDayLots").cast(pl.Float64)
    ex = []
    for w in TIME_W:
        ex.append((pl.col(f"VW_Lots_{w}").cast(pl.Float64) / prev * 1e4).alias(f"VWN_LotsShare_{w}"))  # bp of prev-day lots
    for w in TIME_W + VOL_W:
        lots = pl.col(f"VW_Lots_{w}").cast(pl.Float64)
        px = pl.col(f"VW_Px_{w}").cast(pl.Float64)
        ex += [
            pl.when(lots > 0).then(pl.col(f"VW_NetLots_{w}").cast(pl.Float64) / lots).otherwise(None).alias(f"VWN_Imb_{w}"),
            pl.when(lots > 0).then(pl.col(f"VW_HftLots_{w}").cast(pl.Float64) / lots).otherwise(None).alias(f"VWN_HftShare_{w}"),
            ((mid / px - 1) * 1e4).alias(f"VWN_Mid_{w}"),
            ((pl.col(f"VW_BuyPx_{w}").cast(pl.Float64) - pl.col(f"VW_SellPx_{w}").cast(pl.Float64)) / px * 1e4).alias(f"VWN_BuySell_{w}"),
            (pl.col(f"VW_AvgLots_{w}").cast(pl.Float64) + 1).log().alias(f"VWN_logAvgLots_{w}"),
        ]
    for w in VOL_W:
        ex.append((pl.col(f"VW_Elapsed_{w}").cast(pl.Float64) + 1).log().alias(f"VWN_logElapsed_{w}"))
    return df.with_columns(ex)


def feature_list(df: pl.DataFrame) -> list[str]:
    return BASE_FEATS + [c for c in df.columns if c.startswith(("VWR_", "VWZ_", "VWN_"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260101")
    ap.add_argument("--end", default="20261231")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    dates = sorted(p.name[:8] for p in VWAP_DIR.glob("*_vwapFeature.parquet") if a.start <= p.name[:8] <= a.end)
    dates = [d for d in dates if (POINT_DIR / f"{d}.parquet").exists()]
    print("dates", len(dates), dates[:1], dates[-1:])
    ics = []
    n_rows = []
    for d in dates:
        df = add_vwap(load_points(d), d)
        feats = feature_list(df)
        df = df.with_columns(pl.col("Date").str.to_date("%Y%m%d"))
        n_rows.append((d, df.height, df["QuoteCode"].n_unique(), df["VW_Px_v1"].null_count()))
        for lab in ["net", "netM"]:
            ic = daily_ic(df, feats, lab).with_columns(pl.lit(lab).alias("label"))
            ics.append(ic)
    ic = pl.concat(ics)
    ic.write_parquet(OUT_DIR / f"daily_ic_{a.tag}.parquet")
    pl.DataFrame(n_rows, schema=["date", "rows", "stocks", "vwap_null"], orient="row").write_csv(OUT_DIR / f"coverage_{a.tag}.csv")
    periods = {"all": ("2026-01-01", "2026-12-31"), "JanMar": ("2026-01-01", "2026-03-31"),
               "AprJul": ("2026-04-01", "2026-07-31"), "AugSep": ("2026-08-01", "2026-09-30")}
    outs = []
    for lab in ["net", "netM"]:
        s = summarize_ic(ic.filter(pl.col("label") == lab), periods=periods).with_columns(pl.lit(lab).alias("label"))
        outs.append(s)
    summ = pl.concat(outs)
    summ.write_csv(OUT_DIR / f"summary_{a.tag}.csv")
    with pl.Config(tbl_rows=200, tbl_cols=10, fmt_str_lengths=40, tbl_width_chars=200):
        for lab in ["net", "netM"]:
            print(f"\n===== label={lab}: top |t| over all days =====")
            print(summ.filter((pl.col("label") == lab) & (pl.col("period") == "all")).sort(pl.col("t").abs(), descending=True).head(40))


if __name__ == "__main__":
    main()
