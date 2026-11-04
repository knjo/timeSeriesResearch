"""Build the negFill research signal frame (notebook cells 2-8, 11) for a date range and attach vwap features.

Output: artifacts/signal_<start>_<end>.parquet (study-local). Never writes to the formal data/ directory.
Usage: uv run python src/research/timeSeries/test/20260924_vwap_feature/build_signal.py --start 20260101 --end 20260923
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = PROJECT_ROOT / "src/research/timeSeries/data"
VWAP_DIR = Path("/media/kevin/SSD2/Data/vwapFeature")
ART = Path(__file__).resolve().parent / "artifacts"

TIME_W = ["1s", "30s", "60s", "300s"]
VOL_W = ["v01", "v05", "v1", "v3", "v5"]

# columns needed by the notebook model / gates / backtest (cells 11-34)
KEEP = [
    "Date", "QuoteCode", "ChannelSeq", "TransTime", "TrialMatch", "InOut", "Overshoot", "TotalFillLots", "FillLots",
    "BidPrice1", "BidLots1", "BidLots2", "BidLots4", "BidLots5", "AskPrice1", "AskLots1", "AskLots2", "AskLots4", "AskLots5",
    "RecordHigh", "FutureHigh", "FutureLow", "RefPrice", "Close", "Open", "Spread", "TickSize", "OTC",
    "ToLow", "ToHigh", "Low_High", "B1_A1B1", "B1_B1B5", "A1_A1A5", "RemainSeconds", "MD_ElaspeTime_30", "MD_L1Rate_30",
    "SpreadPairElapsed", "FillLots_atLow", "day_amount_rank", "AmountRank_canDayTrade", "AmountRank",
    "avg_bidLots1", "avg_askLots1", "hft_strick_makerSpreadBP", "MarketCap", "d_intraday_return", "dt_major_count",
    "sell_branch_breadth_rank", "net_buy_gini", "top1_net_buy_amount", "trading_turnover_bf",
    "txf_beta_60d", "txf_to_1330_bp", "txf_residual_vol_to_1330", "txf_residual_vol_0900_1330_60d", "txf_residual_vol_0900_1330_20d",
]


def vwap_feature_cols() -> list[str]:
    f = sorted(VWAP_DIR.glob("*_vwapFeature.parquet"))[0]
    return [c for c in pl.scan_parquet(f).collect_schema().names() if c.startswith(("VW_", "VWR_", "VWZ_"))]


def derive_vwap(df: pl.DataFrame) -> pl.DataFrame:
    mid = (pl.col("BidPrice1") + pl.col("AskPrice1")) / 2
    prev = pl.col("PrevDayLots").cast(pl.Float64)
    ex = []
    for w in TIME_W:
        ex.append((pl.col(f"VW_Lots_{w}").cast(pl.Float64) / prev * 1e4).alias(f"VWN_LotsShare_{w}"))
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


def load_day(date_compact: str) -> pl.DataFrame:
    """Notebook cell 2 filters + labels; returns polars frame with KEEP columns."""
    avail = pl.scan_parquet(DATA_DIR / f"{date_compact}.parquet").collect_schema().names()
    cols = [c for c in KEEP if c in avail and c != "Date"]
    df = pl.read_parquet(DATA_DIR / f"{date_compact}.parquet", columns=cols)
    b1 = pl.col("BidPrice1")
    df = df.filter(b1 > 0).with_columns([
        ((b1 - pl.col("RefPrice")) / pl.col("RefPrice")).round(6).alias("ToRef"),
        ((b1 - pl.col("Open")) / pl.col("Open")).round(6).alias("ToOpen"),
    ])
    stop = pl.col("FutureHigh") > pl.col("RefPrice") * 1.08
    label = pl.when(stop).then((b1 - pl.col("RefPrice") * 1.08 - pl.col("TickSize") * 2) / b1 * 1e4).otherwise((b1 - pl.col("Close")) / b1 * 1e4)
    df = df.with_columns(label.alias("TakerSell_CloseBP"))
    df = df.filter((pl.col("ToRef") > -0.015) & (pl.col("ToRef") < 0.05)
                   & ((pl.col("day_amount_rank") <= 150) | (pl.col("AmountRank_canDayTrade") <= 150)))
    df = df.with_columns(pl.lit(date_compact).alias("Date"))
    for c in KEEP:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    return df.select(KEEP + ["ToRef", "ToOpen", "TakerSell_CloseBP"])


def attach_vwap(df: pl.DataFrame, date_compact: str, cols: list[str]) -> pl.DataFrame:
    f = VWAP_DIR / f"{date_compact}_vwapFeature.parquet"
    v = (pl.scan_parquet(f).select(["QuoteCode", "ChannelSeq", "PrevDayLots"] + cols)
           .join(df.select(["QuoteCode", "ChannelSeq"]).unique().lazy(), on=["QuoteCode", "ChannelSeq"], how="inner").collect())
    out = derive_vwap(df.join(v, on=["QuoteCode", "ChannelSeq"], how="left"))
    drop = [c for c in out.columns if c.startswith("VW_")]  # raw window stats not used as model features
    return out.drop(drop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    ART.mkdir(exist_ok=True)
    dates = sorted(p.stem for p in DATA_DIR.glob("*.parquet") if a.start <= p.stem <= a.end and (VWAP_DIR / f"{p.stem}_vwapFeature.parquet").exists())
    print("dates", len(dates), dates[:1], dates[-1:])
    vcols = vwap_feature_cols()
    parts = [attach_vwap(load_day(d), d, vcols) for d in dates]
    df = pl.concat(parts, how="vertical_relaxed").to_pandas()

    # notebook cells 4, 6, 8: residual label, time/trial filters, net / netM
    df["residual_TakerSell_CloseBP"] = (df.TakerSell_CloseBP + df.txf_beta_60d * df.txf_to_1330_bp) / (df.txf_residual_vol_to_1330 * 1e4)
    df = df[(df.TransTime.dt.time < datetime.time(12, 30)) & (df.TransTime.dt.time > datetime.time(9, 0, 30)) & (df.TrialMatch == 0)]
    df = df[df.TakerSell_CloseBP.abs() < 10000].sort_values(["Date", "QuoteCode", "TransTime"]).reset_index(drop=True)
    df["TakerSell_CloseBP_net"] = df.residual_TakerSell_CloseBP - df.groupby(["Date", "QuoteCode"]).residual_TakerSell_CloseBP.transform("mean")
    df["TakerSell_CloseBP_netM"] = df.residual_TakerSell_CloseBP - df.groupby(["Date"]).residual_TakerSell_CloseBP.transform("mean")
    # cell 11 derived features actually used by the 14-feature model
    df["B1_B12"] = df.BidLots1 / (df.BidLots1 + df.BidLots2)
    df["A1_A12"] = df.AskLots1 / (df.AskLots1 + df.AskLots2)
    df["B45_AB45"] = (df.BidLots4 + df.BidLots5) / (df.AskLots4 + df.AskLots5 + df.BidLots4 + df.BidLots5)
    df["MD_ElaspeTime_30_re"] = np.log(df.MD_ElaspeTime_30 + 1)
    df["MD_L1Rate_30_re"] = df.MD_L1Rate_30
    df["OTC"] = df["OTC"].astype(float)
    out = ART / f"signal_{a.start}_{a.end}.parquet"
    df.to_parquet(out, index=False)
    print("rows", len(df), "days", df.Date.nunique(), "->", out)


if __name__ == "__main__":
    main()
