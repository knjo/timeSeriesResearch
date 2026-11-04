"""Daily Spearman IC of vwap features vs followHFTMaker multiLabel labels, per side, using the module's own loaders."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("FOLLOW_HFT_MAKER_LEAN_INPUT", "1")
PROJECT_ROOT = Path("/home/kevin/Project/HFT")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from src.research.followHFTMaker import followHFTMaker_multiLabel as fm  # noqa: E402
from vwap_ic_lib import VWAP_DIR, daily_ic, summarize_ic, vwap_cols  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out_maker"
TIME_W = ["1s", "30s", "60s", "300s"]
VOL_W = ["v01", "v05", "v1", "v3", "v5"]
LABELS = {
    "buy": ["MakerBuy_follow5m_vwap_late_bp", "MakerBuy_CloseBP", "MakerBuy_CloseBP_net"],
    "sell": ["MakerSell_follow5m_vwap_late_bp", "MakerSell_CloseBP", "MakerSell_CloseBP_net"],
}


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


def attach_vwap(pdf: pd.DataFrame) -> pd.DataFrame:
    cols = ["PrevDayLots"] + vwap_cols()
    parts = []
    for d, g in pdf.groupby("Date", sort=True):
        f = VWAP_DIR / f"{d}_vwapFeature.parquet"
        keys = pl.DataFrame({"QuoteCode": g["QuoteCode"].astype(str).tolist(), "ChannelSeq": g["ChannelSeq"].to_numpy(dtype="uint64")}).unique()
        v = (pl.scan_parquet(f).select(["QuoteCode", "ChannelSeq"] + cols)
               .join(keys.lazy(), on=["QuoteCode", "ChannelSeq"], how="inner").collect())
        base = pl.DataFrame({
            "QuoteCode": g["QuoteCode"].astype(str).tolist(),
            "ChannelSeq": g["ChannelSeq"].to_numpy(dtype="uint64"),
            "BidPrice1": g["BidPrice1"].to_numpy(dtype=float),
            "AskPrice1": g["AskPrice1"].to_numpy(dtype=float),
            "_i": g.index.to_numpy(),
        })
        j = derive_vwap(base.join(v, on=["QuoteCode", "ChannelSeq"], how="left"))
        keep = [c for c in j.columns if c.startswith(("VWR_", "VWZ_", "VWN_"))] + ["_i"]
        parts.append(j.select(keep).to_pandas().set_index("_i"))
    extra = pd.concat(parts).reindex(pdf.index)
    return pd.concat([pdf, extra], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260101")
    ap.add_argument("--end", default="20261231")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    vdates = {p.name[:8] for p in VWAP_DIR.glob("*_vwapFeature.parquet")}
    files = sorted(p for p in fm.POINT_DATA_DIR.glob("*.parquet") if a.start <= p.stem <= a.end and p.stem in vdates)
    print("dates", len(files), files[:1], files[-1:])
    days = [d for d in (fm._read_one_day(p) for p in files) if d is not None]
    signal_df = pd.concat(days, ignore_index=True)
    buy_df, sell_df, feature_cols = fm.add_features_and_base_labels(signal_df)
    buy_df, sell_df = fm.add_future_space_labels(buy_df, sell_df, legacy_time_int=True)
    print("base features", len(feature_cols), "buy", len(buy_df), "sell", len(sell_df))
    ics, cov = [], []
    for side, df, exec_col in [("buy", buy_df, "delay_100ms_Bid1_FillSeconds"), ("sell", sell_df, "delay_100ms_Ask1_FillSeconds")]:
        df = df[df[exec_col] < 600].reset_index(drop=True)
        df = attach_vwap(df)
        feats = feature_cols + [c for c in df.columns if c.startswith(("VWR_", "VWZ_", "VWN_"))]
        sub = df[["Date", "QuoteCode"] + feats + LABELS[side]].copy()
        sub["Date"] = sub["Date"].astype(str)
        sub["QuoteCode"] = sub["QuoteCode"].astype(str)
        pdf = pl.from_pandas(sub).with_columns(pl.col("Date").cast(pl.String).str.to_date("%Y%m%d"))
        cov.append(pdf.group_by("Date").agg(pl.len().alias("rows"), pl.col("VWN_Mid_v1").null_count().alias("vwap_null")).with_columns(pl.lit(side).alias("side")))
        for lab in LABELS[side]:
            ics.append(daily_ic(pdf, feats, lab).with_columns(pl.lit(lab).alias("label"), pl.lit(side).alias("side")))
    ic = pl.concat(ics)
    ic.write_parquet(OUT_DIR / f"daily_ic_{a.tag}.parquet")
    pl.concat(cov).sort(["side", "Date"]).write_csv(OUT_DIR / f"coverage_{a.tag}.csv")
    periods = {"all": ("2026-01-01", "2026-12-31"), "JanMar": ("2026-01-01", "2026-03-31"),
               "AprJul": ("2026-04-01", "2026-07-31"), "AugSep": ("2026-08-01", "2026-09-30")}
    outs = []
    for (side, lab), sub in ic.group_by(["side", "label"]):
        outs.append(summarize_ic(sub, periods=periods).with_columns(pl.lit(side).alias("side"), pl.lit(lab).alias("label")))
    summ = pl.concat(outs)
    summ.write_csv(OUT_DIR / f"summary_{a.tag}.csv")
    with pl.Config(tbl_rows=60, tbl_cols=10, fmt_str_lengths=40, tbl_width_chars=200):
        for side in ["buy", "sell"]:
            lab = LABELS[side][0]
            print(f"\n===== side={side} label={lab}: top |t| (all) =====")
            print(summ.filter((pl.col("side") == side) & (pl.col("label") == lab) & (pl.col("period") == "all")).sort(pl.col("t").abs(), descending=True).head(30))


if __name__ == "__main__":
    main()
