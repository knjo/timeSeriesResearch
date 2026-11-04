"""Helpers: join vwapFeature rows to point frames and compute daily Spearman IC by month."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

VWAP_DIR = Path("/media/kevin/SSD2/Data/vwapFeature")
KEY = ["QuoteCode", "ChannelSeq"]


def vwap_cols() -> list[str]:
    f = sorted(VWAP_DIR.glob("*_vwapFeature.parquet"))[0]
    return [c for c in pl.scan_parquet(f).collect_schema().names() if c.startswith(("VW_", "VWR_", "VWZ_"))]


def join_vwap(points: pl.DataFrame, date_col: str = "Date", cols: list[str] | None = None) -> pl.DataFrame:
    """points must have Date (date or 'YYYYMMDD' str), QuoteCode, ChannelSeq. Left-join vwap columns per day."""
    cols = cols or vwap_cols()
    out = []
    dates = points[date_col].unique().sort().to_list()
    for d in dates:
        dc = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
        f = VWAP_DIR / f"{dc}_vwapFeature.parquet"
        p = points.filter(pl.col(date_col) == d)
        if not f.exists():
            out.append(p.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in cols]))
            continue
        keys = p.select(KEY).unique()
        v = (pl.scan_parquet(f).select(KEY + cols)
               .with_columns(pl.col("ChannelSeq").cast(keys.schema["ChannelSeq"]))
               .join(keys.lazy(), on=KEY, how="inner").collect())
        out.append(p.join(v, on=KEY, how="left"))
    return pl.concat(out, how="vertical_relaxed")


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def daily_ic(df: pl.DataFrame, features: list[str], label: str, date_col: str = "Date", min_n: int = 30) -> pl.DataFrame:
    """Spearman IC per date per feature. Returns long frame: Date, feature, ic, n."""
    rows = []
    for (d,), g in df.group_by([date_col], maintain_order=True):
        y_all = g[label].cast(pl.Float64).to_numpy()
        for f in features:
            x = g[f].cast(pl.Float64).to_numpy()
            m = np.isfinite(x) & np.isfinite(y_all)
            if m.sum() < min_n:
                continue
            xr, yr = _rank(x[m]), _rank(y_all[m])
            if xr.std() == 0 or yr.std() == 0:
                continue
            ic = float(np.corrcoef(xr, yr)[0, 1])
            rows.append((d, f, ic, int(m.sum())))
    return pl.DataFrame(rows, schema=[date_col, "feature", "ic", "n"], orient="row")


def summarize_ic(ic: pl.DataFrame, date_col: str = "Date", periods: dict[str, tuple[str, str]] | None = None) -> pl.DataFrame:
    """Mean IC, t-stat (mean/std*sqrt(days)), hit rate, per feature per period."""
    periods = periods or {"all": ("2000-01-01", "2100-01-01")}
    ic = ic.with_columns(pl.col(date_col).cast(pl.Date))
    outs = []
    for name, (a, b) in periods.items():
        sub = ic.filter((pl.col(date_col) >= pl.lit(a).str.to_date("%Y-%m-%d")) & (pl.col(date_col) <= pl.lit(b).str.to_date("%Y-%m-%d")))
        s = sub.group_by("feature").agg([
            pl.col("ic").mean().alias("ic_mean"),
            (pl.col("ic").mean() / pl.col("ic").std() * pl.len().sqrt()).alias("t"),
            (pl.col("ic") > 0).mean().alias("hit"),
            pl.len().alias("days"),
            pl.col("n").mean().alias("n_per_day"),
        ]).with_columns(pl.lit(name).alias("period"))
        outs.append(s)
    return pl.concat(outs).select(["period", "feature", "ic_mean", "t", "hit", "days", "n_per_day"])
