"""Overlap / similarity between two signals' point sets (the Phase-2 rulers).

The discovery protocol asks every candidate to report its tick, stock-day and
daily-PnL overlap with existing events.  This module gives one function for
all of them so the numbers are computed the same way for every pair:

1. **exact tick overlap** — same (Date, QuoteCode, ChannelSeq); reported as the
   share of A inside B, of B inside A, and Jaccard.
2. **stock-day overlap** — same (Date, QuoteCode); the execution-relevant view,
   since two shorts in the same stock on the same day share inventory risk.
3. **time-proximity overlap** — an A tick with a B tick on the same stock-day
   within ``±window_seconds``; catches "the same event a few ticks apart".
4. **daily label correlation** — Pearson / Spearman between the two signals'
   equal-weighted daily mean label, plus the mean label of A's exact-disjoint
   and stock-day-disjoint subsets so the *incremental* part of A can be judged
   on its own.

Inputs are polars frames with ``Date, QuoteCode, ChannelSeq, TransTime`` and,
for ruler 4, a label column.  Nothing here reads the store; pass whatever
layer you want to compare (index keys, thresholded rows, final selections).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl


KEYS = ["Date", "QuoteCode", "ChannelSeq"]
STOCK_DAY = ["Date", "QuoteCode"]


@dataclass
class OverlapReport:
    a_name: str
    b_name: str
    a_rows: int
    b_rows: int
    tick_shared: int
    a_in_b_share: float
    b_in_a_share: float
    tick_jaccard: float
    a_stock_days: int
    b_stock_days: int
    stock_day_shared: int
    a_stock_day_in_b_share: float
    b_stock_day_in_a_share: float
    stock_day_jaccard: float
    proximity: dict[int, float] = field(default_factory=dict)
    daily_corr_pearson: float | None = None
    daily_corr_spearman: float | None = None
    common_days: int = 0
    a_label_mean: float | None = None
    a_exact_disjoint_rows: int | None = None
    a_exact_disjoint_label_mean: float | None = None
    a_stock_day_disjoint_rows: int | None = None
    a_stock_day_disjoint_label_mean: float | None = None
    daily: pl.DataFrame | None = field(default=None, repr=False)

    def as_table(self) -> pl.DataFrame:
        rows = [
            ("rows", self.a_rows, self.b_rows),
            ("exact tick shared", self.tick_shared, self.tick_shared),
            ("exact tick share (of own)", round(self.a_in_b_share, 4), round(self.b_in_a_share, 4)),
            ("exact tick Jaccard", round(self.tick_jaccard, 4), round(self.tick_jaccard, 4)),
            ("stock-days", self.a_stock_days, self.b_stock_days),
            ("stock-day shared", self.stock_day_shared, self.stock_day_shared),
            ("stock-day share (of own)", round(self.a_stock_day_in_b_share, 4), round(self.b_stock_day_in_a_share, 4)),
            ("stock-day Jaccard", round(self.stock_day_jaccard, 4), round(self.stock_day_jaccard, 4)),
        ]
        for window, share in self.proximity.items():
            rows.append((f"A tick with B tick within ±{window}s", round(share, 4), None))
        if self.daily_corr_pearson is not None:
            rows += [
                ("daily label corr (pearson / spearman)", round(self.daily_corr_pearson, 4), round(self.daily_corr_spearman, 4)),
                ("common days", self.common_days, self.common_days),
                ("A label mean: all / exact-disjoint / stock-day-disjoint",
                 round(self.a_label_mean, 3),
                 f"{round(self.a_exact_disjoint_label_mean, 3)} ({self.a_exact_disjoint_rows:,}) / "
                 f"{round(self.a_stock_day_disjoint_label_mean, 3)} ({self.a_stock_day_disjoint_rows:,})"),
            ]
        return pl.DataFrame(
            {"metric": [r[0] for r in rows], self.a_name: [str(r[1]) for r in rows], self.b_name: [str(r[2]) for r in rows]}
        )


def _require(frame: pl.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def _proximity_share(a: pl.DataFrame, b: pl.DataFrame, window_seconds: int) -> float:
    """Share of A ticks that have a B tick on the same stock-day within ±window."""

    if a.is_empty() or b.is_empty():
        return 0.0
    left = a.select([*STOCK_DAY, "TransTime"]).with_row_index("_a").sort("TransTime")
    right = b.select([*STOCK_DAY, pl.col("TransTime").alias("_b_time")]).sort("_b_time")
    joined = left.join_asof(
        right,
        left_on="TransTime",
        right_on="_b_time",
        by=STOCK_DAY,
        strategy="nearest",
        tolerance=f"{window_seconds}s",
    )
    return float(joined["_b_time"].is_not_null().mean())


def overlap_report(
    a: pl.DataFrame,
    b: pl.DataFrame,
    *,
    a_name: str = "A",
    b_name: str = "B",
    label_col: str | None = None,
    proximity_windows: Sequence[int] = (60, 300),
) -> OverlapReport:
    _require(a, KEYS, a_name)
    _require(b, KEYS, b_name)
    a_keys = a.select(KEYS).unique()
    b_keys = b.select(KEYS).unique()
    shared = a_keys.join(b_keys, on=KEYS, how="inner").height
    a_sd = a.select(STOCK_DAY).unique()
    b_sd = b.select(STOCK_DAY).unique()
    sd_shared = a_sd.join(b_sd, on=STOCK_DAY, how="inner").height
    union = a_keys.height + b_keys.height - shared
    sd_union = a_sd.height + b_sd.height - sd_shared
    report = OverlapReport(
        a_name=a_name,
        b_name=b_name,
        a_rows=a_keys.height,
        b_rows=b_keys.height,
        tick_shared=shared,
        a_in_b_share=shared / max(a_keys.height, 1),
        b_in_a_share=shared / max(b_keys.height, 1),
        tick_jaccard=shared / max(union, 1),
        a_stock_days=a_sd.height,
        b_stock_days=b_sd.height,
        stock_day_shared=sd_shared,
        a_stock_day_in_b_share=sd_shared / max(a_sd.height, 1),
        b_stock_day_in_a_share=sd_shared / max(b_sd.height, 1),
        stock_day_jaccard=sd_shared / max(sd_union, 1),
    )
    if "TransTime" in a.columns and "TransTime" in b.columns:
        for window in proximity_windows:
            report.proximity[int(window)] = _proximity_share(a, b, int(window))
    if label_col is not None and label_col in a.columns and label_col in b.columns:
        a = a.filter(pl.col(label_col).is_finite())
        b = b.filter(pl.col(label_col).is_finite())
        da = a.group_by("Date").agg(pl.col(label_col).mean().alias("a")).sort("Date")
        db = b.group_by("Date").agg(pl.col(label_col).mean().alias("b")).sort("Date")
        daily = da.join(db, on="Date", how="inner").drop_nulls()
        report.daily = daily
        report.common_days = daily.height
        if daily.height >= 3:
            x, y = daily["a"].to_numpy(), daily["b"].to_numpy()
            report.daily_corr_pearson = float(np.corrcoef(x, y)[0, 1])
            rx, ry = daily["a"].rank().to_numpy(), daily["b"].rank().to_numpy()
            report.daily_corr_spearman = float(np.corrcoef(rx, ry)[0, 1])
        else:
            report.daily_corr_pearson = float("nan")
            report.daily_corr_spearman = float("nan")
        a_lab = a
        report.a_label_mean = float(a_lab[label_col].mean()) if a_lab.height else float("nan")
        exact_disjoint = a_lab.join(b_keys, on=KEYS, how="anti")
        report.a_exact_disjoint_rows = exact_disjoint.height
        report.a_exact_disjoint_label_mean = (
            float(exact_disjoint[label_col].mean()) if exact_disjoint.height else float("nan")
        )
        sd_disjoint = a_lab.join(b_sd, on=STOCK_DAY, how="anti")
        report.a_stock_day_disjoint_rows = sd_disjoint.height
        report.a_stock_day_disjoint_label_mean = (
            float(sd_disjoint[label_col].mean()) if sd_disjoint.height else float("nan")
        )
    return report


__all__ = ["KEYS", "STOCK_DAY", "OverlapReport", "overlap_report"]
