"""Point-in-time defensive stock-day list rebuilt from the frozen theme_strength artifacts.

Follows DEFENSIVE_STOCK_SELECTION.md section 4: on each decision date D, rank valid
themes by ``score__candidate_directional`` (capacity weighting), take the top 20%
(``floor(n / 5)``, at least 1) as the defensive side and, for monotonicity checks,
the bottom 20% as the weak side, plus the remaining valid themes as ``middle``
(membership-only control).  Member weights are ``theme_notional *
capacity_weight / sum(capacity_weight in theme)`` and are summed across selected
themes for the same stock.

Differences from ``theme_strength_research.build_overlap_netted_strategy``:

- no ``theme_relative_oc_bp`` / ``stock_oc_bp`` non-null filter, so the list uses
  D-1 information only (the research function drops rows without a D outcome,
  which is fine for evaluation but is not point-in-time for a universe filter);
- no reliability gate (``filtered=false``), matching the +14.93 BP/day evidence.

Output key is unique on ``(decision_date, side, quote_code)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[5]
THEME_STRENGTH_DIR = (
    PROJECT_ROOT / "src" / "research" / "news_collector" / "artifacts" / "theme_strength"
)
DEFAULT_OUTPUT = Path(__file__).with_name("artifacts") / "v1" / "defensive_stock_daily.parquet"
SCORE_COLUMN = "score__candidate_directional"
WEIGHTING = "capacity"
MIN_DAILY_THEMES = 10
TOP_FRACTION = 5  # top / bottom quintile


def select_theme_sides(theme_daily: pl.DataFrame) -> pl.DataFrame:
    """Return (decision_date, theme_id, side, theme_notional) for top and bottom quintiles."""

    frame = (
        theme_daily.filter(pl.col("weighting") == WEIGHTING)
        .drop_nulls([SCORE_COLUMN])
        .filter(pl.col(SCORE_COLUMN).is_finite())
        .select("decision_date", "theme_id", SCORE_COLUMN)
        .sort(["decision_date", SCORE_COLUMN, "theme_id"])
        .with_columns(
            pl.len().over("decision_date").alias("n_themes"),
            pl.int_range(pl.len()).over("decision_date").alias("rank_asc"),
        )
        .filter(pl.col("n_themes") >= MIN_DAILY_THEMES)
        .with_columns(
            pl.max_horizontal(pl.lit(1), pl.col("n_themes") // TOP_FRACTION).alias("tail")
        )
    )
    is_top = pl.col("rank_asc") >= pl.col("n_themes") - pl.col("tail")
    is_bottom = pl.col("rank_asc") < pl.col("tail")
    top = frame.filter(is_top).with_columns(pl.lit("top").alias("side"), (1.0 / pl.col("tail")).alias("theme_notional"))
    bottom = frame.filter(is_bottom).with_columns(pl.lit("bottom").alias("side"), (1.0 / pl.col("tail")).alias("theme_notional"))
    # middle: valid themes that are in neither quintile; kept only to separate the
    # directional-score effect from plain theme-membership (attention) effect
    middle = frame.filter(~is_top & ~is_bottom).with_columns(
        pl.lit("middle").alias("side"),
        (1.0 / (pl.col("n_themes") - 2 * pl.col("tail")).clip(lower_bound=1)).alias("theme_notional"),
    )
    return pl.concat([top, bottom, middle]).select(
        "decision_date", "theme_id", "side", "theme_notional", "n_themes", SCORE_COLUMN
    )


def build_defensive_stock_daily(theme_strength_dir: Path = THEME_STRENGTH_DIR) -> pl.DataFrame:
    theme_daily = pl.read_parquet(
        theme_strength_dir / "theme_daily.parquet",
        columns=["decision_date", "theme_id", "weighting", SCORE_COLUMN],
    )
    membership = (
        pl.read_parquet(
            theme_strength_dir / "membership_daily.parquet",
            columns=["trade_date", "theme_id", "quote_code"],
        )
        .rename({"trade_date": "decision_date"})
        .unique(["decision_date", "theme_id", "quote_code"])
    )
    stock = pl.read_parquet(
        theme_strength_dir / "stock_daily.parquet",
        columns=["decision_date", "source_date", "quote_code", "capacity_weight"],
    )
    sides = select_theme_sides(theme_daily)
    members = (
        sides.join(membership, on=["decision_date", "theme_id"], how="inner")
        .join(stock, on=["decision_date", "quote_code"], how="inner")
        .filter(
            pl.col("capacity_weight").is_not_null()
            & pl.col("capacity_weight").is_finite()
            & (pl.col("capacity_weight") > 0.0)
        )
        .with_columns(
            (
                pl.col("theme_notional")
                * pl.col("capacity_weight")
                / pl.col("capacity_weight").sum().over(["decision_date", "side", "theme_id"])
            ).alias("member_weight")
        )
    )
    out = (
        members.group_by(["decision_date", "side", "quote_code"], maintain_order=True)
        .agg(
            pl.col("member_weight").sum().alias("defensive_weight"),
            pl.len().alias("selected_theme_count"),
            pl.col("theme_id").sort().alias("selected_theme_ids"),
            pl.first("source_date"),
            pl.first("n_themes"),
        )
        .sort(["decision_date", "side", "quote_code"])
    )
    dup = out.group_by(["decision_date", "side", "quote_code"]).len().filter(pl.col("len") > 1)
    if not dup.is_empty():
        raise ValueError(f"duplicate stock-day keys: {dup.head(3).to_dicts()}")
    bad = out.filter(pl.col("source_date") >= pl.col("decision_date"))
    if not bad.is_empty():
        raise ValueError(f"source_date not strictly prior: {bad.head(3).to_dicts()}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = build_defensive_stock_daily()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)
    summary = (
        out.group_by(["decision_date", "side"])
        .agg(pl.len().alias("n_stocks"), pl.col("defensive_weight").max().alias("max_w"))
        .group_by("side")
        .agg(
            pl.col("decision_date").n_unique().alias("days"),
            pl.col("decision_date").min().alias("first"),
            pl.col("decision_date").max().alias("last"),
            pl.col("n_stocks").mean().alias("mean_stocks"),
            pl.col("max_w").mean().alias("mean_max_w"),
        )
    )
    print(summary)
    print(f"wrote {out.height} rows -> {args.output}")


if __name__ == "__main__":
    main()
