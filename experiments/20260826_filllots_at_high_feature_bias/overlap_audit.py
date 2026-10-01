"""Audit sparse promoted events against the pinned negFill reference event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from src.research.intradaySignal.framework.events import DAY_KEY_COLS, DataRoots, EventDaySources
from src.research.intradaySignal.framework.profiles.filllots_at_high import (
    select_filllots_at_high_feature_candidates,
)
from src.research.intradaySignal.framework.profiles.negfill import select_negfill_events


PROJECT_ROOT = Path(__file__).resolve().parents[5]
REFERENCE_ID = "negfill_spread_bid_move_v1"


def _pair_counts(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, int | float | None]:
    intersection = len(left.join(right, on=list(DAY_KEY_COLS), how="inner"))
    union = len(left) + len(right) - intersection
    return {
        "candidate_ticks": len(left),
        "reference_ticks": len(right),
        "intersection_ticks": intersection,
        "candidate_tick_overlap_rate": intersection / len(left) if len(left) else None,
        "reference_tick_overlap_rate": intersection / len(right) if len(right) else None,
        "tick_jaccard": intersection / union if union else None,
    }


def _stock_counts(left: pl.DataFrame, right: pl.DataFrame) -> dict[str, int | float | None]:
    left_stocks = left.select("QuoteCode").unique()
    right_stocks = right.select("QuoteCode").unique()
    intersection = len(left_stocks.join(right_stocks, on="QuoteCode", how="inner"))
    union = len(left_stocks) + len(right_stocks) - intersection
    return {
        "candidate_stock_days": len(left_stocks),
        "reference_stock_days": len(right_stocks),
        "intersection_stock_days": intersection,
        "candidate_stock_day_overlap_rate": (
            intersection / len(left_stocks) if len(left_stocks) else None
        ),
        "reference_stock_day_overlap_rate": (
            intersection / len(right_stocks) if len(right_stocks) else None
        ),
        "stock_day_jaccard": intersection / union if union else None,
    }


def build_overlap_daily(
    dates: list[str],
    candidate_ids: tuple[str, ...],
) -> pl.DataFrame:
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    rows: list[dict[str, Any]] = []
    for index, date in enumerate(dates, start=1):
        day = EventDaySources(date, roots)
        high = select_filllots_at_high_feature_candidates(
            day,
            candidate_ids=candidate_ids,
        )
        reference = select_negfill_events(day).collect().unique().sort(list(DAY_KEY_COLS))
        for candidate_id in candidate_ids:
            candidate = high[candidate_id].unique().sort(list(DAY_KEY_COLS))
            rows.append(
                {
                    "Date": date,
                    "candidate_id": candidate_id,
                    "reference_id": REFERENCE_ID,
                    **_pair_counts(candidate, reference),
                    **_stock_counts(candidate, reference),
                }
            )
        print(f"[{index}/{len(dates)}] {date}", flush=True)
    return pl.DataFrame(rows).sort("Date", "candidate_id")


def summarize_overlap(daily: pl.DataFrame) -> pl.DataFrame:
    additive = (
        daily.group_by("candidate_id", "reference_id")
        .agg(
            pl.len().alias("sessions"),
            pl.col("candidate_ticks").sum(),
            pl.col("reference_ticks").sum(),
            pl.col("intersection_ticks").sum(),
            pl.col("candidate_stock_days").sum(),
            pl.col("reference_stock_days").sum(),
            pl.col("intersection_stock_days").sum(),
        )
        .with_columns(
            (
                pl.col("intersection_ticks") / pl.col("candidate_ticks")
            ).alias("candidate_tick_overlap_rate"),
            (
                pl.col("intersection_ticks") / pl.col("reference_ticks")
            ).alias("reference_tick_overlap_rate"),
            (
                pl.col("intersection_ticks")
                / (
                    pl.col("candidate_ticks")
                    + pl.col("reference_ticks")
                    - pl.col("intersection_ticks")
                )
            ).alias("tick_jaccard"),
            (
                pl.col("intersection_stock_days")
                / pl.col("candidate_stock_days")
            ).alias("candidate_stock_day_overlap_rate"),
            (
                pl.col("intersection_stock_days")
                / pl.col("reference_stock_days")
            ).alias("reference_stock_day_overlap_rate"),
            (
                pl.col("intersection_stock_days")
                / (
                    pl.col("candidate_stock_days")
                    + pl.col("reference_stock_days")
                    - pl.col("intersection_stock_days")
                )
            ).alias("stock_day_jaccard"),
        )
    )
    return additive.sort("candidate_id")


def effect_correlations(
    daily_effects: pl.DataFrame,
    selected: dict[str, str],
    state_dirs: dict[str, Path],
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    base_ids = {"short": "fh_neg_sample", "long": "fh_pos_sample"}
    for direction in ("short", "long"):
        candidate_id = selected[direction]
        high = daily_effects.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == candidate_id)
        ).select("Date", pl.col("daily_shared_delta_z").alias("high_delta_z"))
        base = daily_effects.filter(
            (pl.col("direction") == direction)
            & (pl.col("candidate_id") == base_ids[direction])
        ).select("Date", pl.col("daily_shared_delta_z").alias("base_delta_z"))
        state = (
            pl.read_parquet(state_dirs[direction] / "daily_stats.parquet")
            .filter(pl.col("candidate_id") == "negative_state")
            .with_columns(
                (
                    pl.col("condition_sum_z") / pl.col("condition_n_z")
                    - pl.col("control_sum_z") / pl.col("control_n_z")
                ).alias("filllow_negative_delta_z")
            )
            .select("Date", "filllow_negative_delta_z")
        )
        joined = high.join(base, on="Date", validate="1:1").join(
            state,
            on="Date",
            validate="1:1",
        )
        rows.append(
            {
                "direction": direction,
                "candidate_id": candidate_id,
                "sessions": len(joined),
                "fillhigh_base_id": base_ids[direction],
                "corr_with_fillhigh_base": joined.select(
                    pl.corr("high_delta_z", "base_delta_z")
                ).item(),
                "corr_with_filllow_negative_state": joined.select(
                    pl.corr("high_delta_z", "filllow_negative_delta_z")
                ).item(),
            }
        )
    return pl.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promotion-json", required=True, type=Path)
    parser.add_argument("--sparse-manifest", required=True, type=Path)
    parser.add_argument("--sparse-report-dir", required=True, type=Path)
    parser.add_argument("--short-state-dir", required=True, type=Path)
    parser.add_argument("--long-state-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--date-step", type=int, default=1)
    args = parser.parse_args()
    if args.date_step < 1:
        parser.error("--date-step must be positive")

    promotion = json.loads(args.promotion_json.read_text(encoding="utf-8"))
    selected = {
        row["direction"]: row["candidate_id"]
        for row in promotion["selected"]
        if row["decision"] == "promoted"
    }
    if set(selected) != {"short", "long"}:
        raise ValueError("overlap audit requires one promoted candidate per direction")
    manifest = json.loads(args.sparse_manifest.read_text(encoding="utf-8"))
    dates = manifest.get("sessions")
    if not isinstance(dates, list) or not dates:
        raise ValueError("sparse manifest has no sessions")
    audit_dates = dates[:: args.date_step]
    if audit_dates[-1] != dates[-1]:
        audit_dates.append(dates[-1])

    candidate_ids = tuple(dict.fromkeys(selected.values()))
    daily = build_overlap_daily(audit_dates, candidate_ids)
    summary = summarize_overlap(daily)
    correlations = effect_correlations(
        pl.read_parquet(args.sparse_report_dir / "daily_effects.parquet"),
        selected,
        {"short": args.short_state_dir, "long": args.long_state_dir},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(args.output_dir / "overlap_daily.parquet")
    summary.write_parquet(args.output_dir / "overlap_summary.parquet")
    correlations.write_parquet(args.output_dir / "effect_correlations.parquet")
    (args.output_dir / "overlap_report.json").write_text(
        json.dumps(
            {
                "contract": "filllots_at_high_overlap_audit/v1",
                "scope": "raw selector keys on a deterministic sparse-session audit",
                "date_step": args.date_step,
                "source_session_count": len(dates),
                "audited_dates": audit_dates,
                "reference_id": REFERENCE_ID,
                "selected": selected,
                "overlap_summary": summary.to_dicts(),
                "effect_correlations": correlations.to_dicts(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
