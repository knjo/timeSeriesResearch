"""Audit ambiguous D-1 liquidity references without reading outcome labels.

The preMarket producer replaces invalid depth and big-lot values with ``1``.
For pool-eligible stock sessions where a stored reference equals one, this
script rebuilds the producer inputs from the preceding raw trading session.
It distinguishes a genuine value of one from an untraceable fallback before
the fresh-event union or depth gate is interpreted.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[5]
REFERENCE_COLUMNS = (
    "avg_bidLots1",
    "avg_askLots1",
    "big_buy_lots",
    "big_sell_lots",
)


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _dated_paths(directory: Path, suffix: str) -> dict[str, Path]:
    return {
        path.name[:8]: path
        for path in sorted(directory.glob(f"*{suffix}"))
        if len(path.name) >= 8 and path.name[:8].isdigit()
    }


def _previous_session(date: str, available: list[str]) -> str:
    prior = [candidate for candidate in available if candidate < date]
    if not prior:
        raise ValueError(f"{date}: no preceding tickFeature session")
    return prior[-1]


def _pool_reference_rows(path: Path, date: str) -> pl.DataFrame:
    return pl.read_parquet(
        path,
        columns=(
            "QuoteCode",
            "day_amount_rank",
            "nextday_allow_day_trade_mark",
            *REFERENCE_COLUMNS,
        ),
    ).filter(
        (pl.col("nextday_allow_day_trade_mark") == "X")
        & (pl.col("day_amount_rank") <= 150)
    ).with_columns(pl.lit(date).alias("Date"))


def _ambiguous_pool_rows(frame: pl.DataFrame) -> pl.DataFrame:
    ambiguous = pl.any_horizontal(
        [(pl.col(column) - 1.0).abs() < 1e-12 for column in REFERENCE_COLUMNS]
    )
    return frame.filter(ambiguous)


def _rebuild_references(
    tick_path: Path,
    tick_feature_path: Path,
    quote_codes: list[str],
) -> pl.DataFrame:
    if not quote_codes:
        return pl.DataFrame(
            schema={
                "QuoteCode": pl.String,
                **{f"rebuilt_{column}": pl.Float64 for column in REFERENCE_COLUMNS},
            }
        )
    tick = pl.scan_parquet(tick_path).select(
        "QuoteCode",
        "ChannelSeq",
        "InOut",
        "FillLots",
        "BidLots1",
        "AskLots1",
    ).filter(pl.col("QuoteCode").is_in(quote_codes))
    feature_keys = pl.scan_parquet(tick_feature_path).select(
        "QuoteCode",
        "ChannelSeq",
    ).filter(pl.col("QuoteCode").is_in(quote_codes))
    merged = tick.join(
        feature_keys,
        on=["QuoteCode", "ChannelSeq"],
        how="inner",
        validate="1:1",
    ).collect()

    depth = merged.group_by("QuoteCode").agg(
        pl.col("BidLots1").mean().alias("rebuilt_avg_bidLots1"),
        pl.col("AskLots1").mean().alias("rebuilt_avg_askLots1"),
    )

    def rebuild_big(side: int, output_column: str) -> pl.DataFrame:
        fills = merged.filter(pl.col("InOut") == side).select(
            "QuoteCode", "FillLots"
        )
        if fills.is_empty():
            return pl.DataFrame(
                schema={"QuoteCode": pl.String, output_column: pl.Float64}
            )
        thresholds = fills.group_by("QuoteCode").agg(
            pl.col("FillLots").quantile(0.95).alias("_threshold")
        )
        return (
            fills.join(thresholds, on="QuoteCode", how="left", validate="m:1")
            .filter(pl.col("FillLots") >= pl.col("_threshold"))
            .group_by("QuoteCode")
            .agg(pl.col("FillLots").min().cast(pl.Float64).alias(output_column))
        )

    return (
        depth.join(
            rebuild_big(1, "rebuilt_big_buy_lots"),
            on="QuoteCode",
            how="left",
            validate="1:1",
        )
        .join(
            rebuild_big(-1, "rebuilt_big_sell_lots"),
            on="QuoteCode",
            how="left",
            validate="1:1",
        )
    )


def _classify(frame: pl.DataFrame) -> pl.DataFrame:
    status_columns: list[pl.Expr] = []
    for column in REFERENCE_COLUMNS:
        rebuilt = f"rebuilt_{column}"
        stored_is_one = (pl.col(column).cast(pl.Float64) - 1.0).abs() < 1e-12
        rebuilt_is_one = (
            (pl.col(rebuilt).cast(pl.Float64) - 1.0).abs() < 1e-12
        )
        status_columns.append(
            pl.when(~stored_is_one)
            .then(pl.lit("not_ambiguous"))
            .when(pl.col(rebuilt).is_null() | ~pl.col(rebuilt).is_finite())
            .then(pl.lit("unresolved_missing_raw_support"))
            .when(rebuilt_is_one)
            .then(pl.lit("proven_genuine_one"))
            .otherwise(pl.lit("stored_one_mismatch"))
            .alias(f"{column}_lineage_status")
        )
    status_names = [f"{column}_lineage_status" for column in REFERENCE_COLUMNS]
    return frame.with_columns(status_columns).with_columns(
        pl.any_horizontal(
            [
                pl.col(status).is_in(
                    ["unresolved_missing_raw_support", "stored_one_mismatch"]
                )
                for status in status_names
            ]
        ).alias("has_invalid_ambiguous_reference")
    )


def run(
    *,
    project_root: Path,
    start: str,
    end: str,
    output_dir: Path,
) -> tuple[pl.DataFrame, dict[str, object]]:
    data_dir = project_root / "data"
    pre_market = _dated_paths(data_dir / "preMarket", "_preMarketData.parquet")
    tick = _dated_paths(data_dir / "tickData", "_StockTick.parquet")
    tick_feature = _dated_paths(data_dir / "tickFeature", "_tickFeature.parquet")
    txf_tick = _dated_paths(data_dir / "txfTickData", "_TxfTick.parquet")
    complete_sources = set(pre_market) & set(tick) & set(tick_feature) & set(txf_tick)
    sessions = sorted(date for date in complete_sources if start <= date <= end)
    feature_sessions = sorted(tick_feature)
    if not sessions:
        raise ValueError("no preMarket sessions in requested range")

    audited: list[pl.DataFrame] = []
    source_map: dict[str, str] = {}
    eligible_stock_sessions = 0
    for date in sessions:
        pool_rows = _pool_reference_rows(pre_market[date], date)
        eligible_stock_sessions += len(pool_rows)
        ambiguous = _ambiguous_pool_rows(pool_rows)
        if ambiguous.is_empty():
            continue
        source_date = _previous_session(date, feature_sessions)
        if source_date not in tick:
            raise FileNotFoundError(
                f"{date}: preceding tickData source missing for {source_date}"
            )
        rebuilt = _rebuild_references(
            tick[source_date],
            tick_feature[source_date],
            ambiguous["QuoteCode"].unique().sort().to_list(),
        )
        audited.append(
            _classify(
                ambiguous.join(
                    rebuilt,
                    on="QuoteCode",
                    how="left",
                    validate="1:1",
                ).with_columns(pl.lit(source_date).alias("source_Date"))
            )
        )
        source_map[date] = source_date
        print(
            f"{date}: ambiguous_stock_days={len(ambiguous)}, "
            f"source={source_date}",
            flush=True,
        )

    if audited:
        result = pl.concat(audited, how="vertical").sort("Date", "QuoteCode")
    else:
        result = pl.DataFrame()
    invalid = (
        int(result["has_invalid_ambiguous_reference"].sum())
        if not result.is_empty()
        else 0
    )
    status_counts = {
        column: (
            {
                str(row[f"{column}_lineage_status"]): int(row["len"])
                for row in result.group_by(f"{column}_lineage_status")
                .len()
                .iter_rows(named=True)
            }
            if not result.is_empty()
            else {}
        )
        for column in REFERENCE_COLUMNS
    }
    summary: dict[str, object] = {
        "contract": "fh_neg_rejection_reference_lineage_audit/v1",
        "outcome_columns_read": [],
        "start": start,
        "end": end,
        "requested_pre_market_sessions": len(sessions),
        "eligible_pool_stock_sessions": eligible_stock_sessions,
        "sessions_with_ambiguous_one": len(source_map),
        "ambiguous_stock_sessions": len(result),
        "ambiguous_stock_session_rate": (
            len(result) / eligible_stock_sessions
            if eligible_stock_sessions
            else None
        ),
        "invalid_or_mismatched_stock_sessions": invalid,
        "invalid_or_mismatched_stock_session_rate": (
            invalid / eligible_stock_sessions
            if eligible_stock_sessions
            else None
        ),
        "reference_status_counts": status_counts,
        "all_ambiguous_ones_reconstructed": invalid == 0,
        "source_session_map": source_map,
    }
    _atomic_write_parquet(result, output_dir / "reference_lineage.parquet")
    _atomic_write_json(summary, output_dir / "reference_lineage.json")
    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild only preMarket references ambiguously stored as one"
    )
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260826")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "reference_lineage_v1",
    )
    args = parser.parse_args()
    if len(args.start) != 8 or not args.start.isdigit():
        parser.error("--start must be YYYYMMDD")
    if len(args.end) != 8 or not args.end.isdigit():
        parser.error("--end must be YYYYMMDD")
    if args.start > args.end:
        parser.error("--start cannot be after --end")
    return args


if __name__ == "__main__":
    cli = parse_args()
    _, audit_summary = run(
        project_root=PROJECT_ROOT,
        start=cli.start,
        end=cli.end,
        output_dir=cli.output_dir,
    )
    print(json.dumps(audit_summary, indent=2, sort_keys=True))
