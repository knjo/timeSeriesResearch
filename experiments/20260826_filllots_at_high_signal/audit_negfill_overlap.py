"""Diagnostic overlap of the selected FillHigh book with negFill keys."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from run_model_backtest import (
    MODEL_INPUT_PATH,
    SELECTED_SIGNALS_PATH,
    _calendar_sessions,
    _period_calendar,
    _summary_row,
)
from study import ARTIFACT_DIR, TAKER_FEE_BP, TIMING_LOOKBACK_DAYS


REFERENCE_INDEX = Path(__file__).resolve().parents[2] / "data_index.parquet"
OUTPUT_DIR = ARTIFACT_DIR / "negfill_overlap_v1"


def main() -> int:
    if not SELECTED_SIGNALS_PATH.exists():
        raise FileNotFoundError(SELECTED_SIGNALS_PATH)
    if not REFERENCE_INDEX.exists():
        raise FileNotFoundError(REFERENCE_INDEX)

    signals = pl.read_parquet(SELECTED_SIGNALS_PATH)
    event_keys = ["Date", "QuoteCode", "ChannelSeq"]
    stock_day_keys = ["Date", "QuoteCode"]
    selected_keys = signals.select(event_keys).unique()
    selected_stock_days = signals.select(stock_day_keys).unique()
    start = str(signals["Date"].min())
    end = str(signals["Date"].max())

    reference = pl.scan_parquet(REFERENCE_INDEX).filter(
        (pl.col("Date") >= start) & (pl.col("Date") <= end)
    )
    exact_hits = (
        reference.join(selected_keys.lazy(), on=event_keys, how="semi")
        .select(event_keys)
        .unique()
        .collect()
    )
    stock_day_hits = (
        reference.join(
            selected_stock_days.lazy(), on=stock_day_keys, how="semi"
        )
        .select(stock_day_keys)
        .unique()
        .collect()
    )
    annotated = (
        signals.join(
            exact_hits.with_columns(pl.lit(True).alias("exact_negfill_tick")),
            on=event_keys,
            how="left",
            validate="1:1",
        )
        .join(
            stock_day_hits.with_columns(
                pl.lit(True).alias("negfill_stock_day")
            ),
            on=stock_day_keys,
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.col("exact_negfill_tick").fill_null(False),
            pl.col("negfill_stock_day").fill_null(False),
        )
    )

    segments = {
        "all": pl.lit(True),
        "exact_negfill_tick": pl.col("exact_negfill_tick"),
        "exact_tick_disjoint": ~pl.col("exact_negfill_tick"),
        "negfill_stock_day": pl.col("negfill_stock_day"),
        "stock_day_disjoint": ~pl.col("negfill_stock_day"),
    }
    sessions = _calendar_sessions()
    event_dates = (
        pl.read_parquet(MODEL_INPUT_PATH, columns=["Date"])["Date"]
        .unique()
        .sort()
        .to_list()
    )
    model_start = event_dates[TIMING_LOOKBACK_DAYS]
    rows: list[dict[str, object]] = []
    for segment, expression in segments.items():
        subset = annotated.filter(expression)
        for period in ("development", "holdout_2026_ytd", "full_model_era"):
            row = _summary_row(
                subset,
                variant=segment,
                period=period,
                calendar=_period_calendar(sessions, period, model_start),
                fee_bp=TAKER_FEE_BP,
            )
            row["segment"] = segment
            rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated.write_parquet(
        OUTPUT_DIR / "selected_signals_with_overlap.parquet",
        compression="zstd",
    )
    summary = pl.DataFrame(rows).drop("signal_variant").sort(
        ["segment", "period"]
    )
    summary.write_parquet(OUTPUT_DIR / "period_summary.parquet", compression="zstd")
    reference_stat = REFERENCE_INDEX.stat()
    payload = {
        "contract": "filllots_at_high_selected_negfill_overlap/v1",
        "reference": {
            "path": str(REFERENCE_INDEX.resolve()),
            "size": reference_stat.st_size,
            "mtime_ns": reference_stat.st_mtime_ns,
        },
        "selected_events": len(signals),
        "exact_negfill_tick_events": len(exact_hits),
        "exact_negfill_tick_rate": len(exact_hits) / len(signals),
        "selected_stock_days": len(selected_stock_days),
        "negfill_stock_days": len(stock_day_hits),
        "negfill_stock_day_rate": len(stock_day_hits) / len(selected_stock_days),
        "scope": "diagnostic only; the reference index is not a model or execution input",
    }
    (OUTPUT_DIR / "overlap.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
