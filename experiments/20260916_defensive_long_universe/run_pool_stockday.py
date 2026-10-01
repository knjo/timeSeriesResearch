"""Materialize the long taker eligible pool and keep only stock-day aggregates.

Pool and label are the pinned research references used by the 2026-08 long studies:
``tw_equity_daytrade_rank150_090030_123000_bidref_v1`` and
``benchmark.long_taker_tp109.v1`` (same-tick AskPrice1 taker entry, exit at
``1.09 * RefPrice`` when FutureHigh crosses it, otherwise 13:30 Close; hedge by
prior beta * TXF-to-13:30; z by prior residual vol).

Per (Date, QuoteCode) we store row counts and sums of raw / residual / z so any
stock-day condition (here: the defensive list) can be evaluated against the full
pool without re-reading ticks.  Output: ``artifacts/v1/pool_stockday/<date>.parquet``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader
from src.research.intradaySignal.tools.condition_benchmark import reference_pool_spec
from src.research.intradaySignal.framework.condition_study import build_eligible_labeled_pool
from src.research.intradaySignal.framework.events import DataRoots
from src.research.intradaySignal.framework.profiles.benchmark import LONG_TAKER_TP109_V1

PROJECT_ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = Path(__file__).with_name("artifacts") / "v1" / "pool_stockday"
AGG_COLUMNS = ("label_raw_bp", "label_residual_bp", "label_z", "label_hedge_pnl_bp")


def trading_dates(roots: DataRoots, start: str, end: str, step: int) -> list[str]:
    dates = sorted(
        p.name.split("_")[0]
        for p in roots.tick_data_dir.glob("*_StockTick.parquet")
        if start <= p.name.split("_")[0] <= end
    )
    return dates[::step]


def aggregate_day(date: str, roots: DataRoots, loader: TxfDataLoader) -> pl.DataFrame:
    pool = build_eligible_labeled_pool(
        date,
        roots=roots,
        pool_spec=reference_pool_spec(LONG_TAKER_TP109_V1),
        label_spec=LONG_TAKER_TP109_V1,
        txf_loader=loader,
    )
    agg_exprs = [pl.len().alias("n_rows")]
    for column in AGG_COLUMNS:
        finite = pl.col(column).is_finite() & pl.col(column).is_not_null()
        agg_exprs.append(finite.sum().alias(f"n_{column}"))
        agg_exprs.append(pl.when(finite).then(pl.col(column)).otherwise(0.0).sum().alias(f"sum_{column}"))
        agg_exprs.append(
            pl.when(finite).then(pl.col(column).pow(2)).otherwise(0.0).sum().alias(f"sumsq_{column}")
        )
    agg_exprs.extend(
        [
            pl.col("AskPrice1").mean().alias("mean_ask"),
            pl.col("txf_beta_60d").first().alias("txf_beta_60d"),
            pl.col("day_amount_rank").first().alias("day_amount_rank"),
            pl.col("RefPrice").first().alias("RefPrice"),
            pl.col("Close").first().alias("Close"),
        ]
    )
    out = pool.group_by("QuoteCode").agg(agg_exprs).with_columns(pl.lit(date).alias("Date"))
    return out.sort("QuoteCode")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20250211")
    parser.add_argument("--end", default="20260810")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    roots = DataRoots.from_project_root(PROJECT_ROOT)
    loader = TxfDataLoader(roots.txf_tick_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dates = trading_dates(roots, args.start, args.end, args.step)
    print(f"{len(dates)} dates {dates[0]}..{dates[-1]}", flush=True)
    failures: list[tuple[str, str]] = []
    for i, date in enumerate(dates):
        path = args.output_dir / f"{date}.parquet"
        if path.exists():
            continue
        try:
            frame = aggregate_day(date, roots, loader)
        except Exception as exc:  # noqa: BLE001 - keep going, record the day
            failures.append((date, repr(exc)[:200]))
            print(f"[{i + 1}/{len(dates)}] {date} FAILED {exc!r}"[:300], flush=True)
            continue
        tmp = path.with_suffix(".tmp.parquet")
        frame.write_parquet(tmp)
        tmp.replace(path)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[{i + 1}/{len(dates)}] {date} stocks={frame.height} rows={int(frame['n_rows'].sum())}", flush=True)
        del frame
        gc.collect()
    if failures:
        print("failures:", failures, file=sys.stderr)
    print("done", flush=True)


if __name__ == "__main__":
    main()
