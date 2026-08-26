"""negFill-compatible adapter for the generic event dataset framework.

New event families should call :mod:`src.research.timeSeries.framework.dataset`
directly and choose a versioned event index, universe, and merge policy. This
module keeps the historical ``build_merged_data``, ``_merge_one_day`` and
``read_time_series`` entry points so existing negFill notebooks still run.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader
from src.research.timeSeries.framework.dataset import (
    EventDatasetSpec,
    build_event_dataset,
    merge_event_day,
    read_event_dataset,
)
from src.research.timeSeries.framework.events import DataRoots
from src.research.timeSeries.framework.profiles.negfill import (
    negfill_causal_merge_options_v2,
    negfill_legacy_merge_options_v1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROOTS = DataRoots.from_project_root(PROJECT_ROOT)

# Compatibility constants used by older research scripts.
DATA_DIR = PROJECT_ROOT / "data"
TICK_DATA_DIR = ROOTS.tick_data_dir
TICK_FEATURE_DIR = ROOTS.tick_feature_dir
PRE_MARKET_DIR = ROOTS.pre_market_dir
TICK_BAR_DIR = ROOTS.tick_bar_dir
TXF_TICK_DIR = ROOTS.txf_tick_dir
INDEX_PATH = Path(__file__).with_name("data_index.parquet")
OUTPUT_DIR = Path(__file__).with_name("data")
UNIKEY_COLS = ["QuoteCode", "ChannelSeq"]


def _read_index(
    date: str | None = None, *, index_path: Path | str = INDEX_PATH
) -> pl.DataFrame:
    """Read the reference event index, optionally restricted to one date."""

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"event index not found at {path}. Run build_index.py first."
        )
    index = pl.read_parquet(path)
    if date is not None:
        index = index.filter(pl.col("Date").cast(pl.String) == date)
    return index


def _merge_one_day(
    date: str,
    keys: pl.DataFrame,
    txf_loader: TxfDataLoader | None = None,
) -> pl.DataFrame:
    """Historical negFill merge profile (deprecated compatibility entry point).

    The old stock-tick-time tickBar alignment and next-day day-trade universe
    are deliberately named in ``negfill_legacy_merge_options_v1``. Generic
    studies should use ``merge_event_day`` with the causal default instead.
    """

    if "Date" not in keys.columns:
        keys = keys.with_columns(pl.lit(date).alias("Date"))
    return merge_event_day(
        date,
        keys,
        roots=ROOTS,
        options=negfill_legacy_merge_options_v1(),
        txf_loader=txf_loader,
    )


def build_merged_data(
    forcefetch: bool = False,
    start: str | None = None,
    end: str | None = None,
    *,
    index_path: Path | str = INDEX_PATH,
    output_dir: Path | str = OUTPUT_DIR,
    merge_profile: str = "negfill-legacy-v1",
):
    """Materialize the reference event dataset with explicit merge semantics.

    ``negfill-legacy-v1`` preserves historical notebooks. ``causal-v2`` keeps
    the reference day-trade universe but uses completed tickBar slots; use a
    distinct versioned output directory for it.
    """

    if merge_profile == "negfill-legacy-v1":
        options = negfill_legacy_merge_options_v1()
        dataset_version = "negfill-legacy-v1"
    elif merge_profile == "causal-v2":
        options = negfill_causal_merge_options_v2()
        dataset_version = "causal-v2"
    else:
        raise ValueError(f"unknown merge_profile={merge_profile!r}")

    spec = EventDatasetSpec(
        event_id="negfill",
        dataset_version=dataset_version,
        index_path=Path(index_path),
        output_dir=Path(output_dir),
        merge_options=options,
        allow_unmanifested_existing=(merge_profile == "negfill-legacy-v1"),
    )
    report = build_event_dataset(
        spec,
        roots=ROOTS,
        force=forcefetch,
        start=start,
        end=end,
        # Source/index identities now drive refreshes. In particular, first
        # legacy adoption must not rewrite the newest historical partition.
        refresh_latest=False,
    )
    print(
        f"Done. Saved: {len(report.saved_dates)}, "
        f"Skipped: {len(report.skipped_dates)}, profile: {merge_profile}"
    )
    return report


def read_time_series(
    date: str | None = None,
    dates: list[str] | None = None,
    *,
    output_dir: Path | str = OUTPUT_DIR,
) -> pl.DataFrame:
    """Read legacy timeSeries files with their historical union-schema policy."""

    # Preserve the old precedence rule when both were accidentally supplied.
    selected_dates = None if date is not None else dates
    return read_event_dataset(
        output_dir,
        date=date,
        dates=selected_dates,
        strict_files=False,
        schema_policy="union",
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a versioned event dataset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start", default=None, help="YYYYMMDD, inclusive")
    parser.add_argument("--end", default=None, help="YYYYMMDD, inclusive")
    parser.add_argument("--index-path", type=Path, default=INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--merge-profile",
        choices=["negfill-legacy-v1", "causal-v2"],
        default="negfill-legacy-v1",
    )
    args = parser.parse_args()
    build_merged_data(
        forcefetch=args.force,
        start=args.start,
        end=args.end,
        index_path=args.index_path,
        output_dir=args.output_dir,
        merge_profile=args.merge_profile,
    )


if __name__ == "__main__":
    _main()
