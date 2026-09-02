"""Build the deployed timeSeries event index one trading day at a time.

This is the operational builder for the currently deployed negFill event. It
is intentionally independent from :mod:`src.research.timeSeries.framework`,
which is reserved for versioned event research. The deployed artifact remains
``data_index.parquet`` and no research manifest is written beside it.
"""

from __future__ import annotations

import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from src.pipeline_storage import load_pipeline_storage


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_STORAGE = load_pipeline_storage(project_root=PROJECT_ROOT)
TICK_FEATURE_DIR = PIPELINE_STORAGE.tick_feature_dir
OUTPUT_DIR = PROJECT_ROOT / "src" / "research" / "timeSeries"
OUTPUT_PATH = OUTPUT_DIR / "data_index.parquet"

INDEX_COLUMNS = ("Date", "QuoteCode", "ChannelSeq")
_DEFAULT_OUTPUT_MODE = 0o664
_REQUIRED_SOURCE_COLUMNS = (
    "QuoteCode",
    "ChannelSeq",
    "Spread",
    "BidPreMove",
    "FillLots_atLow",
)


class DeployedIndexError(RuntimeError):
    """Raised when the deployed event-index contract cannot be satisfied."""


def _empty_index() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Date": pl.String,
            "QuoteCode": pl.String,
            "ChannelSeq": pl.UInt64,
        }
    )


def _date_from_path(path: Path) -> str:
    date = path.name.split("_", 1)[0]
    _validate_date(date, f"tickFeature filename {path.name}")
    return date


def _validate_date(value: str, label: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid YYYYMMDD date") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise ValueError(f"{label} must be exactly eight YYYYMMDD digits")


def _validate_bounds(start: str | None, end: str | None) -> None:
    if start is not None:
        _validate_date(start, "start")
    if end is not None:
        _validate_date(end, "end")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start must be <= end, got {start} > {end}")


def _validate_index(frame: pl.DataFrame, *, check_unique: bool = False) -> None:
    if tuple(frame.columns) != INDEX_COLUMNS:
        raise DeployedIndexError(
            f"index columns must be {INDEX_COLUMNS}, got {tuple(frame.columns)}"
        )
    expected = {
        "Date": pl.String,
        "QuoteCode": pl.String,
        "ChannelSeq": pl.UInt64,
    }
    mismatches = {
        name: {"expected": dtype, "actual": frame.schema[name]}
        for name, dtype in expected.items()
        if frame.schema[name] != dtype
    }
    if mismatches:
        raise DeployedIndexError(f"index schema mismatch: {mismatches}")
    null_counts = {
        name: frame[name].null_count()
        for name in INDEX_COLUMNS
        if frame[name].null_count()
    }
    if null_counts:
        raise DeployedIndexError(f"index keys cannot contain null values: {null_counts}")
    if check_unique:
        duplicates = (
            frame.group_by(list(INDEX_COLUMNS)).len().filter(pl.col("len") > 1)
        )
        if not duplicates.is_empty():
            raise DeployedIndexError(
                f"index contains duplicate keys: {duplicates.head(5).to_dicts()}"
            )


def _atomic_write_index(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = (
        stat.S_IMODE(path.stat().st_mode) if path.exists() else _DEFAULT_OUTPUT_MODE
    )
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temporary_path, compression="zstd")
        os.chmod(temporary_path, output_mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def filter_one_day(path: Path) -> pl.DataFrame | None:
    """Return deployed negFill event keys from one tickFeature partition."""

    path = Path(path)
    date = _date_from_path(path)
    source = pl.scan_parquet(path)
    available = set(source.collect_schema().names())
    missing = [name for name in _REQUIRED_SOURCE_COLUMNS if name not in available]
    if missing:
        raise DeployedIndexError(f"{path}: missing required columns {missing}")

    selected = (
        source.sort(["QuoteCode", "ChannelSeq"])
        .with_columns(
            pl.col("Spread").diff().over("QuoteCode").alias("_spread_diff")
        )
        .filter(
            (pl.col("_spread_diff").abs() > 0.001)
            & (pl.col("BidPreMove").abs() > 0.001)
            & (pl.col("FillLots_atLow") < 0)
        )
        .select(
            pl.lit(date, dtype=pl.String).alias("Date"),
            pl.col("QuoteCode").cast(pl.String, strict=True),
            pl.col("ChannelSeq").cast(pl.UInt64, strict=True),
        )
        .sort(list(INDEX_COLUMNS))
        .collect()
    )
    _validate_index(selected, check_unique=True)
    return selected if selected.height else None


def build_index(
    forcefetch: bool = False,
    start: str | None = None,
    end: str | None = None,
    *,
    tick_feature_dir: Path | str = TICK_FEATURE_DIR,
    output_path: Path | str = OUTPUT_PATH,
) -> pl.DataFrame:
    """Incrementally append deployed event keys to ``data_index.parquet``.

    The normal daily run scans only source dates newer than the latest indexed
    date. ``forcefetch`` without bounds rebuilds the full index; with bounds it
    replaces only the requested source dates while preserving all other dates.
    """

    _validate_bounds(start, end)
    tick_feature_dir = Path(tick_feature_dir)
    output_path = Path(output_path)

    existing = pl.read_parquet(output_path) if output_path.exists() else _empty_index()
    _validate_index(existing)
    full_rebuild = forcefetch and start is None and end is None
    base = _empty_index() if full_rebuild else existing

    all_files = sorted(tick_feature_dir.glob("*_tickFeature.parquet"))
    if not all_files:
        print(f"No tickFeature files found in {tick_feature_dir}")
        return existing

    bounded_files: list[Path] = []
    for path in all_files:
        date = _date_from_path(path)
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        bounded_files.append(path)

    latest_date = existing["Date"].max() if existing.height else None
    if not forcefetch and latest_date is not None:
        files = [path for path in bounded_files if _date_from_path(path) > latest_date]
    else:
        files = bounded_files

    if not files:
        print(
            "No new tickFeature files"
            + (f" after {latest_date}" if latest_date is not None else "")
            + "; index is up-to-date."
        )
        return existing

    processed_dates = [_date_from_path(path) for path in files]
    if forcefetch and not full_rebuild and existing.height:
        base = existing.filter(~pl.col("Date").is_in(processed_dates))

    print(f"Scanning {len(files)} deployed event source file(s) ...")
    parts: list[pl.DataFrame] = []
    for position, path in enumerate(files, start=1):
        result = filter_one_day(path)
        row_count = 0 if result is None else result.height
        if result is not None:
            parts.append(result)
        print(f"  [{position}/{len(files)}] {_date_from_path(path)}: {row_count} unikeys")

    frames = [frame for frame in (base, *parts) if frame.height]
    index = pl.concat(frames, how="vertical") if frames else _empty_index()
    if forcefetch:
        index = index.sort(list(INDEX_COLUMNS))
    _validate_index(index)
    _atomic_write_index(index, output_path)

    new_count = sum(part.height for part in parts)
    print(
        f"Saved deployed data_index: {index.height} total unikeys "
        f"({new_count} selected in this run) -> {output_path}"
    )
    return index


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build deployed timeSeries data_index")
    parser.add_argument("--force", action="store_true", help="Rebuild selected dates")
    parser.add_argument("--start", default=None, help="YYYYMMDD, inclusive")
    parser.add_argument("--end", default=None, help="YYYYMMDD, inclusive")
    args = parser.parse_args()
    build_index(forcefetch=args.force, start=args.start, end=args.end)


if __name__ == "__main__":
    _main()
