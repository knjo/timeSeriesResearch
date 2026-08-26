"""Materialize the deployed timeSeries event points into daily parquet files.

This operational loader follows the established outer negFill pipeline shape:
read the fixed deployed ``data_index.parquet``, merge the required daily raw
sources, and write ``data/YYYYMMDD.parquet``. It intentionally does not import
the generic timeSeries research framework and never creates research manifests.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from src.dataloader.txfDataLoader import TxfDataLoader, attach_txf_to_1330_return
from src.features.market_beta import add_txf_event_residual_vol


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
TICK_DATA_DIR = DATA_DIR / "tickData"
TICK_FEATURE_DIR = DATA_DIR / "tickFeature"
PRE_MARKET_DIR = DATA_DIR / "preMarket"
TICK_BAR_DIR = DATA_DIR / "tickBar"
TXF_TICK_DIR = DATA_DIR / "txfTickData"
INDEX_PATH = Path(__file__).with_name("data_index.parquet")
OUTPUT_DIR = Path(__file__).with_name("data")
UNIKEY_COLS = ["QuoteCode", "ChannelSeq"]
_DEFAULT_OUTPUT_MODE = 0o664


@dataclass(frozen=True)
class DeploymentDataPaths:
    tick_data_dir: Path
    tick_feature_dir: Path
    pre_market_dir: Path
    tick_bar_dir: Path
    txf_tick_dir: Path

    def __post_init__(self) -> None:
        for field_name in (
            "tick_data_dir",
            "tick_feature_dir",
            "pre_market_dir",
            "tick_bar_dir",
            "txf_tick_dir",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))


@dataclass(frozen=True)
class DeploymentBuildReport:
    saved_dates: tuple[str, ...]
    skipped_dates: tuple[str, ...]
    refreshed_dates: tuple[str, ...]


DATA_PATHS = DeploymentDataPaths(
    tick_data_dir=TICK_DATA_DIR,
    tick_feature_dir=TICK_FEATURE_DIR,
    pre_market_dir=PRE_MARKET_DIR,
    tick_bar_dir=TICK_BAR_DIR,
    txf_tick_dir=TXF_TICK_DIR,
)


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


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
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


def _read_index(
    date: str | None = None, *, index_path: Path | str = INDEX_PATH
) -> pl.DataFrame:
    """Read the deployed event index, optionally restricted to one date."""

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"data_index not found at {path}. Run build_index.py first."
        )
    index = pl.read_parquet(path)
    required = {"Date", *UNIKEY_COLS}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"deployed index is missing columns {missing}: {path}")
    if date is not None:
        index = index.filter(pl.col("Date").cast(pl.String) == date)
    return index


def _merge_one_day(
    date: str,
    keys: pl.DataFrame,
    txf_loader: TxfDataLoader | None = None,
    *,
    data_paths: DeploymentDataPaths = DATA_PATHS,
) -> pl.DataFrame | None:
    """Merge one deployed event day using the established legacy alignment."""

    tick_feature_path = data_paths.tick_feature_dir / f"{date}_tickFeature.parquet"
    if not tick_feature_path.exists():
        print(f"  [SKIP] tickFeature not found: {tick_feature_path.name}")
        return None
    tick_feature = (
        pl.scan_parquet(tick_feature_path)
        .join(keys.lazy(), on=UNIKEY_COLS, how="semi")
        .collect()
    )

    tick_data_path = data_paths.tick_data_dir / f"{date}_StockTick.parquet"
    if not tick_data_path.exists():
        print(f"  [SKIP] tickData not found: {tick_data_path.name}")
        return None
    tick_data = (
        pl.scan_parquet(tick_data_path)
        .join(keys.lazy(), on=UNIKEY_COLS, how="semi")
        .collect()
    )

    merged = tick_data.join(
        tick_feature,
        on=UNIKEY_COLS,
        how="left",
        suffix="_right",
    )

    pre_market_path = data_paths.pre_market_dir / f"{date}_preMarketData.parquet"
    if pre_market_path.exists():
        pre_market = pl.read_parquet(pre_market_path)
        if "ClosePrice" in pre_market.columns:
            if "PreviousClosePrice" not in pre_market.columns:
                pre_market = pre_market.with_columns(
                    pl.col("ClosePrice").alias("PreviousClosePrice")
                )
            pre_market = pre_market.drop("ClosePrice")
        merged = merged.join(pre_market, on="QuoteCode", how="left", suffix="_pm")
    else:
        print(f"  [WARN] preMarket not found: {pre_market_path.name}, skipping")

    tick_bar_path = data_paths.tick_bar_dir / f"{date}_tickBar.parquet"
    if tick_bar_path.exists():
        tick_bar = pl.read_parquet(tick_bar_path)
        rank_columns = [
            column
            for column in tick_bar.columns
            if column.endswith("Rank") or column.endswith("Count")
        ]
        if "AmountRank_canDayTrade" in tick_bar.columns:
            rank_columns.append("AmountRank_canDayTrade")
        selected_columns = [
            column
            for column in ["QuoteCode", "TransTime", "TimeSlot", *rank_columns]
            if column in tick_bar.columns
        ]
        tick_bar = tick_bar.select(list(dict.fromkeys(selected_columns)))

        if "TransTime" in merged.columns and "TransTime" in tick_bar.columns:
            merged = merged.sort(["QuoteCode", "TransTime"])
            tick_bar = (
                tick_bar.sort(["QuoteCode", "TransTime"])
                .rename({"TransTime": "TransTime_tb"})
                .with_columns(
                    (pl.col("TransTime_tb") + pl.duration(microseconds=1)).alias(
                        "TransTime_tb"
                    )
                )
            )
            merged = merged.join_asof(
                tick_bar,
                left_on="TransTime",
                right_on="TransTime_tb",
                by="QuoteCode",
                strategy="backward",
            )
        else:
            print(f"  [WARN] {date}: TransTime column missing, skipping tickBar join")
    else:
        print(f"  [WARN] tickBar not found: {tick_bar_path.name}, skipping")

    if txf_loader is None:
        txf_loader = TxfDataLoader(data_paths.txf_tick_dir)
    merged = attach_txf_to_1330_return(merged, date, txf_loader)
    merged = add_txf_event_residual_vol(merged)
    merged = merged.with_columns(pl.lit(date).alias("Date"))

    if "nextday_allow_day_trade_mark" in merged.columns:
        merged = merged.filter(pl.col("nextday_allow_day_trade_mark") == "X")
    elif "nextday_allow_day_trade_mark_pm" in merged.columns:
        merged = merged.filter(pl.col("nextday_allow_day_trade_mark_pm") == "X")
    else:
        print(
            f"  [WARN] {date}: nextday_allow_day_trade_mark column not found, "
            "no filter applied"
        )
    return merged


def build_merged_data(
    forcefetch: bool = False,
    start: str | None = None,
    end: str | None = None,
    *,
    index_path: Path | str = INDEX_PATH,
    output_dir: Path | str = OUTPUT_DIR,
    data_paths: DeploymentDataPaths = DATA_PATHS,
) -> DeploymentBuildReport:
    """Incrementally materialize deployed event dates into daily parquet files."""

    _validate_bounds(start, end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _read_index(index_path=index_path)
    dates = index["Date"].cast(pl.String).unique().sort().to_list()
    if start is not None:
        dates = [date for date in dates if date >= start]
    if end is not None:
        dates = [date for date in dates if date <= end]

    existing_files = sorted(output_dir.glob("*.parquet"))
    latest_existing = max((path.stem for path in existing_files), default=None)
    txf_loader = TxfDataLoader(data_paths.txf_tick_dir)
    saved: list[str] = []
    skipped: list[str] = []
    refreshed: list[str] = []

    print(f"Total deployed dates in index: {len(dates)}")
    print(f"Output directory: {output_dir}")
    if latest_existing is not None:
        print(f"Last existing date to refresh: {latest_existing}")

    for position, date in enumerate(dates, start=1):
        output_path = output_dir / f"{date}.parquet"
        if output_path.exists() and not forcefetch:
            if date != latest_existing:
                skipped.append(date)
                continue
            refreshed.append(date)
            print(f"  [INFO] {date} is the latest existing date; refreshing it.")

        keys = index.filter(pl.col("Date").cast(pl.String) == date).select(UNIKEY_COLS)
        result = _merge_one_day(
            date,
            keys,
            txf_loader,
            data_paths=data_paths,
        )
        if result is None or result.is_empty():
            print(f"  [{position}/{len(dates)}] {date}: 0 rows after filter")
            continue
        _atomic_write_parquet(result, output_path)
        saved.append(date)
        print(f"  [{position}/{len(dates)}] {date}: {result.height} rows saved")

    print(
        f"Done. Saved: {len(saved)} "
        f"(Refreshed: {len(refreshed)}), Skipped: {len(skipped)}"
    )
    return DeploymentBuildReport(tuple(saved), tuple(skipped), tuple(refreshed))


def read_time_series(
    date: str | None = None,
    dates: list[str] | None = None,
    *,
    output_dir: Path | str = OUTPUT_DIR,
) -> pl.DataFrame:
    """Read deployed daily parquet files with the historical union policy."""

    root = Path(output_dir)
    if date is not None:
        target_files = [root / f"{date}.parquet"]
    elif dates is not None:
        target_files = [root / f"{value}.parquet" for value in dates]
    else:
        target_files = sorted(root.glob("*.parquet"))

    parts: list[pl.DataFrame] = []
    for path in target_files:
        if path.exists():
            parts.append(pl.read_parquet(path))
        else:
            print(f"  [WARN] not found: {path.name}")
    if not parts:
        print("No data loaded.")
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build deployed timeSeries data (per-day)"
    )
    parser.add_argument("--force", action="store_true", help="Rebuild selected dates")
    parser.add_argument("--start", default=None, help="YYYYMMDD, inclusive")
    parser.add_argument("--end", default=None, help="YYYYMMDD, inclusive")
    args = parser.parse_args()
    build_merged_data(forcefetch=args.force, start=args.start, end=args.end)


if __name__ == "__main__":
    _main()
