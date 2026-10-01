"""Generic event-index construction for time-series research.

An event selector receives paths for one trading day and may scan any declared
combination of tickData, tickFeature, preMarket, tickBar, and TXF data.  The
framework owns the stable output contract: one unique, sorted
``(Date, QuoteCode, ChannelSeq)`` row per event.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import polars as pl

from src.pipeline_storage import load_pipeline_storage


SourceName: TypeAlias = Literal[
    "tick_data",
    "tick_feature",
    "pre_market",
    "tick_bar",
    "txf_tick",
    "brench_feature",
]
Frame: TypeAlias = pl.DataFrame | pl.LazyFrame

EVENT_KEY_COLS = ("Date", "QuoteCode", "ChannelSeq")
DAY_KEY_COLS = ("QuoteCode", "ChannelSeq")
EVENT_INDEX_FORMAT_VERSION = 1

_EVENT_SCHEMA = {
    "Date": pl.String,
    "QuoteCode": pl.String,
    "ChannelSeq": pl.UInt64,
}
_INTEGER_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
_SOURCE_FILENAMES: dict[SourceName, str] = {
    "tick_data": "{date}_StockTick.parquet",
    "tick_feature": "{date}_tickFeature.parquet",
    "pre_market": "{date}_preMarketData.parquet",
    "tick_bar": "{date}_tickBar.parquet",
    "txf_tick": "{date}_TxfTick.parquet",
    "brench_feature": "{date}_brenchFeature.parquet",
}
_SOURCE_GLOBS: dict[SourceName, str] = {
    "tick_data": "*_StockTick.parquet",
    "tick_feature": "*_tickFeature.parquet",
    "pre_market": "*_preMarketData.parquet",
    "tick_bar": "*_tickBar.parquet",
    "txf_tick": "*_TxfTick.parquet",
    "brench_feature": "*_brenchFeature.parquet",
}


class EventIndexError(RuntimeError):
    """Base error for event-index construction."""


class EventIndexValidationError(EventIndexError):
    """Raised when an event selector or an existing index violates the key contract."""


class EventIndexManifestError(EventIndexError):
    """Raised when an existing artifact does not match the requested event spec."""


@dataclass(frozen=True)
class DataRoots:
    """Directories containing the canonical daily research inputs."""

    tick_data_dir: Path
    tick_feature_dir: Path
    pre_market_dir: Path
    tick_bar_dir: Path
    txf_tick_dir: Path
    brench_feature_dir: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "tick_data_dir",
            "tick_feature_dir",
            "pre_market_dir",
            "tick_bar_dir",
            "txf_tick_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.brench_feature_dir is not None:
            object.__setattr__(
                self, "brench_feature_dir", Path(self.brench_feature_dir)
            )

    @classmethod
    def from_project_root(cls, project_root: Path) -> DataRoots:
        root = Path(project_root)
        config_path = root / "config" / "pipeline.yaml"
        if config_path.is_file():
            storage = load_pipeline_storage(
                project_root=root,
                config_path=config_path,
            )
            return cls(
                tick_data_dir=storage.tick_dir,
                tick_feature_dir=storage.tick_feature_dir,
                pre_market_dir=storage.pre_market_dir,
                tick_bar_dir=storage.tick_bar_dir,
                txf_tick_dir=storage.txf_tick_dir,
                brench_feature_dir=storage.brench_feature_dir,
            )

        data_dir = root / "data"
        return cls(
            tick_data_dir=data_dir / "tickData",
            tick_feature_dir=data_dir / "tickFeature",
            pre_market_dir=data_dir / "preMarket",
            tick_bar_dir=data_dir / "tickBar",
            txf_tick_dir=data_dir / "txfTickData",
            brench_feature_dir=data_dir / "brenchFeature",
        )

    def source_dir(self, source: SourceName) -> Path:
        if source == "brench_feature":
            if self.brench_feature_dir is None:
                raise EventIndexError(
                    "DataRoots has no brench_feature_dir; brenchFeature is unavailable"
                )
            return self.brench_feature_dir
        return {
            "tick_data": self.tick_data_dir,
            "tick_feature": self.tick_feature_dir,
            "pre_market": self.pre_market_dir,
            "tick_bar": self.tick_bar_dir,
            "txf_tick": self.txf_tick_dir,
        }[source]


@dataclass(frozen=True)
class EventDaySources:
    """Lazy, multi-source input surface passed to an event selector."""

    date: str
    roots: DataRoots

    def __post_init__(self) -> None:
        _validate_date(self.date, "event day")

    def path(self, source: SourceName) -> Path:
        return self.roots.source_dir(source) / _SOURCE_FILENAMES[source].format(
            date=self.date
        )

    def require(self, *sources: SourceName) -> None:
        missing = [str(self.path(source)) for source in sources if not self.path(source).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self.date}: missing required event source(s): {', '.join(missing)}"
            )

    def scan(self, source: SourceName) -> pl.LazyFrame:
        path = self.path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{self.date}: event source not found: {path}")
        return pl.scan_parquet(path)

    def scan_tick_data(self) -> pl.LazyFrame:
        return self.scan("tick_data")

    def scan_tick_feature(self) -> pl.LazyFrame:
        return self.scan("tick_feature")

    def scan_pre_market(self) -> pl.LazyFrame:
        return self.scan("pre_market")

    def scan_tick_bar(self) -> pl.LazyFrame:
        return self.scan("tick_bar")

    def scan_txf_tick(self) -> pl.LazyFrame:
        return self.scan("txf_tick")

    def scan_brench_feature(self) -> pl.LazyFrame:
        return self.scan("brench_feature")


class EventSelector(Protocol):
    """Select event tick keys from any combination of declared day sources."""

    def __call__(self, day: EventDaySources) -> Frame: ...


@dataclass(frozen=True)
class EventIndexSpec:
    """Versioned definition and destination of one event family."""

    event_id: str
    selector_version: str
    selector: EventSelector
    index_path: Path
    required_sources: tuple[SourceName, ...] = ("tick_feature",)
    calendar_source: SourceName = "tick_feature"
    allow_unmanifested_existing: bool = False

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id cannot be empty")
        if not self.selector_version.strip():
            raise ValueError("selector_version cannot be empty")
        if not callable(self.selector):
            raise TypeError("selector must be callable")
        if len(set(self.required_sources)) != len(self.required_sources):
            raise ValueError("required_sources cannot contain duplicates")
        object.__setattr__(self, "index_path", Path(self.index_path))

    @property
    def manifest_path(self) -> Path:
        return self.index_path.with_suffix(".manifest.json")


@dataclass(frozen=True)
class EventIndexBuildReport:
    """Auditable summary of an event-index build or resume."""

    event_id: str
    selector_version: str
    index_path: Path
    manifest_path: Path
    row_count: int
    unique_key_count: int
    event_date_count: int
    processed_date_count: int
    start_date: str | None
    end_date: str | None
    scanned_dates: tuple[str, ...]
    refreshed_dates: tuple[str, ...]
    skipped_dates: tuple[str, ...]

    def read_index(self) -> pl.DataFrame:
        return pl.read_parquet(self.index_path)


def empty_event_index() -> pl.DataFrame:
    """Return an empty frame with the canonical event-index schema."""

    return pl.DataFrame(schema=_EVENT_SCHEMA)


def normalize_event_keys(frame: Frame, *, date: str | None = None) -> pl.DataFrame:
    """Normalize selector output and enforce the canonical event-key contract."""

    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise EventIndexValidationError(
            "event selector must return a polars DataFrame or LazyFrame"
        )

    missing = [column for column in DAY_KEY_COLS if column not in frame.columns]
    if missing:
        raise EventIndexValidationError(
            f"event selector is missing required key column(s): {missing}"
        )

    schema = frame.schema
    if schema["QuoteCode"] != pl.String:
        raise EventIndexValidationError(
            f"QuoteCode must be String, got {schema['QuoteCode']}"
        )
    if schema["ChannelSeq"] not in _INTEGER_DTYPES:
        raise EventIndexValidationError(
            f"ChannelSeq must be an integer type, got {schema['ChannelSeq']}"
        )

    if "Date" not in frame.columns:
        if date is None:
            raise EventIndexValidationError(
                "Date is absent from selector output and no event day was supplied"
            )
        _validate_date(date, "event day")
        frame = frame.with_columns(pl.lit(date, dtype=pl.String).alias("Date"))
    elif frame.schema["Date"] != pl.String:
        raise EventIndexValidationError(f"Date must be String, got {frame.schema['Date']}")

    frame = frame.select(EVENT_KEY_COLS)
    try:
        frame = frame.with_columns(pl.col("ChannelSeq").cast(pl.UInt64, strict=True))
    except Exception as exc:
        raise EventIndexValidationError("ChannelSeq cannot be normalized to UInt64") from exc

    if date is not None:
        wrong_date = frame.filter(pl.col("Date") != date)
        if wrong_date.height:
            sample = wrong_date.head(5).to_dicts()
            raise EventIndexValidationError(
                f"selector output contains rows outside {date}: {sample}"
            )

    frame = frame.sort(EVENT_KEY_COLS)
    validate_event_index(frame, require_sorted=True)
    return frame


def validate_event_index(
    frame: pl.DataFrame,
    *,
    require_sorted: bool = True,
) -> None:
    """Fail closed if a frame is not a canonical event index."""

    if not isinstance(frame, pl.DataFrame):
        raise EventIndexValidationError("event index must be a polars DataFrame")
    if tuple(frame.columns) != EVENT_KEY_COLS:
        raise EventIndexValidationError(
            f"event index columns must be {EVENT_KEY_COLS}, got {tuple(frame.columns)}"
        )
    for column, expected_dtype in _EVENT_SCHEMA.items():
        actual_dtype = frame.schema[column]
        if actual_dtype != expected_dtype:
            raise EventIndexValidationError(
                f"event index {column} must be {expected_dtype}, got {actual_dtype}"
            )

    null_counts = {
        column: frame[column].null_count()
        for column in EVENT_KEY_COLS
        if frame[column].null_count()
    }
    if null_counts:
        raise EventIndexValidationError(f"event keys cannot be null: {null_counts}")

    duplicates = (
        frame.group_by(EVENT_KEY_COLS)
        .len()
        .filter(pl.col("len") > 1)
        .head(5)
    )
    if duplicates.height:
        raise EventIndexValidationError(
            f"event index contains duplicate keys: {duplicates.to_dicts()}"
        )

    for value in frame["Date"].unique().to_list():
        _validate_date(value, "event index Date")

    if require_sorted and not frame.equals(frame.sort(EVENT_KEY_COLS)):
        raise EventIndexValidationError(
            f"event index must be sorted by {EVENT_KEY_COLS}"
        )


def build_event_index(
    spec: EventIndexSpec,
    *,
    roots: DataRoots,
    force: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> EventIndexBuildReport:
    """Build or incrementally refresh a versioned event index.

    Existing dates are skipped only when their selector version and declared
    source identities still match.  ``force`` with no date bounds performs a
    full rebuild; with bounds it replaces only that inclusive range.
    """

    _validate_bounds(start, end)
    spec.index_path.parent.mkdir(parents=True, exist_ok=True)

    index_existed = spec.index_path.is_file()
    full_rebuild = force and start is None and end is None
    existing, prior_manifest, legacy_adoption = _load_existing(
        spec, full_rebuild=full_rebuild
    )
    prior_processed = set(prior_manifest.get("processed_dates", ())) if prior_manifest else set()
    if existing.height and not prior_processed:
        prior_processed = set(existing["Date"].unique().to_list())

    calendar_dates = _calendar_dates(roots, spec.calendar_source, start=start, end=end)
    identity_dates = set(calendar_dates)
    if legacy_adoption:
        # Adoption is deliberately metadata-only.  Bind every pre-existing date
        # to its current declared inputs, including dates outside start/end, so
        # a later source mutation can trigger a real selector refresh.
        identity_dates.update(prior_processed)
    current_identities = {
        date: _day_source_identities(EventDaySources(date, roots), spec.required_sources)
        for date in sorted(identity_dates)
    }
    prior_identities = prior_manifest.get("source_identities", {}) if prior_manifest else {}
    stale_dates = {
        date
        for date in calendar_dates
        if date in prior_processed
        and date in prior_identities
        and prior_identities[date] != current_identities[date]
    }

    if legacy_adoption:
        # The first adoption call must never rewrite a legacy artifact.  Dates
        # absent from that artifact are handled on the next ordinary build.
        scan_dates = []
    elif force:
        scan_dates = calendar_dates
    else:
        scan_dates = [
            date
            for date in calendar_dates
            if date not in prior_processed or date in stale_dates
        ]
    scan_set = set(scan_dates)

    if full_rebuild:
        base = empty_event_index()
        processed_dates: set[str] = set()
        source_identities: dict[str, object] = {}
        unverified_dates: set[str] = set()
    else:
        base = existing.filter(~pl.col("Date").is_in(scan_dates)) if scan_dates else existing
        processed_dates = prior_processed - scan_set
        source_identities = {
            date: identity
            for date, identity in prior_identities.items()
            if date not in scan_set
        }
        unverified_dates = set(prior_manifest.get("unverified_dates", ()))
        if legacy_adoption:
            unverified_dates = set(prior_processed)
            source_identities.update(
                {
                    date: current_identities[date]
                    for date in sorted(prior_processed)
                }
            )

    parts: list[pl.DataFrame] = []
    for date in scan_dates:
        day = EventDaySources(date, roots)
        day.require(*spec.required_sources)
        try:
            selected = spec.selector(day)
        except Exception as exc:
            raise EventIndexError(f"{date}: event selector failed") from exc
        parts.append(normalize_event_keys(selected, date=date))
        processed_dates.add(date)
        source_identities[date] = current_identities[date]
        unverified_dates.discard(date)

    frames = [base, *parts]
    result = pl.concat(frames, how="vertical") if len(frames) > 1 else frames[0]
    result = result.sort(EVENT_KEY_COLS)
    validate_event_index(result, require_sorted=True)

    index_changed = not index_existed or bool(scan_dates) or full_rebuild
    if index_changed:
        _atomic_write_parquet(result, spec.index_path)
    rows_by_date = {
        row["Date"]: row["len"]
        for row in result.group_by("Date").len().sort("Date").to_dicts()
    }
    manifest = {
        "format_version": EVENT_INDEX_FORMAT_VERSION,
        "event_id": spec.event_id,
        "selector_version": spec.selector_version,
        "key_columns": list(EVENT_KEY_COLS),
        "key_schema": {name: str(dtype) for name, dtype in _EVENT_SCHEMA.items()},
        "processed_dates": sorted(processed_dates),
        "source_identities": dict(sorted(source_identities.items())),
        "rows_by_date": rows_by_date,
        "row_count": result.height,
        "unique_key_count": result.height,
        "event_date_count": len(rows_by_date),
        "start_date": result["Date"].min() if result.height else None,
        "end_date": result["Date"].max() if result.height else None,
        "legacy_adopted": legacy_adoption
        or bool(prior_manifest.get("legacy_adopted", False)),
        "selector_reproduction_verified": not unverified_dates,
        "unverified_dates": sorted(unverified_dates),
    }
    if manifest != prior_manifest:
        _atomic_write_json(manifest, spec.manifest_path)

    skipped_dates = tuple(date for date in calendar_dates if date not in scan_set)
    return EventIndexBuildReport(
        event_id=spec.event_id,
        selector_version=spec.selector_version,
        index_path=spec.index_path,
        manifest_path=spec.manifest_path,
        row_count=result.height,
        unique_key_count=result.height,
        event_date_count=len(rows_by_date),
        processed_date_count=len(processed_dates),
        start_date=manifest["start_date"],
        end_date=manifest["end_date"],
        scanned_dates=tuple(scan_dates),
        refreshed_dates=tuple(date for date in scan_dates if date in stale_dates),
        skipped_dates=skipped_dates,
    )


def _load_existing(
    spec: EventIndexSpec,
    *,
    full_rebuild: bool,
) -> tuple[pl.DataFrame, dict[str, object], bool]:
    if full_rebuild:
        return empty_event_index(), {}, False

    index_exists = spec.index_path.is_file()
    manifest_exists = spec.manifest_path.is_file()
    if manifest_exists and not index_exists:
        raise EventIndexManifestError(
            f"orphan event-index manifest without parquet: {spec.manifest_path}"
        )
    if not index_exists:
        return empty_event_index(), {}, False

    if not manifest_exists and not spec.allow_unmanifested_existing:
        raise EventIndexManifestError(
            f"existing index has no manifest: {spec.index_path}; use force=True to rebuild"
        )

    existing_raw = pl.read_parquet(spec.index_path)
    if manifest_exists:
        manifest = _read_json(spec.manifest_path)
        _validate_manifest_identity(manifest, spec)
        validate_event_index(existing_raw, require_sorted=True)
        if manifest.get("row_count") != existing_raw.height:
            raise EventIndexManifestError(
                "event-index manifest row_count does not match the parquet artifact"
            )
        return existing_raw, manifest, False

    # Explicit compatibility bridge for a known legacy artifact.  It still has
    # to satisfy the canonical contract before it can be adopted.  Do not sort,
    # cast, or rewrite it implicitly: a non-canonical artifact needs an explicit
    # migration rather than a surprising in-place mutation.
    validate_event_index(existing_raw, require_sorted=True)
    return existing_raw, {}, True


def _validate_manifest_identity(
    manifest: dict[str, object],
    spec: EventIndexSpec,
) -> None:
    expected = {
        "format_version": EVENT_INDEX_FORMAT_VERSION,
        "event_id": spec.event_id,
        "selector_version": spec.selector_version,
        "key_columns": list(EVENT_KEY_COLS),
        "key_schema": {name: str(dtype) for name, dtype in _EVENT_SCHEMA.items()},
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise EventIndexManifestError(
            f"existing event index is incompatible with the requested spec: {mismatches}"
        )


def _calendar_dates(
    roots: DataRoots,
    source: SourceName,
    *,
    start: str | None,
    end: str | None,
) -> list[str]:
    source_dir = roots.source_dir(source)
    paths = sorted(source_dir.glob(_SOURCE_GLOBS[source]))
    if not paths:
        raise FileNotFoundError(
            f"no {source} calendar files found in {source_dir}"
        )

    dates: list[str] = []
    for path in paths:
        date = path.name.split("_", 1)[0]
        _validate_date(date, f"calendar filename {path.name}")
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        dates.append(date)
    return sorted(set(dates))


def _day_source_identities(
    day: EventDaySources,
    sources: tuple[SourceName, ...],
) -> dict[str, object]:
    day.require(*sources)
    identities: dict[str, object] = {}
    for source in sources:
        path = day.path(source)
        stat = path.stat()
        identities[source] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return identities


def _validate_bounds(start: str | None, end: str | None) -> None:
    if start is not None:
        _validate_date(start, "start")
    if end is not None:
        _validate_date(end, "end")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start must be <= end, got {start} > {end}")


def _validate_date(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise EventIndexValidationError(f"{label} must be a YYYYMMDD string, got {value!r}")
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise EventIndexValidationError(
            f"{label} must be a valid YYYYMMDD date, got {value!r}"
        ) from exc
    if parsed.strftime("%Y%m%d") != value:
        raise EventIndexValidationError(
            f"{label} must be exactly eight YYYYMMDD digits, got {value!r}"
        )


def _read_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EventIndexManifestError(f"cannot read event-index manifest: {path}") from exc
    if not isinstance(value, dict):
        raise EventIndexManifestError(f"event-index manifest must be an object: {path}")
    return value


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        frame.write_parquet(temp_path, compression="zstd")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_json(value: dict[str, object], path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
